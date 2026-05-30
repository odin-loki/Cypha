"""Full CyphaLM stack: embed -> SSM -> DIF -> GRIA."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cypha_lm.config import CyphaLMConfig
from cypha_lm.embeddings.izaac_embed import IzaacEmbedding
from cypha_lm.expert_field.cypha_dif import CyphaDIF
from cypha_lm.projection.gria_projection import GRIAProjection
from cypha_lm.temporal.cellai_ssm import CellAISSM


class CyphaLM:
    """
    Pipeline: token -> IzaacEmbedding -> CellAISSM -> CyphaDIF -> GRIAProjection.

    Online CE gradients on GRIA; NIG updates in DIF; optional SSM online rule.
    """

    def __init__(self, config: CyphaLMConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self.embed = IzaacEmbedding(
            vocab_size=config.vocab_size,
            d_embed=config.d_embed,
            seed=config.seed,
        )
        self.ssm = CellAISSM(
            d_input=config.d_embed,
            d_state=config.d_state,
            tau_fast=config.tau_fast,
            tau_slow=config.tau_slow,
            n_layers=config.ssm_layers,
            seed=config.seed + 1,
        )
        self.dif = CyphaDIF(config)
        ctx_dim = self.ssm.context_dim
        fd = config.field_dim
        rng = self._rng
        self._proj_ssm = rng.standard_normal((fd, ctx_dim)) * 0.02
        self._proj_dif = rng.standard_normal((fd, fd)) * 0.02
        self._proj_embed = rng.standard_normal((fd, config.d_embed)) * 0.02
        gria_in = fd
        self.gria = GRIAProjection(
            d_input=gria_in,
            vocab_size=config.vocab_size,
            alpha_init=config.alpha_init,
            alpha_learnable=config.alpha_learnable,
            seed=config.seed + 2,
        )
        self._token_counts = np.ones(config.vocab_size, dtype=np.float64)
        self.gria.set_unigram_prior(self._token_counts)
        self._last_epistemic = 0.0
        self._last_aleatoric = 0.0

    def reset_context(self) -> None:
        self.ssm.reset()

    def _project_context(self, context: np.ndarray) -> np.ndarray:
        return self._proj_ssm @ np.asarray(context, dtype=np.float64).ravel()

    def _gria_input(self, field_x: np.ndarray, dif_out: dict) -> np.ndarray:
        mean = np.asarray(dif_out["mean"], dtype=np.float64).ravel()
        mean = self._proj_dif @ np.resize(mean, self.config.field_dim)
        u = float(dif_out["epistemic_var"])
        return mean + u * field_x * 0.01

    def _forward_context(self, token_id: int, update_dif: bool = False) -> dict[str, Any]:
        e = self.embed.embed(token_id)
        ctx = self.ssm.step(e)
        field_x = self._project_context(ctx)
        dif_out = self.dif.predict(field_x)
        v = self._gria_input(field_x, dif_out)
        log_probs = self.gria.forward(v)
        self._last_epistemic = float(dif_out["epistemic_var"])
        self._last_aleatoric = float(dif_out["aleatoric_var"])
        return {
            "log_probs": log_probs,
            "v": v,
            "field_x": field_x,
            "ctx": ctx,
            "dif_out": dif_out,
            "embedding": e,
        }

    def _ssm_online_update(self, e: np.ndarray, ctx: np.ndarray) -> None:
        """Simple Hebbian-style nudge on fast weights (optional)."""
        if not self.config.train_ssm:
            return
        lr = self.config.ssm_lr
        e = np.asarray(e, dtype=np.float64).ravel()
        for layer in range(self.ssm.n_layers):
            h = self.ssm._h[layer]
            delta = lr * np.outer(h, e)
            self.ssm.W_fast[layer] += delta * 0.01

    def train_step(self, token_id: int, next_token_id: int) -> dict[str, Any]:
        fwd = self._forward_context(token_id, update_dif=False)
        v = fwd["v"]
        field_x = fwd["field_x"]
        nxt = int(next_token_id)
        log_probs = fwd["log_probs"]
        loss = float(-log_probs[nxt])
        gw, ga, gb = self.gria.cross_entropy_gradients(v, nxt)
        lr = self.config.gria_lr
        self.gria.update_weights(gw, lr)
        self.gria.update_alpha(ga, lr)
        self.gria.update_bias(gb, lr)
        if self.config.online:
            target_vec = self._proj_embed @ self.embed.embed(nxt)
            self.dif.train_step(field_x, target_vec)
        self._ssm_online_update(fwd["embedding"], fwd["ctx"])
        self._token_counts[nxt] += 1.0
        alpha_gria = self.gria.grand_unified_law_alpha(fwd["field_x"], np.exp(log_probs))
        return {
            "loss": loss,
            "epistemic_var": self._last_epistemic,
            "aleatoric_var": self._last_aleatoric,
            "active_experts": int(fwd["dif_out"]["active_experts"]),
            "alpha_gria": float(alpha_gria),
        }

    def train_sequence(self, token_ids: list[int]) -> dict[str, Any]:
        ids = [int(t) for t in token_ids]
        losses, epi, ale, active, alpha = [], [], [], [], []
        self.reset_context()
        for t in range(len(ids) - 1):
            m = self.train_step(ids[t], ids[t + 1])
            losses.append(m["loss"])
            epi.append(m["epistemic_var"])
            ale.append(m["aleatoric_var"])
            active.append(m["active_experts"])
            alpha.append(m["alpha_gria"])
        return {
            "loss": np.asarray(losses, dtype=np.float64),
            "epistemic_var": np.asarray(epi, dtype=np.float64),
            "aleatoric_var": np.asarray(ale, dtype=np.float64),
            "active_experts": np.asarray(active, dtype=np.int64),
            "alpha_gria": np.asarray(alpha, dtype=np.float64),
        }

    def predict_next(self, token_id: int) -> dict[str, Any]:
        fwd = self._forward_context(int(token_id))
        log_probs = fwd["log_probs"]
        top_k = 10
        idx = np.argsort(log_probs)[::-1][:top_k]
        return {
            "log_probs": log_probs,
            "epistemic_var": self._last_epistemic,
            "aleatoric_var": self._last_aleatoric,
            "top_k_tokens": [int(i) for i in idx],
            "top_k_probs": [float(np.exp(log_probs[i])) for i in idx],
        }

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int = 100,
        temperature: float = 1.0,
        uncertainty_threshold: float | None = None,
    ) -> dict[str, Any]:
        from cypha_lm.model.generation import (
            temperature_sample,
            uncertainty_gated_sample,
        )

        self.reset_context()
        prompt = [int(t) for t in prompt_ids]
        for tid in prompt[:-1]:
            self._forward_context(tid)
        generated: list[int] = []
        losses, epi, ale = [], [], []
        last = prompt[-1] if prompt else 0
        temp = max(float(temperature), 1e-6)
        for _ in range(max_tokens):
            pred = self.predict_next(last)
            ep = float(pred["epistemic_var"])
            epi.append(ep)
            ale.append(float(pred["aleatoric_var"]))
            if uncertainty_threshold is not None and ep > float(uncertainty_threshold):
                break
            log_probs = np.asarray(pred["log_probs"], dtype=np.float64)
            if temperature <= 1e-6:
                tid = int(pred["top_k_tokens"][0])
            else:
                probs = np.exp(log_probs / temp)
                probs /= probs.sum() + 1e-12
                tid = int(self._rng.choice(self.config.vocab_size, p=probs))
            generated.append(tid)
            losses.append(float(-log_probs[tid]))
            last = tid
        return {
            "generated_ids": generated,
            "per_step_metrics": {
                "loss": np.asarray(losses, dtype=np.float64),
                "epistemic_var": np.asarray(epi, dtype=np.float64),
                "aleatoric_var": np.asarray(ale, dtype=np.float64),
            },
        }

    def compression_profile(self) -> dict[str, Any]:
        spec = self.gria.alpha_spectrum()
        expert_alpha = self.dif.alpha_per_expert()
        total = self._last_epistemic + self._last_aleatoric + 1e-12
        return {
            "mean_epistemic_var": self._last_epistemic,
            "mean_aleatoric_var": self._last_aleatoric,
            "mean_alpha": spec["mean"],
            "expert_alpha_spectrum": expert_alpha,
            "gria_alpha_spectrum": spec,
            "n_experts": self.dif.expert_count(),
            "lossless_fraction": float(self._last_epistemic / total),
            "lossy_fraction": float(1.0 - self._last_epistemic / total),
        }

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: CyphaLM._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [CyphaLM._json_safe(v) for v in obj]
        return obj

    def save(self, path: str) -> None:
        base = Path(path)
        if base.suffix in (".json", ".npz"):
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        json_path = base.with_suffix(".json")
        npz_path = base.with_suffix(".npz")
        meta = self._json_safe(
            {
                "config": asdict(self.config),
                "token_counts": self._token_counts,
                "gria": self.gria.get_state(),
                "ssm": self.ssm.get_state(),
                "dif": self.dif.get_state(),
            }
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        np.savez(
            npz_path,
            proj_ssm=self._proj_ssm,
            proj_dif=self._proj_dif,
            proj_embed=self._proj_embed,
        )

    @classmethod
    def load(cls, path: str) -> "CyphaLM":
        p = Path(path)
        json_path = p if p.suffix == ".json" else p.with_suffix(".json")
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)
        cfg = CyphaLMConfig(**meta["config"])
        model = cls(cfg)
        model._token_counts = np.asarray(meta["token_counts"], dtype=np.float64)
        model.gria.set_state(meta["gria"])
        model.ssm.set_state(meta["ssm"])
        model.dif.set_state(meta["dif"])
        npz_path = json_path.with_suffix(".npz")
        if npz_path.exists():
            data = np.load(npz_path)
            model._proj_ssm = np.asarray(data["proj_ssm"], dtype=np.float64)
            model._proj_dif = np.asarray(data["proj_dif"], dtype=np.float64)
            if "proj_embed" in data:
                model._proj_embed = np.asarray(data["proj_embed"], dtype=np.float64)
        return model
