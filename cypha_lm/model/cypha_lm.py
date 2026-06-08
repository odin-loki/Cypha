"""Full CyphaLM stack: embed -> SSM -> DIF -> GRIA."""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cypha_views.runner import iter_view_epochs
from cypha_views.schedule import resolve_schedule
from cypha_views.types import ViewSchedule, ViewSpec

from cypha_lm.array_backend import ArrayBackend, asnumpy
from cypha_lm.config import CyphaLMConfig
from cypha_lm.embeddings.izaac_embed import IzaacEmbedding
from cypha_lm.embeddings.view_embed import ViewEmbedding
from cypha_lm.expert_field.cypha_dif import CyphaDIF
from cypha_lm.projection.gria_projection import GRIAProjection
from cypha_lm.projection.ngram_fusion import NgramFusion
from cypha_lm.model.char_lstm_head import CharLSTMHead, blend_log_probs, blend_logit_grad
from cypha_lm.temporal.cellai_ssm import CellAISSM


class CyphaLM:
    """
    Pipeline: token -> IzaacEmbedding -> CellAISSM -> CyphaDIF -> GRIAProjection.

    Online CE gradients on GRIA; NIG updates in DIF; optional SSM online rule.
    """

    def __init__(self, config: CyphaLMConfig) -> None:
        self.config = config
        self._backend = ArrayBackend(config.device)
        xp = self._backend.xp
        self._rng = np.random.default_rng(config.seed)
        self.embed = IzaacEmbedding(
            vocab_size=config.vocab_size,
            d_embed=config.d_embed,
            seed=config.seed,
            xp=xp,
        )
        self.ssm = CellAISSM(
            d_input=config.d_embed,
            d_state=config.d_state,
            tau_fast=config.tau_fast,
            tau_slow=config.tau_slow,
            n_layers=config.ssm_layers,
            seed=config.seed + 1,
            use_spectral_pde=config.use_spectral_pde,
            use_multiscale=config.use_multiscale,
            use_sparse_hebbian=config.use_sparse_hebbian,
            xp=xp,
        )
        self.dif = CyphaDIF(config)
        ctx_dim = self.ssm.context_dim
        fd = config.field_dim
        rng = self._rng
        self._proj_ssm = xp.asarray(rng.standard_normal((fd, ctx_dim)) * 0.02)
        self._proj_dif = xp.asarray(rng.standard_normal((fd, fd)) * 0.02)
        self._proj_embed = xp.asarray(rng.standard_normal((fd, config.d_embed)) * 0.02)
        n_embeds = 1 + max(0, int(config.ngram_context))
        ngram_in = (0 if config.context_mode == "ablation_no_ssm" else fd) + n_embeds * config.d_embed
        self._ngram_fuse_split = bool(config.ngram_fuse_split) and config.context_mode in (
            "gria_ngram",
            "hybrid_gria_lstm",
            "ablation_no_ssm",
        )
        self._ngram_fusion: NgramFusion | None = None
        if self._ngram_fuse_split:
            embed_in = n_embeds * config.d_embed if config.context_mode != "ablation_no_ssm" else ngram_in
            field_in = fd if config.context_mode != "ablation_no_ssm" else 0
            fuse_mode = config.ngram_fusion if config.ngram_fusion in ("sum", "gated") else "sum"
            if field_in > 0:
                self._ngram_fusion = NgramFusion(
                    fd,
                    field_in,
                    embed_in,
                    mode=fuse_mode,
                    n_positions=n_embeds,
                    position_weights=bool(config.ngram_position_weights),
                    seed=config.seed + 4,
                    xp=xp,
                )
                self._proj_ngram_field = None
                self._proj_ngram_embed = None
            else:
                self._proj_ngram_field = None
                self._proj_ngram_embed = xp.asarray(rng.standard_normal((fd, embed_in)) * 0.02)
            self._proj_ngram = xp.asarray(rng.standard_normal((fd, ngram_in)) * 0.02)
        else:
            self._proj_ngram_field = None
            self._proj_ngram_embed = None
            self._proj_ngram = xp.asarray(rng.standard_normal((fd, ngram_in)) * 0.02)
        vid = max(0, int(config.view_id_dim))
        self._view_slots = max(1, int(config.max_view_slots)) if vid > 0 else 0
        if vid > 0:
            self.view_emb = ViewEmbedding(
                self._view_slots,
                vid,
                seed=config.seed + 3,
                learnable=bool(config.view_learnable),
                xp=xp,
            )
        else:
            self.view_emb = None
        gria_in = fd + vid
        self.gria = GRIAProjection(
            d_input=gria_in,
            vocab_size=config.vocab_size,
            alpha_init=config.alpha_init,
            alpha_learnable=config.alpha_learnable,
            seed=config.seed + 2,
            xp=xp,
        )
        self._token_counts = np.ones(config.vocab_size, dtype=np.float64)
        if config.laplace_smoothing > 0:
            self.gria.set_laplace_prior(
                self._token_counts, smoothing=config.laplace_smoothing
            )
        else:
            self.gria.set_unigram_prior(self._token_counts)
        self._embed_history: deque[Any] = deque(maxlen=1 + max(0, int(config.ngram_context)))
        self._bptt_buffer: list[tuple[Any, Any, Any]] = []
        self._last_epistemic = 0.0
        self._last_aleatoric = 0.0
        self._current_view_id = 0
        self._current_view_slot = 0
        self._hybrid_blend_logit = float(config.hybrid_blend_logit)
        if config.context_mode == "hybrid_gria_lstm":
            self.lstm_head = CharLSTMHead(
                config.vocab_size,
                int(config.lstm_hidden),
                seed=config.seed + 5,
            )
            self._lstm_h, self._lstm_c = self.lstm_head.reset_state()
        else:
            self.lstm_head = None
            self._lstm_h = None
            self._lstm_c = None

    @property
    def device(self) -> str:
        return self._backend.device

    def reset_context(self) -> None:
        self.ssm.reset()
        self._embed_history.clear()
        self._bptt_buffer.clear()
        if self.lstm_head is not None:
            self._lstm_h, self._lstm_c = self.lstm_head.reset_state()

    def _project_context(self, context) -> Any:
        xp = self._backend.xp
        ctx = xp.asarray(context, dtype=xp.float64).ravel()
        return self._proj_ssm @ ctx

    def _record_embedding(self, embedding) -> None:
        self._embed_history.appendleft(embedding)

    def _ngram_embedding_vector(self) -> Any:
        xp = self._backend.xp
        fd = self.config.field_dim
        d = self.config.d_embed
        n_prev = max(0, int(self.config.ngram_context))
        parts: list[Any] = []
        for i in range(n_prev + 1):
            if i < len(self._embed_history):
                parts.append(xp.asarray(self._embed_history[i], dtype=xp.float64).ravel())
            else:
                parts.append(xp.zeros(d, dtype=xp.float64))
        return xp.concatenate(parts)

    def _ngram_gria_vector(self, field_x) -> Any:
        xp = self._backend.xp
        mode = self.config.context_mode
        embeds = self._ngram_embedding_vector()
        if self._ngram_fusion is not None:
            return self._ngram_fusion.forward(
                xp.asarray(field_x, dtype=xp.float64).ravel(), embeds
            )
        if self._ngram_fuse_split and self._proj_ngram_embed is not None:
            field_part = self._proj_ngram_field @ xp.asarray(
                field_x, dtype=xp.float64
            ).ravel() if self._proj_ngram_field is not None else xp.zeros(self.config.field_dim)
            embed_part = self._proj_ngram_embed @ embeds
            return field_part + embed_part
        if mode == "ablation_no_ssm":
            concat = embeds
        else:
            concat = xp.concatenate(
                [xp.asarray(field_x, dtype=xp.float64).ravel(), embeds]
            )
        return self._proj_ngram @ concat

    def _view_vector(self) -> Any:
        xp = self._backend.xp
        if self.view_emb is None:
            return xp.zeros(max(0, int(self.config.view_id_dim)), dtype=xp.float64)
        return self.view_emb.forward(self._current_view_slot)

    def _augment_gria_input(self, v: Any) -> Any:
        xp = self._backend.xp
        vid = max(0, int(self.config.view_id_dim))
        if vid <= 0:
            return v
        return xp.concatenate(
            [xp.asarray(v, dtype=xp.float64).ravel(), self._view_vector()]
        )

    def _gria_input(self, field_x, dif_out: dict) -> Any:
        xp = self._backend.xp
        mode = self.config.context_mode
        if mode in ("gria_ngram", "hybrid_gria_lstm", "ablation_no_ssm"):
            return self._ngram_gria_vector(field_x)
        if mode == "ssm_only":
            return xp.asarray(field_x, dtype=xp.float64).ravel()
        mean = xp.asarray(dif_out["mean"], dtype=xp.float64).ravel()
        mean = self._proj_dif @ xp.resize(mean, self.config.field_dim)
        if mode == "ablation_no_dif":
            return mean
        u = float(dif_out["epistemic_var"])
        return mean + u * field_x * 0.01

    def _forward_context(self, token_id: int, update_dif: bool = False) -> dict[str, Any]:
        xp = self._backend.xp
        mode = self.config.context_mode
        e = self.embed.embed(token_id)
        self._record_embedding(e)
        if mode == "ablation_no_ssm":
            ctx = xp.zeros(self.ssm.context_dim, dtype=xp.float64)
            field_x = xp.zeros(self.config.field_dim, dtype=xp.float64)
        else:
            ctx = self.ssm.step(e)
            field_x = self._project_context(ctx)
        if mode in ("ssm_only", "ablation_no_ssm"):
            dif_out = {
                "mean": xp.zeros(self.config.field_dim, dtype=xp.float64),
                "epistemic_var": 0.0,
                "aleatoric_var": 0.0,
                "routing_probs": np.array([], dtype=np.float64),
                "active_experts": 0,
            }
        else:
            dif_out = self.dif.predict(asnumpy(field_x))
        v = self._gria_input(field_x, dif_out)
        v = self._augment_gria_input(v)
        log_probs = self.gria.forward(v)
        log_probs_gria = log_probs
        log_probs_lstm = None
        if self.lstm_head is not None and self._lstm_h is not None and self._lstm_c is not None:
            log_l, self._lstm_h, self._lstm_c = self.lstm_head.forward(
                int(token_id), self._lstm_h, self._lstm_c
            )
            log_probs_lstm = log_l
            log_g = asnumpy(log_probs_gria)
            log_probs = blend_log_probs(log_g, log_l, self._hybrid_blend_logit)
            if self._backend.is_cuda:
                log_probs = self._backend.xp.asarray(log_probs, dtype=self._backend.xp.float64)
            else:
                log_probs = np.asarray(log_probs, dtype=np.float64)
        self._last_epistemic = float(dif_out["epistemic_var"])
        self._last_aleatoric = float(dif_out["aleatoric_var"])
        return {
            "log_probs": log_probs,
            "log_probs_gria": log_probs_gria,
            "log_probs_lstm": log_probs_lstm,
            "v": v,
            "field_x": field_x,
            "ctx": ctx,
            "dif_out": dif_out,
            "embedding": e,
        }

    def _ssm_online_update(self, e, ctx) -> None:
        """Simple Hebbian-style nudge on fast weights (optional, layer 0 only)."""
        if not self.config.train_ssm:
            return
        xp = self._backend.xp
        lr = self.config.ssm_lr
        e = xp.asarray(e, dtype=xp.float64).ravel()
        if self.ssm.n_layers < 1:
            return
        h = self.ssm._h[0]
        in_dim = self.ssm._layer_input_dims[0]
        if e.size != in_dim:
            return
        delta = lr * xp.outer(h, e)
        norm = float(xp.linalg.norm(delta))
        if norm > 0.1:
            delta = delta * (0.1 / norm)
        self.ssm.W_fast[0] += delta * 0.01

    def _grad_v_field_core(self, grad_v: Any) -> Any:
        """GRIA input may include view_id tail; backprop to field path uses core dims only."""
        xp = self._backend.xp
        g = xp.asarray(grad_v, dtype=xp.float64).ravel()
        fd = self.config.field_dim
        return g[:fd]

    def _bptt_ssm_update(self, fwd: dict[str, Any], next_token_id: int) -> None:
        """Approximate truncated BPTT: backprop GRIA loss to SSM layer-0 fast weights."""
        steps = int(self.config.bptt_steps)
        if steps <= 0 or self.config.context_mode == "ablation_no_ssm":
            return
        xp = self._backend.xp
        v = fwd["v"]
        nxt = int(next_token_id)
        grad_v = self._grad_v_field_core(self.gria.grad_v_cross_entropy(v, nxt))
        mode = self.config.context_mode
        fd = self.config.field_dim

        if mode == "ssm_only":
            grad_field = grad_v
        elif mode in ("gria_ngram", "hybrid_gria_lstm"):
            if self._ngram_fusion is not None:
                grad_field = self._ngram_fusion.grad_field_x(grad_v)
            elif self._ngram_fuse_split and self._proj_ngram_field is not None:
                grad_field = self._proj_ngram_field.T @ grad_v
            else:
                grad_concat = self._proj_ngram.T @ grad_v
                grad_field = grad_concat[:fd]
        elif mode == "full":
            u = float(fwd["dif_out"]["epistemic_var"])
            grad_field = grad_v * (u * 0.01)
        else:
            return

        grad_ctx = self._proj_ssm.T @ grad_field
        if self.ssm.n_layers < 1:
            return
        e = xp.asarray(fwd["embedding"], dtype=xp.float64).ravel()
        in_dim = self.ssm._layer_input_dims[0]
        d_state = self.ssm.d_state
        if e.size != in_dim or grad_ctx.size != self.ssm.context_dim:
            return
        grad_h = grad_ctx[:d_state]
        lf = float(np.clip(self.ssm.lambda_fast, 0.01, 0.999))
        delta = (1.0 - lf) * xp.outer(grad_h, e)
        self._bptt_buffer.append(delta.copy())
        if len(self._bptt_buffer) < steps:
            return
        avg = xp.zeros_like(self.ssm.W_fast[0])
        for d in self._bptt_buffer:
            avg += d
        avg /= float(len(self._bptt_buffer))
        self.ssm.W_fast[0] -= float(self.config.ssm_lr) * avg * 0.001
        self._bptt_buffer.clear()

    def train_step(self, token_id: int, next_token_id: int, *, gria_lr: float | None = None) -> dict[str, Any]:
        fwd = self._forward_context(token_id, update_dif=False)
        v = fwd["v"]
        field_x = fwd["field_x"]
        nxt = int(next_token_id)
        log_probs = fwd["log_probs"]
        if self._backend.is_cuda:
            loss = float(-asnumpy(log_probs)[nxt])
        else:
            loss = float(-log_probs[nxt])
        gw, ga, gb = self.gria.cross_entropy_gradients(v, nxt)
        lr = float(self.config.gria_lr if gria_lr is None else gria_lr)
        self.gria.update_weights(gw, lr)
        self.gria.update_alpha(ga, lr)
        self.gria.update_bias(gb, lr)
        if self.view_emb is not None and self.config.view_learnable:
            fd = self.config.field_dim
            xp = self._backend.xp
            grad_v = self.gria.grad_v_cross_entropy(v, nxt)
            grad_view = xp.asarray(grad_v, dtype=xp.float64).ravel()[fd:]
            norm = float(xp.linalg.norm(grad_view))
            if norm > 0.05:
                grad_view = grad_view * (0.05 / norm)
            if gria_lr is not None and self.config.gria_lr > 0:
                view_lr = float(self.config.view_lr) * (float(gria_lr) / float(self.config.gria_lr))
            else:
                view_lr = float(self.config.view_lr)
            self.view_emb.update(self._current_view_slot, grad_view, view_lr)
        if self.config.online and self.config.context_mode not in (
            "ssm_only",
            "ablation_no_ssm",
        ):
            target_vec = self._proj_embed @ self.embed.embed(nxt)
            self.dif.train_step(asnumpy(field_x), asnumpy(target_vec))
        if self.config.bptt_steps > 0:
            self._bptt_ssm_update(fwd, nxt)
        elif self.config.train_ssm:
            self._ssm_online_update(fwd["embedding"], fwd["ctx"])
        if self.lstm_head is not None and fwd.get("log_probs_lstm") is not None:
            self.lstm_head.backward(nxt, float(self.config.lstm_lr))
            if self.config.hybrid_blend_learnable:
                log_g = asnumpy(fwd["log_probs_gria"])
                log_l = asnumpy(fwd["log_probs_lstm"])
                grad_logit = blend_logit_grad(log_g, log_l, self._hybrid_blend_logit, nxt)
                self._hybrid_blend_logit -= float(self.config.hybrid_blend_lr) * grad_logit
        self._token_counts[nxt] += 1.0
        if self.config.laplace_smoothing > 0:
            self.gria.set_laplace_prior(
                self._token_counts, smoothing=self.config.laplace_smoothing
            )
        alpha_gria = self.gria.grand_unified_law_alpha(
            asnumpy(fwd["field_x"]), np.exp(asnumpy(log_probs))
        )
        return {
            "loss": loss,
            "epistemic_var": self._last_epistemic,
            "aleatoric_var": self._last_aleatoric,
            "active_experts": int(fwd["dif_out"]["active_experts"]),
            "alpha_gria": float(alpha_gria),
        }

    def train_sequence_views(
        self,
        token_ids: list[int],
        schedule: list[ViewSpec] | ViewSchedule | None = None,
    ) -> dict[str, Any]:
        """Train over an explicit or config-resolved multi-view schedule."""
        ids = [int(t) for t in token_ids]
        if schedule is None:
            view_schedule = resolve_schedule(
                self.config.view_schedule,
                seed=self.config.seed,
                train_epochs=self.config.train_epochs,
            )
        elif isinstance(schedule, ViewSchedule):
            view_schedule = schedule
        else:
            view_schedule = ViewSchedule(views=list(schedule), seed=self.config.seed)

        losses, epi, ale, active, alpha = [], [], [], [], []
        base_lr = float(self.config.gria_lr)
        block_size = max(1, int(self.config.view_block_size))
        last_macro = -1
        log_every = int(os.environ.get("CYPHA_LM_TRAIN_LOG_EVERY", "25000"))
        step_i = 0
        for view_spec, macro_index, segment, reset_before in iter_view_epochs(
            ids,
            view_schedule,
            char_newline_id=None,
            block_size=block_size,
        ):
            macro_lr = base_lr * (float(self.config.gria_lr_decay) ** macro_index)
            self._current_view_id = hash(view_spec.view_id) & 0x7FFFFFFF
            if self.view_emb is not None:
                self._current_view_slot = self.view_emb.slot_for_view(view_spec.name)
            if macro_index != last_macro:
                self.reset_context()
                last_macro = macro_index
            elif reset_before:
                self.reset_context()
                self._embed_history.clear()
                self._bptt_buffer.clear()
            for t in range(len(segment) - 1):
                m = self.train_step(segment[t], segment[t + 1], gria_lr=macro_lr)
                losses.append(m["loss"])
                epi.append(m["epistemic_var"])
                ale.append(m["aleatoric_var"])
                active.append(m["active_experts"])
                alpha.append(m["alpha_gria"])
                step_i += 1
                if log_every > 0 and step_i % log_every == 0:
                    print(
                        f"[CyphaLM] train step {step_i} view={view_spec.name} "
                        f"loss={m['loss']:.4f}",
                        flush=True,
                    )
        return {
            "loss": np.asarray(losses, dtype=np.float64),
            "epistemic_var": np.asarray(epi, dtype=np.float64),
            "aleatoric_var": np.asarray(ale, dtype=np.float64),
            "active_experts": np.asarray(active, dtype=np.int64),
            "alpha_gria": np.asarray(alpha, dtype=np.float64),
        }

    def train_sequence(self, token_ids: list[int]) -> dict[str, Any]:
        return self.train_sequence_views(token_ids)

    def predict_next(self, token_id: int) -> dict[str, Any]:
        fwd = self._forward_context(int(token_id))
        log_probs = asnumpy(fwd["log_probs"])
        dif_out = fwd["dif_out"]
        routing = np.asarray(dif_out["routing_probs"], dtype=np.float64)
        top_k = 10
        idx = np.argsort(log_probs)[::-1][:top_k]
        dominant = int(np.argmax(routing)) if routing.size else 0
        return {
            "log_probs": log_probs,
            "epistemic_var": self._last_epistemic,
            "aleatoric_var": self._last_aleatoric,
            "top_k_tokens": [int(i) for i in idx],
            "top_k_probs": [float(np.exp(log_probs[i])) for i in idx],
            "active_experts": int(dif_out["active_experts"]),
            "routing_probs": routing.tolist(),
            "dominant_expert": dominant,
        }

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int = 100,
        temperature: float = 1.0,
        uncertainty_threshold: float | None = None,
        *,
        strategy: str = "temperature",
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> dict[str, Any]:
        from cypha_lm.model.generation import autoregressive_decode

        strat = str(strategy)
        if uncertainty_threshold is not None and strat == "temperature":
            strat = "uncertainty_gated"
        out = autoregressive_decode(
            self,
            [int(t) for t in prompt_ids],
            int(max_tokens),
            strategy=strat,  # type: ignore[arg-type]
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            epistemic_threshold=uncertainty_threshold,
        )
        steps = out["per_step"]
        losses = [s["loss"] for s in steps if s.get("token_id") is not None]
        epi = [s["epistemic_var"] for s in steps]
        ale = [s["aleatoric_var"] for s in steps]
        return {
            "generated_ids": out["generated_ids"],
            "per_step_metrics": {
                "loss": np.asarray(losses, dtype=np.float64),
                "epistemic_var": np.asarray(epi, dtype=np.float64),
                "aleatoric_var": np.asarray(ale, dtype=np.float64),
            },
            "per_step": steps,
            "halted_on_uncertainty": out["halted_on_uncertainty"],
            "strategy": out["strategy"],
        }

    def stream_generate(
        self,
        prompt_ids: list[int],
        max_tokens: int = 100,
        temperature: float = 1.0,
        uncertainty_threshold: float | None = None,
        *,
        strategy: str = "temperature",
        top_k: int = 40,
        top_p: float = 0.9,
    ):
        """Yield one chunk dict per token (for REST SSE / NDJSON)."""
        from cypha_lm.model.generation import stream_generate as _stream

        strat = str(strategy)
        if uncertainty_threshold is not None and strat == "temperature":
            strat = "uncertainty_gated"
        yield from _stream(
            self,
            [int(t) for t in prompt_ids],
            int(max_tokens),
            strategy=strat,  # type: ignore[arg-type]
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            epistemic_threshold=uncertainty_threshold,
        )

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
                "view_emb": self.view_emb.get_state() if self.view_emb is not None else None,
                "hybrid_blend_logit": self._hybrid_blend_logit,
                "lstm": self.lstm_head.get_state() if self.lstm_head is not None else None,
            }
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        npz_kwargs = dict(
            proj_ssm=asnumpy(self._proj_ssm),
            proj_dif=asnumpy(self._proj_dif),
            proj_embed=asnumpy(self._proj_embed),
            proj_ngram=asnumpy(self._proj_ngram),
        )
        if self._proj_ngram_field is not None:
            npz_kwargs["proj_ngram_field"] = asnumpy(self._proj_ngram_field)
        if self._proj_ngram_embed is not None:
            npz_kwargs["proj_ngram_embed"] = asnumpy(self._proj_ngram_embed)
        if self._ngram_fusion is not None:
            nf = self._ngram_fusion
            npz_kwargs["ngram_W_field"] = asnumpy(nf.W_field)
            npz_kwargs["ngram_W_embed"] = asnumpy(nf.W_embed)
            npz_kwargs["ngram_W_gate"] = asnumpy(nf.W_gate)
            if nf.pos_weights is not None:
                npz_kwargs["ngram_pos_weights"] = asnumpy(nf.pos_weights)
        if self.view_emb is not None:
            npz_kwargs["view_embed"] = asnumpy(self.view_emb.table)
        np.savez(npz_path, **npz_kwargs)

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
            xp = model._backend.xp
            model._proj_ssm = xp.asarray(data["proj_ssm"], dtype=xp.float64)
            model._proj_dif = xp.asarray(data["proj_dif"], dtype=xp.float64)
            if "proj_embed" in data:
                model._proj_embed = xp.asarray(data["proj_embed"], dtype=xp.float64)
            if "proj_ngram" in data:
                model._proj_ngram = xp.asarray(data["proj_ngram"], dtype=xp.float64)
            if "proj_ngram_field" in data and model._proj_ngram_field is not None:
                model._proj_ngram_field = xp.asarray(
                    data["proj_ngram_field"], dtype=xp.float64
                )
            if "proj_ngram_embed" in data and model._proj_ngram_embed is not None:
                model._proj_ngram_embed = xp.asarray(
                    data["proj_ngram_embed"], dtype=xp.float64
                )
            if model._ngram_fusion is not None:
                nf = model._ngram_fusion
                if "ngram_W_field" in data:
                    nf.W_field = xp.asarray(data["ngram_W_field"], dtype=xp.float64)
                if "ngram_W_embed" in data:
                    nf.W_embed = xp.asarray(data["ngram_W_embed"], dtype=xp.float64)
                if "ngram_W_gate" in data:
                    nf.W_gate = xp.asarray(data["ngram_W_gate"], dtype=xp.float64)
                if "ngram_pos_weights" in data and nf.pos_weights is not None:
                    nf.pos_weights = xp.asarray(data["ngram_pos_weights"], dtype=xp.float64)
            if "view_embed" in data and model.view_emb is not None:
                model.view_emb.table = xp.asarray(data["view_embed"], dtype=xp.float64)
        if meta.get("view_emb") is not None and model.view_emb is not None:
            model.view_emb.set_state(meta["view_emb"])
        if meta.get("hybrid_blend_logit") is not None:
            model._hybrid_blend_logit = float(meta["hybrid_blend_logit"])
        if meta.get("lstm") is not None and model.lstm_head is not None:
            model.lstm_head.set_state(meta["lstm"])
        return model
