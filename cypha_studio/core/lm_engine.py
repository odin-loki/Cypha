"""CyphaLM inference and streaming generation for CyphaStudio REST."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


class LMEngine:
    """
    Wraps a loaded :class:`cypha_lm.model.cypha_lm.CyphaLM` for REST generation.

    Separate from :class:`cypha_studio.core.inference.InferenceEngine` (CyphaDIF
    vector classifier). Both can be attached to the same FastAPI app.
    """

    def __init__(self, model: Any, *, source_path: str | None = None) -> None:
        self.model = model
        self.source_path = source_path
        self.n_generations = 0
        self.n_stream_tokens = 0
        self._loaded_at = time.time()

    @classmethod
    def from_checkpoint(cls, path: str) -> LMEngine:
        from cypha_lm.model.cypha_lm import CyphaLM

        model = CyphaLM.load(path)
        return cls(model, source_path=str(Path(path).resolve()))

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> LMEngine:
        from cypha_lm.config import CyphaLMConfig
        from cypha_lm.model.cypha_lm import CyphaLM

        cfg = CyphaLMConfig(**(config or {}))
        return cls(CyphaLM(cfg))

    def predict_next(self, token_id: int) -> dict[str, Any]:
        pred = self.model.predict_next(int(token_id))
        lp = np.asarray(pred["log_probs"], dtype=np.float64)
        return {
            "token_id": int(token_id),
            "log_probs": lp.tolist(),
            "epistemic_var": float(pred["epistemic_var"]),
            "aleatoric_var": float(pred["aleatoric_var"]),
            "top_k_tokens": pred["top_k_tokens"],
            "top_k_probs": pred["top_k_probs"],
            "active_experts": int(pred.get("active_experts", 0)),
            "dominant_expert": int(pred.get("dominant_expert", 0)),
            "routing_probs": pred.get("routing_probs", []),
        }

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 64,
        temperature: float = 0.9,
        strategy: str = "temperature",
        top_k: int = 40,
        top_p: float = 0.9,
        uncertainty_threshold: float | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        out = self.model.generate(
            [int(t) for t in prompt_ids],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            uncertainty_threshold=uncertainty_threshold,
            strategy=strategy,
            top_k=int(top_k),
            top_p=float(top_p),
        )
        self.n_generations += 1
        out["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        out["n_tokens"] = len(out.get("generated_ids", []))
        return out

    def stream_generate(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 64,
        temperature: float = 0.9,
        strategy: str = "temperature",
        top_k: int = 40,
        top_p: float = 0.9,
        uncertainty_threshold: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        for chunk in self.model.stream_generate(
            [int(t) for t in prompt_ids],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            uncertainty_threshold=uncertainty_threshold,
            strategy=strategy,
            top_k=int(top_k),
            top_p=float(top_p),
        ):
            if chunk.get("token_id") is not None:
                self.n_stream_tokens += 1
            yield chunk

    def compression_profile(self) -> dict[str, Any]:
        return self.model.compression_profile()

    def summary(self) -> dict[str, Any]:
        prof = self.compression_profile()
        return {
            "loaded": True,
            "source_path": self.source_path,
            "vocab_size": int(self.model.config.vocab_size),
            "field_dim": int(self.model.config.field_dim),
            "n_generations": self.n_generations,
            "n_stream_tokens": self.n_stream_tokens,
            "n_experts": int(prof.get("n_experts", 0)),
            "mean_alpha": float(prof.get("mean_alpha", float("nan"))),
            "uptime_s": time.time() - self._loaded_at,
        }
