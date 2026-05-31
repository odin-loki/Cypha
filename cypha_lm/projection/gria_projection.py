"""GRIA alpha-projection: context vector to vocabulary log-probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from cypha_lm.array_backend import asnumpy, to_xp


def _softmax_logits(logits: Any, *, xp: Any = np) -> Any:
    z = logits - xp.max(logits)
    exp = xp.exp(z)
    return exp / (xp.sum(exp) + 1e-12)


def _entropy_from_probs(probs: np.ndarray, n_bins: int = 32) -> float:
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 0.0
    p = np.abs(p)
    total = p.sum()
    if total <= 0:
        return 0.0
    hist, _ = np.histogram(p, bins=n_bins, range=(0.0, float(np.max(p)) + 1e-12))
    hist = hist.astype(np.float64) + 1e-12
    hist /= hist.sum()
    return float(-np.sum(hist * np.log(hist)))


class GRIAProjection:
    """
    alpha_k * (W @ v)_k + (1 - alpha_k) * bias_k  ->  log-softmax probs.
    """

    def __init__(
        self,
        d_input: int,
        vocab_size: int,
        alpha_init: float = 0.5,
        alpha_learnable: bool = True,
        seed: int = 42,
        xp: Any = None,
    ) -> None:
        self.d_input = int(d_input)
        self.vocab_size = int(vocab_size)
        self.alpha_learnable = bool(alpha_learnable)
        self._xp = xp if xp is not None else np
        xp_mod = self._xp
        rng = np.random.default_rng(seed)
        scale = 0.01
        self.W = xp_mod.asarray(rng.standard_normal((vocab_size, d_input)) * scale)
        self.alpha = xp_mod.full(vocab_size, float(alpha_init), dtype=xp_mod.float64)
        self.bias = xp_mod.zeros(vocab_size, dtype=xp_mod.float64)

    def logits(self, v: Any) -> Any:
        xp = self._xp
        v = xp.asarray(v, dtype=xp.float64).ravel()
        if v.size != self.d_input:
            v = xp.resize(v, self.d_input)
        z = self.W @ v
        return self.alpha * z + (1.0 - self.alpha) * self.bias

    def forward(self, v: Any) -> Any:
        """Log-probabilities over vocabulary."""
        xp = self._xp
        logits = self.logits(v)
        probs = _softmax_logits(logits, xp=xp)
        return xp.log(probs + 1e-12)

    def cross_entropy_gradients(
        self, v: Any, target_id: int
    ) -> tuple[Any, Any, Any]:
        xp = self._xp
        v = xp.asarray(v, dtype=xp.float64).ravel()
        if v.size != self.d_input:
            v = xp.resize(v, self.d_input)
        logits = self.logits(v)
        probs = _softmax_logits(logits, xp=xp)
        d_logits = probs.copy()
        d_logits[int(target_id)] -= 1.0
        z = self.W @ v
        grad_W = xp.outer(d_logits * self.alpha, v)
        grad_alpha = d_logits * (z - self.bias)
        grad_bias = d_logits * (1.0 - self.alpha)
        return grad_W, grad_alpha, grad_bias

    def update_alpha(self, grad_alpha: Any, lr: float = 1e-3) -> None:
        if not self.alpha_learnable:
            return
        xp = self._xp
        g = xp.asarray(grad_alpha, dtype=xp.float64).ravel()
        if g.size != self.vocab_size:
            g = xp.resize(g, self.vocab_size)
        self.alpha -= lr * g
        xp.clip(self.alpha, 0.01, 0.99, out=self.alpha)

    def update_weights(self, grad_W: Any, lr: float = 1e-3) -> None:
        g = self._xp.asarray(grad_W, dtype=self._xp.float64)
        if g.shape != self.W.shape:
            raise ValueError(f"grad_W shape {g.shape} != W shape {self.W.shape}")
        self.W -= lr * g

    def update_bias(self, grad_bias: Any, lr: float = 1e-3) -> None:
        xp = self._xp
        g = xp.asarray(grad_bias, dtype=xp.float64).ravel()
        if g.size != self.vocab_size:
            g = xp.resize(g, self.vocab_size)
        self.bias -= lr * g

    def set_unigram_prior(self, token_counts: np.ndarray) -> None:
        xp = self._xp
        counts = xp.asarray(token_counts, dtype=xp.float64).ravel()
        if counts.size < self.vocab_size:
            counts = xp.resize(counts, self.vocab_size)
        counts = counts[: self.vocab_size] + 1.0
        probs = counts / counts.sum()
        self.bias = xp.log(probs + 1e-12)

    def alpha_spectrum(self) -> dict:
        a = asnumpy(self.alpha)
        hist, edges = np.histogram(a, bins=10, range=(0.0, 1.0))
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "histogram": hist.astype(int).tolist(),
            "histogram_edges": edges.tolist(),
        }

    def grand_unified_law_alpha(
        self, activations: np.ndarray, outputs: np.ndarray
    ) -> float:
        """alpha = 1 - H(outputs) / H(activations)."""
        h_act = _entropy_from_probs(activations)
        h_out = _entropy_from_probs(outputs)
        if h_act <= 1e-12:
            return 0.5
        return float(np.clip(1.0 - h_out / h_act, 0.0, 1.0))

    def get_state(self) -> dict:
        return {
            "W": asnumpy(self.W),
            "alpha": asnumpy(self.alpha),
            "bias": asnumpy(self.bias),
            "alpha_learnable": self.alpha_learnable,
        }

    def set_state(self, state: dict) -> None:
        xp = self._xp
        self.W = xp.asarray(state["W"], dtype=xp.float64)
        self.alpha = xp.asarray(state["alpha"], dtype=xp.float64)
        self.bias = xp.asarray(state["bias"], dtype=xp.float64)
        self.alpha_learnable = bool(state.get("alpha_learnable", True))
