"""GRIA alpha-projection: context vector to vocabulary log-probabilities."""

from __future__ import annotations

import numpy as np


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    exp = np.exp(z)
    return exp / (np.sum(exp) + 1e-12)


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
    ) -> None:
        self.d_input = int(d_input)
        self.vocab_size = int(vocab_size)
        self.alpha_learnable = bool(alpha_learnable)
        rng = np.random.default_rng(seed)
        scale = 0.01
        self.W = rng.standard_normal((vocab_size, d_input)) * scale
        self.alpha = np.full(vocab_size, float(alpha_init), dtype=np.float64)
        self.bias = np.zeros(vocab_size, dtype=np.float64)

    def logits(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).ravel()
        if v.size != self.d_input:
            v = np.resize(v, self.d_input)
        z = self.W @ v
        return self.alpha * z + (1.0 - self.alpha) * self.bias

    def forward(self, v: np.ndarray) -> np.ndarray:
        """Log-probabilities over vocabulary."""
        logits = self.logits(v)
        probs = _softmax_logits(logits)
        return np.log(probs + 1e-12)

    def cross_entropy_gradients(
        self, v: np.ndarray, target_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Manual CE gradients: dL/dW, dL/dalpha, dL/dbias for one target token.
        """
        v = np.asarray(v, dtype=np.float64).ravel()
        if v.size != self.d_input:
            v = np.resize(v, self.d_input)
        logits = self.logits(v)
        probs = _softmax_logits(logits)
        d_logits = probs.copy()
        d_logits[int(target_id)] -= 1.0
        z = self.W @ v
        grad_W = np.outer(d_logits * self.alpha, v)
        grad_alpha = d_logits * (z - self.bias)
        grad_bias = d_logits * (1.0 - self.alpha)
        return grad_W, grad_alpha, grad_bias

    def update_alpha(self, grad_alpha: np.ndarray, lr: float = 1e-3) -> None:
        if not self.alpha_learnable:
            return
        g = np.asarray(grad_alpha, dtype=np.float64).ravel()
        if g.size != self.vocab_size:
            g = np.resize(g, self.vocab_size)
        self.alpha -= lr * g
        np.clip(self.alpha, 0.01, 0.99, out=self.alpha)

    def update_weights(self, grad_W: np.ndarray, lr: float = 1e-3) -> None:
        g = np.asarray(grad_W, dtype=np.float64)
        if g.shape != self.W.shape:
            raise ValueError(f"grad_W shape {g.shape} != W shape {self.W.shape}")
        self.W -= lr * g

    def update_bias(self, grad_bias: np.ndarray, lr: float = 1e-3) -> None:
        g = np.asarray(grad_bias, dtype=np.float64).ravel()
        if g.size != self.vocab_size:
            g = np.resize(g, self.vocab_size)
        self.bias -= lr * g

    def set_unigram_prior(self, token_counts: np.ndarray) -> None:
        counts = np.asarray(token_counts, dtype=np.float64).ravel()
        if counts.size < self.vocab_size:
            counts = np.resize(counts, self.vocab_size)
        counts = counts[: self.vocab_size] + 1.0
        probs = counts / counts.sum()
        self.bias = np.log(probs + 1e-12)

    def alpha_spectrum(self) -> dict:
        a = self.alpha
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
            "W": self.W.copy(),
            "alpha": self.alpha.copy(),
            "bias": self.bias.copy(),
            "alpha_learnable": self.alpha_learnable,
        }

    def set_state(self, state: dict) -> None:
        self.W = np.asarray(state["W"], dtype=np.float64)
        self.alpha = np.asarray(state["alpha"], dtype=np.float64)
        self.bias = np.asarray(state["bias"], dtype=np.float64)
        self.alpha_learnable = bool(state.get("alpha_learnable", True))
