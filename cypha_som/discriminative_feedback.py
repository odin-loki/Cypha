"""Discriminative feedback from CyphaDIF LLRs to encoder updates (Upgrade U4)."""

from __future__ import annotations

import numpy as np


class DiscriminativeFeedback:
    def __init__(self, beta: float = 0.1) -> None:
        self.beta = float(beta)

    def compute_d(self, delta_mu: np.ndarray, sigma_diag_inv: np.ndarray) -> np.ndarray:
        """delta_mu: (K, d), sigma_diag_inv: (d,)"""
        delta_mu = np.asarray(delta_mu, dtype=np.float64)
        sigma_diag_inv = np.asarray(sigma_diag_inv, dtype=np.float64).ravel()
        importance = np.sum(np.abs(delta_mu) * sigma_diag_inv[None, :], axis=0)
        s = float(importance.sum()) + 1e-9
        return importance / s

    def modulate(self, dW: np.ndarray, d: np.ndarray) -> np.ndarray:
        dW = np.asarray(dW, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64).ravel()
        if dW.ndim == 2 and d.size == dW.shape[1]:
            return dW + self.beta * (d[None, :] * dW)
        if dW.ndim == 1 and d.size == dW.size:
            return dW + self.beta * (d * dW)
        return dW
