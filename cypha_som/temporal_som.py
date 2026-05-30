"""Temporal SOM for SSM decay-rate regimes (Upgrade U6)."""

from __future__ import annotations

from typing import Optional

import numpy as np


class TemporalSOM:
    def __init__(
        self,
        M: int = 8,
        L_max: int = 16,
        eta_ts: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.M = int(M)
        self.L_max = int(L_max)
        self.eta_ts = float(eta_ts)
        self._rng = rng or np.random.default_rng(42)
        self.centroids = self._rng.standard_normal((self.M, self.L_max)) * 0.1
        # each unit owns fast/slow decay multipliers in (0.5, 1.5)
        self.Lambda = self._rng.uniform(0.85, 1.15, (self.M, 2))
        self._x_hist: list[np.ndarray] = []

    def _autocorr_features(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        self._x_hist.append(x.copy())
        if len(self._x_hist) > self.L_max + 2:
            self._x_hist = self._x_hist[-(self.L_max + 2) :]
        feats = np.zeros(self.L_max, dtype=np.float64)
        if len(self._x_hist) < 3:
            return feats
        seq = np.stack(self._x_hist, axis=0)
        mean = seq.mean(axis=0) + 1e-9
        for lag in range(1, min(self.L_max + 1, seq.shape[0])):
            a = seq[-1]
            b = seq[-1 - lag]
            feats[lag - 1] = float(np.mean((a - mean) * (b - mean)) / (np.var(seq) + 1e-9))
        return feats

    def step(self, x: np.ndarray, train: bool = True) -> tuple[int, float, float]:
        r = self._autocorr_features(x)
        dists = np.linalg.norm(self.centroids - r, axis=1)
        bmu = int(np.argmin(dists))
        if train:
            h = np.exp(-dists / (np.median(dists) + 1e-9))
            h /= h.sum() + 1e-12
            for m in range(self.M):
                self.centroids[m] += self.eta_ts * h[m] * (r - self.centroids[m])
        lam_fast, lam_slow = float(self.Lambda[bmu, 0]), float(self.Lambda[bmu, 1])
        return bmu, lam_fast, lam_slow
