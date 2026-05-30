"""Dynamic Hebbian graph for CellAI state diffusion (Upgrade U5)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class DynamicHebbianGraph:
    def __init__(
        self,
        n: int,
        eta_edge: float = 0.01,
        lambda_decay: float = 0.001,
        theta_prune: float = 0.01,
        theta_form: float = 0.3,
        gamma: float = 0.1,
    ) -> None:
        self.n = int(n)
        self.eta_edge = float(eta_edge)
        self.lambda_decay = float(lambda_decay)
        self.theta_prune = float(theta_prune)
        self.theta_form = float(theta_form)
        self.gamma = float(gamma)
        self.edges: Dict[Tuple[int, int], float] = {}
        self._ring_init()

    def _ring_init(self) -> None:
        for i in range(self.n):
            j = (i + 1) % self.n
            self.edges[(min(i, j), max(i, j))] = 0.5

    def update(self, a: np.ndarray) -> None:
        a = np.asarray(a, dtype=np.float64).ravel()
        n = min(self.n, a.size)
        for i in range(n):
            for j in range(i + 1, n):
                key = (i, j)
                co = float(a[i] * a[j])
                if key in self.edges:
                    e = self.edges[key] + self.eta_edge * co
                    e *= 1.0 - self.lambda_decay
                    if e < self.theta_prune:
                        del self.edges[key]
                    else:
                        self.edges[key] = e
                elif co > self.theta_form:
                    self.edges[key] = self.theta_form

    def normalized_adjacency(self) -> np.ndarray:
        A = np.zeros((self.n, self.n), dtype=np.float64)
        for (i, j), w in self.edges.items():
            A[i, j] = w
            A[j, i] = w
        deg = A.sum(axis=1) + 1e-12
        D_inv = np.diag(1.0 / deg)
        return D_inv @ A

    def spectral_radius(self) -> float:
        A = self.normalized_adjacency()
        vals = np.linalg.eigvals(A)
        return float(np.max(np.abs(vals)))

    def diffuse(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size < self.n:
            x = np.resize(x, self.n)
        x = x[: self.n]
        A = self.normalized_adjacency()
        x_d = x + self.gamma * (A @ x)
        if not np.all(np.isfinite(x_d)):
            return x
        return x_d
