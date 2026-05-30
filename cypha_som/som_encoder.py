"""Online SOM over encoder features (Upgrade U2)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class OnlineSOMEncoder:
    def __init__(
        self,
        d_in: int,
        k: int = 16,
        eta0: float = 0.3,
        sigma0: float = 4.0,
        T: int = 10000,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.d_in = int(d_in)
        self.k = int(k)
        self.n_units = self.k * self.k
        self.eta0 = float(eta0)
        self.sigma0 = float(sigma0)
        self.T = int(T)
        self.t = 0
        rng = rng or np.random.default_rng(42)
        self.W = rng.standard_normal((self.n_units, self.d_in)) * 0.1
        self.positions = np.array(
            [[i, j] for i in range(self.k) for j in range(self.k)], dtype=np.float64
        )

    def encode(self, z: np.ndarray, train: bool = True) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).ravel()
        if z.size != self.d_in:
            z = np.resize(z, self.d_in)
        dists = np.linalg.norm(self.W - z, axis=1)
        bmu = int(np.argmin(dists))
        if train:
            eta = self.eta0 * np.exp(-self.t / max(self.T, 1))
            sigma = self.sigma0 * np.exp(-self.t / max(self.T, 1))
            d2 = np.sum((self.positions - self.positions[bmu]) ** 2, axis=1)
            h = np.exp(-d2 / (2.0 * sigma * sigma + 1e-12))
            self.W += eta * h[:, None] * (z - self.W)
            self.t += 1
        return self.W[bmu].copy()

    def batch_encode(self, Z: np.ndarray, train: bool = True) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        out = np.zeros((Z.shape[0], self.d_in), dtype=np.float64)
        for i in range(Z.shape[0]):
            out[i] = self.encode(Z[i], train=train)
        return out


def _encoder_base_class() -> type:
    from Cypha import Encoder
    return Encoder


class SOMWrappedEncoder(_encoder_base_class()):
    """Wraps a Cypha Encoder: x -> base -> SOM smoothed features."""

    def __init__(self, base_encoder: Any, k: int = 16, T: int = 10000) -> None:
        super().__init__()
        self._base = base_encoder
        self._som = OnlineSOMEncoder(base_encoder.dim, k=k, T=T)

    @property
    def dim(self) -> int:
        return self._base.dim

    def __call__(self, x: Any) -> np.ndarray:
        z = np.asarray(self._base(x), dtype=np.float64).ravel()
        return self._som.encode(z, train=True)

    def batch_encode(self, X: np.ndarray) -> np.ndarray:
        Z = self._base.batch_encode(X)
        return self._som.batch_encode(Z, train=True)

    @property
    def inner(self) -> Any:
        return self._base
