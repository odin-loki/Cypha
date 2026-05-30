"""GRIA alpha live topology controller (Upgrade U3)."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Optional

import numpy as np


def _entropy_matrix(X: np.ndarray, n_bins: int = 16) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0:
        return 0.0
    flat = X.ravel()
    hist, _ = np.histogram(flat, bins=n_bins)
    hist = hist.astype(np.float64) + 1e-12
    hist /= hist.sum()
    return float(-np.sum(hist * np.log(hist)))


class GRIAController:
    def __init__(
        self,
        window: int = 200,
        low: float = 0.35,
        high: float = 0.65,
        delta_ssm: float = 0.01,
        control_interval: int = 50,
    ) -> None:
        self.window = int(window)
        self.low = float(low)
        self.high = float(high)
        self.delta_ssm = float(delta_ssm)
        self.control_interval = int(control_interval)
        self._inp_buf: Deque[float] = deque(maxlen=window)
        self._act_buf: Deque[float] = deque(maxlen=window)
        self._step = 0

    def push(self, x: np.ndarray, activations: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        a = np.asarray(activations, dtype=np.float64).ravel()
        self._inp_buf.append(float(np.std(x) + 1e-9))
        self._act_buf.append(float(np.std(a) + 1e-9))

    def alpha(self) -> float:
        if len(self._inp_buf) < min(32, self.window):
            return 0.5
        H_x = _entropy_matrix(np.array(self._inp_buf, dtype=np.float64))
        H_f = _entropy_matrix(np.array(self._act_buf, dtype=np.float64))
        return float(np.clip(1.0 - H_f / (H_x + 1e-9), 0.0, 1.0))

    def act(
        self,
        node_id: int,
        gng: Any,
        ssm_adjust: Optional[Callable[[float], None]] = None,
    ) -> str:
        self._step += 1
        if self._step % self.control_interval != 0:
            return "skip"
        a = self.alpha()
        action = "hold"
        if a < self.low:
            gng.force_insert(node_id)
            action = "split"
            if ssm_adjust is not None:
                ssm_adjust(+self.delta_ssm)
        elif a > self.high:
            gng.merge_with_nearest(node_id)
            action = "merge"
            if ssm_adjust is not None:
                ssm_adjust(-self.delta_ssm)
        return action
