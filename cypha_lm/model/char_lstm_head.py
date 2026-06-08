"""Character LSTM head for CyphaLM hybrid / model-class research (C2)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _sigmoid(x: float | np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def blend_log_probs(
    log_g: np.ndarray,
    log_l: np.ndarray,
    blend_logit: float,
) -> np.ndarray:
    """Convex blend in probability space: alpha*GRIA + (1-alpha)*LSTM."""
    alpha = float(_sigmoid(blend_logit))
    p_g = np.exp(np.asarray(log_g, dtype=np.float64))
    p_l = np.exp(np.asarray(log_l, dtype=np.float64))
    p = alpha * p_g + (1.0 - alpha) * p_l
    return np.log(p + 1e-12)


def blend_logit_grad(
    log_g: np.ndarray,
    log_l: np.ndarray,
    blend_logit: float,
    target_id: int,
) -> float:
    """d/d(blend_logit) of CE loss on blended distribution."""
    alpha = float(_sigmoid(blend_logit))
    p_g = np.exp(np.asarray(log_g, dtype=np.float64))
    p_l = np.exp(np.asarray(log_l, dtype=np.float64))
    p_t = alpha * p_g[int(target_id)] + (1.0 - alpha) * p_l[int(target_id)]
    if p_t <= 0:
        return 0.0
    g_t = p_g[int(target_id)]
    l_t = p_l[int(target_id)]
    d_loss_d_alpha = -(g_t - l_t) / p_t
    d_alpha_d_logit = alpha * (1.0 - alpha)
    return float(d_loss_d_alpha * d_alpha_d_logit)


class CharLSTMHead:
    """Single-layer char LSTM with one-step backprop (online BPTT-1)."""

    def __init__(self, vocab_size: int, hidden: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        scale = 0.02
        self.vocab_size = int(vocab_size)
        self.hidden = int(hidden)
        h = self.hidden
        self.E = rng.standard_normal((self.vocab_size, h)) * scale
        self.Wx = rng.standard_normal((4 * h, h)) * scale
        self.Wh = rng.standard_normal((4 * h, h)) * scale
        self.b = np.zeros(4 * h, dtype=np.float64)
        self.Wy = rng.standard_normal((self.vocab_size, h)) * scale
        self.by = np.zeros(self.vocab_size, dtype=np.float64)
        self._cache: dict[str, np.ndarray] | None = None

    def reset_state(self) -> tuple[np.ndarray, np.ndarray]:
        h = np.zeros(self.hidden, dtype=np.float64)
        c = np.zeros(self.hidden, dtype=np.float64)
        self._cache = None
        return h, c

    def _forward_step(
        self,
        token_id: int,
        h: np.ndarray,
        c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        x = self.E[int(token_id)]
        gates = self.Wx @ x + self.Wh @ h + self.b
        i = _sigmoid(gates[: self.hidden])
        f = _sigmoid(gates[self.hidden : 2 * self.hidden])
        g = np.tanh(gates[2 * self.hidden : 3 * self.hidden])
        o = _sigmoid(gates[3 * self.hidden :])
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        logits = self.Wy @ h_new + self.by
        probs = _softmax(logits)
        log_probs = np.log(probs + 1e-12)
        cache = {
            "token_id": int(token_id),
            "x": x,
            "h": h,
            "c": c,
            "i": i,
            "f": f,
            "g": g,
            "o": o,
            "c_new": c_new,
            "h_new": h_new,
            "logits": logits,
            "probs": probs,
        }
        return h_new, c_new, log_probs, cache

    def _backward_step(self, cache: dict[str, np.ndarray], target_id: int) -> dict[str, np.ndarray]:
        probs = cache["probs"]
        h_new = cache["h_new"]
        c_new = cache["c_new"]
        i, f, g, o = cache["i"], cache["f"], cache["g"], cache["o"]
        h, c = cache["h"], cache["c"]
        x = cache["x"]
        token_id = cache["token_id"]

        d_logits = probs.copy()
        d_logits[int(target_id)] -= 1.0

        dWy = np.outer(d_logits, h_new)
        dby = d_logits
        dh_new = self.Wy.T @ d_logits

        do = dh_new * np.tanh(c_new)
        dc_new = dh_new * o * (1.0 - np.tanh(c_new) ** 2)
        df = dc_new * c
        di = dc_new * g
        dg = dc_new * i
        dc_prev = dc_new * f

        di_raw = di * i * (1.0 - i)
        df_raw = df * f * (1.0 - f)
        dg_raw = dg * (1.0 - g ** 2)
        do_raw = do * o * (1.0 - o)
        dgates = np.concatenate([di_raw, df_raw, dg_raw, do_raw])

        dWx = np.outer(dgates, x)
        dWh = np.outer(dgates, h)
        db = dgates
        dx = self.Wx.T @ dgates
        dh_prev = self.Wh.T @ dgates
        dE = np.zeros_like(self.E)
        dE[token_id] = dx

        return {
            "dE": dE,
            "dWx": dWx,
            "dWh": dWh,
            "db": db,
            "dWy": dWy,
            "dby": dby,
            "dh_prev": dh_prev,
            "dc_prev": dc_prev,
        }

    def _apply_grads(self, grads: dict[str, np.ndarray], lr: float) -> None:
        self.E -= lr * grads["dE"]
        self.Wx -= lr * grads["dWx"]
        self.Wh -= lr * grads["dWh"]
        self.b -= lr * grads["db"]
        self.Wy -= lr * grads["dWy"]
        self.by -= lr * grads["dby"]

    def forward(
        self,
        token_id: int,
        h: np.ndarray,
        c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_new, c_new, log_probs, cache = self._forward_step(token_id, h, c)
        self._cache = cache
        return log_probs, h_new, c_new

    def backward(self, target_id: int, lr: float) -> None:
        if self._cache is None:
            return
        grads = self._backward_step(self._cache, int(target_id))
        self._apply_grads(grads, lr)
        self._cache = None

    def get_state(self) -> dict[str, Any]:
        return {
            "E": self.E.tolist(),
            "Wx": self.Wx.tolist(),
            "Wh": self.Wh.tolist(),
            "b": self.b.tolist(),
            "Wy": self.Wy.tolist(),
            "by": self.by.tolist(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.E = np.asarray(state["E"], dtype=np.float64)
        self.Wx = np.asarray(state["Wx"], dtype=np.float64)
        self.Wh = np.asarray(state["Wh"], dtype=np.float64)
        self.b = np.asarray(state["b"], dtype=np.float64)
        self.Wy = np.asarray(state["Wy"], dtype=np.float64)
        self.by = np.asarray(state["by"], dtype=np.float64)
