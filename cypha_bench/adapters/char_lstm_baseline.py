"""Character LSTM baseline (NumPy only) for cypha_bench language-model domains."""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


class _CharLSTM:
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
        dc = dc_new
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

    def train_step(
        self,
        token_id: int,
        next_token_id: int,
        h: np.ndarray,
        c: np.ndarray,
        lr: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        h_new, c_new, _log_probs, cache = self._forward_step(token_id, h, c)
        grads = self._backward_step(cache, next_token_id)
        self._apply_grads(grads, lr)
        return h_new, c_new

    def predict_log_probs(
        self,
        token_id: int,
        h: np.ndarray,
        c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_new, c_new, log_probs, _cache = self._forward_step(token_id, h, c)
        return log_probs, h_new, c_new


def char_lstm_baseline_bpc(
    train_ids: list[int],
    test_ids: list[int],
    vocab_size: int,
    *,
    hidden: int = 128,
    n_train_steps: int | None = None,
    seed: int = 42,
) -> float:
    """Train a 1-layer char LSTM and return held-out bits-per-character."""
    if len(train_ids) < 2:
        return float("nan")
    if len(test_ids) < 2:
        return float("nan")

    limit = n_train_steps if n_train_steps is not None else min(len(train_ids) - 1, 10_000)
    limit = max(1, min(limit, len(train_ids) - 1))

    model = _CharLSTM(vocab_size, hidden, seed)
    h = np.zeros(hidden, dtype=np.float64)
    c = np.zeros(hidden, dtype=np.float64)
    lr = 0.05

    for t in range(limit):
        h, c = model.train_step(int(train_ids[t]), int(train_ids[t + 1]), h, c, lr)

    bits: list[float] = []
    h = np.zeros(hidden, dtype=np.float64)
    c = np.zeros(hidden, dtype=np.float64)
    for i in range(len(test_ids) - 1):
        log_probs, h, c = model.predict_log_probs(int(test_ids[i]), h, c)
        nxt = int(test_ids[i + 1])
        if 0 <= nxt < vocab_size:
            bits.append(float(-log_probs[nxt] / np.log(2)))
    return float(np.mean(bits)) if bits else float("nan")
