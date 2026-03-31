#!/usr/bin/env python3
"""
Emit ``parity_fixtures/dif_regressor_train_step/`` for native ``DIFRegressor``-shaped online steps.

**Routing:** Python ``DIFRegressor.train_step`` — cold-start hash while ``len(classes) < max(n_experts,4)``
and ``step <= K*20``; then ``CyphaDIF.infer`` (argmax LLR from ``memory.classify``) picks the expert label.

**Replay:** ``replay_ratio > 0`` on the embedded ``CyphaDIF``; sidecar ``replay_u01`` records the replay
RNG stream (gate + ``PriorityReplayBuffer.sample`` batch draws) for native ``TrainStepExtras`` parity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Cypha import DIFRegressor, VectorEncoder, cypha_load_binary, cypha_save_binary

_OUT = _ROOT / "parity_fixtures" / "dif_regressor_train_step"
_N_STEPS = 12
_REPLAY_RATIO = 0.3
_REPLAY_RNG_SEED = 424242


class RecordingReplayRng:
    """Wraps a NumPy generator and records every scalar/batch ``random`` draw."""

    def __init__(self, inner: np.random.Generator) -> None:
        self._inner = inner
        self.values: list[float] = []

    def random(self, *args, **kwargs):  # noqa: ANN002
        if kwargs.get("out") is not None:
            out = kwargs["out"]
            n = int(args[0])
            for j in range(n):
                v = float(self._inner.random())
                self.values.append(v)
                out[j] = v
            return out
        v = float(self._inner.random())
        self.values.append(v)
        return v


class ListReplayRng:
    """Replays a fixed stream (must match recorded order)."""

    def __init__(self, xs: list[float]) -> None:
        self.xs = xs
        self.i = 0

    def random(self, *args, **kwargs):  # noqa: ANN002
        if kwargs.get("out") is not None:
            out = kwargs["out"]
            n = int(args[0])
            for j in range(n):
                if self.i >= len(self.xs):
                    raise RuntimeError("ListReplayRng exhausted (batch)")
                out[j] = self.xs[self.i]
                self.i += 1
            return out
        if self.i >= len(self.xs):
            raise RuntimeError("ListReplayRng exhausted (scalar)")
        v = float(self.xs[self.i])
        self.i += 1
        return v


def main() -> None:
    rng_data = np.random.default_rng(2026)
    rec = RecordingReplayRng(np.random.default_rng(_REPLAY_RNG_SEED))
    reg = DIFRegressor(
        encoder=VectorEncoder(4),
        field_dim=32,
        n_experts=8,
        target_lr=0.06,
        rng=np.random.default_rng(42),
        replay_ratio=_REPLAY_RATIO,
        replay_rng=rec,
    )

    _OUT.mkdir(parents=True, exist_ok=True)
    cypha_save_binary(reg.clf.save_state(), str(_OUT / "before.cypha"))
    F = reg.clf.memory.world.F_field.astype(np.float64)
    (_OUT / "f_field.json").write_text(json.dumps(F.tolist()), encoding="utf-8")

    xs = rng_data.standard_normal((_N_STEPS, 4)).astype(np.float64)
    ys = rng_data.standard_normal(_N_STEPS).astype(np.float64)
    steps = []
    for i in range(_N_STEPS):
        x = xs[i]
        y = float(ys[i])
        sc = getattr(reg, "_step_count", 0)
        k_tgt = max(reg.n_experts, 4)
        with reg.clf.memory._lock:
            n_ex = len(reg.clf.memory._classes)
        if n_ex < k_tgt and (sc + 1) <= k_tgt * 20:
            exp_ex = "_e%d" % ((sc + 1) % k_tgt)
        else:
            pred, _conf = reg.clf.infer(x)
            exp_ex = pred if pred != "__unknown__" else "_e0"
        loss = reg.train_step(x, y)
        steps.append(
            {
                "x": x.tolist(),
                "y": y,
                "expected_loss": float(loss),
                "expected_expert": exp_ex,
            }
        )

    qx = rng_data.standard_normal(4).astype(np.float64)
    y_pred, unc = reg.predict(qx)

    replay_u01 = rec.values

    sidecar = {
        "fixture_schema": 2,
        "n_steps": _N_STEPS,
        "d_latent": 4,
        "field_dim": 32,
        "n_experts": int(reg.n_experts),
        "target_lr": float(reg.target_lr),
        "target_dim": int(reg._target_dim or 1),
        "world_lr": float(reg.clf.world_lr),
        "delta_lr": float(reg.clf.delta_lr),
        "temperature": float(reg.clf.temperature),
        "ood_sigma": float(reg.clf.ood_sigma),
        "enc_lr": float(reg.clf.enc_lr),
        "replay_ratio": float(_REPLAY_RATIO),
        "replay_cap": 10000,
        "align_every": 500,
        "temp_recalib_every": 0,
        "replay_u01": replay_u01,
        "steps": steps,
        "predict_x": qx.tolist(),
        "expected_y_pred": float(np.asarray(y_pred, dtype=np.float64).ravel()[0]),
        "expected_uncertainty": float(unc),
        "final_expert_mu": {k: np.asarray(v, dtype=np.float64).tolist() for k, v in reg._expert_mu.items()},
        "final_expert_var": {k: float(v) for k, v in reg._expert_var.items()},
        "final_expert_n": {k: int(v) for k, v in reg._expert_n.items()},
    }
    (_OUT / "sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    st = cypha_load_binary(str(_OUT / "before.cypha"))
    rec2 = ListReplayRng(list(replay_u01))
    reg2 = DIFRegressor(
        encoder=VectorEncoder(4),
        field_dim=32,
        n_experts=8,
        target_lr=0.06,
        rng=np.random.default_rng(42),
        replay_ratio=_REPLAY_RATIO,
        replay_rng=rec2,
    )
    reg2.clf.load_state(st)
    for i, s in enumerate(steps):
        loss2 = reg2.train_step(np.asarray(s["x"], dtype=np.float64), float(s["y"]))
        if abs(loss2 - s["expected_loss"]) > 1e-9:
            raise RuntimeError(f"self-check loss step {i}: got {loss2} expected {s['expected_loss']}")
    if rec2.i != len(replay_u01):
        raise RuntimeError(f"self-check replay_u01: consumed {rec2.i} expected {len(replay_u01)}")

    print(f"Wrote {_OUT}/ ({_N_STEPS} steps: cold hash + warm infer + replay_ratio={_REPLAY_RATIO})")


if __name__ == "__main__":
    main()
