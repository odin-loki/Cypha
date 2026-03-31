#!/usr/bin/env python3
"""
Emit ``parity_fixtures/mke_train_extended/`` for ``mke_train_step_parity`` extended mode.

Multi-step ``MKERegressor.train_step`` with ``replay_ratio > 0``, ``enc_lr > 0``, and a recorded
``replay_u01`` stream (same pattern as ``generate_dif_train_replay_fixture.py``).
``replay_rng`` is set via ``MKERegressor.from_data(..., replay_rng=...)`` (forwards to ``CyphaDIF``).

Before each step, ``_sync_world_log_norm_like_load_state`` aligns live ``WorldPrior`` with ``load_state``;
the native harness calls ``CyphaDifMemoryState::refresh_world_log_norm_from_v``.

``replay_warmup`` repopulates the buffer after ``before.cypha`` (replay is not on disk). Expected router
loss uses: save state **before** the step, ``mke.train_step`` (records replay draws), then a probe
with ``ListReplayRng(segment)`` so it replays the same inner ``clf.train_step`` randomness as live.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from Cypha import (  # noqa: E402
    CyphaDIF,
    MKERegressor,
    PriorityReplayBuffer,
    _ENC_LR,
    _EPS,
    _LOG2PI,
    _MIN_VAR,
    _PRIORITY_EPS,
    _REPLAY_CAP,
    _nig_R_eff,
    _softmax_batch,
    cypha_load_binary,
    cypha_save_binary,
)

_OUT = _ROOT / "parity_fixtures" / "mke_train_extended"
# Multi-step surface for native ``mke_train_step_parity`` (enc_lr>0, replay_ratio>0, replay_u01).
_N_EXT = 12
_REPLAY_RATIO_FIX = 0.35
_REPLAY_INNER_SEED = 884422


def _flatten_p(P: np.ndarray) -> list[float]:
    return np.asarray(P, dtype=np.float64).reshape(-1, order="C").tolist()


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


def _sync_world_log_norm_like_load_state(clf: CyphaDIF) -> None:
    w = clf.memory.world
    w._log_norm = -0.5 * (
        w.d * _LOG2PI + float(np.sum(np.log(np.maximum(w.v, _MIN_VAR))))
    )
    w._log_n_ctr = 0


def _snapshot_replay_entries(r: PriorityReplayBuffer) -> list[tuple[np.ndarray, np.ndarray, str, float]]:
    with r._lock:
        out: list[tuple[np.ndarray, np.ndarray, str, float]] = []
        for i in range(r._buf_len):
            h, f, lbl, loss_v, _ins = r._buf[i]
            out.append(
                (
                    np.asarray(h, dtype=np.float64).copy(),
                    np.asarray(f, dtype=np.float64).copy(),
                    str(lbl),
                    float(loss_v),
                )
            )
        return out


def _replay_warmup_json(entries: list[tuple[np.ndarray, np.ndarray, str, float]]) -> list[dict]:
    rows: list[dict] = []
    for h, f, lbl, loss_v in entries:
        rows.append(
            {
                "h": h.tolist(),
                "f": f.tolist(),
                "label": lbl,
                "loss_v": float(loss_v),
            }
        )
    return rows


def _rebuild_replay_from_entries(clf: CyphaDIF, entries: list[tuple[np.ndarray, np.ndarray, str, float]]) -> None:
    clf.replay = PriorityReplayBuffer(capacity=_REPLAY_CAP, rng=clf._replay_rng)
    for h, f, lbl, loss_v in entries:
        loss_arg = max(float(loss_v) - _PRIORITY_EPS, 0.0)
        clf.replay.push(h, f, lbl, loss=loss_arg)
    clf._buf_len_cache = len(clf.replay)


def main() -> None:
    rng = np.random.default_rng(9027)
    d_in = 3
    D = 32
    K = 2
    field_dim = 24
    rng_seed = 901

    X_seed = rng.standard_normal((100, d_in))
    y_seed = X_seed[:, 0] * 0.6 + rng.standard_normal(100) * 0.25

    rec = RecordingReplayRng(np.random.default_rng(_REPLAY_INNER_SEED))
    mke = MKERegressor.from_data(
        X_seed,
        y_seed,
        K=K,
        D=D,
        field_dim=field_dim,
        rng_seed=rng_seed,
        replay_rng=rec,
    )
    mke.clf.replay = PriorityReplayBuffer(capacity=_REPLAY_CAP, rng=mke.clf._replay_rng)
    mke.clf._buf_len_cache = len(mke.clf.replay)
    mke.clf.enc_lr = 0.0
    mke.clf.replay_ratio = 0.0
    mke.forgetting_factor = 1.0
    mke.nig_alpha = 0.0

    for _ in range(14):
        xw = rng.standard_normal(d_in)
        yw = float(rng.standard_normal())
        mke.train_step(xw, yw)

    with mke.clf.memory._lock:
        if len(mke.clf.memory._classes) == 0:
            raise RuntimeError("fixture needs nonempty router after warmup")
    if len(mke.clf.replay) < 10:
        raise RuntimeError("fixture needs replay buffer len >= 10")

    warmup_entries = _snapshot_replay_entries(mke.clf.replay)
    replay_warmup = _replay_warmup_json(warmup_entries)

    total_steps_start = int(mke.clf._total_steps)
    enc_update_count_start = int(mke.clf.encoder._update_count)

    _OUT.mkdir(parents=True, exist_ok=True)
    mke.clf.replay = PriorityReplayBuffer(capacity=_REPLAY_CAP, rng=mke.clf._replay_rng)
    mke.clf._buf_len_cache = len(mke.clf.replay)

    cypha_save_binary(mke.clf.save_state(), str(_OUT / "before.cypha"))
    f_field = np.ascontiguousarray(mke.clf.memory.world.F_field, dtype=np.float64)
    (_OUT / "f_field.json").write_text(json.dumps(f_field.tolist(), indent=2), encoding="utf-8")

    rec = RecordingReplayRng(np.random.default_rng(_REPLAY_INNER_SEED))
    mke.clf._replay_rng = rec
    mke.clf.replay = PriorityReplayBuffer(capacity=_REPLAY_CAP, rng=rec)
    _rebuild_replay_from_entries(mke.clf, warmup_entries)
    mke.clf.enc_lr = float(_ENC_LR)
    mke.clf.replay_ratio = float(_REPLAY_RATIO_FIX)

    step_docs: list[dict] = []
    replay_u01_all: list[float] = []
    for _ in range(_N_EXT):
        _sync_world_log_norm_like_load_state(mke.clf)
        x = rng.standard_normal(d_in)
        y = float(rng.standard_normal() * 0.4 + 0.15)
        phi = np.asarray(mke.enc(x), dtype=np.float64).ravel()

        w_before = {k: v.copy() for k, v in mke._w.items()}
        P_before = {k: v.copy() for k, v in mke._P.items()}

        LLR, routing_labs = mke.clf.score_matrix(phi.reshape(1, -1), use_field=True)
        labs_list = list(routing_labs)
        mem_order = list(mke.clf.memory._classes.keys())
        if labs_list != mem_order:
            raise RuntimeError(f"routing lab order {labs_list} != memory order {mem_order}")

        p = _softmax_batch(LLR / (mke.clf.temperature + _EPS))[0]
        y_hat = sum(float(p[i]) * float(mke._w_for(labs_list[i]) @ phi) for i in range(len(labs_list)))
        err = float(y) - y_hat
        err_sq = err * err

        gh_scales: list[float] = []
        for i, lbl in enumerate(labs_list):
            pi = float(p[i])
            if pi < 0.02:
                gh_scales.append(0.0)
                continue
            if lbl in mke._chi:
                R_eff = _nig_R_eff(err_sq, mke._R_base, mke._chi[lbl], mke._psi[lbl])
                gh_scales.append(float(min(1.0, mke._R_base / max(R_eff, mke._R_base))))
            else:
                gh_scales.append(1.0)

        pred, _ = mke.clf.infer(x)
        if pred == "__unknown__":
            pred = labs_list[int(np.argmax(p))]
        router_train_label = str(pred)

        fd, tmp_cypha = tempfile.mkstemp(suffix=".cypha")
        os.close(fd)
        try:
            cypha_save_binary(mke.clf.save_state(), tmp_cypha)
            buf_before = _snapshot_replay_entries(mke.clf.replay)

            pos0 = len(rec.values)
            err_sq_out = float(mke.train_step(x, y))
            segment = rec.values[pos0:]
            replay_u01_all.extend(segment)

            probe = CyphaDIF(
                mke.enc,
                field_dim=field_dim,
                rng=np.random.default_rng(992),
                replay_rng=ListReplayRng(list(segment)),
                enc_lr=float(mke.clf.enc_lr),
                replay_ratio=float(mke.clf.replay_ratio),
            )
            probe.load_state(cypha_load_binary(tmp_cypha))
            _rebuild_replay_from_entries(probe, buf_before)
            expected_router_loss = float(
                probe.train_step(np.asarray(x, dtype=np.float64), router_train_label)
            )
        finally:
            try:
                os.unlink(tmp_cypha)
            except OSError:
                pass

        if not np.isclose(err_sq_out, err_sq, rtol=0.0, atol=1e-9):
            raise RuntimeError(f"err_sq mismatch {err_sq_out} vs recomputed {err_sq}")

        enc_w_after = np.ascontiguousarray(mke.clf.encoder.W, dtype=np.float64).reshape(-1, order="C")
        w_after = {k: v.copy() for k, v in mke._w.items()}
        P_after = {k: v.copy() for k, v in mke._P.items()}

        step_docs.append(
            {
                "x": np.asarray(x, dtype=np.float64).tolist(),
                "y": float(y),
                "expected_phi": phi.tolist(),
                "routing_labs": labs_list,
                "routing_probs": [float(v) for v in p],
                "expected_err_sq": float(err_sq),
                "gh_scales": gh_scales,
                "router_train_label": router_train_label,
                "expected_router_loss": float(expected_router_loss),
                "w_before": {k: v.astype(np.float64).tolist() for k, v in w_before.items()},
                "P_before": {k: _flatten_p(v) for k, v in P_before.items()},
                "w_after": {k: v.astype(np.float64).tolist() for k, v in w_after.items()},
                "P_after": {k: _flatten_p(v) for k, v in P_after.items()},
                "enc_w_rowmajor": enc_w_after.tolist(),
            }
        )

    doc = {
        "fixture_schema": 2,
        "n_extended_steps": _N_EXT,
        "description": (
            f"Multi-step MKERegressor.train_step ({_N_EXT} steps, enc_lr>0, replay_ratio>0, replay_u01); "
            "expected_router_loss from live train_step replay segment + probe ListReplayRng"
        ),
        "d_in": d_in,
        "D_rff": D,
        "total_steps_start": total_steps_start,
        "enc_update_count_start": enc_update_count_start,
        "replay_warmup": replay_warmup,
        "replay_u01": replay_u01_all,
        "rff_W_rowmajor": np.ascontiguousarray(mke.enc.W, dtype=np.float64).reshape(-1, order="C").tolist(),
        "rff_b": np.asarray(mke.enc.b, dtype=np.float64).tolist(),
        "temperature": float(mke.clf.temperature),
        "forgetting_factor": float(mke.forgetting_factor),
        "world_lr": float(mke.clf.world_lr),
        "delta_lr": float(mke.clf.delta_lr),
        "ood_sigma": float(mke.clf.ood_sigma),
        "enc_lr": float(mke.clf.enc_lr),
        "replay_ratio": float(_REPLAY_RATIO_FIX),
        "replay_cap": 10000,
        "align_every": 500,
        "temp_recalib_every": 0,
        "rng_seed": 42,
        "steps": step_docs,
    }
    (_OUT / "sidecar.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(
        f"Wrote {_OUT / 'sidecar.json'} (+ before.cypha, f_field.json); "
        f"warmup_entries={len(replay_warmup)} len(replay_u01)={len(replay_u01_all)}"
    )


if __name__ == "__main__":
    main()
