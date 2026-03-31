#!/usr/bin/env python3
"""
Emit ``parity_fixtures/mke_train_step/`` for ``mke_train_step_parity``.

One ``MKERegressor.train_step`` after warmup: RFF φ → ``W_proj @ φ`` matches native ``batch_encode``
when the buffer passed to ``dif_train_step_vector`` is φ (``enc_lr=0``, ``replay_ratio=0``).

Writes ``before.cypha``, ``f_field.json``, ``sidecar.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from Cypha import (  # noqa: E402
    CyphaDIF,
    MKERegressor,
    _EPS,
    _nig_R_eff,
    _softmax_batch,
    cypha_load_binary,
    cypha_save_binary,
)

_OUT = _ROOT / "parity_fixtures" / "mke_train_step"


def _flatten_p(P: np.ndarray) -> list[float]:
    return np.asarray(P, dtype=np.float64).reshape(-1, order="C").tolist()


def main() -> None:
    rng = np.random.default_rng(9026)
    d_in = 3
    D = 32
    K = 2
    field_dim = 24
    rng_seed = 901

    X_seed = rng.standard_normal((100, d_in))
    y_seed = X_seed[:, 0] * 0.6 + rng.standard_normal(100) * 0.25

    mke = MKERegressor.from_data(
        X_seed,
        y_seed,
        K=K,
        D=D,
        field_dim=field_dim,
        rng_seed=rng_seed,
    )
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

    _OUT.mkdir(parents=True, exist_ok=True)

    x = rng.standard_normal(d_in)
    y = float(rng.standard_normal() * 0.4 + 0.15)

    # Snapshot router **before** this (x, y) step — matches ``MKERegressor.train_step`` entry.
    cypha_save_binary(mke.clf.save_state(), str(_OUT / "before.cypha"))
    f_field = np.ascontiguousarray(mke.clf.memory.world.F_field, dtype=np.float64)
    (_OUT / "f_field.json").write_text(json.dumps(f_field.tolist(), indent=2), encoding="utf-8")

    total_steps_start = int(mke.clf._total_steps)

    # After save, avoid touching ``mke.clf`` until ``mke.train_step`` so the
    # live router matches a fresh ``cypha_load_binary`` (native path). Routing scores use a
    # reload of ``before.cypha``, not the in-memory object (infer/score can advance caches).
    clf_snap = CyphaDIF(
        mke.enc,
        field_dim=field_dim,
        rng=np.random.default_rng(997),
        enc_lr=0.0,
        replay_ratio=0.0,
    )
    clf_snap.load_state(cypha_load_binary(str(_OUT / "before.cypha")))

    phi = np.asarray(mke.enc(x), dtype=np.float64).ravel()
    LLR, routing_labs = clf_snap.score_matrix(phi.reshape(1, -1), use_field=True)
    labs_list = list(routing_labs)
    mem_order = list(clf_snap.memory._classes.keys())
    if labs_list != mem_order:
        raise RuntimeError(f"routing lab order {labs_list} != memory order {mem_order}")

    p = _softmax_batch(LLR / (clf_snap.temperature + _EPS))[0]
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

    w_before = {k: v.copy() for k, v in mke._w.items()}
    P_before = {k: v.copy() for k, v in mke._P.items()}

    pred, _ = clf_snap.infer(x)
    if pred == "__unknown__":
        pred = labs_list[int(np.argmax(p))]

    clf_loss_probe = CyphaDIF(
        mke.enc,
        field_dim=field_dim,
        rng=np.random.default_rng(998),
        enc_lr=0.0,
        replay_ratio=0.0,
    )
    clf_loss_probe.load_state(cypha_load_binary(str(_OUT / "before.cypha")))
    pred_probe, _ = clf_loss_probe.infer(x)
    if pred_probe == "__unknown__":
        pred_probe = labs_list[int(np.argmax(p))]
    if pred_probe != pred:
        raise RuntimeError(f"infer label reload mismatch: {pred_probe!r} vs {pred!r}")
    expected_router_loss = float(clf_loss_probe.train_step(x, pred_probe))

    err_sq_out = float(mke.train_step(x, y))

    if not np.isclose(err_sq_out, err_sq, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"err_sq mismatch {err_sq_out} vs recomputed {err_sq}")

    w_after = {k: v.copy() for k, v in mke._w.items()}
    P_after = {k: v.copy() for k, v in mke._P.items()}

    doc = {
        "fixture_schema": 1,
        "description": "One MKERegressor.train_step: RFF + expert RLS + CyphaDIF.train_step(pred); native uses phi buffer + dif_train_step_vector",
        "d_in": d_in,
        "D_rff": D,
        "total_steps_start": total_steps_start,
        "rff_W_rowmajor": np.ascontiguousarray(mke.enc.W, dtype=np.float64).reshape(-1, order="C").tolist(),
        "rff_b": np.asarray(mke.enc.b, dtype=np.float64).tolist(),
        "x": np.asarray(x, dtype=np.float64).tolist(),
        "y": float(y),
        "expected_phi": phi.tolist(),
        "routing_labs": labs_list,
        "expected_err_sq": float(err_sq),
        "router_train_label": str(pred_probe),
        "expected_router_loss": float(expected_router_loss),
        "temperature": float(clf_snap.temperature),
        "forgetting_factor": float(mke.forgetting_factor),
        "label_order": labs_list,
        "K": len(labs_list),
        "world_lr": float(clf_snap.world_lr),
        "delta_lr": float(clf_snap.delta_lr),
        "ood_sigma": float(clf_snap.ood_sigma),
        "enc_lr": 0.0,
        "replay_ratio": 0.0,
        "replay_cap": 10000,
        "align_every": 500,
        "temp_recalib_every": 0,
        "rng_seed": 42,
        "routing_probs": [float(v) for v in p],
        "gh_scales": gh_scales,
        "w_before": {k: v.astype(np.float64).tolist() for k, v in w_before.items()},
        "P_before": {k: _flatten_p(v) for k, v in P_before.items()},
        "w_after": {k: v.astype(np.float64).tolist() for k, v in w_after.items()},
        "P_after": {k: _flatten_p(v) for k, v in P_after.items()},
    }
    (_OUT / "sidecar.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT / 'sidecar.json'} (+ before.cypha, f_field.json); router_label={pred_probe!r}")


if __name__ == "__main__":
    main()
