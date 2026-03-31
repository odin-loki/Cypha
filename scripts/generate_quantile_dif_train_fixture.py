#!/usr/bin/env python3
"""
Emit ``parity_fixtures/quantile_dif_train/`` for ``quantile_dif_train_parity``.

Replays a quantile-style label sequence (``_ts_*``) on a fresh ``CyphaDIF`` with
``replay_ratio=0`` so priority replay never runs — no Python ``np.random`` vs native
``std::mt19937`` mismatch even when the buffer has 10+ samples.

Writes ``before.cypha``, ``f_field.json``, ``sidecar.json`` (steps + expected field-conditioned LLR).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Cypha import CyphaDIF, VectorEncoder, cypha_save_binary

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "parity_fixtures" / "quantile_dif_train"


def main() -> None:
    d_in = 4
    field_dim = 24
    k_bins = 3
    # >10 steps fills replay buffer; replay_ratio=0 skips sampling (Python + native).
    n = 15
    seed_clf = 7755
    seed_data = 7756

    rng_clf = np.random.default_rng(seed_clf)
    rng_data = np.random.default_rng(seed_data)

    # enc_lr=0 avoids contrastive / deliberate encoder drift while we parity memory+field+context.
    clf = CyphaDIF(
        VectorEncoder(d_in),
        field_dim=field_dim,
        rng=rng_clf,
        enc_lr=0.0,
        replay_ratio=0.0,
    )
    _OUT.mkdir(parents=True, exist_ok=True)
    cypha_save_binary(clf.save_state(), str(_OUT / "before.cypha"))
    f_field = np.ascontiguousarray(clf.memory.world.F_field, dtype=np.float64)
    (_OUT / "f_field.json").write_text(json.dumps(f_field.tolist(), indent=2), encoding="utf-8")

    x_mat = rng_data.standard_normal((n, d_in)).astype(np.float64)
    y_norm = x_mat[:, 0] * 0.5 + x_mat[:, 1] * 0.3 + rng_data.standard_normal(n) * 0.12
    quantiles = np.quantile(y_norm, np.linspace(0.0, 1.0, k_bins + 1))
    perm = rng_data.permutation(n)

    steps: list[dict] = []
    for i in perm:
        ii = int(i)
        k = int(np.searchsorted(quantiles[1:-1], float(y_norm[ii])))
        steps.append({"x": x_mat[ii].tolist(), "label": f"_ts_{k}"})

    expected_step_losses: list[float] = []
    for s in steps:
        expected_step_losses.append(
            float(clf.train_step(np.asarray(s["x"], dtype=np.float64), s["label"]))
        )

    rows = [x_mat[j] for j in range(n)]
    h = clf.batch_encode(rows)
    llr, labs = clf.score_matrix(h, use_field=True)

    doc = {
        "fixture_schema": 1,
        "description": "Quantile-style DIF train replay (replay_ratio=0, enc_lr=0); LLR vs batch_llr_from_x",
        "d_in": d_in,
        "field_dim": field_dim,
        "n": n,
        "n_steps": len(steps),
        "K": int(llr.shape[1]),
        "label_order": labs,
        "world_lr": float(clf.world_lr),
        "delta_lr": float(clf.delta_lr),
        "enc_lr": 0.0,
        "ood_sigma": float(clf.ood_sigma),
        "temperature": float(clf.temperature),
        "replay_ratio": 0.0,
        "replay_cap": 10000,
        "align_every": 500,
        "temp_recalib_every": 0,
        "rng_seed": seed_clf,
        "expected_step_losses": expected_step_losses,
        "steps": steps,
        "x_rowmajor": x_mat.ravel(order="C").tolist(),
        "expected_llr_rowmajor": np.ascontiguousarray(llr, dtype=np.float64).ravel(order="C").tolist(),
    }
    (_OUT / "sidecar.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT / 'sidecar.json'} (+ before.cypha, f_field.json)")


if __name__ == "__main__":
    main()
