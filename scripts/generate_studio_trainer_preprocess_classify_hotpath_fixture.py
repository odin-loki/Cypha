#!/usr/bin/env python3
"""
Emit ``parity_fixtures/studio_trainer_preprocess_classify_hotpath/`` for
``preprocess_train_classify_parity``.

Mirrors Studio **Trainer**: ``Preprocessor.fit`` on raw **X**, epoch permutations on indices,
``train_step(preprocessor.transform_one(x_raw), label)``. Native path applies the same
``preprocessor.json`` then ``dif_train_classify_sequence``.

Uses ``enc_lr=0``, ``replay_ratio=0`` to minimize cross-RNG surface; focus is preprocess + train I/O.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Cypha import CyphaDIF, VectorEncoder, cypha_save_binary
from cypha_studio.core.dataset import Preprocessor

_OUT = _ROOT / "parity_fixtures" / "studio_trainer_preprocess_classify_hotpath"


def main() -> None:
    raw_dim = 8
    field_dim = 24
    k_bins = 3
    n_samples = 7
    n_epochs = 2
    trainer_seed = 41
    rng_data = np.random.default_rng(7001)
    rng_clf = np.random.default_rng(7002)

    X_raw = rng_data.standard_normal((n_samples, raw_dim)).astype(np.float64)
    pre = Preprocessor(scale=True, pca_dim=4, seed=1337)
    pre.fit(X_raw)
    X_pp = np.asarray(pre.transform(X_raw), dtype=np.float64)
    d_pp = int(X_pp.shape[1])

    clf = CyphaDIF(
        VectorEncoder(d_pp),
        field_dim=field_dim,
        rng=rng_clf,
        enc_lr=0.0,
        replay_ratio=0.0,
    )
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "preprocessor.json").write_text(json.dumps(pre.save_state(), indent=2), encoding="utf-8")
    cypha_save_binary(clf.save_state(), str(_OUT / "before.cypha"))
    f_field = np.ascontiguousarray(clf.memory.world.F_field, dtype=np.float64)
    (_OUT / "f_field.json").write_text(json.dumps(f_field.tolist(), indent=2), encoding="utf-8")

    y_score = X_pp[:, 0] * 0.4 + X_pp[:, 1] * 0.2 + rng_data.standard_normal(n_samples) * 0.08
    quantiles = np.quantile(y_score, np.linspace(0.0, 1.0, k_bins + 1))

    steps: list[dict] = []
    expected_step_losses: list[float] = []
    for epoch in range(n_epochs):
        perm = np.random.default_rng(trainer_seed + epoch).permutation(n_samples)
        for idx in perm:
            ii = int(idx)
            k = int(np.searchsorted(quantiles[1:-1], float(y_score[ii])))
            lab = f"_prep_{k}"
            steps.append({"x_raw": X_raw[ii].tolist(), "label": lab})
            expected_step_losses.append(float(clf.train_step(X_pp[ii], lab)))

    rows = [X_pp[j] for j in range(n_samples)]
    h = clf.batch_encode(rows)
    llr, labs = clf.score_matrix(h, use_field=True)

    doc = {
        "fixture_schema": 1,
        "description": "Preprocessor fit/transform + Trainer.fit order train_step (enc_lr=0, replay_ratio=0)",
        "d_raw": raw_dim,
        "d_in": d_pp,
        "field_dim": field_dim,
        "n": n_samples,
        "n_steps": len(steps),
        "n_epochs": n_epochs,
        "trainer_seed": trainer_seed,
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
        "rng_seed": 7002,
        "expected_step_losses": expected_step_losses,
        "steps": steps,
        "x_rowmajor": X_pp.ravel(order="C").tolist(),
        "expected_llr_rowmajor": np.ascontiguousarray(llr, dtype=np.float64).ravel(order="C").tolist(),
    }
    (_OUT / "sidecar.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT} (n_steps={len(steps)}, d_raw={raw_dim}, d_in={d_pp})")


if __name__ == "__main__":
    main()
