#!/usr/bin/env python3
"""
Emit ``parity_fixtures/studio_trainer_gh_classify_hotpath/`` for ``quantile_dif_train_parity``.

Mirrors ``Trainer.fit`` when ``config.gh_protect`` is set: epoch permutations + ``gh_train_step``
(``chi`` / ``psi`` threaded). ``enc_lr=0`` and ``replay_ratio=0`` keep RNG surface small; GH + NIG
adaptation is the focus.

Sidecar: ``use_gh``, ``gh_inv_v_clean``, ``gh_r_base``, ``nig_alpha``, ``expected_chi_end`` /
``expected_psi_end``, plus standard multi-step fields consumed by native
``dif_gh_train_classify_sequence``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Cypha import CyphaDIF, VectorEncoder, cypha_save_binary

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "parity_fixtures" / "studio_trainer_gh_classify_hotpath"

_EPS = 1e-8


def main() -> None:
    d_in = 4
    field_dim = 24
    k_bins = 3
    n_samples = 5
    n_epochs = 2
    trainer_seed = 43
    rng_data_seed = 9002
    clf_rng_seed = 8845

    rng_data = np.random.default_rng(rng_data_seed)
    rng_clf = np.random.default_rng(clf_rng_seed)

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

    gh_inv = np.asarray(clf.memory.world.inv_v, dtype=np.float64).copy()
    gh_r_base = float(1.0 / (float(gh_inv.mean()) + _EPS))

    x_mat = rng_data.standard_normal((n_samples, d_in)).astype(np.float64)
    y_norm = x_mat[:, 0] * 0.5 + x_mat[:, 1] * 0.3 + rng_data.standard_normal(n_samples) * 0.11
    quantiles = np.quantile(y_norm, np.linspace(0.0, 1.0, k_bins + 1))

    steps: list[dict] = []
    expected_step_losses: list[float] = []
    chi, psi = 1.0, 1.0
    nig_alpha = 0.98

    for epoch in range(n_epochs):
        perm = np.random.default_rng(trainer_seed + epoch).permutation(n_samples)
        for idx in perm:
            ii = int(idx)
            k = int(np.searchsorted(quantiles[1:-1], float(y_norm[ii])))
            lab = f"_ghp_{k}"
            x = x_mat[ii]
            steps.append({"x": x.tolist(), "label": lab})
            loss, _r_eff, chi, psi = clf.gh_train_step(x, lab, chi, psi, nig_alpha)
            expected_step_losses.append(float(loss))

    rows = [x_mat[j] for j in range(n_samples)]
    h = clf.batch_encode(rows)
    llr, labs = clf.score_matrix(h, use_field=True)

    doc = {
        "fixture_schema": 1,
        "description": "Studio Trainer.fit + gh_train_step; enc_lr=0 replay_ratio=0; chi/psi threaded",
        "use_gh": True,
        "gh_inv_v_clean": gh_inv.tolist(),
        "gh_r_base": gh_r_base,
        "chi_start": 1.0,
        "psi_start": 1.0,
        "nig_alpha": nig_alpha,
        "expected_chi_end": float(chi),
        "expected_psi_end": float(psi),
        "d_in": d_in,
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
        "rng_seed": clf_rng_seed,
        "expected_step_losses": expected_step_losses,
        "steps": steps,
        "x_rowmajor": x_mat.ravel(order="C").tolist(),
        "expected_llr_rowmajor": np.ascontiguousarray(llr, dtype=np.float64).ravel(order="C").tolist(),
    }
    (_OUT / "sidecar.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT / 'sidecar.json'} (+ before.cypha, f_field.json), n_steps={len(steps)}")


if __name__ == "__main__":
    main()
