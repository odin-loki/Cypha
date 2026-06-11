#!/usr/bin/env python3
"""
Emit ``parity_fixtures/kernel_llr/sidecar.json`` for ``kernel_llr_parity``.

Captures ``KernelMemory`` phi / score_all / update and blended LLRs vs linear
``score_matrix`` (Python ``CyphaDIF`` with ``use_kernel_llr=True``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Cypha import CyphaDIF, VectorEncoder
_OUT = _ROOT / "parity_fixtures" / "kernel_llr"


def _kernel_snapshot(km) -> dict:
    M = km.M
    d = km.feat_dim
    basis = km._basis[: km._n_basis].copy()
    full = np.zeros((M, d), dtype=np.float64)
    full[: km._n_basis] = basis
    weights = {lbl: w.tolist() for lbl, w in km._weights.items()}
    return {
        "feat_dim": int(d),
        "M": int(M),
        "gamma": float(km.gamma),
        "n_basis": int(km._n_basis),
        "n_seen": int(km._n_seen),
        "basis_rowmajor": full.ravel(order="C").tolist(),
        "weights": weights,
    }


def main() -> None:
    d_in = 4
    field_dim = 8
    seed = 4242
    blend = 0.5
    rng = np.random.default_rng(seed)

    clf = CyphaDIF(
        VectorEncoder(d_in),
        field_dim=field_dim,
        rng=rng,
        enc_lr=0.0,
        delta_lr=0.05,
        use_kernel_llr=True,
    )
    km = clf._kernel_mem
    assert km is not None

    labels = ["pos", "neg"]
    train_h = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float64),
        np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float64),
        np.array([0.8, 0.2, 0.1, 0.0], dtype=np.float64),
        np.array([0.2, 0.8, 0.0, 0.1], dtype=np.float64),
    ]
    train_y = ["pos", "neg", "pos", "neg", "pos", "neg"]
    for h, y in zip(train_h, train_y):
        clf.memory.train(h, y, h_field=clf.field.h, context_prior={}, temperature=clf.temperature,
                         ood_sigma=clf.ood_sigma, world_lr=0.0, delta_lr=0.05)
        km.update(h, y, labels, lr=0.05)

    if km._n_basis < 4:
        raise SystemExit("kernel basis too thin after training")

    h_test = np.stack(
        [
            np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float64),
            np.array([0.05, 0.95, 0.0, 0.0], dtype=np.float64),
            np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float64),
        ],
        axis=0,
    )
    n_test = h_test.shape[0]
    K = len(labels)

    phi_rows = np.stack([km._phi(h_test[i]) for i in range(n_test)], axis=0)
    kernel_scores = np.stack([km.score_all(h_test[i], labels) for i in range(n_test)], axis=0)

    linear_llr, _ = clf.score_matrix(h_test, use_field=True)
    blended = (1.0 - blend) * linear_llr + blend * kernel_scores

    km_before = _kernel_snapshot(km)

    # Deterministic update while basis is still filling (n_basis < M → no RNG).
    from Cypha import KernelMemory

    st = km_before
    km_up = KernelMemory(st["feat_dim"], M=st["M"], rng=rng)
    km_up._n_basis = st["n_basis"]
    km_up._n_seen = st["n_seen"]
    km_up._basis = np.asarray(st["basis_rowmajor"], dtype=np.float64).reshape(st["M"], st["feat_dim"])
    km_up._weights = {k: np.asarray(v, dtype=np.float64) for k, v in st["weights"].items()}
    h_up = np.array([0.7, 0.3, 0.05, 0.05], dtype=np.float64)
    lr = 0.05
    n_basis_before = int(km_up._n_basis)
    km_up.update(h_up, "pos", labels, lr=lr)
    weights_after = {lbl: km_up._weights[lbl].tolist() for lbl in labels}

    doc = {
        "fixture_schema": 1,
        "source": "Cypha.py KernelMemory + score_matrix blend",
        "labels": labels,
        "blend": blend,
        "kernel_state": km_before,
        "n_test": int(n_test),
        "K": K,
        "h_test_rowmajor": h_test.ravel(order="C").tolist(),
        "expected_phi_rowmajor": phi_rows.ravel(order="C").tolist(),
        "expected_kernel_scores_rowmajor": kernel_scores.ravel(order="C").tolist(),
        "linear_llr_rowmajor": linear_llr.ravel(order="C").tolist(),
        "expected_blended_rowmajor": blended.ravel(order="C").tolist(),
        "update_step": {
            "h": h_up.tolist(),
            "label": "pos",
            "all_labels": labels,
            "lr": lr,
            "n_basis_before": n_basis_before,
            "n_basis_after": int(km_up._n_basis),
            "weights_after": weights_after,
        },
    }

    _OUT.mkdir(parents=True, exist_ok=True)
    side = _OUT / "sidecar.json"
    side.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {side}")


if __name__ == "__main__":
    main()
