#!/usr/bin/env python3
"""
Emit ``parity_fixtures/rff_regression/sidecar.json`` for CTest ``native_regression_rff``.

Matches native ``rff_encode_batch_rowmajor``, ``ridge_fit_bias``, ``linear_predict_with_bias``,
``mke_expert_linear_dots`` (RFF + ridge + MKERegressor-style expert dots).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "parity_fixtures" / "rff_regression" / "sidecar.json"


def main() -> int:
    rng = np.random.default_rng(2026)
    d_in, D, N = 4, 20, 40
    X = rng.standard_normal((N, d_in))
    W = rng.standard_normal((D, d_in))
    b = rng.uniform(0.0, 2.0 * math.pi, size=D)
    scale = math.sqrt(2.0 / D)
    phi = scale * np.cos(X @ W.T + b)
    y = 0.5 * X[:, 0] + 0.1 * rng.standard_normal(N)
    y_mean = float(y.mean())
    y_std = float(max(float(y.std()), 1e-12))
    yn = (y - y_mean) / y_std
    lam = 1e-2 * D
    p = D + 1
    phi_aug = np.c_[phi, np.ones(N)]
    ab = phi_aug.T @ phi_aug + np.diag(np.append(np.full(D, lam), 0.0))
    rhs = phi_aug.T @ yn
    coef = np.linalg.solve(ab, rhs)
    pred_norm = phi @ coef[:D] + coef[D]
    y_pred = pred_norm * y_std + y_mean

    d_phi = np.asarray(phi[0], dtype=np.float64)
    K = 5
    w_exp = rng.standard_normal((K, D))
    dots = (w_exp @ d_phi).tolist()

    doc = {
        "fixture_schema": 1,
        "rff_ridge": {
            "n": N,
            "d_in": d_in,
            "D": D,
            "lam": lam,
            "y_mean": y_mean,
            "y_std": y_std,
            "X": X.ravel().tolist(),
            "W": W.ravel().tolist(),
            "b": b.tolist(),
            "y_raw": y.tolist(),
            "expected_phi_rowmajor": phi.ravel().tolist(),
            "expected_coef": coef.tolist(),
            "expected_y_pred": y_pred.tolist(),
        },
        "mke_dots": {
            "d_feat": D,
            "K": K,
            "phi": d_phi.tolist(),
            "W_experts_rowmajor": w_exp.ravel().tolist(),
            "expected_dots": dots,
        },
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
