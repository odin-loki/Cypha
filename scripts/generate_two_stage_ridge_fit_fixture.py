#!/usr/bin/env python3
"""
Emit ``parity_fixtures/two_stage_ridge_fit/sidecar.json`` for
``regression_two_stage_ridge_fit_parity``.

Mirrors ``TwoStageDIFRegressor.fit`` **ridge blocks** given fixed LLR (no CyphaDIF training).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "parity_fixtures" / "two_stage_ridge_fit" / "sidecar.json"


def main() -> None:
    rng = np.random.default_rng(303)
    n, K, d_in, D2 = 24, 3, 6, 12
    llr = rng.standard_normal((n, K))
    X = rng.standard_normal((n, d_in))
    y = rng.standard_normal(n) * 0.5 + 0.3 * X[:, 0]
    y_mean = float(y.mean())
    y_std = float(max(float(y.std()), 1e-8))
    yn = (y - y_mean) / y_std
    lam1, lam2 = 0.03, 0.08
    W2 = rng.standard_normal((D2, d_in))
    b_rff = rng.uniform(0.0, 2.0 * math.pi, size=D2)

    F1 = np.c_[llr, X, np.ones(n)]
    lam1s = lam1 * n
    w_full = np.linalg.solve(F1.T @ F1 + lam1s * np.eye(K + d_in + 1), F1.T @ yn)
    w1 = w_full[:-1].copy()
    b1 = float(w_full[-1])
    y_s1 = F1 @ w_full
    res = yn - y_s1
    scale = math.sqrt(2.0 / D2)
    phi = scale * np.cos(X @ W2.T + b_rff)
    phi_aug = np.c_[phi, np.ones(n)]
    lam2s = lam2 * n
    w2_full = np.linalg.solve(phi_aug.T @ phi_aug + lam2s * np.eye(D2 + 1), phi_aug.T @ res)
    w2 = w2_full[:-1].copy()
    b2 = float(w2_full[-1])
    yn_hat = y_s1 + phi @ w2 + b2

    doc = {
        "fixture_schema": 1,
        "n": n,
        "K": K,
        "d_in": d_in,
        "D2": D2,
        "lam1": lam1,
        "lam2": lam2,
        "llr_rowmajor": llr.astype(np.float64).ravel(order="C").tolist(),
        "X_rowmajor": X.astype(np.float64).ravel(order="C").tolist(),
        "y_raw": y.astype(np.float64).tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
        "enc2_W": W2.astype(np.float64).ravel(order="C").tolist(),
        "enc2_b": b_rff.astype(np.float64).tolist(),
        "expected_w1": w1.astype(np.float64).tolist(),
        "expected_b1": b1,
        "expected_w2": w2.astype(np.float64).tolist(),
        "expected_b2": b2,
        "expected_yn_hat": yn_hat.astype(np.float64).tolist(),
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}")


if __name__ == "__main__":
    main()
