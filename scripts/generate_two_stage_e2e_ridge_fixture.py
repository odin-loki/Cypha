#!/usr/bin/env python3
"""
Emit ``parity_fixtures/two_stage_e2e_ridge/sidecar.json``.

Fits a real ``TwoStageDIFRegressor`` (quantile ``CyphaDIF`` router + ridge stages), then exports
training **LLR**, **X**, **y**, RFF params, and fitted **w1/b1/w2/b2**. Native
``two_stage_dif_ridge_fit_from_llr`` must reproduce the same coefficients — proves the C++ ridge
blocks match Python on **router-produced** LLR (not synthetic).

Repo root must be on ``PYTHONPATH``. Regenerate when ``TwoStageDIFRegressor`` / LLR numerics change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Cypha import TwoStageDIFRegressor  # noqa: E402

_OUT = _ROOT / "parity_fixtures" / "two_stage_e2e_ridge" / "sidecar.json"


def main() -> None:
    rng = np.random.default_rng(424242)
    n, d_in = 38, 5
    X = rng.standard_normal((n, d_in))
    y = 1.15 * X[:, 0] + 0.42 * X[:, 1] + 0.2 * rng.standard_normal(n)

    K, D2 = 4, 14
    lam1, lam2 = 0.025, 0.06
    seed = 991
    reg = TwoStageDIFRegressor(K=K, lam1=lam1, lam2=lam2, D=D2, seed=seed)
    reg.fit(X, y, field_dim=40)

    if reg.clf is None or reg._w1 is None or reg._enc2 is None:
        raise RuntimeError("fit failed")
    LLR, _ = reg.clf.score_matrix(reg.clf.batch_encode(X))
    LLR = np.asarray(LLR, dtype=np.float64)
    if LLR.shape != (n, K):
        raise RuntimeError(f"unexpected LLR shape {LLR.shape}")

    yn = (y - reg._y_mean) / max(reg._y_std, 1e-8)
    F1 = np.c_[LLR, X, np.ones(n)]
    lam1s = lam1 * n
    w_full = np.linalg.solve(F1.T @ F1 + lam1s * np.eye(K + d_in + 1), F1.T @ yn)
    y_s1 = F1 @ w_full
    res = yn - y_s1
    enc = reg._enc2
    PHI = enc.batch_encode(X)
    PHIb = np.c_[PHI, np.ones(n)]
    lam2s = lam2 * n
    w2_full = np.linalg.solve(PHIb.T @ PHIb + lam2s * np.eye(D2 + 1), PHIb.T @ res)
    yn_hat = y_s1 + PHI @ w2_full[:-1] + w2_full[-1]

    np.testing.assert_allclose(w_full[:-1], reg._w1, rtol=0, atol=1e-10)
    np.testing.assert_allclose(w_full[-1], reg._b1, rtol=0, atol=1e-10)
    np.testing.assert_allclose(w2_full[:-1], reg._w2, rtol=0, atol=1e-10)
    np.testing.assert_allclose(w2_full[-1], reg._b2, rtol=0, atol=1e-10)

    doc = {
        "fixture_schema": 1,
        "description": "LLR from TwoStageDIFRegressor.fit (quantile CyphaDIF); native ridge must match",
        "n": int(n),
        "K": int(K),
        "d_in": int(d_in),
        "D2": int(D2),
        "lam1": float(lam1),
        "lam2": float(lam2),
        "llr_rowmajor": LLR.ravel(order="C").tolist(),
        "X_rowmajor": np.asarray(X, dtype=np.float64).ravel(order="C").tolist(),
        "y_raw": np.asarray(y, dtype=np.float64).tolist(),
        "y_mean": float(reg._y_mean),
        "y_std": float(reg._y_std),
        "enc2_W": np.asarray(enc.W, dtype=np.float64).ravel(order="C").tolist(),
        "enc2_b": np.asarray(enc.b, dtype=np.float64).tolist(),
        "expected_w1": np.asarray(reg._w1, dtype=np.float64).tolist(),
        "expected_b1": float(reg._b1),
        "expected_w2": np.asarray(reg._w2, dtype=np.float64).tolist(),
        "expected_b2": float(reg._b2),
        "expected_yn_hat": np.asarray(yn_hat, dtype=np.float64).tolist(),
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}")


if __name__ == "__main__":
    main()
