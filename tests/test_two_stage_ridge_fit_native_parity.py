"""
Native: ``regression_two_stage_ridge_fit_parity`` vs ``parity_fixtures/two_stage_ridge_fit/sidecar.json``.

Closed-form two-stage ridge (``two_stage_dif_ridge_fit_from_llr``) and batched normalized predict
(``two_stage_dif_predict_batch``). CTest: ``native_regression_two_stage_ridge_fit`` (milestone **≥ 7**).
Override: ``CYPHA_TWO_STAGE_RIDGE_FIT_PARITY_BIN``.

Also checks the committed sidecar stays consistent with NumPy (no C++ build required).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_SIDE = _ROOT / "parity_fixtures" / "two_stage_ridge_fit" / "sidecar.json"


def test_two_stage_ridge_fit_sidecar_numpy_consistent():
    if not _SIDE.is_file():
        pytest.skip("two_stage_ridge_fit sidecar missing")
    j = json.loads(_SIDE.read_text(encoding="utf-8"))
    n, K, d_in, D2 = j["n"], j["K"], j["d_in"], j["D2"]
    llr = np.asarray(j["llr_rowmajor"], dtype=np.float64).reshape(n, K)
    X = np.asarray(j["X_rowmajor"], dtype=np.float64).reshape(n, d_in)
    y = np.asarray(j["y_raw"], dtype=np.float64)
    y_mean, y_std = float(j["y_mean"]), float(j["y_std"])
    yn = (y - y_mean) / max(y_std, 1e-8)
    lam1, lam2 = float(j["lam1"]), float(j["lam2"])
    W2 = np.asarray(j["enc2_W"], dtype=np.float64).reshape(D2, d_in)
    b_rff = np.asarray(j["enc2_b"], dtype=np.float64)
    F1 = np.c_[llr, X, np.ones(n)]
    lam1s = lam1 * n
    w_full = np.linalg.solve(F1.T @ F1 + lam1s * np.eye(K + d_in + 1), F1.T @ yn)
    np.testing.assert_allclose(w_full[:-1], np.asarray(j["expected_w1"], dtype=np.float64), rtol=0, atol=1e-12)
    assert abs(float(w_full[-1]) - float(j["expected_b1"])) < 1e-12
    y_s1 = F1 @ w_full
    res = yn - y_s1
    scale = math.sqrt(2.0 / D2)
    phi = scale * np.cos(X @ W2.T + b_rff)
    phi_aug = np.c_[phi, np.ones(n)]
    lam2s = lam2 * n
    w2_full = np.linalg.solve(phi_aug.T @ phi_aug + lam2s * np.eye(D2 + 1), phi_aug.T @ res)
    np.testing.assert_allclose(w2_full[:-1], np.asarray(j["expected_w2"], dtype=np.float64), rtol=0, atol=1e-12)
    assert abs(float(w2_full[-1]) - float(j["expected_b2"])) < 1e-12
    yn_hat = y_s1 + phi @ w2_full[:-1] + w2_full[-1]
    np.testing.assert_allclose(yn_hat, np.asarray(j["expected_yn_hat"], dtype=np.float64), rtol=0, atol=1e-11)


def test_two_stage_ridge_fit_parity_subprocess():
    if not _SIDE.is_file():
        pytest.skip(
            "parity_fixtures/two_stage_ridge_fit/sidecar.json missing — "
            "run scripts/generate_two_stage_ridge_fit_fixture.py"
        )
    r = run_native_executable(
        "regression_two_stage_ridge_fit_parity",
        [_SIDE],
        timeout=60,
        env_override="CYPHA_TWO_STAGE_RIDGE_FIT_PARITY_BIN",
    )
    if r is None:
        pytest.skip("regression_two_stage_ridge_fit_parity not built (cmake native/build)")
    assert r.returncode == 0, (r.stdout, r.stderr)
