"""
Native M4b: ``regression_rff_parity`` vs ``parity_fixtures/rff_regression/sidecar.json``.

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

_SIDE = _ROOT / "parity_fixtures" / "rff_regression" / "sidecar.json"


def test_rff_regression_sidecar_numpy_consistent():
    if not _SIDE.is_file():
        pytest.skip("parity_fixtures/rff_regression/sidecar.json missing — run scripts/generate_rff_regression_fixture.py")
    doc = json.loads(_SIDE.read_text(encoding="utf-8"))
    rr = doc["rff_ridge"]
    n, d_in, D = rr["n"], rr["d_in"], rr["D"]
    X = np.asarray(rr["X"], dtype=np.float64).reshape(n, d_in)
    W = np.asarray(rr["W"], dtype=np.float64).reshape(D, d_in)
    b = np.asarray(rr["b"], dtype=np.float64)
    scale = math.sqrt(2.0 / D)
    phi = scale * np.cos(X @ W.T + b)
    exp_phi = np.asarray(rr["expected_phi_rowmajor"], dtype=np.float64).reshape(n, D)
    np.testing.assert_allclose(phi, exp_phi, rtol=0, atol=1e-12)

    lam = float(rr["lam"])
    y_mean, y_std = float(rr["y_mean"]), float(rr["y_std"])
    y = np.asarray(rr["y_raw"], dtype=np.float64)
    yn = (y - y_mean) / y_std
    phi_aug = np.c_[phi, np.ones(n)]
    ab = phi_aug.T @ phi_aug + np.diag(np.append(np.full(D, lam), 0.0))
    coef = np.linalg.solve(ab, phi_aug.T @ yn)
    np.testing.assert_allclose(coef, np.asarray(rr["expected_coef"], dtype=np.float64), rtol=0, atol=1e-10)
    pred = (phi @ coef[:D] + coef[D]) * y_std + y_mean
    np.testing.assert_allclose(pred, np.asarray(rr["expected_y_pred"], dtype=np.float64), rtol=0, atol=1e-10)

    md = doc["mke_dots"]
    d_feat, K = md["d_feat"], md["K"]
    ph1 = np.asarray(md["phi"], dtype=np.float64)
    wexp = np.asarray(md["W_experts_rowmajor"], dtype=np.float64).reshape(K, d_feat)
    dots = wexp @ ph1
    np.testing.assert_allclose(dots, np.asarray(md["expected_dots"], dtype=np.float64), rtol=0, atol=1e-12)


def test_regression_rff_parity_subprocess():
    if not _SIDE.is_file():
        pytest.skip("parity_fixtures/rff_regression/sidecar.json missing")
    r = run_native_executable(
        "regression_rff_parity",
        [_SIDE],
        timeout=60,
        env_override="CYPHA_REGRESSION_RFF_PARITY_BIN",
    )
    if r is None:
        pytest.skip("regression_rff_parity not built (cmake native/build or native/build-exp; WSL ELF ok on Windows)")
    assert r.returncode == 0, (r.stdout, r.stderr)
