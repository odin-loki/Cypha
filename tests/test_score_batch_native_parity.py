"""
``score_batch_parity`` vs ``parity_fixtures/score_batch/sidecar.json``.

Goldens: ``cypha_accel.score_batch.project_features`` + ``fused_score_llr`` (same
fused LLR path as ``CyphaDIF.score_matrix``). CTest: ``native_score_batch``.
Override: ``CYPHA_SCORE_BATCH_PARITY_BIN``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cypha_accel.score_batch import fused_score_llr, project_features
from tests.native_subprocess import run_native_executable

_SIDE = _ROOT / "parity_fixtures" / "score_batch" / "sidecar.json"


def test_score_batch_sidecar_python_reference():
    if not _SIDE.is_file():
        pytest.skip("run scripts/generate_score_batch_fixture.py")
    j = json.loads(_SIDE.read_text(encoding="utf-8"))
    n, d, K = int(j["n"]), int(j["d"]), int(j["K"])
    F = np.asarray(j["F_rowmajor"], dtype=np.float64).reshape(n, d)
    W = np.asarray(j["W_enc_rowmajor"], dtype=np.float64).reshape(d, d)
    mu0 = np.asarray(j["mu0"], dtype=np.float64)
    inv_v = np.asarray(j["inv_v"], dtype=np.float64)
    D = np.asarray(j["D_rowmajor"], dtype=np.float64).reshape(K, d)
    D_sq = np.asarray(j["D_sq"], dtype=np.float64)
    u_k = np.asarray(j["u_k"], dtype=np.float64)
    ctx = np.asarray(j["ctx"], dtype=np.float64)
    H = project_features(F, W)
    LLR = fused_score_llr(H, mu0, inv_v, D, D_sq, u_k, ctx)
    np.testing.assert_allclose(H.ravel(order="C"), j["expected_H_rowmajor"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(LLR.ravel(order="C"), j["expected_LLR_rowmajor"], rtol=0, atol=1e-12)


def test_score_batch_parity_subprocess():
    if not _SIDE.is_file():
        pytest.skip("run scripts/generate_score_batch_fixture.py")
    r = run_native_executable(
        "score_batch_parity",
        [_SIDE],
        timeout=60,
        env_override="CYPHA_SCORE_BATCH_PARITY_BIN",
    )
    if r is None:
        pytest.skip("score_batch_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
