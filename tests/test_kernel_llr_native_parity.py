"""
``kernel_llr_parity`` vs ``parity_fixtures/kernel_llr/sidecar.json``.

Sidecar from ``scripts/generate_kernel_llr_fixture.py``. CTest: ``native_kernel_llr``.
Override: ``CYPHA_KERNEL_LLR_PARITY_BIN``.
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

from tests.native_subprocess import run_native_executable

_SIDE = _ROOT / "parity_fixtures" / "kernel_llr" / "sidecar.json"


def test_kernel_llr_sidecar_self_consistent():
    if not _SIDE.is_file():
        pytest.skip("run scripts/generate_kernel_llr_fixture.py")
    j = json.loads(_SIDE.read_text(encoding="utf-8"))
    feat_dim = j["kernel_state"]["feat_dim"]
    M = j["kernel_state"]["M"]
    n_test = j["n_test"]
    K = j["K"]
    blend = j["blend"]
    linear = np.asarray(j["linear_llr_rowmajor"], dtype=np.float64).reshape(n_test, K)
    kernel = np.asarray(j["expected_kernel_scores_rowmajor"], dtype=np.float64).reshape(n_test, K)
    exp_blend = np.asarray(j["expected_blended_rowmajor"], dtype=np.float64).reshape(n_test, K)
    got = (1.0 - blend) * linear + blend * kernel
    np.testing.assert_allclose(got, exp_blend, rtol=0, atol=1e-12)
    phi = np.asarray(j["expected_phi_rowmajor"], dtype=np.float64).reshape(n_test, M)
    assert phi.shape == (n_test, M)


def test_kernel_llr_parity_subprocess():
    if not _SIDE.is_file():
        pytest.skip("run scripts/generate_kernel_llr_fixture.py")
    r = run_native_executable(
        "kernel_llr_parity",
        [_SIDE],
        timeout=60,
        env_override="CYPHA_KERNEL_LLR_PARITY_BIN",
    )
    if r is None:
        pytest.skip("kernel_llr_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
