"""
``batch_llr_parity`` vs ``parity_fixtures/batch_llr/sidecar.json``.

Sidecar must match ``expected.npz`` (x_input, llr). CTest: ``native_batch_llr``.
Override: ``CYPHA_BATCH_LLR_PARITY_BIN``.
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

_SIDE = _ROOT / "parity_fixtures" / "batch_llr" / "sidecar.json"
_NPZ = _ROOT / "parity_fixtures" / "expected.npz"


def test_batch_llr_sidecar_matches_expected_npz():
    if not _SIDE.is_file() or not _NPZ.is_file():
        pytest.skip("batch_llr sidecar or expected.npz missing")
    j = json.loads(_SIDE.read_text(encoding="utf-8"))
    z = np.load(_NPZ)
    x = np.asarray(z["x_input"], dtype=np.float64)
    llr = np.asarray(z["llr"], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(j["x_rowmajor"], dtype=np.float64), x.ravel(order="C"), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(j["expected_llr_rowmajor"], dtype=np.float64), llr.ravel(order="C"), rtol=0, atol=0
    )


def test_batch_llr_parity_subprocess():
    if not _SIDE.is_file():
        pytest.skip("run scripts/generate_batch_llr_fixture.py")
    r = run_native_executable(
        "batch_llr_parity",
        [_SIDE],
        timeout=60,
        env_override="CYPHA_BATCH_LLR_PARITY_BIN",
    )
    if r is None:
        pytest.skip("batch_llr_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
