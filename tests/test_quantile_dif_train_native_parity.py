"""
``quantile_dif_train_parity`` vs ``parity_fixtures/quantile_dif_train/``.

Multi-step ``dif_train_step_vector`` replay (``N < 10``, no replay sampling) then ``batch_llr_from_x``.
CTest: ``native_quantile_dif_train``. Override: ``CYPHA_QUANTILE_DIF_TRAIN_PARITY_BIN``.
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

_FIX = _ROOT / "parity_fixtures" / "quantile_dif_train"
_SIDE = _FIX / "sidecar.json"


def test_quantile_dif_train_sidecar_geometry():
    if not _SIDE.is_file():
        pytest.skip("run scripts/generate_quantile_dif_train_fixture.py")
    j = json.loads(_SIDE.read_text(encoding="utf-8"))
    n, d_in, k = int(j["n"]), int(j["d_in"]), int(j["K"])
    x = np.asarray(j["x_rowmajor"], dtype=np.float64)
    llr = np.asarray(j["expected_llr_rowmajor"], dtype=np.float64)
    assert x.size == n * d_in
    assert llr.size == n * k
    assert len(j["label_order"]) == k
    assert len(j["steps"]) == int(j["n_steps"])
    assert len(j["expected_step_losses"]) == len(j["steps"])


def test_quantile_dif_train_parity_subprocess():
    if not _SIDE.is_file() or not (_FIX / "before.cypha").is_file():
        pytest.skip("run scripts/generate_quantile_dif_train_fixture.py")
    r = run_native_executable(
        "quantile_dif_train_parity",
        [_FIX],
        timeout=120,
        env_override="CYPHA_QUANTILE_DIF_TRAIN_PARITY_BIN",
    )
    if r is None:
        pytest.skip("quantile_dif_train_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
