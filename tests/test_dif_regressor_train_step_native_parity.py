"""
``dif_regressor_train_step_parity`` — native ``dif_train_step_vector`` + expert target EMA vs Python ``DIFRegressor``.

Cold hash + warm LLR-argmax routing (12 steps, ``replay_ratio>0`` + sidecar ``replay_u01``). CTest: ``native_dif_regressor_train_step``.
Override: ``CYPHA_DIF_REGRESSOR_TRAIN_STEP_PARITY_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures" / "dif_regressor_train_step"


def test_dif_regressor_train_step_parity_subprocess():
    if not (_FIX / "sidecar.json").is_file():
        pytest.skip("run scripts/generate_dif_regressor_train_step_fixture.py")
    r = run_native_executable(
        "dif_regressor_train_step_parity",
        [_FIX],
        timeout=120,
        env_override="CYPHA_DIF_REGRESSOR_TRAIN_STEP_PARITY_BIN",
    )
    if r is None:
        pytest.skip("dif_regressor_train_step_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
