"""
Subprocess ``regression_two_stage_ridge_fit_parity`` on ``parity_fixtures/two_stage_e2e_ridge/sidecar.json``.

LLR comes from a real ``TwoStageDIFRegressor.fit`` (quantile router). Same binary/env as
``test_two_stage_ridge_fit_native_parity`` (``CYPHA_TWO_STAGE_RIDGE_FIT_PARITY_BIN``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_SIDE = _ROOT / "parity_fixtures" / "two_stage_e2e_ridge" / "sidecar.json"


def test_two_stage_e2e_ridge_parity_subprocess():
    if not _SIDE.is_file():
        pytest.skip(
            "parity_fixtures/two_stage_e2e_ridge/sidecar.json missing — "
            "run scripts/generate_two_stage_e2e_ridge_fixture.py (PYTHONPATH=repo root)"
        )
    r = run_native_executable(
        "regression_two_stage_ridge_fit_parity",
        [_SIDE],
        timeout=90,
        env_override="CYPHA_TWO_STAGE_RIDGE_FIT_PARITY_BIN",
    )
    if r is None:
        pytest.skip("regression_two_stage_ridge_fit_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
