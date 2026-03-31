"""
Native: ``regression_two_stage_pipeline_parity`` vs ``parity_fixtures/two_stage_pipeline/sidecar.json``.

End-to-end native ``CyphaInferModel`` LLR + stage-2 RFF + ``two_stage_dif_predict`` combine.
CTest: ``native_regression_two_stage_pipeline``. Override binary with ``CYPHA_TWO_STAGE_PIPELINE_PARITY_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_SIDE = _ROOT / "parity_fixtures" / "two_stage_pipeline" / "sidecar.json"


def test_two_stage_pipeline_parity_subprocess():
    if not _SIDE.is_file():
        pytest.skip(
            "parity_fixtures/two_stage_pipeline/sidecar.json missing — "
            "run scripts/generate_two_stage_pipeline_fixture.py (repo root on PYTHONPATH)"
        )
    r = run_native_executable(
        "regression_two_stage_pipeline_parity",
        [_SIDE],
        timeout=60,
        env_override="CYPHA_TWO_STAGE_PIPELINE_PARITY_BIN",
    )
    if r is None:
        pytest.skip(
            "regression_two_stage_pipeline_parity not built (cmake native/build; WSL ELF ok on Windows)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
