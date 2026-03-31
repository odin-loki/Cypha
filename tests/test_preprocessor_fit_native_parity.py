"""
``preprocessor_fit_parity`` — native ``PreprocessorState::fit_from_design_matrix`` (scale on/off + PCA).

Runs ``preprocessor_fit/`` and ``preprocessor_fit_no_scale/``. CTest: ``native_preprocessor_fit``.
Override: ``CYPHA_PREPROCESSOR_FIT_PARITY_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures" / "preprocessor_fit"
_FIX_NS = _ROOT / "parity_fixtures" / "preprocessor_fit_no_scale"


def test_preprocessor_fit_parity_subprocess():
    for base in (_FIX, _FIX_NS):
        for name in ("design.json", "expected_preprocessor.json", "probe.json"):
            if not (base / name).is_file():
                pytest.skip("run scripts/generate_preprocessor_fit_fixture.py")
    r = run_native_executable(
        "preprocessor_fit_parity",
        [_FIX, _FIX_NS],
        timeout=120,
        env_override="CYPHA_PREPROCESSOR_FIT_PARITY_BIN",
    )
    if r is None:
        pytest.skip("preprocessor_fit_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
