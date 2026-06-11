"""
``gh_infer_deliberation_parity`` vs ``parity_fixtures/gh_infer_deliberation/``.

CTest: ``native_gh_infer_deliberation``. Override: ``CYPHA_GH_INFER_PARITY_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures" / "gh_infer_deliberation"


def test_gh_infer_deliberation_parity_subprocess():
    if not (_FIX / "sidecar.json").is_file():
        pytest.skip("parity_fixtures/gh_infer_deliberation missing")
    r = run_native_executable(
        "gh_infer_deliberation_parity",
        [_FIX],
        timeout=60,
        env_override="CYPHA_GH_INFER_PARITY_BIN",
    )
    if r is None:
        pytest.skip("gh_infer_deliberation_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
