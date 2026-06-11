"""``merge_from_parity`` vs ``parity_fixtures/merge_from/``. CTest: ``native_merge_from``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures" / "merge_from"


def test_merge_from_parity_subprocess():
    for name in ("sidecar.json", "self_before.cypha", "other.cypha", "self_after.cypha"):
        if not (_FIX / name).is_file():
            pytest.skip("run scripts/generate_merge_from_fixture.py")
    r = run_native_executable(
        "merge_from_parity",
        [_FIX],
        timeout=120,
        env_override="CYPHA_MERGE_FROM_PARITY_BIN",
    )
    if r is None:
        pytest.skip("merge_from_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
