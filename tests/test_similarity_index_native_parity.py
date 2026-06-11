"""``similarity_index_parity`` vs ``parity_fixtures/similarity_index/``. CTest: ``native_similarity_index``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures" / "similarity_index"


def test_similarity_index_parity_subprocess():
    for name in ("sidecar.json", "reference.cypha", "f_field.json"):
        if not (_FIX / name).is_file():
            pytest.skip("run scripts/generate_similarity_index_fixture.py")
    r = run_native_executable(
        "similarity_index_parity",
        [_FIX],
        timeout=120,
        env_override="CYPHA_SIMILARITY_INDEX_PARITY_BIN",
    )
    if r is None:
        pytest.skip("similarity_index_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
