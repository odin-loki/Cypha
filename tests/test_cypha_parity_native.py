"""
``cypha_parity``: ``reference.cypha`` + ``native_parity.bin`` (LLR / probs / gates + v2 tail).

CTest: ``native_parity``. Override: ``CYPHA_CYPHA_PARITY_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures"
_REF = _FIX / "reference.cypha"
_BIN = _FIX / "native_parity.bin"


def test_cypha_parity_subprocess():
    if not _REF.is_file() or not _BIN.is_file():
        pytest.skip("reference.cypha or native_parity.bin missing under parity_fixtures/")
    r = run_native_executable(
        "cypha_parity",
        [_REF, _BIN],
        timeout=120,
        env_override="CYPHA_CYPHA_PARITY_BIN",
    )
    if r is None:
        pytest.skip(
            "cypha_parity not built (cmake native/; set CYPHA_CYPHA_PARITY_BIN; "
            "Windows: WSL ELF under native/build-wsl or native/build)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
