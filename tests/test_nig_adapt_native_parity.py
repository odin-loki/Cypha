"""
``nig_adapt_parity``: ``nig_adapt_session_chi`` vs three fixed ``_nig_adapt`` goldens.

CTest: ``native_nig_adapt``. Override: ``CYPHA_NIG_ADAPT_PARITY_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable


def test_nig_adapt_parity_subprocess():
    r = run_native_executable(
        "nig_adapt_parity",
        [],
        timeout=30,
        env_override="CYPHA_NIG_ADAPT_PARITY_BIN",
    )
    if r is None:
        pytest.skip(
            "nig_adapt_parity not built (cmake native/; set CYPHA_NIG_ADAPT_PARITY_BIN; "
            "Windows: WSL ELF under native/build-wsl or native/build)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
