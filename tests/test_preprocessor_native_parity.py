"""
``preprocessor_parity`` vs ``parity_fixtures/preprocessor/``.

CTest: ``native_preprocessor``. Override: ``CYPHA_PREPROCESSOR_PARITY_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures" / "preprocessor"


def test_preprocessor_parity_subprocess():
    for name in ("preprocessor.json", "sidecar.json"):
        if not (_FIX / name).is_file():
            pytest.skip("preprocessor fixtures missing — run scripts/generate_preprocessor_parity.py")
    r = run_native_executable(
        "preprocessor_parity",
        [_FIX],
        timeout=60,
        env_override="CYPHA_PREPROCESSOR_PARITY_BIN",
    )
    if r is None:
        pytest.skip(
            "preprocessor_parity not built (cmake native/; set CYPHA_PREPROCESSOR_PARITY_BIN; "
            "Windows: WSL ELF under native/build-wsl or native/build)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
