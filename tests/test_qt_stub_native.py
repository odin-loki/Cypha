"""
``cypha_qt_stub``: headless Qt Core + ``load_cypha_from_buffer`` on ``reference.cypha``.

CTest: ``native_qt_stub_load_reference`` (only when ``-DCYPHA_BUILD_QT=ON`` and Qt6 is found).
Override: ``CYPHA_QT_STUB_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_REF = _ROOT / "parity_fixtures" / "reference.cypha"


def test_qt_stub_load_reference_subprocess():
    if not _REF.is_file():
        pytest.skip("parity_fixtures/reference.cypha missing")
    r = run_native_executable(
        "cypha_qt_stub",
        [_REF],
        timeout=60,
        env_override="CYPHA_QT_STUB_BIN",
        extra_env={"QT_QPA_PLATFORM": "offscreen"},
    )
    if r is None:
        pytest.skip(
            "cypha_qt_stub not built (cmake native/ -DCYPHA_BUILD_QT=ON; qt6-base-dev on Linux; "
            "set CYPHA_QT_STUB_BIN; Windows: WSL ELF or MinGW .exe if built)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "load_cypha_from_buffer" in (r.stdout or ""), r.stdout
