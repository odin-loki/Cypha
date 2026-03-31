"""
``cypha_qt_shell``: Qt Widgets + native ``CyphaInferModel`` classify (``--smoke``);
CLI text (``--help``) documents loss export, ``/predict`` options, Y-lock, and training log.

CTest: ``native_qt_shell_smoke`` (when ``-DCYPHA_BUILD_QT=ON`` and Qt6 Widgets is found).
Override: ``CYPHA_QT_SHELL_BIN``.
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


def test_qt_shell_smoke_subprocess():
    if not _REF.is_file():
        pytest.skip("parity_fixtures/reference.cypha missing")
    r = run_native_executable(
        "cypha_qt_shell",
        ["--smoke", _REF],
        timeout=90,
        env_override="CYPHA_QT_SHELL_BIN",
        extra_env={"QT_QPA_PLATFORM": "offscreen"},
    )
    if r is None:
        pytest.skip(
            "cypha_qt_shell not built (cmake native/ -DCYPHA_BUILD_QT=ON with Qt6 Widgets; "
            "set CYPHA_QT_SHELL_BIN; Windows: WSL ELF or MinGW .exe)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "smoke OK" in (r.stdout or ""), r.stdout


def test_qt_shell_help_documents_loss_export_and_predict_options():
    """Regression: keep loss PNG/SVG/CSV + EMA + Y lock + training log + return_explanation in --help."""
    r = run_native_executable(
        "cypha_qt_shell",
        ["--help"],
        timeout=30,
        env_override="CYPHA_QT_SHELL_BIN",
        extra_env={"QT_QPA_PLATFORM": "offscreen"},
    )
    if r is None:
        pytest.skip(
            "cypha_qt_shell not built (cmake native/ -DCYPHA_BUILD_QT=ON with Qt6 Widgets; "
            "set CYPHA_QT_SHELL_BIN; Windows: WSL ELF or MinGW .exe)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
    out = (r.stdout or "") + (r.stderr or "")
    for needle in ("PNG", "SVG", "CSV", "return_explanation", "EMA", "Y lock", "training log"):
        assert needle in out, f"Expected {needle!r} in --help output:\n{out}"
