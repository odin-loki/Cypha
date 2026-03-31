"""Main window QSettings — `geometry` + `windowState` written on close (INI isolated in test)."""
from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 not installed (cypha_studio/requirements.txt)")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _restore_native_qsettings_format():
    """Avoid leaking IniFormat to other test modules in the same process."""
    from PySide6.QtCore import QSettings

    yield
    QSettings.setDefaultFormat(QSettings.Format.NativeFormat)


def test_main_window_close_writes_geometry_and_window_state(qapp, tmp_path):
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )

    from cypha_studio.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    qapp.processEvents()
    w.close()
    qapp.processEvents()

    s = QSettings("Cypha", "CyphaStudio")
    assert s.value("geometry") is not None
    assert s.value("windowState") is not None
