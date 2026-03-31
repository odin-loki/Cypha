"""
GUI interaction tests via pytest-qt (qtbot): real clicks and dialog buttons.

Requires: PySide6 (cypha_studio/requirements.txt) and pip install pytest-qt (not listed in requirements-verify.txt).
Run: pytest tests/test_gui_qtbot.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 not installed (cypha_studio/requirements.txt)")
pytest.importorskip("pytestqt", reason="pytest-qt not installed (pip install pytest-qt)")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _tiny_classifier():
    from Cypha import CyphaDIF, VectorEncoder

    rng = np.random.default_rng(42)
    clf = CyphaDIF(VectorEncoder(4), field_dim=48, rng=rng)
    for i in range(60):
        x = rng.standard_normal(4)
        clf.train_step(x, str(i % 3))
    return clf


def test_chat_send_via_send_button(qtbot):
    from PySide6.QtCore import Qt
    from cypha_studio.gui.main_window import AppState
    from cypha_studio.gui.chat_widget import ChatWidget

    state = AppState()
    chat = ChatWidget(state)
    qtbot.addWidget(chat)
    chat.show()
    qtbot.keyClicks(chat._input, "1,2,3,4")
    qtbot.mouseClick(chat._send_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: chat._msg_layout.count() == 3, timeout=2000)
    chat.close()


def test_chat_clear_via_button(qtbot):
    from PySide6.QtCore import Qt
    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.gui.main_window import AppState
    from cypha_studio.gui.chat_widget import ChatWidget

    state = AppState()
    eng = InferenceEngine(_tiny_classifier(), None)
    state.engine = eng
    state.session = InferenceSession(eng)

    chat = ChatWidget(state)
    qtbot.addWidget(chat)
    chat.show()
    qtbot.keyClicks(chat._input, "0.1,0.2,-0.1,0.3")
    qtbot.mouseClick(chat._send_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: chat._msg_layout.count() >= 3, timeout=2000)
    qtbot.mouseClick(chat._clear_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: chat._msg_layout.count() == 1, timeout=2000)
    chat.close()


def test_train_config_ok_persists_field_dim(qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog, QDialogButtonBox
    from cypha_studio.gui.main_window import AppState
    from cypha_studio.gui.widgets import TrainConfigDialog

    state = AppState()
    dlg = TrainConfigDialog(state)
    qtbot.addWidget(dlg)
    dlg.show()
    target = min(512, max(64, dlg._field_dim.value() + 16))
    dlg._field_dim.setValue(target)
    box = dlg.findChild(QDialogButtonBox)
    assert box is not None
    ok = box.button(QDialogButtonBox.StandardButton.Ok)
    qtbot.mouseClick(ok, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dlg.result() == QDialog.DialogCode.Accepted, timeout=2000)
    assert getattr(state, "_train_config").field_dim == target


def test_main_window_focus_chat_shortcut_handler(qtbot):
    """Shortcut slot requests focus on chat input (use focusWidget() — hasFocus() is flaky offscreen)."""
    from cypha_studio.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.activateWindow()
    w.raise_()
    w.model_combo.setFocus()
    qtbot.wait(30)
    w._shortcut_focus_chat()
    qtbot.wait(30)
    assert w.focusWidget() == w.chat_widget._input
    w.close()


def test_main_window_train_toolbar_no_engine(qtbot):
    """Toolbar Train with no engine hits the early-return path (would show QMessageBox)."""
    from unittest import mock

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMessageBox
    from cypha_studio.gui.main_window import MainWindow

    with mock.patch.object(QMessageBox, "information", return_value=QMessageBox.StandardButton.Ok):
        w = MainWindow()
        qtbot.addWidget(w)
        w.show()
        qtbot.mouseClick(w.btn_train, Qt.MouseButton.LeftButton)
    w.close()
