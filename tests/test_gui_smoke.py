"""
Smoke-test CyphaStudio Qt GUI without a visible display.

Uses QT_QPA_PLATFORM=offscreen when available (Linux CI / headless).
On a desktop, unset QT_QPA_PLATFORM to see the window briefly.

Run: pytest tests/test_gui_smoke.py -v
Requires: pip install -r cypha_studio/requirements.txt (PySide6, pyqtgraph)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Must set before Qt imports
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 not installed (cypha_studio/requirements.txt)")

# Repo root on path for cypha_studio
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


def _tiny_classifier():
    """Small CyphaDIF for 4-D chat input tests."""
    from Cypha import CyphaDIF, VectorEncoder

    rng = np.random.default_rng(42)
    clf = CyphaDIF(VectorEncoder(4), field_dim=48, rng=rng)
    for i in range(60):
        x = rng.standard_normal(4)
        clf.train_step(x, str(i % 3))
    return clf


def test_main_window_construct_show_process_events(qapp):
    from cypha_studio.gui.main_window import MainWindow

    w = MainWindow()
    assert w.windowTitle() == "CyphaStudio"
    w.show()
    for _ in range(20):
        qapp.processEvents()
    # Docks / central widget exist
    assert w.centralWidget() is not None
    w.close()
    qapp.processEvents()


def test_train_config_dialog_opens(qapp):
    """TrainConfigDialog: modal exec is heavy; smoke-open with show + events."""
    from cypha_studio.gui.main_window import AppState
    from cypha_studio.gui.dialogs import TrainConfigDialog

    state = AppState()
    dlg = TrainConfigDialog(state)
    dlg.show()
    for _ in range(10):
        qapp.processEvents()
    dlg.close()
    qapp.processEvents()


def test_chat_widget_send_without_model_shows_error(qapp):
    from cypha_studio.gui.main_window import AppState
    from cypha_studio.gui.chat_widget import ChatWidget

    state = AppState()
    chat = ChatWidget(state)
    assert chat._msg_layout.count() == 1  # bottom stretch only
    chat._input.setText("1,2,3,4")
    chat._on_send()
    for _ in range(5):
        qapp.processEvents()
    # user bubble + error bubble + stretch
    assert chat._msg_layout.count() == 3


def test_chat_widget_send_with_model(qapp):
    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.gui.main_window import AppState
    from cypha_studio.gui.chat_widget import ChatWidget

    state = AppState()
    eng = InferenceEngine(_tiny_classifier(), None)
    state.engine = eng
    state.session = InferenceSession(eng)

    chat = ChatWidget(state)
    chat._input.setText("0.1,0.2,-0.1,0.3")
    chat._on_send()
    for _ in range(10):
        qapp.processEvents()
    summ = state.session.summary()
    assert summ.get("n_predictions", 0) >= 1


def test_confidence_widget_updates_on_bus_prediction(qapp):
    from cypha_studio.core.inference import Prediction
    from cypha_studio.gui.widgets import ConfidenceWidget
    from cypha_studio.server.local_server import SignalBus

    cw = ConfidenceWidget()
    cw.show()
    pred = Prediction(
        label="0",
        confidence=0.85,
        all_scores={"0": 2.0, "1": -0.5, "2": -1.0},
        anomaly_score=0.15,
        is_ood=False,
    )
    SignalBus.instance().emit_prediction(pred)
    for _ in range(8):
        qapp.processEvents()
    cw.close()
    assert "0" in cw._class_bars


def test_message_bubble_system_role(qapp):
    from cypha_studio.gui.chat_widget import MessageBubble

    b = MessageBubble("system", "hello", prediction=None)
    b.show()
    for _ in range(3):
        qapp.processEvents()
    b.close()
