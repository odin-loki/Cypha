"""
Training monitor + dataset panel (PySide6 offscreen).

Run: pytest tests/test_gui_training_dataset.py -v
"""
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


def test_training_widget_training_steps_accumulate(qapp):
    from cypha_studio.gui.widgets import TrainingWidget
    from cypha_studio.server.local_server import SignalBus

    bus = SignalBus.instance()
    tw = TrainingWidget()
    tw.show()
    n = 120
    for step in range(1, n + 1):
        bus.emit_training_step(step, 1.0 / (1.0 + step * 0.01), "0", step % 2 == 0)
        qapp.processEvents()
    assert len(tw._steps) == n
    assert tw._steps[-1] == n
    assert "120" in tw._summary.text()
    tw.close()


def test_dataset_widget_load_csv_sets_train_state(qapp, tmp_path):
    from cypha_studio.gui.main_window import AppState
    from cypha_studio.gui.widgets import DatasetWidget

    # Enough rows for stratified split defaults (train/val/test per class).
    lines = ["f0,f1,f2,target"]
    for i in range(40):
        c = i % 2
        lines.append(f"{i},{i},{i},{c}")
    p = tmp_path / "loadme.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state = AppState()
    dw = DatasetWidget(state)
    dw.show()
    dw.load_file(str(p))
    qapp.processEvents()
    assert getattr(state, "_train_ds", None) is not None
    assert getattr(state, "_val_ds", None) is not None
    assert dw.dataset is not None
    assert dw.dataset.n_samples == 40
    dw.close()
