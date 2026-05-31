"""TrainingWidget series compression for pyqtgraph."""
from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 not installed (cypha_studio/requirements.txt)")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def test_compress_xy_identity_when_short(qapp):
    from cypha_studio.gui.widgets import TrainingWidget

    tw = TrainingWidget()
    xs = list(range(10))
    ys = [float(i * i) for i in xs]
    ox, oy = tw._compress_xy(xs, ys, cap=100)
    assert ox == xs and oy == ys
    tw.close()


def test_compress_xy_reduces_length(qapp):
    from cypha_studio.gui.widgets import TrainingWidget

    tw = TrainingWidget()
    n = 5000
    xs = list(range(n))
    ys = [float(i % 7) for i in xs]
    ox, oy = tw._compress_xy(xs, ys, cap=200)
    assert len(ox) <= 200 and len(ox) == len(oy)
    assert ox[0] == 0.0 and ox[-1] == float(n - 1)
    tw.close()
