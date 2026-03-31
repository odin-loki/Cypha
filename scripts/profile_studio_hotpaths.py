#!/usr/bin/env python3
"""
cProfile CyphaStudio GUI and API hot paths (not cold import — see profile_gui_startup.py).

  python scripts/profile_studio_hotpaths.py training --steps 3000
  python scripts/profile_studio_hotpaths.py chat --rounds 200
  python scripts/profile_studio_hotpaths.py dataset
  python scripts/profile_studio_hotpaths.py registry --iterations 50
  python scripts/profile_studio_hotpaths.py api --predicts 400

Requires: pip install -r cypha_studio/requirements.txt (+ requirements-verify.txt for FastAPI tests).
GUI modes use QT_QPA_PLATFORM=offscreen by default.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _pstats_top(pr: cProfile.Profile, n: int = 60) -> str:
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf)
    st.strip_dirs().sort_stats("cumtime").print_stats(n)
    return buf.getvalue()


def _tiny_classifier():
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder

    rng = np.random.default_rng(42)
    clf = CyphaDIF(VectorEncoder(4), field_dim=48, rng=rng)
    for i in range(60):
        x = rng.standard_normal(4)
        clf.train_step(x, str(i % 3))
    return clf


def run_training(steps: int, app) -> None:
    from cypha_studio.gui.widgets import TrainingWidget
    from cypha_studio.server.local_server import SignalBus

    bus = SignalBus.instance()
    tw = TrainingWidget()
    tw.show()
    for step in range(1, steps + 1):
        loss = 1.0 / (1.0 + step * 0.001)
        bus.emit_training_step(step, loss, "0", step % 3 != 0)
        app.processEvents()
    tw.close()


def run_chat(rounds: int, app) -> None:
    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.gui.chat_widget import ChatWidget
    from cypha_studio.gui.main_window import AppState

    state = AppState()
    eng = InferenceEngine(_tiny_classifier(), None)
    state.engine = eng
    state.session = InferenceSession(eng)
    chat = ChatWidget(state)
    chat.show()
    vec = "0.1,0.2,-0.1,0.3"
    for _ in range(rounds):
        chat._input.setText(vec)
        chat._on_send()
        app.processEvents()
    chat.close()


def run_dataset(app) -> None:
    from sklearn.datasets import load_iris

    from cypha_studio.gui.main_window import AppState
    from cypha_studio.gui.widgets import DatasetWidget

    iris = load_iris()
    X = iris.data
    y = iris.target
    hdr = "f0,f1,f2,f3,target"
    lines = [hdr]
    for i in range(len(X)):
        lines.append(
            ",".join(str(float(v)) for v in X[i])
            + ","
            + str(int(y[i]))
        )
    text = "\n".join(lines) + "\n"
    path = Path(tempfile.mkdtemp()) / "iris_profile.csv"
    path.write_text(text, encoding="utf-8")

    state = AppState()
    dw = DatasetWidget(state)
    dw.show()
    dw.load_file(str(path))
    app.processEvents()
    dw.close()


def run_registry(iterations: int, root: str | None) -> None:
    from cypha_studio.core.registry import ModelRegistry

    reg = ModelRegistry(root) if root else ModelRegistry()
    for _ in range(iterations):
        reg.list_models()


def run_api(predicts: int) -> None:
    import pytest

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry
    from cypha_studio.server.api import create_app

    eng = InferenceEngine(_tiny_classifier(), None)
    sess = InferenceSession(eng)
    app = create_app(engine=eng, registry=ModelRegistry(), session=sess)
    client = TestClient(app)
    body = {"input": [0.1, 0.2, -0.1, 0.3], "use_gh": False, "return_explanation": False}
    for _ in range(predicts):
        r = client.post("/predict", json=body)
        assert r.status_code == 200, r.text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    p_tr = sub.add_parser("training", help="TrainingWidget + training_step signals")
    p_tr.add_argument("--steps", type=int, default=2500)

    p_ch = sub.add_parser("chat", help="ChatWidget inference + bubbles")
    p_ch.add_argument("--rounds", type=int, default=150)

    sub.add_parser("dataset", help="DatasetWidget.load_file (temp iris CSV)")

    p_reg = sub.add_parser("registry", help="ModelRegistry.list_models loop")
    p_reg.add_argument("--iterations", type=int, default=40)
    p_reg.add_argument(
        "--registry-root",
        default="",
        help="Pass to ModelRegistry(root); default ~/.cypha/models",
    )

    p_api = sub.add_parser("api", help="FastAPI TestClient POST /predict loop")
    p_api.add_argument("--predicts", type=int, default=300)

    ap.add_argument(
        "-o",
        "--output",
        default="",
        help="Write pstats (cumtime, top 60) to this file",
    )

    args = ap.parse_args()
    sys.path.insert(0, str(_ROOT))

    pr = cProfile.Profile()
    pr.enable()
    try:
        if args.mode == "registry":
            run_registry(
                args.iterations,
                args.registry_root or None,
            )
        elif args.mode == "api":
            run_api(args.predicts)
        else:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance() or QApplication([])
            if args.mode == "training":
                run_training(args.steps, app)
            elif args.mode == "chat":
                run_chat(args.rounds, app)
            elif args.mode == "dataset":
                run_dataset(app)
            else:
                raise SystemExit(f"unknown mode {args.mode!r}")
            app.processEvents()
    finally:
        pr.disable()

    text = _pstats_top(pr, 60)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
