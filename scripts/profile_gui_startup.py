#!/usr/bin/env python3
"""
Profile CyphaStudio GUI cold start (MainWindow + docks + processEvents).

  python scripts/profile_gui_startup.py
  python scripts/profile_gui_startup.py -o artifacts/profiles/gui_startup_cprofile.txt

Uses QT_QPA_PLATFORM=offscreen by default so it runs without a display.
Requires: pip install -r cypha_studio/requirements.txt
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run_once() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(_ROOT))

    from PySide6.QtWidgets import QApplication

    from cypha_studio.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.show()
    for _ in range(40):
        app.processEvents()
    w.close()
    app.processEvents()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--output",
        default="",
        help="Write pstats (cumtime, top 60) to this file",
    )
    args = ap.parse_args()

    pr = cProfile.Profile()
    pr.enable()
    try:
        _run_once()
    finally:
        pr.disable()

    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf)
    st.strip_dirs().sort_stats("cumtime").print_stats(60)
    text = buf.getvalue()
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
