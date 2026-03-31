#!/usr/bin/env python3
"""
Optional: tracemalloc diff around the same training-widget hot path as
``profile_studio_hotpaths.py training`` (small step count by default).

  python scripts/profile_studio_memory.py --steps 800

Does not require memray; for memray see docs/studio/OPTIONAL_MEMORY_AND_LOAD.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import tracemalloc
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    import importlib.util

    from PySide6.QtWidgets import QApplication

    spec = importlib.util.spec_from_file_location(
        "profile_studio_hotpaths",
        _ROOT / "scripts" / "profile_studio_hotpaths.py",
    )
    assert spec and spec.loader
    hot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hot)

    app = QApplication.instance() or QApplication([])

    tracemalloc.start(25)
    s0 = tracemalloc.take_snapshot()
    hot.run_training(args.steps, app)
    s1 = tracemalloc.take_snapshot()

    stats = s1.compare_to(s0, "lineno")
    print("Top tracemalloc diffs (lineno):")
    for st in stats[:25]:
        print(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
