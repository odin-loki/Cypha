"""Ensure cypha_bench and repo root are importable."""

from __future__ import annotations

import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parent


def ensure_paths() -> None:
    for path in (REPO_ROOT, BENCH_ROOT):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
