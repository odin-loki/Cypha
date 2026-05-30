from __future__ import annotations

import os
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parent
DATA_DIR = BENCH_ROOT / "data"
FIGURES_DIR = BENCH_ROOT / "report" / "figures"
TABLES_DIR = BENCH_ROOT / "report" / "tables"

for path in (REPO_ROOT, BENCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def is_fast() -> bool:
    return os.environ.get("CYPHA_BENCH_FAST", "0") == "1"


def scale(default: int, fast: int | None = None) -> int:
    if is_fast():
        return fast if fast is not None else max(default // 5, 1)
    return default
