#!/usr/bin/env python3
"""Run D04 CyphaLM hybrid refresh @ 300k (skip ablation retrains; stub from sweep)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "cypha_bench"))

os.environ.setdefault("CYPHA_BENCH_FULL_CORPUS", "1")
os.environ.setdefault("CYPHA_BENCH_FULL_N_TRAIN", "300000")
os.environ.setdefault("CYPHA_BENCH_SKIP_LM_ABLATIONS", "1")

from cypha_bench.domains.d04_generation_language import run


def main() -> int:
    print("[D04] hybrid refresh @ 300k (ablations stubbed from sweep)", flush=True)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
