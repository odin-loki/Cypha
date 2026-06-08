#!/usr/bin/env python3
"""Run D17 Phase 1c (full corpus @ 300k): 17A, 17B, 17D, 17K only."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "cypha_bench"))

os.environ.setdefault("CYPHA_BENCH_FULL_CORPUS", "1")
os.environ.setdefault("CYPHA_BENCH_FULL_N_TRAIN", "300000")

from bench_common import finalize_domain
from cypha_bench.domains import d17_cyphalm_integration as d17

_LOG = Path(
    os.environ.get(
        "CYPHA_BENCH_PHASE1C_LOG",
        str(_REPO / "cypha_bench" / "config" / "d17_phase1c_summary.json"),
    )
)


def main() -> int:
    experiments: dict = {}
    steps = [
        ("17A_bits_per_character", d17.experiment_17a_bpc),
        ("17B_alpha_spectrum", d17.experiment_17b_alpha_spectrum),
        ("17D_online_adaptation", d17.experiment_17d_online_adaptation),
        ("17K_long_range_context", d17.experiment_17k_long_range_context),
    ]
    skip = os.environ.get("CYPHA_BENCH_SKIP_17A", "0") == "1"
    if skip:
        prior = _REPO / "cypha_bench" / "config" / "d17_phase1c_17a.json"
        if prior.exists():
            import json as _json

            stub = _json.loads(prior.read_text(encoding="utf-8"))
            experiments["17A_bits_per_character"] = {
                "cyphalm_bpc": stub.get("cyphalm_bpc"),
                "train_tokens": stub.get("n_train"),
                "profile": {"context_mode": stub.get("context_mode")},
                "phase1c_stub": True,
            }
            print(f"=== loaded 17A from {prior} ===", flush=True)
    for label, fn in steps:
        if skip and label.startswith("17A"):
            continue
        print(f"\n=== {label} ===", flush=True)
        t0 = time.perf_counter()
        try:
            out = fn()
            experiments[label] = out
            print(f"=== {label} done in {time.perf_counter() - t0:.0f}s ===", flush=True)
        except Exception as exc:
            print(f"=== {label} FAILED: {exc} ===", flush=True)
            raise
    if experiments.get("17A_bits_per_character") and not experiments["17A_bits_per_character"].get("skipped"):
        d17._save_learning_figure(experiments["17A_bits_per_character"])
    result = finalize_domain("d17", experiments)
    _LOG.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote summary to {_LOG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
