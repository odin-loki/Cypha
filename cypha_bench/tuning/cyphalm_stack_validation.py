#!/usr/bin/env python3
"""Validate stacked best configs from axis + component studies."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from cypha_bench.adapters.cyphalm_bench import (
    bigram_baseline_bpc,
    prepare_lm_corpus,
    require_real_corpus,
)
from cypha_bench.tuning.cyphalm_view_iteration_sweep import _eval_bpc

_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_stack_validation.json"

STACKS: list[dict[str, Any]] = [
    {
        "id": "profile_d17",
        "label": "D17 profile (gria_ngram, same_order)",
        "mode": {"label": "profile", "view_schedule": "same_order", "train_epochs": 2},
        "overrides": {},
    },
    {
        "id": "schedule_b_profile",
        "label": "D17 + schedule_b",
        "mode": {"label": "sched_b", "view_schedule": "schedule_b", "train_epochs": 2},
        "overrides": {},
    },
    {
        "id": "axes_best",
        "label": "schedule_b + gria_lr_decay=0.3 + ngram_context=3",
        "mode": {"label": "axes_best", "view_schedule": "schedule_b", "train_epochs": 2},
        "overrides": {"gria_lr_decay": 0.3, "ngram_context": 3},
    },
    {
        "id": "axes_frozen_alpha",
        "label": "axes_best + frozen alpha",
        "mode": {"label": "frozen", "view_schedule": "schedule_b", "train_epochs": 2},
        "overrides": {
            "gria_lr_decay": 0.3,
            "ngram_context": 3,
            "alpha_learnable": False,
        },
    },
    {
        "id": "component_frozen",
        "label": "gria_ngram + frozen alpha (component study winner @ 8k)",
        "mode": {"label": "comp", "view_schedule": "same_order", "train_epochs": 2},
        "overrides": {"alpha_learnable": False},
    },
]

FULL_N_TRAIN = [40000, 70000, 300000]
FAST_N_TRAIN = [8000, 40000]


def run_validation(*, n_train_grid: list[int], n_eval: int = 2000) -> dict[str, Any]:
    corpus = prepare_lm_corpus(prefer_wikitext=True, max_train_chars=10_000_000)
    require_real_corpus(corpus.source, domain="cyphalm_stack_validation")
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for n_train in n_train_grid:
        if n_train > len(corpus.train_ids) - 1:
            continue
        limit = min(n_train, len(corpus.train_ids) - 1)
        bigram = bigram_baseline_bpc(
            corpus.train_ids[: limit + 1],
            corpus.eval_ids,
            corpus.vocab_size,
        )
        for stack in STACKS:
            tag = f"{stack['id']} n={n_train}"
            try:
                row = _eval_bpc(
                    corpus,
                    n_train=n_train,
                    n_eval=n_eval,
                    mode=stack["mode"],
                    extra_overrides=stack["overrides"],
                )
                row["stack_id"] = stack["id"]
                row["label"] = stack["label"]
                row["bigram_bpc"] = float(bigram)
                row["delta_vs_bigram"] = row["held_out_bpc"] - bigram
                results.append(row)
                print(
                    f"[{tag}] bpc={row['held_out_bpc']:.4f} "
                    f"d_bi={row['delta_vs_bigram']:+.4f} ({row['train_seconds']:.0f}s)"
                )
            except Exception as exc:
                print(f"[{tag}] FAIL: {exc}")

    ok = [r for r in results if "held_out_bpc" in r]
    best = min(ok, key=lambda r: r["held_out_bpc"]) if ok else None
    return {
        "corpus": corpus.source,
        "grid_n_train": n_train_grid,
        "n_eval": n_eval,
        "stacks": [s["id"] for s in STACKS],
        "elapsed_s": time.perf_counter() - t0,
        "results": results,
        "best": best,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--n-eval", type=int, default=2000)
    args = ap.parse_args()
    grid = FAST_N_TRAIN if args.fast else FULL_N_TRAIN
    out = run_validation(n_train_grid=grid, n_eval=args.n_eval)
    if args.write:
        _OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {_OUT}")
    b = out.get("best")
    if b:
        print(
            f"\nBest: {b['stack_id']} @ n={b['n_train']} "
            f"BPC={b['held_out_bpc']:.4f} d_bi={b['delta_vs_bigram']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
