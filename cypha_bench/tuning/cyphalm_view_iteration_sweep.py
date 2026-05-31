#!/usr/bin/env python3
"""Sweep training length × view schedule to find optimal iterations per presentation mode."""

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
    eval_held_out_bpc,
    load_cyphalm_config,
    make_cyphalm,
    prepare_lm_corpus,
    require_real_corpus,
)
from cypha_bench.common.paths import is_fast

_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_view_iteration_sweep.json"

DEFAULT_N_TRAIN = [2000, 4000, 8000, 12000, 16000, 24000, 32000, 40000]
FAST_N_TRAIN = [2000, 4000, 8000]

# Training presentation modes to compare (hypothesis: same-order repetition hurts at high n).
TRAINING_MODES: list[dict[str, Any]] = [
    {"label": "same_order_e1", "view_schedule": "same_order", "train_epochs": 1},
    {"label": "same_order_e2", "view_schedule": "same_order", "train_epochs": 2},
    {"label": "schedule_a", "view_schedule": "schedule_a", "train_epochs": 2},
    {"label": "schedule_b", "view_schedule": "schedule_b", "train_epochs": 2},
]


def _eval_bpc(
    corpus,
    *,
    n_train: int,
    n_eval: int,
    mode: dict[str, Any],
) -> dict[str, Any]:
    overrides = {
        "view_schedule": mode["view_schedule"],
        "train_epochs": int(mode.get("train_epochs", 2)),
    }
    merged = load_cyphalm_config(overrides, profile="d17")
    model, _ = make_cyphalm(merged, profile=None)
    limit = min(n_train, len(corpus.train_ids) - 1)
    t0 = time.perf_counter()
    model.train_sequence(corpus.train_ids[: limit + 1])
    train_s = time.perf_counter() - t0
    bpc = eval_held_out_bpc(model, corpus.eval_ids, n_eval=n_eval)
    return {
        "held_out_bpc": float(bpc),
        "n_train": int(limit),
        "train_seconds": float(train_s),
        "view_schedule": mode["view_schedule"],
        "train_epochs": overrides["train_epochs"],
        "label": mode["label"],
    }


def _best_per_mode(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_label.setdefault(row["label"], []).append(row)
    best: dict[str, Any] = {}
    for label, rows in by_label.items():
        winner = min(rows, key=lambda r: r["held_out_bpc"])
        best[label] = {
            "optimal_n_train": winner["n_train"],
            "held_out_bpc": winner["held_out_bpc"],
            "train_seconds": winner["train_seconds"],
        }
    return best


def _crossover(results: list[dict[str, Any]]) -> dict[str, Any]:
    """At each n_train, which mode wins (lowest BPC)?"""
    by_n: dict[int, list[dict[str, Any]]] = {}
    for row in results:
        by_n.setdefault(row["n_train"], []).append(row)
    winners = {}
    for n, rows in sorted(by_n.items()):
        w = min(rows, key=lambda r: r["held_out_bpc"])
        winners[str(n)] = {
            "label": w["label"],
            "held_out_bpc": w["held_out_bpc"],
            "delta_vs_same_order_e2": float(
                w["held_out_bpc"]
                - next(
                    (r["held_out_bpc"] for r in rows if r["label"] == "same_order_e2"),
                    float("nan"),
                )
            ),
        }
    return winners


def run_sweep(
    *,
    n_train_grid: list[int],
    n_eval: int = 2000,
    modes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    corpus = prepare_lm_corpus(prefer_wikitext=True)
    require_real_corpus(corpus.source, domain="cyphalm_view_iteration_sweep")
    train_slice = corpus.train_ids

    modes = modes or TRAINING_MODES
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for n_train in n_train_grid:
        limit = min(n_train, len(corpus.train_ids) - 1)
        bigram_n = bigram_baseline_bpc(
            train_slice[: limit + 1],
            corpus.eval_ids,
            corpus.vocab_size,
        )
        for mode in modes:
            try:
                row = _eval_bpc(corpus, n_train=n_train, n_eval=n_eval, mode=mode)
                row["bigram_bpc"] = float(bigram_n)
                row["delta_vs_bigram"] = row["held_out_bpc"] - bigram_n
                results.append(row)
                print(
                    f"[{mode['label']}] n={n_train} bpc={row['held_out_bpc']:.4f} "
                    f"d_bi={row['delta_vs_bigram']:+.4f} ({row['train_seconds']:.1f}s)"
                )
            except Exception as exc:
                print(f"[{mode['label']}] n={n_train} FAIL: {exc}")

    out = {
        "corpus": corpus.source,
        "n_eval": n_eval,
        "elapsed_s": time.perf_counter() - t0,
        "grid_n_train": n_train_grid,
        "modes": [m["label"] for m in modes],
        "results": results,
        "best_per_mode": _best_per_mode(results),
        "winner_by_n_train": _crossover(results),
    }
    overall = min(results, key=lambda r: r["held_out_bpc"]) if results else None
    if overall:
        out["global_best"] = {
            "label": overall["label"],
            "n_train": overall["n_train"],
            "held_out_bpc": overall["held_out_bpc"],
            "delta_vs_bigram": overall["delta_vs_bigram"],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CyphaLM train-length × view-schedule sweep")
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument(
        "--n-train",
        type=int,
        nargs="+",
        default=None,
        help="Training lengths to sweep (default: full or fast grid)",
    )
    ap.add_argument("--fast", action="store_true", help="Use fast n_train grid")
    ap.add_argument("--write", action="store_true", help="Write JSON to config/")
    args = ap.parse_args()

    if args.n_train:
        grid = args.n_train
    elif args.fast or is_fast():
        grid = FAST_N_TRAIN
        print(f"Using fast grid: {grid}")
    else:
        grid = DEFAULT_N_TRAIN

    out = run_sweep(n_train_grid=grid, n_eval=args.n_eval)
    if args.write or not is_fast():
        _OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {_OUT}")
    if out.get("global_best"):
        g = out["global_best"]
        print(
            f"\nGlobal best: {g['label']} @ n_train={g['n_train']} "
            f"BPC={g['held_out_bpc']:.4f} (d_bigram {g['delta_vs_bigram']:+.4f})"
        )
    print("\nBest per mode:")
    for label, b in out.get("best_per_mode", {}).items():
        print(f"  {label}: n={b['optimal_n_train']} BPC={b['held_out_bpc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
