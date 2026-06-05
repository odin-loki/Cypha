#!/usr/bin/env python3
"""Single-axis sweeps under schedule_b @ 70k — next levers after hyperparam grid."""

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
from cypha_bench.common.paths import is_fast
from cypha_bench.tuning.cyphalm_view_iteration_sweep import _eval_bpc

_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_beat_bigram_axes.json"

DEFAULT_N_TRAIN = 70000
FAST_N_TRAIN = 8000

AXIS_VALUES: dict[str, list] = {
    "gria_lr_decay": [0.0, 0.3, 0.5, 0.7, 0.9, 1.0],
    "laplace_smoothing": [0.01, 0.1, 0.5, 1.0, 5.0, 20.0],
    "ngram_context": [1, 2, 3, 4],
    "train_ssm": [False, True],
}

SCHEDULE_MODES: list[dict[str, Any]] = [
    {"label": "schedule_b", "view_schedule": "schedule_b", "train_epochs": 2},
    {"label": "schedule_c", "view_schedule": "schedule_c", "train_epochs": 2},
    {"label": "schedule_a", "view_schedule": "schedule_a", "train_epochs": 2},
]

BASE_MODE = {"label": "schedule_b", "view_schedule": "schedule_b", "train_epochs": 2}


def _run_axis(
    corpus,
    *,
    axis: str,
    values: list,
    n_train: int,
    n_eval: int,
    mode: dict[str, Any],
) -> list[dict[str, Any]]:
    limit = min(n_train, len(corpus.train_ids) - 1)
    bigram_n = bigram_baseline_bpc(
        corpus.train_ids[: limit + 1],
        corpus.eval_ids,
        corpus.vocab_size,
    )
    rows: list[dict[str, Any]] = []
    for val in values:
        overrides = {axis: val}
        tag = f"{mode['label']} n={n_train} {axis}={val}"
        try:
            row = _eval_bpc(
                corpus,
                n_train=n_train,
                n_eval=n_eval,
                mode=mode,
                extra_overrides=overrides,
            )
            row["bigram_bpc"] = float(bigram_n)
            row["delta_vs_bigram"] = row["held_out_bpc"] - bigram_n
            row["overrides"] = overrides
            rows.append(row)
            print(
                f"[{tag}] bpc={row['held_out_bpc']:.4f} "
                f"d_bi={row['delta_vs_bigram']:+.4f} ({row['train_seconds']:.0f}s)"
            )
        except Exception as exc:
            print(f"[{tag}] FAIL: {exc}")
    return rows


def _run_schedules(
    corpus,
    *,
    n_train: int,
    n_eval: int,
    modes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    limit = min(n_train, len(corpus.train_ids) - 1)
    bigram_n = bigram_baseline_bpc(
        corpus.train_ids[: limit + 1],
        corpus.eval_ids,
        corpus.vocab_size,
    )
    rows: list[dict[str, Any]] = []
    for mode in modes:
        tag = f"{mode['label']} n={n_train}"
        try:
            row = _eval_bpc(corpus, n_train=n_train, n_eval=n_eval, mode=mode)
            row["bigram_bpc"] = float(bigram_n)
            row["delta_vs_bigram"] = row["held_out_bpc"] - bigram_n
            row["overrides"] = {}
            rows.append(row)
            print(
                f"[{tag}] bpc={row['held_out_bpc']:.4f} "
                f"d_bi={row['delta_vs_bigram']:+.4f} ({row['train_seconds']:.0f}s)"
            )
        except Exception as exc:
            print(f"[{tag}] FAIL: {exc}")
    return rows


def run_axes(
    *,
    axes: list[str],
    n_train: int,
    n_eval: int,
    include_schedules: bool,
) -> dict[str, Any]:
    corpus = prepare_lm_corpus(prefer_wikitext=True, max_train_chars=10_000_000)
    require_real_corpus(corpus.source, domain="cyphalm_beat_bigram_axes")
    t0 = time.perf_counter()

    by_axis: dict[str, Any] = {}
    for axis in axes:
        print(f"\n=== axis: {axis} ===")
        rows = _run_axis(
            corpus,
            axis=axis,
            values=AXIS_VALUES[axis],
            n_train=n_train,
            n_eval=n_eval,
            mode=BASE_MODE,
        )
        best = min(rows, key=lambda r: r["held_out_bpc"]) if rows else None
        by_axis[axis] = {"history": rows, "best": best}

    schedules = None
    if include_schedules:
        print(f"\n=== schedules @ n={n_train} ===")
        sched_rows = _run_schedules(
            corpus, n_train=n_train, n_eval=n_eval, modes=SCHEDULE_MODES
        )
        best_sched = min(sched_rows, key=lambda r: r["held_out_bpc"]) if sched_rows else None
        schedules = {"history": sched_rows, "best": best_sched}

    all_rows = [r for a in by_axis.values() for r in a.get("history", [])]
    if schedules:
        all_rows.extend(schedules["history"])
    global_best = min(all_rows, key=lambda r: r["held_out_bpc"]) if all_rows else None

    return {
        "corpus": corpus.source,
        "n_train": n_train,
        "n_eval": n_eval,
        "base_mode": BASE_MODE,
        "axes": by_axis,
        "schedules": schedules,
        "best_held_out": global_best,
        "elapsed_s": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Beat-bigram axis sweeps @ schedule_b / 70k")
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument(
        "--axes",
        default="all",
        help="Comma-separated axis names or 'all' (default: all)",
    )
    ap.add_argument("--schedules", action="store_true", help="Compare schedule_a/b/c")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    n_train = args.n_train
    if n_train is None:
        n_train = FAST_N_TRAIN if (args.fast or is_fast()) else DEFAULT_N_TRAIN

    if args.axes == "all":
        axes = list(AXIS_VALUES.keys())
    else:
        axes = [a.strip() for a in args.axes.split(",") if a.strip()]
        for a in axes:
            if a not in AXIS_VALUES:
                raise SystemExit(f"Unknown axis: {a}")

    print(f"Axes: {axes} | n_train={n_train} | schedules={args.schedules}")

    out = run_axes(
        axes=axes,
        n_train=n_train,
        n_eval=args.n_eval,
        include_schedules=args.schedules,
    )

    if args.write or not is_fast():
        _OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {_OUT}")

    best = out.get("best_held_out")
    if best:
        print(
            f"\nGlobal best: {best['label']} BPC={best['held_out_bpc']:.4f} "
            f"d_bi={best['delta_vs_bigram']:+.4f} overrides={best.get('overrides', {})}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
