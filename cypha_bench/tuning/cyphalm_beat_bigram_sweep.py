#!/usr/bin/env python3
"""Combined sweep to close the bigram gap: schedule_b × extended n_train × key hyperparams."""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from cypha_bench.adapters.cyphalm_bench import (
    bigram_baseline_bpc,
    load_cyphalm_config,
    prepare_lm_corpus,
    require_real_corpus,
)
from cypha_bench.common.paths import is_fast
from cypha_bench.config.load_cyphalm_profile import write_cyphalm_profile
from cypha_bench.tuning.cyphalm_view_iteration_sweep import _eval_bpc

_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_beat_bigram_sweep.json"

FULL_N_TRAIN = [70000, 100000, 150000]
FAST_N_TRAIN = [40000, 70000]

GRIA_LR = [0.06, 0.08, 0.10]
TAU_SLOW = [20.0, 50.0, 100.0]
LAPLACE_SMOOTHING = [0.0, 1.0, 2.0]

SCHEDULE_B = {"label": "schedule_b", "view_schedule": "schedule_b", "train_epochs": 2}
SCHEDULE_A = {"label": "schedule_a", "view_schedule": "schedule_a", "train_epochs": 2}


def _hp_grid(*, axis_laplace: bool) -> dict[str, list]:
    grid: dict[str, list] = {
        "gria_lr": GRIA_LR,
        "tau_slow": TAU_SLOW,
    }
    if axis_laplace:
        grid["laplace_smoothing"] = LAPLACE_SMOOTHING
    return grid


def _iter_hp_configs(grid: dict[str, list]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]


def _grid_size(
    *,
    n_train: list[int],
    modes: list[dict[str, Any]],
    hp_grid: dict[str, list],
) -> int:
    hp_n = 1
    for vals in hp_grid.values():
        hp_n *= len(vals)
    return len(modes) * len(n_train) * hp_n


def run_sweep(
    *,
    n_train_grid: list[int],
    n_eval: int = 2000,
    modes: list[dict[str, Any]] | None = None,
    hp_grid: dict[str, list] | None = None,
) -> dict[str, Any]:
    corpus = prepare_lm_corpus(
        prefer_wikitext=True,
        max_train_chars=10_000_000,
    )
    require_real_corpus(corpus.source, domain="cyphalm_beat_bigram_sweep")

    modes = modes or [SCHEDULE_B]
    hp_grid = hp_grid or _hp_grid(axis_laplace=False)
    hp_configs = _iter_hp_configs(hp_grid)

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for n_train in n_train_grid:
        if n_train > len(corpus.train_ids) - 1:
            print(f"SKIP n_train={n_train}: exceeds corpus ({len(corpus.train_ids) - 1} tokens)")
            continue
        limit = min(n_train, len(corpus.train_ids) - 1)
        bigram_n = bigram_baseline_bpc(
            corpus.train_ids[: limit + 1],
            corpus.eval_ids,
            corpus.vocab_size,
        )
        for mode in modes:
            for hp in hp_configs:
                tag = (
                    f"{mode['label']} n={n_train} "
                    f"lr={hp.get('gria_lr')} tau={hp.get('tau_slow')}"
                )
                if "laplace_smoothing" in hp:
                    tag += f" lap={hp['laplace_smoothing']}"
                try:
                    row = _eval_bpc(
                        corpus,
                        n_train=n_train,
                        n_eval=n_eval,
                        mode=mode,
                        extra_overrides=hp,
                    )
                    row["bigram_bpc"] = float(bigram_n)
                    row["delta_vs_bigram"] = row["held_out_bpc"] - bigram_n
                    row["overrides"] = {**hp}
                    results.append(row)
                    print(
                        f"[{tag}] bpc={row['held_out_bpc']:.4f} "
                        f"d_bi={row['delta_vs_bigram']:+.4f} ({row['train_seconds']:.0f}s)"
                    )
                except Exception as exc:
                    print(f"[{tag}] FAIL: {exc}")

    results.sort(key=lambda r: r["delta_vs_bigram"])
    best = results[0] if results else None

    out: dict[str, Any] = {
        "corpus": corpus.source,
        "corpus_train_tokens": len(corpus.train_ids),
        "n_eval": n_eval,
        "grid_n_train": n_train_grid,
        "modes": [m["label"] for m in modes],
        "hp_grid": hp_grid,
        "grid_size": _grid_size(n_train=n_train_grid, modes=modes, hp_grid=hp_grid),
        "elapsed_s": time.perf_counter() - t0,
        "results": results,
    }
    if best:
        out["best"] = {
            "label": best["label"],
            "n_train": best["n_train"],
            "held_out_bpc": best["held_out_bpc"],
            "bigram_bpc": best["bigram_bpc"],
            "delta_vs_bigram": best["delta_vs_bigram"],
            "train_seconds": best["train_seconds"],
            "overrides": best["overrides"],
            "view_schedule": best["view_schedule"],
            "train_epochs": best["train_epochs"],
        }
    return out


def _write_best_profile(best: dict[str, Any]) -> None:
    overrides = {
        "view_schedule": best["view_schedule"],
        "train_epochs": best["train_epochs"],
        **best["overrides"],
    }
    merged = load_cyphalm_config(overrides, profile="d17")
    meta = {
        "source": "cyphalm_beat_bigram_sweep",
        "corpus": "wikitext2",
        "n_train": best["n_train"],
        "delta_vs_bigram": best["delta_vs_bigram"],
        "held_out_bpc": best["held_out_bpc"],
        "bigram_bpc": best["bigram_bpc"],
    }
    for profile_key in ("d17", "llm"):
        path = write_cyphalm_profile(merged, profile=profile_key, meta=meta)
        print(f"Updated {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CyphaLM combined sweep to beat bigram baseline at extended n_train"
    )
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--write", action="store_true", help="Write JSON to config/")
    ap.add_argument(
        "--include-schedule-a",
        action="store_true",
        help="Also sweep schedule_a (default: schedule_b only)",
    )
    ap.add_argument(
        "--axis",
        choices=("laplace_smoothing",),
        default=None,
        help="Optional single-axis extension to the grid",
    )
    ap.add_argument(
        "--write-profile",
        action="store_true",
        help="If best beats bigram (delta<0), write d17/llm profiles",
    )
    args = ap.parse_args()

    n_train_grid = FAST_N_TRAIN if (args.fast or is_fast()) else FULL_N_TRAIN
    modes = [SCHEDULE_B]
    if args.include_schedule_a:
        modes.append(SCHEDULE_A)
    hp_grid = _hp_grid(axis_laplace=args.axis == "laplace_smoothing")

    grid_n = _grid_size(n_train=n_train_grid, modes=modes, hp_grid=hp_grid)
    print(f"Grid: {grid_n} configs | n_train={n_train_grid} | modes={[m['label'] for m in modes]}")

    out = run_sweep(
        n_train_grid=n_train_grid,
        n_eval=args.n_eval,
        modes=modes,
        hp_grid=hp_grid,
    )

    if args.write or not is_fast():
        _OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {_OUT}")

    best = out.get("best")
    if best:
        beats = best["delta_vs_bigram"] < 0
        print(
            f"\nBest: {best['label']} @ n_train={best['n_train']} "
            f"BPC={best['held_out_bpc']:.4f} bigram={best['bigram_bpc']:.4f} "
            f"delta={best['delta_vs_bigram']:+.4f} ({best['train_seconds']:.0f}s)"
        )
        print(f"  overrides: {best['overrides']}")
        if beats:
            print("  -> beats bigram baseline")
            if args.write_profile:
                _write_best_profile(best)
        elif args.write_profile:
            print("  -> does not beat bigram; skipping profile write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
