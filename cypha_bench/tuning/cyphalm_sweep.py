#!/usr/bin/env python3
"""CyphaLM hyperparameter sweep — full grid or single-axis — per corpus/domain profile."""

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
    DEFAULT_CYPHALM_CONFIG,
    bigram_baseline_bpc,
    eval_held_out_bpc,
    load_cyphalm_config,
    make_cyphalm,
    prepare_lm_corpus,
    require_real_corpus,
)
from cypha_bench.config.load_cyphalm_profile import write_cyphalm_profile

_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_profile_sweep.json"
_AXIS_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_profile_axis_sweeps.json"

FULL_GRID = {
    "gria_lr": [0.04, 0.06, 0.08],
    "tau_slow": [20.0, 50.0, 100.0],
    "train_ssm": [False, True],
    "n_experts": [0, 4],
}

AXIS_VALUES: dict[str, list] = {
    "gria_lr": [0.03, 0.04, 0.06, 0.08, 0.10],
    "tau_slow": [10.0, 20.0, 50.0, 100.0, 200.0],
    "tau_fast": [0.5, 1.0, 2.0, 5.0],
    "n_experts": [0, 1, 2, 4, 8],
    "train_ssm": [False, True],
    "ssm_lr": [0.0005, 0.001, 0.002, 0.005],
    "gria_lr_decay": [0.0, 0.3, 0.5, 0.7, 0.9, 1.0],
    "laplace_smoothing": [0.01, 0.1, 0.5, 1.0, 5.0, 20.0],
    "ngram_context": [1, 2, 3, 4],
    "bptt_steps": [0, 32, 64, 128],
}


def _corpus(prefer_wikitext: bool):
    corpus = prepare_lm_corpus(prefer_wikitext=prefer_wikitext)
    require_real_corpus(corpus.source, domain="cyphalm_sweep")
    return corpus


def _score(overrides: dict, *, prefer_wikitext: bool, n_train: int, n_eval: int, base: dict) -> dict:
    merged = {**base, **overrides}
    corpus = _corpus(prefer_wikitext)
    model, cfg = make_cyphalm(merged, profile=None)
    limit = min(n_train, len(corpus.train_ids) - 1)
    model.train_sequence(corpus.train_ids[:limit])
    bpc = eval_held_out_bpc(model, corpus.eval_ids, n_eval=n_eval)
    train_slice = corpus.train_ids[:limit]
    bigram = bigram_baseline_bpc(train_slice, corpus.eval_ids, corpus.vocab_size)
    return {
        "cyphalm_bpc": float(bpc),
        "bigram_bpc": float(bigram),
        "delta_vs_bigram": float(bpc - bigram),
        "overrides": overrides,
        "merged_keys": {k: getattr(cfg, k, merged.get(k)) for k in overrides},
    }


def _iter_configs(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]


def run_sweep(
    *,
    prefer_wikitext: bool,
    n_train: int,
    n_eval: int,
    grid: dict[str, list],
    base: dict,
    label: str,
) -> dict:
    results: list[dict] = []
    t0 = time.perf_counter()
    for overrides in _iter_configs(grid):
        try:
            row = _score(overrides, prefer_wikitext=prefer_wikitext, n_train=n_train, n_eval=n_eval, base=base)
            row["label"] = label
            results.append(row)
            print(f"[{label}] delta={row['delta_vs_bigram']:+.3f} bpc={row['cyphalm_bpc']:.3f} {overrides}")
        except Exception as exc:
            print(f"[{label}] SKIP {overrides}: {exc}")
    results.sort(key=lambda r: r["delta_vs_bigram"])
    return {
        "label": label,
        "corpus": "wikitext2" if prefer_wikitext else "gutenberg",
        "n_train": n_train,
        "n_eval": n_eval,
        "elapsed_s": time.perf_counter() - t0,
        "best": results[0] if results else None,
        "all": results,
    }


def _write_best(profile_key: str, base: dict, best: dict, corpus: str) -> None:
    merged = {**base, **best["overrides"]}
    meta = {
        "source": "cyphalm_sweep",
        "corpus": corpus,
        "delta_vs_bigram": best["delta_vs_bigram"],
        "cyphalm_bpc": best["cyphalm_bpc"],
        "bigram_bpc": best["bigram_bpc"],
    }
    path = write_cyphalm_profile(merged, profile=profile_key, meta=meta)
    print(f"Updated {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="CyphaLM sweep → domain profiles")
    ap.add_argument("--corpus", choices=("d17", "d04", "both"), default="both")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--axis", choices=[*AXIS_VALUES.keys(), "none"], default="none")
    ap.add_argument("--write-profile", action="store_true")
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-axis", action="store_true")
    ap.add_argument(
        "--view-schedule",
        default=None,
        help="view_schedule preset for training (default: profile value or same_order)",
    )
    args = ap.parse_args()

    profile_cfg = load_cyphalm_config()
    base = {**DEFAULT_CYPHALM_CONFIG, **profile_cfg}
    view_sched = args.view_schedule or profile_cfg.get("view_schedule", "same_order")
    base["view_schedule"] = view_sched
    full_out: dict[str, Any] = {}
    axis_out: dict[str, Any] = {}

    jobs: list[tuple[str, bool, str]] = []
    if args.corpus in ("d17", "both"):
        jobs.append(("d17", True, "d17"))
    if args.corpus in ("d04", "both"):
        jobs.append(("d04", False, "d04"))

    for tag, prefer_wt, profile_key in jobs:
        if not args.skip_full and args.axis == "none":
            full = run_sweep(
                prefer_wikitext=prefer_wt,
                n_train=args.n_train,
                n_eval=args.n_eval,
                grid=FULL_GRID,
                base=base,
                label=f"{tag}_full",
            )
            full_out[tag] = full
            if full.get("best") and args.write_profile:
                _write_best(profile_key, base, full["best"], full["corpus"])
                if tag == "d17":
                    _write_best("llm", base, full["best"], full["corpus"])

        if not args.skip_axis:
            axis_runs: dict[str, Any] = {}
            axes = [args.axis] if args.axis != "none" else list(AXIS_VALUES.keys())
            for ax in axes:
                one = run_sweep(
                    prefer_wikitext=prefer_wt,
                    n_train=args.n_train,
                    n_eval=args.n_eval,
                    grid={ax: AXIS_VALUES[ax]},
                    base=base,
                    label=f"{tag}_axis_{ax}",
                )
                axis_runs[ax] = one
                b = one.get("best")
                if b:
                    print(
                        f"  [{tag}] best {ax}={b['overrides'].get(ax)} "
                        f"delta={b['delta_vs_bigram']:+.3f}"
                    )
            axis_out[tag] = axis_runs

    _OUT.write_text(json.dumps(full_out, indent=2), encoding="utf-8")
    _AXIS_OUT.write_text(json.dumps(axis_out, indent=2), encoding="utf-8")
    print(f"\nWrote {_OUT} and {_AXIS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
