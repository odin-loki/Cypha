#!/usr/bin/env python3
"""Find CyphaLM training limit: held-out BPC vs train steps until plateau."""

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
from cypha_bench.tuning.cyphalm_view_iteration_sweep import TRAINING_MODES, _eval_bpc

_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_convergence_limit.json"

CONVERGENCE_MODES = [
    {"label": "same_order_e2", "view_schedule": "same_order", "train_epochs": 2},
    {"label": "schedule_b", "view_schedule": "schedule_b", "train_epochs": 2},
]

# Continue beyond prior 40k cap until plateau.
EXTENDED_N_TRAIN = [
    40000,
    50000,
    60000,
    70000,
    80000,
    90000,
    100000,
    120000,
    150000,
    200000,
    250000,
]
FAST_N_TRAIN = [40000, 60000, 80000, 100000, 120000]

PLATEAU_TOL = 0.004


def _plateau_hit(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return plateau info if last two steps show no improvement (< tol)."""
    if len(history) < 3:
        return None
    a, b, c = history[-3], history[-2], history[-1]
    d1 = b["held_out_bpc"] - a["held_out_bpc"]
    d2 = c["held_out_bpc"] - b["held_out_bpc"]
    if d1 > PLATEAU_TOL and d2 > PLATEAU_TOL:
        return {
            "kind": "overtrain",
            "limit_step": b["n_train"],
            "best": min(history, key=lambda r: r["held_out_bpc"]),
        }
    if abs(d1) < PLATEAU_TOL and abs(d2) < PLATEAU_TOL:
        return {
            "kind": "plateau",
            "limit_step": c["n_train"],
            "best": min(history, key=lambda r: r["held_out_bpc"]),
        }
    return None


def _sweep_mode(
    corpus,
    mode: dict[str, Any],
    *,
    grid: list[int],
    n_eval: int,
    early_stop: bool = True,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    plateau_info = None
    t0 = time.perf_counter()

    for n_train in grid:
        if n_train > len(corpus.train_ids) - 1:
            break
        row = _eval_bpc(corpus, n_train=n_train, n_eval=n_eval, mode=mode)
        limit = min(n_train, len(corpus.train_ids) - 1)
        bigram_n = bigram_baseline_bpc(
            corpus.train_ids[: limit + 1],
            corpus.eval_ids,
            corpus.vocab_size,
        )
        row["bigram_bpc"] = float(bigram_n)
        row["delta_vs_bigram"] = row["held_out_bpc"] - bigram_n
        history.append(row)
        print(
            f"[{mode['label']}] n={n_train} bpc={row['held_out_bpc']:.4f} "
            f"d_bi={row.get('delta_vs_bigram', 0):+.4f} ({row['train_seconds']:.0f}s)"
        )
        if early_stop:
            plateau_info = _plateau_hit(history)
            if plateau_info:
                print(f"  -> {plateau_info['kind']} at n~{plateau_info['limit_step']}")
                break

    best = min(history, key=lambda r: r["held_out_bpc"]) if history else {}
    last = history[-1] if history else {}
    still_improving = False
    if len(history) >= 2:
        still_improving = (last["held_out_bpc"] - history[-2]["held_out_bpc"]) < -PLATEAU_TOL

    return {
        "label": mode["label"],
        "history": history,
        "plateau": plateau_info,
        "best": best,
        "final": last,
        "still_improving_at_end": still_improving,
        "elapsed_s": time.perf_counter() - t0,
    }


def run_convergence(
    *,
    grid: list[int],
    n_eval: int = 2000,
    modes: list[dict[str, Any]] | None = None,
    early_stop: bool = True,
) -> dict[str, Any]:
    corpus = prepare_lm_corpus(
        prefer_wikitext=True,
        max_train_chars=10_000_000,
    )
    require_real_corpus(corpus.source, domain="cyphalm_convergence_limit")
    corpus_tokens = len(corpus.train_ids)

    modes = modes or CONVERGENCE_MODES
    t0 = time.perf_counter()
    by_mode: dict[str, Any] = {}

    for mode in modes:
        print(f"\n=== {mode['label']} ===")
        by_mode[mode["label"]] = _sweep_mode(
            corpus,
            mode,
            grid=grid,
            n_eval=n_eval,
            early_stop=early_stop,
        )

    out = {
        "corpus": corpus.source,
        "corpus_train_tokens": corpus_tokens,
        "grid_n_train": grid,
        "n_eval": n_eval,
        "plateau_tol_bpc": PLATEAU_TOL,
        "prior_sweep_cap": 40000,
        "elapsed_s": time.perf_counter() - t0,
        "modes": by_mode,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Find CyphaLM training convergence limit")
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    grid = FAST_N_TRAIN if (args.fast or is_fast()) else EXTENDED_N_TRAIN

    out = run_convergence(
        grid=grid,
        n_eval=args.n_eval,
        early_stop=not args.no_early_stop,
    )
    if args.write or not is_fast():
        _OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {_OUT}")

    for label, data in out["modes"].items():
        b = data.get("best") or {}
        p = data.get("plateau")
        print(
            f"\n{label}: best BPC {b.get('held_out_bpc', float('nan')):.4f} @ n={b.get('n_train')} "
            f"improving_at_end={data.get('still_improving_at_end')} "
            f"plateau={p.get('kind') if p else 'not_detected'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
