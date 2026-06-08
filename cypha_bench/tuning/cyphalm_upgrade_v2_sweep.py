#!/usr/bin/env python3
"""Upgrade V2 sweep: learnable views (Track A) vs fixed baseline."""

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
    make_cyphalm,
    prepare_lm_corpus,
    require_real_corpus,
)
from cypha_bench.common.paths import is_fast

_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_upgrade_v2_sweep.json"

CELLS: list[dict[str, Any]] = [
    {"id": "baseline", "view_learnable": False, "ngram_fusion": "sum"},
    {"id": "learnable_view", "view_learnable": True, "ngram_fusion": "sum"},
    {"id": "gated", "view_learnable": False, "ngram_fusion": "gated"},
]


def _run_cell(
    corpus,
    *,
    cell: dict[str, Any],
    n_train: int,
    n_eval: int,
    profile: str,
) -> dict[str, Any]:
    overrides = {
        "view_learnable": bool(cell["view_learnable"]),
        "view_id_dim": 8,
        "ngram_fusion": str(cell.get("ngram_fusion", "sum")),
    }
    model, cfg = make_cyphalm(overrides, profile=profile)
    limit = min(n_train, len(corpus.train_ids) - 1)
    train_ids = corpus.train_ids[: limit + 1]
    t0 = time.perf_counter()
    model.train_sequence(train_ids)
    train_s = time.perf_counter() - t0
    bpc = eval_held_out_bpc(model, corpus.eval_ids, n_eval=n_eval)
    return {
        "cell_id": cell["id"],
        "held_out_bpc": float(bpc),
        "train_seconds": train_s,
        "n_train": limit,
        "view_learnable": cfg.view_learnable,
        "ngram_fusion": cfg.ngram_fusion,
        "context_mode": cfg.context_mode,
        "view_schedule": cfg.view_schedule,
    }


def run_sweep(
    *,
    n_train: int,
    n_eval: int = 2000,
    profile: str = "d17",
    cells: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    corpus = prepare_lm_corpus(prefer_wikitext=True, max_train_chars=10_000_000)
    require_real_corpus(corpus.source, domain="cyphalm_upgrade_v2_sweep")
    train_slice = corpus.train_ids[: min(n_train, len(corpus.train_ids) - 1)]
    bigram = bigram_baseline_bpc(train_slice, corpus.eval_ids, corpus.vocab_size)

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for cell in cells or CELLS:
        print(f"  cell={cell['id']} n_train={n_train}", flush=True)
        row = _run_cell(corpus, cell=cell, n_train=n_train, n_eval=n_eval, profile=profile)
        row["delta_vs_bigram"] = row["held_out_bpc"] - bigram
        rows.append(row)
        print(f"    bpc={row['held_out_bpc']:.4f} d_bi={row['delta_vs_bigram']:.4f}", flush=True)

    best = min(rows, key=lambda r: r["held_out_bpc"])
    baseline = next((r for r in rows if r["cell_id"] == "baseline"), None)
    learnable = next((r for r in rows if r["cell_id"] == "learnable_view"), None)
    improvement = None
    if baseline and learnable:
        improvement = float(baseline["held_out_bpc"] - learnable["held_out_bpc"])

    return {
        "corpus": corpus.source,
        "n_train": n_train,
        "n_eval": n_eval,
        "profile": profile,
        "bigram_bpc": float(bigram),
        "cells": rows,
        "best": best,
        "learnable_minus_baseline_bpc": improvement,
        "elapsed_s": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CyphaLM Upgrade V2 sweep (learnable views)")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--profile", default="d17")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--cells", nargs="*", default=None, help="Cell ids to run (default: all)")
    ap.add_argument("--out", type=Path, default=None, help="Output JSON path")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    use_fast = args.fast or is_fast()
    n_train = args.n_train or (40_000 if use_fast else 300_000)
    out_path = args.out or _OUT
    cell_list = CELLS
    if args.cells:
        wanted = set(args.cells)
        cell_list = [c for c in CELLS if c["id"] in wanted]

    print(f"Upgrade V2 sweep | n_train={n_train} | profile={args.profile} | cells={[c['id'] for c in cell_list]}", flush=True)
    out = run_sweep(n_train=n_train, n_eval=args.n_eval, profile=args.profile, cells=cell_list)

    if args.write or not use_fast:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}", flush=True)

    imp = out.get("learnable_minus_baseline_bpc")
    if imp is not None:
        print(f"Learnable vs fixed: {imp:+.4f} BPC (positive = learnable better)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
