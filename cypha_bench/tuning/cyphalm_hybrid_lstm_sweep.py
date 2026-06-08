#!/usr/bin/env python3
"""Hybrid GRIA + char-LSTM sweep (model-class C2)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from cypha_bench.adapters.char_lstm_baseline import char_lstm_baseline_bpc
from cypha_bench.adapters.cyphalm_bench import (
    bigram_baseline_bpc,
    eval_held_out_bpc,
    make_cyphalm,
    prepare_lm_corpus,
    require_real_corpus,
)
from cypha_bench.common.paths import is_fast

_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_hybrid_lstm_sweep.json"

CELLS = [
    {"id": "gria_ngram", "context_mode": "gria_ngram"},
    {"id": "hybrid_gria_lstm", "context_mode": "hybrid_gria_lstm"},
    {"id": "char_lstm", "context_mode": "char_lstm"},
]


def _run_cell(corpus, *, cell: dict, n_train: int, n_eval: int, profile: str) -> dict[str, Any]:
    overrides = {"context_mode": cell["context_mode"]}
    if cell["context_mode"] == "hybrid_gria_lstm":
        overrides.update(
            {
                "view_id_dim": 8,
                "view_learnable": False,
                "ngram_fusion": "sum",
            }
        )
    model, cfg = make_cyphalm(overrides, profile=profile)
    limit = min(n_train, len(corpus.train_ids) - 1)
    train_ids = corpus.train_ids[: limit + 1]
    t0 = time.perf_counter()
    model.train_sequence(train_ids)
    train_s = time.perf_counter() - t0
    bpc = eval_held_out_bpc(model, corpus.eval_ids, n_eval=n_eval)
    row: dict[str, Any] = {
        "cell_id": cell["id"],
        "held_out_bpc": float(bpc),
        "train_seconds": train_s,
        "n_train": limit,
        "context_mode": cfg.context_mode,
    }
    if cell["context_mode"] == "hybrid_gria_lstm":
        row["hybrid_blend_logit"] = float(getattr(model, "_hybrid_blend_logit", 0.0))
        row["hybrid_gria_weight"] = float(1.0 / (1.0 + __import__("math").exp(-row["hybrid_blend_logit"])))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="CyphaLM hybrid GRIA+LSTM sweep")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--profile", default="d17")
    ap.add_argument(
        "--corpus",
        choices=("wikitext", "gutenberg"),
        default=None,
        help="Training corpus (default: wikitext for d17, gutenberg for d04)",
    )
    ap.add_argument(
        "--cells",
        default=None,
        help="Comma-separated cell ids (default: all)",
    )
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    use_fast = args.fast or is_fast()
    n_train = args.n_train or (40_000 if use_fast else 300_000)
    out_path = args.out or _OUT
    corpus_name = args.corpus or ("gutenberg" if args.profile == "d04" else "wikitext")

    corpus = prepare_lm_corpus(
        prefer_wikitext=(corpus_name == "wikitext"),
        max_train_chars=10_000_000,
    )
    require_real_corpus(corpus.source, domain="cyphalm_hybrid_lstm_sweep")
    train_slice = corpus.train_ids[: min(n_train, len(corpus.train_ids) - 1)]
    bigram = bigram_baseline_bpc(train_slice, corpus.eval_ids, corpus.vocab_size)

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    cell_list = CELLS
    if args.cells:
        wanted = {c.strip() for c in args.cells.split(",") if c.strip()}
        cell_list = [c for c in CELLS if c["id"] in wanted]
        if not cell_list:
            raise SystemExit(f"No matching cells in {wanted}")
    for cell in cell_list:
        print(f"  cell={cell['id']} n_train={n_train}", flush=True)
        row = _run_cell(corpus, cell=cell, n_train=n_train, n_eval=args.n_eval, profile=args.profile)
        row["delta_vs_bigram"] = row["held_out_bpc"] - bigram
        rows.append(row)
        print(f"    bpc={row['held_out_bpc']:.4f} d_bi={row['delta_vs_bigram']:.4f}", flush=True)

    char_lstm = char_lstm_baseline_bpc(
        train_slice, corpus.eval_ids, corpus.vocab_size, n_train_steps=len(train_slice)
    )
    gria = next((r for r in rows if r["cell_id"] == "gria_ngram"), None)
    hybrid = next((r for r in rows if r["cell_id"] == "hybrid_gria_lstm"), None)
    char_only = next((r for r in rows if r["cell_id"] == "char_lstm"), None)
    out = {
        "corpus": corpus.source,
        "n_train": n_train,
        "n_eval": args.n_eval,
        "bigram_bpc": float(bigram),
        "char_lstm_baseline_bpc": float(char_lstm),
        "cells": rows,
        "hybrid_minus_gria_bpc": (
            float(gria["held_out_bpc"] - hybrid["held_out_bpc"]) if gria and hybrid else None
        ),
        "char_lstm_minus_hybrid_bpc": (
            float(char_only["held_out_bpc"] - hybrid["held_out_bpc"])
            if char_only and hybrid
            else None
        ),
        "elapsed_s": time.perf_counter() - t0,
    }

    if args.write or not use_fast:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
