#!/usr/bin/env python3
"""Phase 2C: sparse Hebbian SSM toggle on hybrid CyphaLM."""

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

_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_hebbian_phase2_sweep.json"


def _run(*, corpus, n_train: int, n_eval: int, profile: str, use_hebbian: bool) -> dict[str, Any]:
    overrides = {
        "context_mode": "hybrid_gria_lstm",
        "use_sparse_hebbian": use_hebbian,
    }
    model, cfg = make_cyphalm(overrides, profile=profile)
    limit = min(n_train, len(corpus.train_ids) - 1)
    train_ids = corpus.train_ids[: limit + 1]
    t0 = time.perf_counter()
    model.train_sequence(train_ids)
    train_s = time.perf_counter() - t0
    bpc = eval_held_out_bpc(model, corpus.eval_ids, n_eval=n_eval)
    return {
        "use_sparse_hebbian": use_hebbian,
        "held_out_bpc": float(bpc),
        "train_seconds": train_s,
        "n_train": limit,
        "context_mode": cfg.context_mode,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Cypha Tests 2C — sparse Hebbian SSM sweep")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--profile", default="d17")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    use_fast = args.fast or is_fast()
    n_train = args.n_train or (40_000 if use_fast else 300_000)
    out_path = args.out or _OUT
    corpus_name = "gutenberg" if args.profile == "d04" else "wikitext"
    corpus = prepare_lm_corpus(
        prefer_wikitext=(corpus_name == "wikitext"),
        max_train_chars=10_000_000,
    )
    require_real_corpus(corpus.source, domain="cyphalm_hebbian_phase2_sweep")
    train_slice = corpus.train_ids[: min(n_train, len(corpus.train_ids) - 1)]
    bigram = bigram_baseline_bpc(train_slice, corpus.eval_ids, corpus.vocab_size)

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for hebbian in (False, True):
        label = "hebbian_on" if hebbian else "hebbian_off"
        print(f"  cell={label} n_train={n_train}", flush=True)
        row = _run(
            corpus=corpus,
            n_train=n_train,
            n_eval=args.n_eval,
            profile=args.profile,
            use_hebbian=hebbian,
        )
        row["cell_id"] = label
        row["delta_vs_bigram"] = row["held_out_bpc"] - bigram
        rows.append(row)
        print(f"    bpc={row['held_out_bpc']:.4f}", flush=True)

    off = next(r for r in rows if r["cell_id"] == "hebbian_off")
    on = next(r for r in rows if r["cell_id"] == "hebbian_on")
    out = {
        "corpus": corpus.source,
        "n_train": n_train,
        "bigram_bpc": float(bigram),
        "cells": rows,
        "hebbian_on_minus_off_bpc": float(on["held_out_bpc"] - off["held_out_bpc"]),
        "elapsed_s": time.perf_counter() - t0,
    }

    if args.write or not use_fast:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
