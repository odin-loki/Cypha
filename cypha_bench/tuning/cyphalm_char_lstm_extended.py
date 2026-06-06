#!/usr/bin/env python3
"""Char-LSTM baseline at extended train lengths (model-class research M1)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from cypha_bench.adapters.char_lstm_baseline import char_lstm_baseline_bpc
from cypha_bench.adapters.cyphalm_bench import (
    bigram_baseline_bpc,
    prepare_lm_corpus,
    require_real_corpus,
)

_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_char_lstm_extended.json"

GRID = [40000, 70000, 150000, 300000]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    corpus = prepare_lm_corpus(prefer_wikitext=True, max_train_chars=10_000_000)
    require_real_corpus(corpus.source, domain="cyphalm_char_lstm_extended")
    results = []
    t0 = time.perf_counter()

    for n_train in GRID:
        limit = min(n_train, len(corpus.train_ids) - 1)
        if limit < 1000:
            continue
        train_slice = corpus.train_ids[:limit]
        print(f"[char_lstm] n={limit} ...")
        t1 = time.perf_counter()
        bpc = char_lstm_baseline_bpc(
            train_slice,
            corpus.eval_ids,
            corpus.vocab_size,
            n_train_steps=limit,
        )
        bigram = bigram_baseline_bpc(train_slice, corpus.eval_ids, corpus.vocab_size)
        row = {
            "n_train": limit,
            "char_lstm_bpc": float(bpc),
            "bigram_bpc": float(bigram),
            "delta_vs_bigram": float(bpc - bigram),
            "elapsed_s": time.perf_counter() - t1,
        }
        results.append(row)
        print(f"  bpc={bpc:.4f} d_bi={row['delta_vs_bigram']:+.4f}")

    out = {
        "corpus": corpus.source,
        "grid_n_train": GRID,
        "n_eval": args.n_eval,
        "elapsed_s": time.perf_counter() - t0,
        "results": results,
        "best": min(results, key=lambda r: r["char_lstm_bpc"]) if results else None,
    }
    if args.write:
        _OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
