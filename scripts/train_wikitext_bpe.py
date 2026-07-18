#!/usr/bin/env python3
"""Train a tiny GPT-style char BPE for CyphaLM (merges.txt + vocab.json).

Usage:
  python scripts/train_wikitext_bpe.py --text-file bench/data/wikitext2/wiki.train.tokens \\
      --out-dir bench/data/wikitext2/bpe --merges 2000

If --text-file is missing, uses a built-in English sample so smoke still works.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def train_bpe(text: str, num_merges: int) -> tuple[list[tuple[str, str]], dict[str, int]]:
    # Character-level tokens (one Unicode char each), then greedy pair merges.
    tokens = [ch for ch in text if ch.isprintable() or ch in "\n\t "]
    if not tokens:
        tokens = list("the cat sat on the mat. ")
    vocab: dict[str, int] = {}
    for ch in sorted(set(tokens)):
        vocab[ch] = len(vocab)
    merges: list[tuple[str, str]] = []
    seq = tokens[:]
    for _ in range(num_merges):
        pairs = Counter(zip(seq, seq[1:]))
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        if a + b in vocab:
            # Already merged somehow; skip
            merged = a + b
        else:
            merged = a + b
            vocab[merged] = len(vocab)
            merges.append((a, b))
        # Apply merge left-to-right
        out: list[str] = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
    return merges, vocab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", default="")
    ap.add_argument("--out-dir", default="bench/data/wikitext2/bpe")
    ap.add_argument("--merges", type=int, default=2000)
    ap.add_argument("--max-chars", type=int, default=2_000_000)
    args = ap.parse_args()

    text = ""
    if args.text_file and os.path.isfile(args.text_file):
        text = Path(args.text_file).read_text(encoding="utf-8", errors="ignore")[: args.max_chars]
    if not text:
        text = (
            "the cat sat on the mat. the cat ran to the mat. "
            "wikitext language modeling with byte pair encoding. "
            * 200
        )
        print("train_wikitext_bpe: using built-in sample (no --text-file)")

    merges, vocab = train_bpe(text, args.merges)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merges_path = out / "merges.txt"
    vocab_path = out / "vocab.json"
    with merges_path.open("w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=0)
        f.write("\n")
    print(f"wrote {merges_path} ({len(merges)} merges)")
    print(f"wrote {vocab_path} (vocab={len(vocab)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
