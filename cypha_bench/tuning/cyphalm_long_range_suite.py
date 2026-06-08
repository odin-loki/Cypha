#!/usr/bin/env python3
"""CyphaLM long-range context suite (Cypha Tests.txt Phase 1 mapped to CyphaLM)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from cypha_bench.adapters.cyphalm_bench import prepare_lm_corpus, require_real_corpus
from cypha_bench.adapters.cyphalm_long_range import run_long_range_suite, save_long_range_figures
from cypha_bench.common.paths import is_fast

_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_long_range_suite.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="CyphaLM long-range context test suite")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--profile", default="d17")
    ap.add_argument(
        "--corpus",
        choices=("wikitext", "gutenberg"),
        default=None,
        help="Training corpus (default: wikitext for d17, gutenberg for d04)",
    )
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true", help="Skip 3× SSM ablation retrains")
    ap.add_argument("--figures", action="store_true", help="Write report figures")
    ap.add_argument("--out", type=Path, default=None, help="Output JSON path")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    use_fast = args.fast or is_fast()
    n_train = args.n_train or (8000 if use_fast else 40000)
    out_path = args.out or _OUT
    corpus_name = args.corpus or ("gutenberg" if args.profile == "d04" else "wikitext")

    corpus = prepare_lm_corpus(
        prefer_wikitext=(corpus_name == "wikitext"),
        max_train_chars=10_000_000,
    )
    require_real_corpus(corpus.source, domain="cyphalm_long_range_suite")

    print(
        f"Long-range suite | n_train={n_train} | profile={args.profile} | "
        f"fast={use_fast} | skip_ablation={args.skip_ablation}",
        flush=True,
    )
    out = run_long_range_suite(
        corpus,
        n_train=n_train,
        n_eval=args.n_eval,
        profile=args.profile,
        fast=use_fast,
        skip_ablation=args.skip_ablation,
    )

    if args.write or not is_fast():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}", flush=True)

    if args.figures:
        save_long_range_figures(out)

    ctx = out.get("context_length_bpc") or {}
    if ctx:
        best_ctx = min(ctx.items(), key=lambda kv: kv[1])
        print(f"Best context window: {best_ctx[0]} -> {best_ctx[1]:.4f} BPC", flush=True)
    sh = out.get("sequential_vs_shuffled") or {}
    if sh:
        print(
            f"Forward {sh.get('forward_bpc', float('nan')):.4f} vs "
            f"block-shuffled {sh.get('block_shuffled_bpc', float('nan')):.4f} BPC",
            flush=True,
        )
        if sh.get("char_shuffled_bpc") is not None:
            print(
                f"  char-shuffled {sh['char_shuffled_bpc']:.4f} BPC "
                f"(delta {sh.get('delta_char_shuffled_minus_forward', float('nan')):+.4f})",
                flush=True,
            )
    ab = out.get("ssm_ablation_sequential") or {}
    if ab.get("ssm_contribution_bpc") is not None:
        print(f"SSM contribution (no_ssm - gria): {ab['ssm_contribution_bpc']:.4f} BPC", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
