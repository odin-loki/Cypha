#!/usr/bin/env python3
"""Systematic CyphaLM component ablation — architecture, toggles, SSM combos, upgrades."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from cypha_bench.adapters.cyphalm_bench import prepare_lm_corpus, require_real_corpus
from cypha_bench.adapters.cyphalm_component_study import all_cells, run_component_study
from cypha_bench.common.paths import is_fast

_OUT = _REPO / "cypha_bench" / "config" / "cyphalm_component_ablation.json"

PHASE_CHOICES = ("architecture", "toggle", "ssm_combo", "upgrade")


def main() -> int:
    ap = argparse.ArgumentParser(description="CyphaLM systematic component ablation study")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--fast", action="store_true", help="Subset of cells + smaller n_train")
    ap.add_argument(
        "--phase",
        action="append",
        choices=PHASE_CHOICES,
        help="Run only these phases (repeatable); default: all",
    )
    ap.add_argument("--profile", default="d17", help="Default profile for cells without explicit profile")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    use_fast = args.fast or is_fast()
    n_train = args.n_train
    if n_train is None:
        n_train = 8000 if use_fast else 40000

    phases = set(args.phase) if args.phase else None
    cells = all_cells(fast=use_fast)
    if phases:
        cells = [c for c in cells if c["phase"] in phases]

    print(f"Component ablation: {len(cells)} cells | n_train={n_train} | fast={use_fast}")

    corpus = prepare_lm_corpus(prefer_wikitext=True, max_train_chars=10_000_000)
    require_real_corpus(corpus.source, domain="cyphalm_component_ablation")

    out = run_component_study(
        corpus,
        n_train=n_train,
        n_eval=args.n_eval,
        phases=phases,
        fast=use_fast,
        default_profile=args.profile,
    )

    if args.write or not is_fast():
        _OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {_OUT}")

    best = out.get("global_best")
    if best:
        print(
            f"\nGlobal best: [{best['id']}] BPC={best['held_out_bpc']:.4f} "
            f"d_bi={best['delta_vs_bigram']:+.4f}"
        )
        print(f"  {best['label']}")

    comps = out.get("comparisons") or {}
    if comps:
        print("\nKey comparisons (positive = first worse than second):")
        for k, v in comps.items():
            print(f"  {k}: {v:+.4f} BPC")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
