#!/usr/bin/env python3
"""Demo: embed query → CyphaDIF route with epistemic gate (Branch A)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from cypha_bench.adapters.branch_a_documents import run_branch_a_documents
from cypha_studio.core.branch_a_router import BranchARouter
from cypha_studio.core.ollama_client import ollama_available


def main() -> int:
    ap = argparse.ArgumentParser(description="Branch A routing demo")
    ap.add_argument("query", nargs="?", default="How do I compile Linux kernel modules?")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--quick-bench", action="store_true", help="Print full Branch A bench summary")
    ap.add_argument("--generate", action="store_true", help="Dispatch to CyphaLM or Ollama after route")
    ap.add_argument("--backend", default=os.environ.get("CYPHA_BRANCH_A_EMBED_BACKEND", "auto"))
    ap.add_argument("--n-train", type=int, default=1200)
    args = ap.parse_args()

    if args.quick_bench:
        out = run_branch_a_documents(n_samples=800, backend="auto")
        print(f"accuracy={out['cypha_accuracy']:.4f}")
        print(f"ood_epistemic={out['gutenberg_ood']['mean_epistemic_ood']:.4f}")
        return 0

    print(f"Training mini router on 20 Newsgroups (backend={args.backend})...", flush=True)
    router = BranchARouter(
        epistemic_threshold=args.threshold,
        n_train_samples=args.n_train,
        backend=args.backend,
    )
    info = router.train()
    print(f"  trained in {info['train_seconds']:.1f}s, backend={info['embedding_backend']}")

    if args.generate:
        from cypha_studio.core.lm_engine import LMEngine

        lm = None
        ckpt = os.environ.get("CYPHA_LM_CHECKPOINT", "").strip()
        if ckpt:
            try:
                lm = LMEngine.from_checkpoint(ckpt)
                print(f"CyphaLM loaded from {ckpt}")
            except Exception as exc:
                print(f"CyphaLM load skipped: {exc}")
        result = router.dispatch_generate(args.query, lm)
        print(f"query: {args.query!r}")
        print(f"  route: {result['route']}")
        gen = result.get("generation") or {}
        print(f"  generation provider: {gen.get('provider')}")
        if gen.get("text"):
            print(f"  text: {gen['text'][:500]}")
        if gen.get("error"):
            print(f"  error: {gen['error']}")
        return 0

    result = router.route(args.query)
    print(f"query: {args.query!r}")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if result["action"] == "fallback_llm":
        if ollama_available():
            print("Ollama reachable — run with --generate to call fallback LLM.")
        else:
            print("Ollama not reachable (set CYPHA_OLLAMA_URL or start Ollama).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
