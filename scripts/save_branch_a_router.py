#!/usr/bin/env python3
"""Train Branch A router and write checkpoint for fast REST/Studio cold-start."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from cypha_studio.core.branch_a_router import BranchARouter, default_checkpoint_base


def main() -> int:
    ap = argparse.ArgumentParser(description="Train and save Branch A router checkpoint")
    ap.add_argument("--n-train", type=int, default=int(os.environ.get("CYPHA_BRANCH_A_N_TRAIN", "1200")))
    ap.add_argument("--backend", default=os.environ.get("CYPHA_BRANCH_A_EMBED_BACKEND", "auto"))
    ap.add_argument(
        "--out",
        default=str(default_checkpoint_base()),
        help="Checkpoint base path (writes .json + .npz)",
    )
    args = ap.parse_args()

    router = BranchARouter(n_train_samples=args.n_train, backend=args.backend)
    info = router.train()
    path = router.save_checkpoint(args.out)
    print(f"saved {path}")
    print(f"  accuracy proxy: {info['n_classes']} classes, dim={info['input_dim']}")
    print(f"  backend={info['embedding_backend']}, train_s={info['train_seconds']:.1f}")
    print(f"Set CYPHA_BRANCH_A_CHECKPOINT={args.out} to skip retrain on startup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
