#!/usr/bin/env python3
"""Train Branch A router and write native ``branch_a_router.json`` + ``router_model.cypha``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from Cypha import cypha_save_binary
from cypha_studio.core.branch_a_router import BranchARouter


def hashing_projection_matrix(texts: list[str], *, n_features: int = 512, n_components: int = 128, seed: int = 42):
    hv = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
    )
    raw = hv.transform(texts).astype(np.float64)
    n_comp = min(n_components, max(2, raw.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    svd.fit(raw)
    return svd.components_.astype(np.float64).reshape(-1).tolist(), n_comp


def main() -> int:
    ap = argparse.ArgumentParser(description="Export native Branch A router checkpoint")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=400)
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    router = BranchARouter(n_train_samples=args.n_train, backend="hashing")
    info = router.train()
    assert router._model is not None and router._mean is not None and router._std is not None

    from cypha_bench.domains.d09_documents import _load_20news

    _, _, _, _, texts = _load_20news(args.n_train)
    proj, n_comp = hashing_projection_matrix(texts)

    cypha_path = out_dir / "router_model.cypha"
    cypha_save_binary(router._model.dif.save_state(), str(cypha_path))

    meta = {
        "version": 1,
        "format": "native",
        "model_cypha": cypha_path.name,
        "mean": router._mean.astype(np.float64).tolist(),
        "std": router._std.astype(np.float64).tolist(),
        "epistemic_threshold": router.epistemic_threshold,
        "embedding_backend": router._embedding_backend or "hashing_svd",
        "hash_n_features": 512,
        "hash_n_components": int(router._mean.shape[0]),
        "hash_seed": 42,
        "projection": proj,
        "n_train_samples": int(info["n_train_samples"]),
        "train_seconds": float(info["train_seconds"]),
        "n_classes": int(info["n_classes"]),
    }
    json_path = out_dir / "branch_a_router.json"
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
