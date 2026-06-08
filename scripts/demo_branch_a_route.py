#!/usr/bin/env python3
"""Demo: embed query → CyphaDIF route with epistemic gate (Branch A)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from cypha_bench.adapters.branch_a_documents import run_branch_a_documents
from cypha_bench.adapters.frozen_text_embeddings import embed_texts
from cypha_bench.adapters.bench_models import BenchClassifier
from cypha_bench.common.metrics import online_train_classifier, standardize_train_test
from cypha_bench.config.load_profile import classification_params, load_profile
from cypha_bench.domains.d09_documents import _load_20news
from sklearn.model_selection import train_test_split


def _train_router(n_samples: int = 1200) -> tuple[BenchClassifier, np.ndarray, np.ndarray]:
    _, _, _, y, texts = _load_20news(n_samples)
    x, _ = embed_texts(texts, backend="auto")
    y = np.asarray(y)
    x_train, _, y_train, _, = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, _ = standardize_train_test(x_train, x_train)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0] = 1.0
    passes = max(1, int(classification_params(load_profile()).get("n_epochs", 1)))
    model = BenchClassifier(x_train.shape[1], seed=42)
    model.dif.encoder._frozen = True
    online_train_classifier(model, x_train, y_train, label_fn=str, passes=passes)
    return model, mean, std


def route(
    model: BenchClassifier,
    mean: np.ndarray,
    std: np.ndarray,
    text: str,
    *,
    epistemic_threshold: float = 0.5,
) -> dict:
    vec, meta = embed_texts([text], backend="auto")
    x = (vec[0] - mean) / std
    label, probs, epistemic = model.predict(x)
    conf = float(np.max(probs)) if len(probs) else 0.0
    abstain = epistemic > epistemic_threshold
    return {
        "label": label,
        "confidence": conf,
        "epistemic_var": float(epistemic),
        "abstain": abstain,
        "embedding_backend": meta.get("backend"),
        "action": "fallback_llm" if abstain else "cypha_route",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Branch A routing demo")
    ap.add_argument("query", nargs="?", default="How do I compile Linux kernel modules?")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--quick-bench", action="store_true", help="Print full Branch A bench summary")
    args = ap.parse_args()

    if args.quick_bench:
        out = run_branch_a_documents(n_samples=800, backend="auto")
        print(f"accuracy={out['cypha_accuracy']:.4f}")
        print(f"ood_epistemic={out['gutenberg_ood']['mean_epistemic_ood']:.4f}")
        return 0

    print("Training mini router on 20 Newsgroups (frozen MiniLM)...", flush=True)
    model, mean, std = _train_router(1200)
    result = route(model, mean, std, args.query, epistemic_threshold=args.threshold)
    print(f"query: {args.query!r}")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
