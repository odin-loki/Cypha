#!/usr/bin/env python3
"""Branch A — CyphaDIF on frozen text embeddings (Cypha Tests.txt)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from cypha_bench.adapters.bench_models import BenchClassifier
from cypha_bench.adapters.frozen_text_embeddings import embed_texts
from cypha_bench.common.metrics import evaluate_classification, online_train_classifier, standardize_train_test
from cypha_bench.config.load_profile import classification_params, load_profile
from cypha_bench.domains.d09_documents import _load_20news

_OUT = _REPO / "cypha_bench" / "config" / "cypha_branch_a_sweep.json"


def _load_corpus(n_samples: int) -> tuple[list[str], np.ndarray]:
    if n_samples <= 2500:
        _, _, x, y, texts = _load_20news(n_samples)
        return list(texts), np.asarray(y)
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    idx = np.arange(len(data.data))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    idx = idx[:n_samples]
    texts = [data.data[i] for i in idx]
    y = np.asarray(data.target)[idx]
    return texts, y


def _cypha_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    freeze_encoder_proj: bool = False,
) -> dict[str, Any]:
    passes = max(1, int(classification_params(load_profile()).get("n_epochs", 1)))
    t0 = time.perf_counter()
    clf = BenchClassifier(x_train.shape[1], seed=42)
    if freeze_encoder_proj:
        clf.dif.encoder._frozen = True
    online_train_classifier(clf, x_train, y_train, label_fn=str, passes=passes)
    scores = evaluate_classification(clf, x_test, y_test, label_fn=str)
    ep_in = [float(clf.predict(xi)[2]) for xi in x_test[: min(100, len(x_test))]]
    return {
        "accuracy": float(scores["accuracy"]),
        "f1_macro": float(scores.get("f1_macro", scores["accuracy"])),
        "mean_epistemic_in": float(np.mean(ep_in)) if ep_in else None,
        "train_seconds": time.perf_counter() - t0,
        "freeze_encoder_proj": freeze_encoder_proj,
    }


def _lr_eval(x_train, y_train, x_test, y_test) -> dict[str, Any]:
    t0 = time.perf_counter()
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(x_train, y_train)
    acc = float(lr.score(x_test, y_test))
    return {
        "accuracy": acc,
        "f1_macro": acc,
        "train_seconds": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Branch A frozen embedding sweep")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--backend", default="auto", choices=("auto", "sentence_transformers", "hashing"))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    texts, y = _load_corpus(args.n_samples)
    x_raw, embed_meta = embed_texts(texts, backend=args.backend)
    x_train, x_test, y_train, y_test = train_test_split(
        x_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, x_test = standardize_train_test(x_train, x_test)

    print(f"Branch A | backend={embed_meta.get('backend')} | dim={x_train.shape[1]}", flush=True)

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for cell_id, freeze in (
        ("cypha_frozen_embed_online_proj", False),
        ("cypha_frozen_embed_frozen_proj", True),
        ("logreg_frozen_embed", None),
    ):
        print(f"  cell={cell_id}", flush=True)
        if cell_id == "logreg_frozen_embed":
            row = _lr_eval(x_train, y_train, x_test, y_test)
        else:
            row = _cypha_eval(
                x_train, y_train, x_test, y_test, freeze_encoder_proj=bool(freeze)
            )
        row["cell_id"] = cell_id
        rows.append(row)
        print(f"    acc={row['accuracy']:.4f}", flush=True)

    # TF-IDF+SVD reference (D09-style)
    _, _, x_tfidf, y_ref, _ = _load_20news(args.n_samples)
    xt_train, xt_test, yt_train, yt_test = train_test_split(
        x_tfidf, y_ref, test_size=0.2, random_state=42, stratify=y_ref
    )
    xt_train, xt_test = standardize_train_test(xt_train, xt_test)
    print("  cell=cypha_tfidf_svd_reference", flush=True)
    ref = _cypha_eval(xt_train, yt_train, xt_test, yt_test, freeze_encoder_proj=False)
    ref["cell_id"] = "cypha_tfidf_svd_reference"
    rows.append(ref)
    print(f"    acc={ref['accuracy']:.4f}", flush=True)

    cypha_online = next(r for r in rows if r["cell_id"] == "cypha_frozen_embed_online_proj")
    out = {
        "n_samples": args.n_samples,
        "n_classes": int(len(np.unique(y))),
        "embedding": embed_meta,
        "cells": rows,
        "frozen_embed_minus_tfidf_acc": float(
            cypha_online["accuracy"] - ref["accuracy"]
        ),
        "elapsed_s": time.perf_counter() - t0,
    }
    out_path = args.out or _OUT
    if args.write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
