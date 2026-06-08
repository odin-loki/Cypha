#!/usr/bin/env python3
"""Cypha Tests 2A — contrastive vs Hebbian encoder updates on classification."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from cypha_bench.adapters.bench_models import BenchClassifier
from cypha_bench.common.metrics import evaluate_classification, online_train_classifier, standardize_train_test
from cypha_bench.config.load_profile import classification_params, load_profile
from cypha_bench.domains.d09_documents import _load_20news

_OUT = _REPO / "cypha_bench" / "config" / "cypha_encoder_phase2a_sweep.json"

MODES = ("contrastive", "hebbian")


def _run_task(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
    passes: int,
) -> dict[str, Any]:
    os.environ["CYPHA_ENCODER_UPDATE"] = mode
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, x_test = standardize_train_test(x_train, x_test)
    dim = x_train.shape[1]
    t0 = time.perf_counter()
    clf = BenchClassifier(dim, seed=42)
    if mode in MODES:
        clf.dif.encoder_update_mode = mode
    online_train_classifier(clf, x_train, y_train, label_fn=str, passes=passes)
    scores = evaluate_classification(clf, x_test, y_test, label_fn=str)
    return {
        "task": name,
        "encoder_update_mode": mode,
        "accuracy": float(scores["accuracy"]),
        "f1_macro": float(scores.get("f1_macro", scores["accuracy"])),
        "train_seconds": time.perf_counter() - t0,
        "n_train": int(len(x_train)),
        "n_features": int(dim),
    }


def _classification_tasks() -> list[tuple[str, np.ndarray, np.ndarray]]:
    tasks: list[tuple[str, np.ndarray, np.ndarray]] = []
    x, y = make_classification(
        n_samples=2000, n_features=10, n_informative=5, n_redundant=2, random_state=42
    )
    tasks.append(("linearly_separable_2class", x, y))
    x, y = make_blobs(n_samples=2000, n_features=8, centers=4, cluster_std=1.5, random_state=42)
    tasks.append(("4_gaussian_blobs", x, y))
    x, y = make_classification(
        n_samples=2000, n_features=100, n_informative=10, n_redundant=40, random_state=42
    )
    tasks.append(("high_dim_noisy", x, y))
    news_enc, news_svd, x_news, y_news, _ = _load_20news(1200)
    tasks.append(("20newsgroups_svd100", x_news, y_news))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description="Cypha Tests 2A encoder update sweep")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    passes = max(1, int(classification_params(load_profile()).get("n_epochs", 1)))
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for mode in MODES:
        for name, x, y in _classification_tasks():
            print(f"  task={name} mode={mode}", flush=True)
            row = _run_task(name, x, y, mode=mode, passes=passes)
            rows.append(row)
            print(f"    acc={row['accuracy']:.4f}", flush=True)

    summary: dict[str, dict[str, float]] = {}
    for name, _, _ in _classification_tasks():
        c = next(r for r in rows if r["task"] == name and r["encoder_update_mode"] == "contrastive")
        h = next(r for r in rows if r["task"] == name and r["encoder_update_mode"] == "hebbian")
        summary[name] = {
            "hebbian_minus_contrastive_acc": float(h["accuracy"] - c["accuracy"]),
            "contrastive_acc": c["accuracy"],
            "hebbian_acc": h["accuracy"],
        }

    out = {
        "encoder_update_modes": list(MODES),
        "passes": passes,
        "rows": rows,
        "summary": summary,
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
