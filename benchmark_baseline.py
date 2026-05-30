#!/usr/bin/env python3
"""
Cypha SOM upgrade benchmark (cypha_som_upgrades.md §3).

Usage:
  python benchmark_baseline.py --dataset classification --seeds 3
  python benchmark_baseline.py --dataset all --upgrade U2 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"


def _cypha_imports():
    from Cypha import CyphaDIF, RFFEncoder, VectorEncoder
    return CyphaDIF, RFFEncoder, VectorEncoder


def _make_clf(X_train: np.ndarray, seed: int, use_rff: bool = True):
    CyphaDIF, RFFEncoder, VectorEncoder = _cypha_imports()
    from cypha_som import config as som_cfg
    from cypha_som.som_encoder import SOMWrappedEncoder

    d = X_train.shape[1]
    rng = np.random.default_rng(seed)
    if use_rff:
        D = min(128, d * 8)
        enc = RFFEncoder(d, D=D, gamma=1.0, seed=seed)
        enc.auto_gamma(X_train[: min(500, len(X_train))])
        if som_cfg.USE_SOM_ENCODER:
            enc = SOMWrappedEncoder(enc, k=8, T=5000)
        field = D
    else:
        enc = VectorEncoder(d)
        field = max(d, 64)
    return CyphaDIF(encoder=enc, field_dim=field, rng=rng)


def _train_eval(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    max_steps: int | None = None,
) -> dict:
    labels = [str(c) for c in sorted(np.unique(y))]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = _make_clf(X_tr, seed)
    n = len(X_tr) if max_steps is None else min(max_steps, len(X_tr))
    idx = np.random.default_rng(seed).permutation(len(X_tr))[:n]

    t0 = time.perf_counter()
    acc_curve: dict[str, float] = {}
    for i, j in enumerate(idx):
        clf.train_step(X_tr[j], str(y_tr[j]))
        step = i + 1
        if step in (100, 1000, 10000) and step <= n:
            preds = [clf.infer(x)[0] for x in X_te[:200]]
            yp = [labels.index(p) if p in labels else 0 for p in preds]
            acc_curve[f"acc_at_{step}"] = float(accuracy_score(y_te[:200], yp))

    train_ms = (time.perf_counter() - t0) / max(n, 1) * 1000.0

    t0 = time.perf_counter()
    preds = [clf.infer(x)[0] for x in X_te]
    infer_ms = (time.perf_counter() - t0) / max(len(X_te), 1) * 1000.0
    yp = [labels.index(p) if p in labels else 0 for p in preds]
    acc = float(accuracy_score(y_te, yp))

    gng_nodes = 0
    if getattr(clf, "_som_hooks", None) and clf._som_hooks.gng is not None:
        gng_nodes = clf._som_hooks.gng.node_count()

    return {
        "accuracy": acc,
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        "gng_nodes": gng_nodes,
        "acc_curve": acc_curve,
    }


def run_classification(seeds: int) -> dict:
    X, y = make_classification(
        n_samples=10000,
        n_features=50,
        n_informative=20,
        n_classes=10,
        random_state=42,
    )
    runs = []
    for s in range(seeds):
        runs.append(_train_eval(X, y, seed=42 + s))
    accs = [r["accuracy"] for r in runs]
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "train_ms_mean": float(np.mean([r["train_ms"] for r in runs])),
        "infer_ms_mean": float(np.mean([r["infer_ms"] for r in runs])),
        "gng_nodes_mean": float(np.mean([r["gng_nodes"] for r in runs])),
        "runs": runs,
    }


def run_drift(seeds: int) -> dict:
    rng = np.random.default_rng(42)
    X1, y1 = make_classification(
        n_samples=5000, n_features=30, n_informative=12, n_classes=5, random_state=1
    )
    X2, y2 = make_classification(
        n_samples=5000, n_features=30, n_informative=12, n_classes=5, random_state=2
    )
    X2 = X2 + 3.0
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    out = []
    for s in range(seeds):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = _make_clf(Xs[:500], seed=s)
        acc_pre, acc_post = [], []
        for i in range(len(Xs)):
            pred, _ = clf.infer(Xs[i])
            if i == 4999:
                labels = [str(c) for c in sorted(np.unique(y))]
                acc_pre.append(1.0 if pred == str(y[i]) else 0.0)
            if i >= 5000 and i < 5500:
                labels = [str(c) for c in sorted(np.unique(y))]
                acc_post.append(1.0 if pred == str(y[i]) else 0.0)
            clf.train_step(Xs[i], str(y[i]))
        out.append({
            "acc_end_phase1": float(np.mean(acc_pre)) if acc_pre else 0.0,
            "acc_early_phase2": float(np.mean(acc_post)) if acc_post else 0.0,
        })
    return {"runs": out}


def run_adversarial() -> dict:
    rng = np.random.default_rng(42)
    n, d = 2000, 41
    labels = np.array([str(i % 4) for i in range(n)])
    X = rng.standard_normal((n, d))
    for c in range(4):
        X[labels == str(c)] += rng.standard_normal(d) * 2.0 + c * 1.5
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    WARMUP = 400

    def _dos_recall(use_gh: bool) -> float:
        CyphaDIF, _, VectorEncoder = _cypha_imports()
        clf = CyphaDIF(encoder=VectorEncoder(d), field_dim=128, rng=rng)
        for i in range(WARMUP):
            clf.train_step(Xs[i], labels[i])
        for _ in range(15):
            x_adv = rng.normal(0, 20, d) - 30
            if use_gh:
                clf.gh_train_step(x_adv, "1")
            else:
                clf.train_step(x_adv, "1")
        dos_idx = [i for i in range(WARMUP, n) if labels[i] == "1"][:80]
        preds = [clf.infer(Xs[i])[0] for i in dos_idx]
        return float(sum(p == "1" for p in preds) / max(len(preds), 1))

    return {
        "dos_recall_standard": _dos_recall(False),
        "dos_recall_gh": _dos_recall(True),
    }


def compare_to_baseline(metrics: dict, baseline: dict) -> dict:
    """§3.4 pass/fail thresholds."""
    b = baseline.get("classification", metrics.get("classification", {}))
    m = metrics.get("classification", {})
    checks = {}
    acc_b = b.get("accuracy_mean", m.get("accuracy_mean", 0))
    acc_m = m.get("accuracy_mean", 0)
    checks["accuracy"] = acc_m >= acc_b - 0.005
    train_b = b.get("train_ms_mean", 1.0)
    train_m = m.get("train_ms_mean", 1.0)
    checks["train_time"] = train_m <= train_b * 1.25 + 1e-9
    std_b = max(b.get("accuracy_std", 0.01), 1e-6)
    std_m = m.get("accuracy_std", 0.0)
    checks["variance"] = std_m <= std_b * 1.5 + 1e-9
    adv_b = baseline.get("adversarial", {})
    adv_m = metrics.get("adversarial", {})
    if adv_b and adv_m:
        checks["dos_recall"] = adv_m.get("dos_recall_gh", 0) >= adv_b.get("dos_recall_gh", 0) - 0.02
    checks["all_pass"] = all(checks.values())
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="classification",
                        choices=["classification", "drift", "adversarial", "all"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", default=str(RESULTS_DIR / "baseline.json"))
    parser.add_argument("--upgrade", default="none",
                        help="none | U1..U6 | all")
    parser.add_argument("--compare", default=None, help="baseline JSON to compare")
    args = parser.parse_args()

    from cypha_som.config import set_upgrade_flags

    set_upgrade_flags(args.upgrade)
    print(f"Flags: upgrade={args.upgrade}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"upgrade": args.upgrade}

    if args.dataset in ("classification", "all"):
        print("Running classification benchmark...", flush=True)
        metrics["classification"] = run_classification(args.seeds)
        print(f"  accuracy={metrics['classification']['accuracy_mean']:.4f} "
              f"+/- {metrics['classification']['accuracy_std']:.4f}")

    if args.dataset in ("drift", "all"):
        print("Running drift benchmark...", flush=True)
        metrics["drift"] = run_drift(min(args.seeds, 2))

    if args.dataset in ("adversarial", "all"):
        print("Running adversarial benchmark...", flush=True)
        metrics["adversarial"] = run_adversarial()
        print(f"  dos_recall_gh={metrics['adversarial']['dos_recall_gh']:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_path}")

    compare_path = args.compare or str(RESULTS_DIR / "baseline.json")
    if args.upgrade.lower() not in ("none", "") and Path(compare_path).exists():
        with open(compare_path, encoding="utf-8") as f:
            baseline = json.load(f)
        if args.upgrade.lower() != "none" and baseline.get("upgrade", "none") == "none":
            checks = compare_to_baseline(metrics, baseline)
            print("Pass/fail vs baseline:", checks)
            metrics["checks"] = checks
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
