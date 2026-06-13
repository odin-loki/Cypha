#!/usr/bin/env python3
"""Benchmark XOR (S3) with linear vs kernel LLR on CyphaDIF.

Usage:
  python scripts/benchmark_xor_kernel_llr.py
  python scripts/benchmark_xor_kernel_llr.py --seeds 5 --kernel-blend 0.7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Cypha import CyphaDIF, VectorEncoder  # noqa: E402


def make_xor(n: int = 4000, d: int = 20, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)
    return x, y


def scale_split(x: np.ndarray, y: np.ndarray, seed: int, test_frac: float = 0.25):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    te, tr = idx[:n_test], idx[n_test:]
    mu = x[tr].mean(axis=0)
    std = x[tr].std(axis=0) + 1e-8
    def norm(ix):
        return (x[ix] - mu) / std
    return norm(tr), norm(te), y[tr], y[te]


def train_online(clf: CyphaDIF, x: np.ndarray, y: np.ndarray, passes: int = 4) -> float:
    t0 = time.perf_counter()
    for _ in range(passes):
        for xi, yi in zip(x, y):
            clf.train_step(xi, str(yi))
    return time.perf_counter() - t0


def eval_acc(clf: CyphaDIF, x: np.ndarray, y: np.ndarray) -> float:
    correct = 0
    for xi, yi in zip(x, y):
        pred, _ = clf.infer(xi)
        correct += int(pred == str(yi))
    return correct / max(len(y), 1)


def run_seed(seed: int, use_kernel: bool, kernel_blend: float, passes: int) -> dict:
    x, y = make_xor(seed=42)
    x_tr, x_te, y_tr, y_te = scale_split(x, y, seed=seed)
    enc = VectorEncoder(dim=x.shape[1])
    clf = CyphaDIF(
        encoder=enc,
        use_kernel_llr=use_kernel,
        rng=np.random.default_rng(seed),
    )
    clf._kernel_blend = kernel_blend
    clf.deliberation_lo = 1.0
    clf.deliberation_hi = 0.0
    wall = train_online(clf, x_tr, y_tr, passes=passes)
    acc = eval_acc(clf, x_te, y_te)
    km = getattr(clf, "_kernel_mem", None)
    n_basis = int(km._n_basis) if km is not None else 0
    return {
        "seed": seed,
        "use_kernel_llr": use_kernel,
        "kernel_blend": kernel_blend,
        "accuracy": acc,
        "train_sec": wall,
        "n_basis": n_basis,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="XOR linear vs kernel LLR benchmark")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--kernel-blend", type=float, default=0.5)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    results = {"linear": [], "kernel": []}
    for seed in range(args.seeds):
        results["linear"].append(run_seed(seed, False, args.kernel_blend, args.passes))
        results["kernel"].append(run_seed(seed, True, args.kernel_blend, args.passes))

    def mean_acc(rows):
        return float(np.mean([r["accuracy"] for r in rows]))

    summary = {
        "dataset": "S3_nonlinear_xor",
        "n_samples": 4000,
        "n_dims": 20,
        "passes": args.passes,
        "kernel_blend": args.kernel_blend,
        "linear_mean_acc": mean_acc(results["linear"]),
        "kernel_mean_acc": mean_acc(results["kernel"]),
        "delta_pp": 100.0 * (mean_acc(results["kernel"]) - mean_acc(results["linear"])),
        "per_seed": results,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
