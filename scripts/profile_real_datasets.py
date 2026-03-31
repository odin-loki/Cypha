#!/usr/bin/env python3
"""
CPU profile CyphaDIF on real sklearn tabular data (cProfile cumulative time).

Writes ``artifacts/profiles/profile_real_cumtime.txt`` by default for port prioritisation.

  python scripts/profile_real_datasets.py          # iris + wine + breast_cancer
  python scripts/profile_real_datasets.py --fast   # iris only, smaller loops
  python scripts/profile_real_datasets.py --digits # include digits (slower)

Requires: numpy, scikit-learn (same as benchmark).
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from Cypha import CyphaDIF, VectorEncoder  # noqa: E402
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler


def _train_and_infer(name: str, X: np.ndarray, y: np.ndarray, n_passes: int, rng: np.random.Generator):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.astype(np.float64))
    d = Xs.shape[1]
    n_classes = len(np.unique(y))
    clf = CyphaDIF(
        encoder=VectorEncoder(d),
        field_dim=max(d, 64),
        rng=np.random.default_rng(42),
    )
    idx = rng.permutation(len(Xs))
    for i in idx:
        clf.train_step(Xs[i], str(int(y[i])))
    # stress batch + serial paths
    xs = [Xs[j] for j in range(min(n_passes, len(Xs)))]
    for _ in range(3):
        clf.batch_infer(xs)
        for x in xs[: min(20, len(xs))]:
            clf.infer(x)
    # score_matrix hot
    H = clf.batch_encode(xs)
    clf.score_matrix(H, use_field=True)


def workload(fast: bool, digits: bool):
    rng = np.random.default_rng(0)
    n_pass = 40 if fast else 150

    iris = load_iris()
    _train_and_infer("iris", iris.data, iris.target, n_pass, rng)

    if not fast:
        wine = load_wine()
        _train_and_infer("wine", wine.data, wine.target, n_pass, rng)
        bc = load_breast_cancer()
        _train_and_infer("breast_cancer", bc.data, bc.target, n_pass, rng)

    if digits and not fast:
        dig = load_digits()
        _train_and_infer("digits", dig.data, dig.target, min(n_pass, 200), rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--digits", action="store_true")
    ap.add_argument(
        "-o",
        "--output",
        default=str(_ROOT / "artifacts" / "profiles" / "profile_real_cumtime.txt"),
        help="Where to write pstats text",
    )
    args = ap.parse_args()

    pr = cProfile.Profile()
    pr.enable()
    workload(args.fast, args.digits)
    pr.disable()

    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumtime").print_stats(45)
    text = buf.getvalue()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# Cypha real-data CPU profile (cumtime)\n"
        f"# fast={args.fast} digits={args.digits}\n\n" + text,
        encoding="utf-8",
    )
    print(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
