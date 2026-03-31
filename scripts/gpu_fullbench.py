#!/usr/bin/env python3
"""
End-to-end inference benchmark: encode → score_matrix → softmax → gate,
plus a **batch_infer** timing line (encode + fused score + gate when CUDA is on).

Compares wall time with CUDA (CuPy) enabled vs forced CPU path, and checks
max |ΔLLR| between the two backends (should be ~1e-12..1e-9 in fp64).

  python scripts/gpu_fullbench.py
  python scripts/gpu_fullbench.py --n 4096 --d 128 --k 32 --repeat 7
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from Cypha import CyphaDIF, VectorEncoder, _probs_from_llr_matrix  # noqa: E402


def _train_clf(d: int, K: int, n_steps: int, seed: int) -> CyphaDIF:
    rng = np.random.default_rng(seed)
    clf = CyphaDIF(VectorEncoder(d), rng=rng)
    labs = [str(i) for i in range(K)]
    for _ in range(n_steps):
        k = int(rng.integers(0, K))
        clf.train_step(rng.standard_normal(d), labs[k])
    return clf


def _bench_one(clf: CyphaDIF, X: np.ndarray, repeat: int) -> tuple[float, float, float]:
    """Returns (mean encode ms, mean score+post ms, mean total ms)."""
    t_enc = t_rest = 0.0
    for _ in range(repeat):
        t0 = time.perf_counter()
        H = clf.batch_encode([X[i] for i in range(len(X))])
        t_enc += time.perf_counter() - t0

        t0 = time.perf_counter()
        LLR, labels = clf.score_matrix(H, use_field=True)
        if labels:
            _ = _probs_from_llr_matrix(LLR, clf.temperature)
            _ = clf.world_gate_vector(H, use_field=True)
        t_rest += time.perf_counter() - t0

    n = repeat
    return t_enc / n * 1000, t_rest / n * 1000, (t_enc + t_rest) / n * 1000


def _bench_batch_infer(clf: CyphaDIF, X: np.ndarray, repeat: int) -> float:
    """Mean wall ms for ``batch_infer`` over the same batch (repeat times)."""
    xs = [X[i] for i in range(len(X))]
    acc = 0.0
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = clf.batch_infer(xs)
        acc += time.perf_counter() - t0
    return acc / repeat * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048, help="batch size")
    ap.add_argument("--d", type=int, default=64, help="input / latent dim")
    ap.add_argument("--k", type=int, default=16, help="number of classes")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--train-steps", type=int, default=400)
    args = ap.parse_args()

    try:
        import cupy  # noqa: F401, WPS433
    except ImportError:
        print("CuPy not installed - install cupy-cuda* to benchmark GPU.\n")
        return 0

    rng = np.random.default_rng(7)
    clf = _train_clf(args.d, args.k, args.train_steps, seed=101)
    X = rng.standard_normal((args.n, args.d))

    import cypha_accel.cuda_util as cu

    H0 = clf.batch_encode([X[i] for i in range(len(X))])
    with mock.patch.object(cu, "cuda_gemm_usable", return_value=False):
        llr_cpu, labs = clf.score_matrix(H0, use_field=True)
    llr_gpu, labs2 = clf.score_matrix(H0, use_field=True)
    assert labs == labs2
    err = float(np.max(np.abs(llr_gpu - llr_cpu))) if llr_cpu.size else 0.0
    print(f"max|LLR_gpu - LLR_cpu| = {err:.3e} (fp64 GEMM)")

    with mock.patch.object(cu, "cuda_gemm_usable", return_value=False):
        ms_enc_cpu, ms_sc_cpu, ms_tot_cpu = _bench_one(clf, X, args.repeat)
        ms_bi_cpu = _bench_batch_infer(clf, X, args.repeat)

    ms_enc_gpu, ms_sc_gpu, ms_tot_gpu = _bench_one(clf, X, args.repeat)
    ms_bi_gpu = _bench_batch_infer(clf, X, args.repeat)

    print(f"Shape N={args.n} d={args.d} K={args.k}  (repeat={args.repeat})")
    print(f"CPU  encode {ms_enc_cpu:.2f} ms  score+softmax+gate {ms_sc_cpu:.2f} ms  total {ms_tot_cpu:.2f} ms")
    print(f"GPU  encode {ms_enc_gpu:.2f} ms  score+softmax+gate {ms_sc_gpu:.2f} ms  total {ms_tot_gpu:.2f} ms")
    print(f"CPU  batch_infer {ms_bi_cpu:.2f} ms   GPU batch_infer {ms_bi_gpu:.2f} ms")
    if ms_tot_gpu > 0:
        print(f"Speedup total: {ms_tot_cpu/ms_tot_gpu:.2f}x")
    return 0 if err < 1e-6 else 2


if __name__ == "__main__":
    raise SystemExit(main())
