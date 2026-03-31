#!/usr/bin/env python3
"""
Optional GPU micro-benchmark for Cypha-like linear algebra (no full DIF on GPU).

- Compares CPU NumPy vs CuPy for: (N,d) @ (d,K)  [same shape as score inner product]
- If CuPy is missing, prints install hints (wheel must match your CUDA version).

  pip install cupy-cuda12x   # or cupy-cuda11x — see https://docs.cupy.dev/en/stable/install.html

Run:
  python scripts/gpu_microbench.py
  python scripts/gpu_microbench.py --n 8192 --d 256 --k 128
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    try:
        import cupy as cp  # noqa: WPS433
    except ImportError:
        print(
            "CuPy not installed — GPU microbench skipped (exit 0).\n"
            "Install a CUDA-matched wheel, e.g.:\n"
            "  pip install cupy-cuda12x\n"
            "Docs: https://docs.cupy.dev/en/stable/install.html"
        )
        return 0

    rng = np.random.default_rng(0)
    X = rng.standard_normal((args.n, args.d)).astype(np.float64)
    W = rng.standard_normal((args.d, args.k)).astype(np.float64)

    # CPU baseline
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        R_cpu = X @ W
    t_cpu = (time.perf_counter() - t0) / args.repeat

    Xg = cp.asarray(X)
    Wg = cp.asarray(W)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        Rg = Xg @ Wg
    cp.cuda.Stream.null.synchronize()
    t_gpu = (time.perf_counter() - t0) / args.repeat

    # One sync outside the timed region would hide per-iter launch cost; keep sync
    # after the loop for correctness. Report an optional "kernel-only" estimate:
    t0k = time.perf_counter()
    _ = Xg @ Wg
    cp.cuda.Stream.null.synchronize()
    t_one = time.perf_counter() - t0k

    R_back = cp.asnumpy(Rg)
    err = float(np.max(np.abs(R_back - R_cpu)))

    print(f"Shape (N,d)@(d,K) = ({args.n},{args.d})@({args.d},{args.k})")
    print(f"CPU mean {t_cpu*1000:.3f} ms / iter")
    print(
        f"GPU mean {t_gpu*1000:.3f} ms / iter ({args.repeat} iters, sync after loop)"
    )
    print(f"GPU single matmul+sync {t_one*1000:.3f} ms (one-shot; less batch overhead)")
    if t_gpu > 0:
        print(f"Speedup vs CPU (loop mean): {t_cpu/t_gpu:.2f}x")
    if t_one > 0:
        print(f"Speedup vs CPU (single-shot): {t_cpu/t_one:.2f}x")
    print(f"max|GPU-CPU| = {err:.3e} (should be ~1e-12..1e-10 for fp64 GEMM)")
    return 0 if err < 1e-8 else 2


if __name__ == "__main__":
    raise SystemExit(main())
