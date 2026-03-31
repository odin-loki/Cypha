"""
Heavy GPU workloads for benchmarking / tuning (CuPy). No effect without CUDA.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from cypha_accel.cuda_util import cuda_gemm_usable


def cupy_gemm_burn(
    n_passes: int,
    n: int = 6144,
    d: int = 256,
    k: int = 128,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Sustained fp64 GEMM on device (no Cypha). For thermal/driver soak + timing.
    """
    if not cuda_gemm_usable() or n_passes <= 0:
        return {"cupy_burn_skipped": True, "n_passes": n_passes}

    import cupy as cp  # type: ignore

    rng = np.random.default_rng(seed)
    A = cp.asarray(rng.standard_normal((n, d)), dtype=np.float64)
    B = cp.asarray(rng.standard_normal((d, k)), dtype=np.float64)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    C = A @ B
    for _ in range(n_passes - 1):
        C = A @ B
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    flops = 2.0 * n * d * k * n_passes
    return {
        "cupy_burn_s": round(elapsed, 6),
        "cupy_burn_passes": n_passes,
        "cupy_burn_shape": [n, d, k],
        "cupy_burn_gflops_s": round((flops / elapsed) / 1e9, 3) if elapsed > 0 else 0.0,
    }
