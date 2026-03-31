"""
Batched cross term:  R @ D.T  with R (N, d), D (K, d) → (N, K).

Prefer `score_batch.fused_score_llr` for Cypha scoring (fuses R build + GEMM on GPU).
This module remains for direct R @ D.T tests and small call sites.
"""
from __future__ import annotations

import numpy as np

from cypha_accel.cuda_util import cuda_gemm_usable

Array = np.ndarray


def cross_r_dT(R: Array, D: Array) -> Array:
    """
    Compute R @ D.T in float64.

    Parameters
    ----------
    R : (N, d)
    D : (K, d)
    """
    R = np.ascontiguousarray(R, dtype=np.float64)
    D = np.ascontiguousarray(D, dtype=np.float64)
    if R.ndim != 2 or D.ndim != 2 or R.shape[1] != D.shape[1]:
        raise ValueError(f"bad shapes R{R.shape} D{D.shape}")

    if cuda_gemm_usable():
        import cupy as cp  # type: ignore

        Rg = cp.asarray(R)
        Dg = cp.asarray(D)
        out = cp.asnumpy(Rg @ Dg.T)
        return np.ascontiguousarray(out, dtype=np.float64)

    return R @ D.T
