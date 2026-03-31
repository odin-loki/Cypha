"""
Fused batch ops for CyphaDIF: LLR core, feature projection.
Uses CuPy when `cuda_gemm_usable()`; otherwise NumPy (same numerics).

Env:
  CYPHA_ACCEL_FP32=1  — use float32 CuPy tensors in device fused paths (faster; tiny numeric drift).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from cypha_accel.cuda_util import cuda_gemm_usable
from cypha_accel.nig_gh import nig_r_eff_vec

Array = np.ndarray


def _use_fp32_device() -> bool:
    return os.environ.get("CYPHA_ACCEL_FP32", "").lower() in ("1", "true", "yes")


def fused_features_to_latent_and_llr(
    F: Array,
    W_enc: Array,
    mu0: Array,
    inv_v: Array,
    D: Array,
    D_sq: Array,
    u_k: Optional[Array] = None,
    ctx_arr: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """
    Single GPU upload of raw features F: H = F @ W_enc.T, then fused LLR(H, …).
    Returns (H, LLR) as float64 numpy (one device sync; avoids round-trip for H before score).

    CPU fallback matches project_features + fused_score_llr numerically.
    """
    F = np.ascontiguousarray(F, dtype=np.float64)
    W_enc = np.ascontiguousarray(W_enc, dtype=np.float64)
    mu0 = np.ascontiguousarray(mu0, dtype=np.float64).ravel()
    inv_v = np.ascontiguousarray(inv_v, dtype=np.float64).ravel()
    D = np.ascontiguousarray(D, dtype=np.float64)
    D_sq = np.ascontiguousarray(D_sq, dtype=np.float64).ravel()
    K = D.shape[0]
    if u_k is None:
        u_k = np.zeros(K, dtype=np.float64)
    else:
        u_k = np.ascontiguousarray(u_k, dtype=np.float64).ravel()
    if ctx_arr is None:
        ctx_arr = np.zeros(K, dtype=np.float64)
    else:
        ctx_arr = np.ascontiguousarray(ctx_arr, dtype=np.float64).ravel()

    if cuda_gemm_usable():
        import cupy as cp  # type: ignore

        dt = cp.float32 if _use_fp32_device() else cp.float64
        Fg = cp.asarray(F, dtype=dt)
        Wg = cp.asarray(W_enc, dtype=dt)
        Hg = Fg @ Wg.T
        mu0g = cp.asarray(mu0, dtype=dt)
        inv_vg = cp.asarray(inv_v, dtype=dt)
        Dg = cp.asarray(D, dtype=dt)
        Rg = (Hg - mu0g) * inv_vg
        cross = Rg @ Dg.T
        bias = -0.5 * cp.asarray(D_sq, dtype=dt) - cp.asarray(u_k, dtype=dt) + cp.asarray(
            ctx_arr, dtype=dt
        )
        LLRg = cross + bias
        H = np.ascontiguousarray(cp.asnumpy(Hg), dtype=np.float64)
        LLR = np.ascontiguousarray(cp.asnumpy(LLRg), dtype=np.float64)
        return H, LLR

    H = project_features(F, W_enc)
    LLR = fused_score_llr(H, mu0, inv_v, D, D_sq, u_k, ctx_arr)
    return H, LLR


def fused_features_to_device_latent_llr(
    F: Array,
    W_enc: Array,
    mu0: Array,
    inv_v: Array,
    D: Array,
    D_sq: Array,
    u_k: Optional[Array] = None,
    ctx_arr: Optional[Array] = None,
):
    """
    GPU-only: return (Hg, LLRg) as CuPy arrays (no host copy of H/LLR).
    Caller must sync/finalize. Returns None if CUDA unavailable.
    """
    if not cuda_gemm_usable():
        return None
    F = np.ascontiguousarray(F, dtype=np.float64)
    W_enc = np.ascontiguousarray(W_enc, dtype=np.float64)
    mu0 = np.ascontiguousarray(mu0, dtype=np.float64).ravel()
    inv_v = np.ascontiguousarray(inv_v, dtype=np.float64).ravel()
    D = np.ascontiguousarray(D, dtype=np.float64)
    D_sq = np.ascontiguousarray(D_sq, dtype=np.float64).ravel()
    K = D.shape[0]
    if u_k is None:
        u_k = np.zeros(K, dtype=np.float64)
    else:
        u_k = np.ascontiguousarray(u_k, dtype=np.float64).ravel()
    if ctx_arr is None:
        ctx_arr = np.zeros(K, dtype=np.float64)
    else:
        ctx_arr = np.ascontiguousarray(ctx_arr, dtype=np.float64).ravel()

    import cupy as cp  # type: ignore

    dt = cp.float32 if _use_fp32_device() else cp.float64
    Fg = cp.asarray(F, dtype=dt)
    Wg = cp.asarray(W_enc, dtype=dt)
    Hg = Fg @ Wg.T
    mu0g = cp.asarray(mu0, dtype=dt)
    inv_vg = cp.asarray(inv_v, dtype=dt)
    Dg = cp.asarray(D, dtype=dt)
    Rg = (Hg - mu0g) * inv_vg
    cross = Rg @ Dg.T
    bias = -0.5 * cp.asarray(D_sq, dtype=dt) - cp.asarray(u_k, dtype=dt) + cp.asarray(
        ctx_arr, dtype=dt
    )
    LLRg = cross + bias
    return Hg, LLRg


def fused_batch_infer_indices_confs_cupy(
    Hg,
    LLRg,
    temperature: float,
    mu0: Array,
    inv_v: Array,
    d_latent: int,
    R_base: float,
    gh_chi: float,
    gh_psi: float,
    eps: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    On device: softmax(LLR/T), GH gate from latent (small mahal D2H for Bessel), argmax conf.
    Returns (best_idx [N], confs [N], gates [N]) int64/float64 numpy (``gates`` matches
    ``world_gate_vector`` for the same latents on device), or None if K<=8 / no GH / no CUDA.
    """
    if not cuda_gemm_usable():
        return None
    if gh_chi <= 0.0 or gh_psi <= 0.0:
        return None
    import cupy as cp  # type: ignore

    K = int(LLRg.shape[1])
    if K <= 8:
        return None

    dt = LLRg.dtype
    T = cp.asarray(temperature + eps, dtype=dt)
    scaled = LLRg / T
    Z = scaled - scaled.max(axis=1, keepdims=True)
    E = cp.exp(Z)
    P = E / (E.sum(axis=1, keepdims=True) + eps)

    mu0g = cp.asarray(mu0, dtype=dt)
    inv_vg = cp.asarray(inv_v, dtype=dt)
    diffs = Hg - mu0g
    r = diffs * inv_vg
    mahal_per_dim = (diffs * r).sum(axis=1) / max(d_latent, 1)
    mp = cp.maximum(mahal_per_dim, 0.0)
    mp_np = cp.asnumpy(mp).astype(np.float64, copy=False)

    R_eff_np = nig_r_eff_vec(mp_np, float(R_base), float(gh_chi), float(gh_psi))
    gates_np = R_base / np.maximum(R_eff_np, R_base)
    gates_g = cp.asarray(gates_np, dtype=dt)

    best_idx = cp.argmax(P, axis=1).astype(cp.int32)
    n = int(P.shape[0])
    row = cp.arange(n, dtype=cp.int32)
    confs = P[row, best_idx] * gates_g
    cp.cuda.Stream.null.synchronize()
    return (
        cp.asnumpy(best_idx).astype(np.int64, copy=False),
        cp.asnumpy(confs).astype(np.float64, copy=False),
        gates_np.astype(np.float64, copy=False),
    )


def fused_score_llr(
    H: Array,
    mu0: Array,
    inv_v: Array,
    D: Array,
    D_sq: Array,
    u_k: Optional[Array] = None,
    ctx_arr: Optional[Array] = None,
) -> Array:
    """
    LLR[i,k] = (R @ D.T)[i,k] - 0.5*D_sq[k] - u_k[k] + ctx[k],  R = (H-μ₀)⊙inv_v.

    Shapes: H (N,d), mu0 (d,), inv_v (d,), D (K,d), D_sq (K,).
    If u_k / ctx_arr are None, they default to zeros (e.g. generation path).
    """
    H = np.ascontiguousarray(H, dtype=np.float64)
    mu0 = np.ascontiguousarray(mu0, dtype=np.float64).ravel()
    inv_v = np.ascontiguousarray(inv_v, dtype=np.float64).ravel()
    D = np.ascontiguousarray(D, dtype=np.float64)
    D_sq = np.ascontiguousarray(D_sq, dtype=np.float64).ravel()
    K = D.shape[0]
    if u_k is None:
        u_k = np.zeros(K, dtype=np.float64)
    else:
        u_k = np.ascontiguousarray(u_k, dtype=np.float64).ravel()
    if ctx_arr is None:
        ctx_arr = np.zeros(K, dtype=np.float64)
    else:
        ctx_arr = np.ascontiguousarray(ctx_arr, dtype=np.float64).ravel()

    if cuda_gemm_usable():
        import cupy as cp  # type: ignore

        Hg = cp.asarray(H)
        mu0g = cp.asarray(mu0)
        inv_vg = cp.asarray(inv_v)
        Dg = cp.asarray(D)
        Rg = (Hg - mu0g) * inv_vg
        cross = Rg @ Dg.T
        bias = -0.5 * cp.asarray(D_sq) - cp.asarray(u_k) + cp.asarray(ctx_arr)
        LLRg = cross + bias
        return np.ascontiguousarray(cp.asnumpy(LLRg), dtype=np.float64)

    R = (H - mu0) * inv_v
    cross = R @ D.T
    return cross - 0.5 * D_sq - u_k + ctx_arr


def project_features(F: Array, W: Array) -> Array:
    """Latent batch: F (N, d) @ W.T  with W (d, d)."""
    F = np.ascontiguousarray(F, dtype=np.float64)
    W = np.ascontiguousarray(W, dtype=np.float64)
    if cuda_gemm_usable():
        import cupy as cp  # type: ignore

        out = cp.asarray(F) @ cp.asarray(W).T
        return np.ascontiguousarray(cp.asnumpy(out), dtype=np.float64)
    return F @ W.T


def softmax_rows_llr(Z: Array, eps: float) -> Array:
    """
    Row softmax matching Cypha `_softmax_batch` for K > 8 (vectorised path).
    For K ≤ 8, callers should use Cypha._softmax_batch for exact serial parity.
    """
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("softmax_rows_llr expects 2d array")
    n, k = Z.shape
    if k <= 8:
        raise ValueError("use Cypha._softmax_batch for K<=8 parity")

    if cuda_gemm_usable():
        import cupy as cp  # type: ignore

        Zg = cp.asarray(Z)
        Z2 = Zg - Zg.max(axis=1, keepdims=True)
        E = cp.exp(Z2)
        P = E / (E.sum(axis=1, keepdims=True) + eps)
        return np.ascontiguousarray(cp.asnumpy(P), dtype=np.float64)

    X2 = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(X2)
    return E / (E.sum(axis=1, keepdims=True) + eps)
