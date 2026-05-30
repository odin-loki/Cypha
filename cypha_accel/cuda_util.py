"""Shared CUDA availability probe (CuPy + at least one device)."""
from __future__ import annotations

from typing import Optional

_cuda_ok: Optional[bool] = None


def cuda_gemm_usable() -> bool:
    """True after first success: CuPy imports, device present, and a tiny GEMM runs."""
    global _cuda_ok
    if _cuda_ok is not None:
        return _cuda_ok
    try:
        import cupy as cp  # type: ignore
    except ImportError:
        _cuda_ok = False
        return False
    try:
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            _cuda_ok = False
            return False
        a = cp.ones((4, 4), dtype=cp.float64)
        b = cp.ones((4, 4), dtype=cp.float64)
        c = a @ b
        cp.cuda.Stream.null.synchronize()
        del a, b, c
        _cuda_ok = True
    except Exception:
        _cuda_ok = False
    return _cuda_ok


def warmup_cuda() -> None:
    """Pay one-time driver/JIT cost before latency-sensitive work (no-op if no CUDA)."""
    if not cuda_gemm_usable():
        return
    try:
        import cupy as cp  # type: ignore

        a = cp.random.standard_normal((96, 96), dtype=cp.float64)
        b = cp.random.standard_normal((96, 96), dtype=cp.float64)
        c = a @ b
        cp.cuda.Stream.null.synchronize()
        del a, b, c
    except Exception:
        global _cuda_ok
        _cuda_ok = False
