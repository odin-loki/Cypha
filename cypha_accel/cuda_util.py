"""Shared CUDA availability probe (CuPy + at least one device)."""
from __future__ import annotations

from typing import Optional

_cuda_ok: Optional[bool] = None


def cuda_gemm_usable() -> bool:
    """True after first success: CuPy imports and getDeviceCount() > 0."""
    global _cuda_ok
    if _cuda_ok is not None:
        return _cuda_ok
    try:
        import cupy as cp  # type: ignore
    except ImportError:
        _cuda_ok = False
        return False
    try:
        _cuda_ok = int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        _cuda_ok = False
    return _cuda_ok


def warmup_cuda() -> None:
    """Pay one-time driver/JIT cost before latency-sensitive work (no-op if no CUDA)."""
    if not cuda_gemm_usable():
        return
    import cupy as cp  # type: ignore

    a = cp.random.standard_normal((96, 96), dtype=cp.float64)
    b = cp.random.standard_normal((96, 96), dtype=cp.float64)
    c = a @ b
    cp.cuda.Stream.null.synchronize()
    del a, b, c
