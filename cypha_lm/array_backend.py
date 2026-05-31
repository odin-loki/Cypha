"""NumPy / CuPy array backend for CyphaLM (optional GPU)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

_CUDA_OK: bool | None = None


def cuda_available() -> bool:
    """True when CuPy can run a tiny GEMM on at least one CUDA device."""
    global _CUDA_OK
    if _CUDA_OK is not None:
        return _CUDA_OK
    try:
        from cypha_accel.cuda_util import cuda_gemm_usable

        _CUDA_OK = cuda_gemm_usable()
    except ImportError:
        try:
            import cupy as cp  # type: ignore

            if int(cp.cuda.runtime.getDeviceCount()) <= 0:
                _CUDA_OK = False
            else:
                a = cp.ones((4, 4), dtype=cp.float64)
                b = cp.ones((4, 4), dtype=cp.float64)
                c = a @ b
                cp.cuda.Stream.null.synchronize()
                del a, b, c
                _CUDA_OK = True
        except Exception:
            _CUDA_OK = False
    return bool(_CUDA_OK)


def resolve_device(device: str | None = None) -> str:
    """Resolve ``auto`` / ``cpu`` / ``cuda`` (also accepts ``gpu``)."""
    if device is None:
        device = os.environ.get("CYPHA_LM_DEVICE", "auto")
    key = str(device).lower().strip()
    if key == "auto":
        return "cuda" if cuda_available() else "cpu"
    if key in ("cuda", "gpu"):
        if not cuda_available():
            raise RuntimeError(
                "CyphaLM device='cuda' but CuPy/GPU is unavailable. "
                "Install cupy-cuda12x (or cupy-cuda11x) and an NVIDIA driver."
            )
        return "cuda"
    if key == "cpu":
        return "cpu"
    raise ValueError(f"Unknown CyphaLM device {device!r} (use auto, cpu, cuda)")


def resolve_cyphalm_device(device: str | None = None) -> str:
    """
    CyphaLM device policy: ``auto`` → CPU.

    Sequential char-LM (one token per step, small matrices, CPU CyphaDIF) is
    faster on CPU than CUDA in practice. Use ``device='cuda'`` explicitly for
    experiments; batch APIs may benefit later.
    """
    if device is None:
        device = os.environ.get("CYPHA_LM_DEVICE", "auto")
    key = str(device).lower().strip()
    if key == "auto":
        return "cpu"
    return resolve_device(key)


def get_xp(device: str) -> Any:
    if device == "cuda":
        import cupy as cp  # type: ignore

        return cp
    return np


def asnumpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "get"):
        return np.asarray(x.get(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def to_xp(x: Any, xp: Any) -> Any:
    if xp is np:
        return np.asarray(x, dtype=np.float64)
    if hasattr(x, "device"):
        return x
    return xp.asarray(x, dtype=np.float64)


class ArrayBackend:
    """Select NumPy or CuPy and expose small transfer helpers."""

    def __init__(self, device: str = "auto") -> None:
        self.device = resolve_cyphalm_device(device)
        self.xp = get_xp(self.device)
        self.is_cuda = self.device == "cuda"

    def sync(self) -> None:
        if self.is_cuda:
            self.xp.cuda.Stream.null.synchronize()

    def asnumpy(self, x: Any) -> np.ndarray:
        return asnumpy(x)

    def to(self, x: Any) -> Any:
        return to_xp(x, self.xp)
