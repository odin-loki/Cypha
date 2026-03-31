"""
``opencl_smoke`` — verify OpenCL backend initialises and computes correctly.

Exit codes from the binary:
  0  OpenCL available + all correctness checks passed  → PASS
  2  No fp64 device found / not compiled in            → SKIP
  1  Device found but computation incorrect            → FAIL

CTest: ``native_opencl_smoke``, ``native_opencl_bench``.
Override: ``CYPHA_OPENCL_SMOKE_BIN``.

Build with ``-DCYPHA_ENABLE_OPENCL=ON`` to exercise the GPU path.  Without that
flag the binary silently falls back to CPU stubs and exits 2 (skip).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable  # noqa: E402


def test_opencl_smoke():
    """OpenCL backend must either pass or skip (no fp64 device is acceptable)."""
    r = run_native_executable(
        "opencl_smoke",
        [],
        timeout=30,
        env_override="CYPHA_OPENCL_SMOKE_BIN",
    )
    if r is None:
        pytest.skip("opencl_smoke binary not built")
    if r.returncode == 2:
        pytest.skip("No fp64-capable OpenCL device found or OpenCL not compiled in")
    assert r.returncode == 0, (
        "opencl_smoke correctness FAILED:\n" + r.stdout + r.stderr
    )
    assert "All OpenCL correctness checks PASSED." in r.stdout


def test_opencl_bench():
    """Benchmark run must succeed (or skip) without crashing."""
    r = run_native_executable(
        "opencl_smoke",
        ["--bench"],
        timeout=60,
        env_override="CYPHA_OPENCL_SMOKE_BIN",
    )
    if r is None:
        pytest.skip("opencl_smoke binary not built")
    if r.returncode == 2:
        pytest.skip("No fp64-capable OpenCL device found")
    assert r.returncode == 0, r.stdout + r.stderr
