"""
``cuda_smoke`` — verify ``cypha::accel`` (CUDA GPU or parallel CPU) vs reference.

Exit codes:
  0  correctness passed
  2  ``--bench`` but no CUDA GPU (skip)
  1  failure

CTest: ``native_cuda_smoke``, ``native_cuda_bench``.
Override: ``CYPHA_CUDA_SMOKE_BIN``.

Build with ``-DCYPHA_ENABLE_CUDA=ON`` for the GPU path; without it the binary
uses ISO C++ ``std::thread`` parallel CPU and still passes correctness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable  # noqa: E402


def test_cuda_smoke():
    r = run_native_executable(
        "cuda_smoke",
        [],
        timeout=30,
        env_override="CYPHA_CUDA_SMOKE_BIN",
    )
    if r is None:
        pytest.skip("cuda_smoke binary not built")
    assert r.returncode == 0, "cuda_smoke correctness FAILED:\n" + r.stdout + r.stderr
    assert "All accel correctness checks PASSED." in r.stdout


def test_cuda_bench():
    r = run_native_executable(
        "cuda_smoke",
        ["--bench"],
        timeout=120,
        env_override="CYPHA_CUDA_SMOKE_BIN",
    )
    if r is None:
        pytest.skip("cuda_smoke binary not built")
    if r.returncode == 2:
        pytest.skip("CUDA bench skipped (no GPU or CUDA not enabled at build time)")
    assert r.returncode == 0, r.stdout + r.stderr
