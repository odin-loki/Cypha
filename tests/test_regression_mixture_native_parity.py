"""
``regression_mixture_parity``: ``predict_mixture_scalar`` vs fixed reference (d=1 mixture).

CTest: ``native_regression_mixture``. Override: ``CYPHA_REGRESSION_MIXTURE_PARITY_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable


def test_regression_mixture_parity_subprocess():
    r = run_native_executable(
        "regression_mixture_parity",
        [],
        timeout=30,
        env_override="CYPHA_REGRESSION_MIXTURE_PARITY_BIN",
    )
    if r is None:
        pytest.skip(
            "regression_mixture_parity not built (cmake native/; set CYPHA_REGRESSION_MIXTURE_PARITY_BIN; "
            "Windows: WSL ELF under native/build-wsl or native/build)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
