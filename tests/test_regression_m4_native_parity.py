"""
Native M4: ``regression_m4_parity`` vs ``parity_fixtures/regression_m4/sidecar.json``.

Includes mixture batch + EMA, RFF RLS ``train_step``, MKE per-expert RLS (with forgetting),
two-stage predict combine, and ``MKERegressor``-style routing softmax + scalar mixture predict.
Requires native regression milestone **≥ 5**. CTest name: ``native_regression_m4``. Skips if the binary is missing
(same discovery as ``test_cypha_rest_smoke``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_SIDE = _ROOT / "parity_fixtures" / "regression_m4" / "sidecar.json"


def test_regression_m4_parity_subprocess():
    if not _SIDE.is_file():
        pytest.skip("parity_fixtures/regression_m4/sidecar.json missing — run scripts/generate_regression_m4_fixture.py")
    r = run_native_executable(
        "regression_m4_parity",
        [_SIDE],
        timeout=30,
        env_override="CYPHA_REGRESSION_M4_PARITY_BIN",
    )
    if r is None:
        pytest.skip("regression_m4_parity not built (cmake native/build or native/build-exp; WSL ELF ok on Windows)")
    assert r.returncode == 0, (r.stdout, r.stderr)
