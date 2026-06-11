"""
Native bench/tune/diagnostics CLI smoke (mirrors CTest ``native_bench_run_list_domains``,
``native_tune_run_smoke``, ``native_diagnostics_run``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_TUNE_CFG = _ROOT / "cypha_bench" / "config" / "cyphalm_hybrid_lstm_tune_smoke.json"
_PARITY = _ROOT / "parity_fixtures"


def test_cypha_bench_run_list_domains():
    r = run_native_executable(
        "cypha_bench_run",
        ["--list-domains"],
        timeout=30,
        env_override="CYPHA_BENCH_RUN_BIN",
    )
    if r is None:
        pytest.skip("cypha_bench_run not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "domain" in (r.stdout or "").lower() or "d0" in (r.stdout or "").lower()


def test_cypha_tune_run_smoke():
    if not _TUNE_CFG.is_file():
        pytest.skip("tune smoke config missing")
    r = run_native_executable(
        "cypha_tune_run",
        ["--config", str(_TUNE_CFG), "--max-cells", "2"],
        timeout=120,
        env_override="CYPHA_TUNE_RUN_BIN",
    )
    if r is None:
        pytest.skip("cypha_tune_run not built")
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_cypha_diagnostics_run_smoke(tmp_path):
    if not _PARITY.is_dir():
        pytest.skip("parity_fixtures missing")
    out = tmp_path / "diagnostics_results"
    r = run_native_executable(
        "cypha_diagnostics_run",
        ["--fixtures", str(_PARITY), "--out", str(out)],
        timeout=60,
        env_override="CYPHA_DIAGNOSTICS_RUN_BIN",
    )
    if r is None:
        pytest.skip("cypha_diagnostics_run not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
