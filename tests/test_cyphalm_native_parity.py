"""
``cyphalm_parity`` / component parity vs ``parity_fixtures/cyphalm_*/sidecar.json``.

CTest: ``native_cyphalm_char_lstm`` (when fixture exists), ``native_cyphalm_parity_suite``.
Override: ``CYPHALM_PARITY_BIN``, ``CYPHALM_CHAR_LSTM_PARITY_BIN``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable  # noqa: E402

_CHAR_SIDE = _ROOT / "parity_fixtures" / "cyphalm_char_lstm" / "sidecar.json"


def test_cyphalm_parity_suite():
    """Meta-runner invokes bundled native CyphaLM parity tools."""
    r = run_native_executable(
        "cyphalm_parity",
        [],
        timeout=120,
        env_override="CYPHALM_PARITY_BIN",
    )
    if r is None:
        pytest.skip("cyphalm_parity binary not built")
    assert r.returncode == 0, "cyphalm_parity failed:\n" + r.stdout + r.stderr
    assert "All CyphaLM native parity checks PASSED." in r.stdout


def test_cyphalm_char_lstm_fixture_parity():
    """Char-LSTM sidecar vs ``cyphalm_char_lstm_parity`` when fixture is present."""
    if not _CHAR_SIDE.is_file():
        pytest.skip("run scripts/generate_cyphalm_native_fixtures.py")
    data = json.loads(_CHAR_SIDE.read_text(encoding="utf-8"))
    assert "expected" in data
    r = run_native_executable(
        "cyphalm_char_lstm_parity",
        [str(_CHAR_SIDE)],
        timeout=60,
        env_override="CYPHALM_CHAR_LSTM_PARITY_BIN",
        cwd=str(_ROOT),
    )
    if r is None:
        pytest.skip("cyphalm_char_lstm_parity binary not built")
    assert r.returncode == 0, r.stdout + r.stderr


def test_cyphalm_bench_native_smoke():
    """Bench CLI emits JSON with a finite BPC on synthetic fallback."""
    r = run_native_executable(
        "cyphalm_bench_native",
        ["--mode", "char_lstm", "--profile", "d17", "--n-train", "500", "--n-eval", "100", "--threads", "0"],
        timeout=120,
        env_override="CYPHALM_BENCH_NATIVE_BIN",
    )
    if r is None:
        pytest.skip("cyphalm_bench_native binary not built")
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads(r.stdout)
    assert "bpc" in out
    assert out.get("mode") == "char_lstm"


_CKPT_SIDE = _ROOT / "parity_fixtures" / "cyphalm_checkpoint" / "char_lstm" / "sidecar.json"


def test_cyphalm_checkpoint_parity():
    """Checkpoint roundtrip + Python char_lstm load lock."""
    if not _CKPT_SIDE.is_file():
        pytest.skip("run scripts/generate_cyphalm_checkpoint_fixture.py")
    r = run_native_executable(
        "cyphalm_checkpoint_parity",
        [str(_CKPT_SIDE)],
        timeout=120,
        env_override="CYPHALM_CHECKPOINT_PARITY_BIN",
    )
    if r is None:
        pytest.skip("cyphalm_checkpoint_parity binary not built")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


def test_cyphalm_model_parity():
    r = run_native_executable(
        "cyphalm_model_parity",
        [],
        timeout=60,
        env_override="CYPHALM_MODEL_PARITY_BIN",
    )
    if r is None:
        pytest.skip("cyphalm_model_parity binary not built")
    assert r.returncode == 0, r.stdout + r.stderr


def test_cyphalm_hebbian_parity():
    r = run_native_executable(
        "cyphalm_hebbian_parity",
        [],
        timeout=60,
        env_override="CYPHALM_HEBBIAN_PARITY_BIN",
    )
    if r is None:
        pytest.skip("cyphalm_hebbian_parity binary not built")
    assert r.returncode == 0, r.stdout + r.stderr


def test_cyphalm_ssm_parity():
    r = run_native_executable(
        "cyphalm_ssm_parity",
        [],
        timeout=60,
        env_override="CYPHALM_SSM_PARITY_BIN",
    )
    if r is None:
        pytest.skip("cyphalm_ssm_parity binary not built")
    assert r.returncode == 0, r.stdout + r.stderr


_SSM_SIDE = _ROOT / "parity_fixtures" / "cyphalm_ssm" / "sidecar.json"


def test_cyphalm_ssm_fixture_parity():
    if not _SSM_SIDE.is_file():
        pytest.skip("run scripts/generate_cyphalm_native_fixtures.py (ssm sidecar)")
    r = run_native_executable(
        "cyphalm_parity",
        [str(_SSM_SIDE)],
        timeout=60,
        env_override="CYPHALM_PARITY_BIN",
        cwd=str(_ROOT),
    )
    if r is None:
        pytest.skip("cyphalm_parity binary not built")
    assert r.returncode == 0, r.stdout + r.stderr
