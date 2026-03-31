"""
``create_model_smoke`` — verify that ``create_fresh_model_root`` produces a
valid .cypha root that both ``CyphaInferModel::from_root`` and
``CyphaDifMemoryState::from_cypha_root`` can load without error, and that a
save/reload roundtrip is byte-compatible.

CTest: ``native_create_model``.
Override: ``CYPHA_CREATE_MODEL_SMOKE_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable  # noqa: E402


def test_create_model_smoke():
    """Fresh model creation + load roundtrip must pass."""
    r = run_native_executable(
        "create_model_smoke",
        [],
        timeout=15,
        env_override="CYPHA_CREATE_MODEL_SMOKE_BIN",
    )
    if r is None:
        pytest.skip("create_model_smoke binary not built")
    assert r.returncode == 0, (
        "create_model_smoke FAILED:\n" + r.stdout + r.stderr
    )
    assert "PASS" in r.stdout
