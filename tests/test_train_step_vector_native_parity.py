"""
``train_step_vector_parity`` vs ``parity_fixtures/train_step_vector/sidecar.json`` (one ``dif_train_step_vector``).

CTest: ``native_train_step_vector``. Override: ``CYPHA_TRAIN_STEP_VECTOR_PARITY_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures" / "train_step_vector"
_SIDE = _FIX / "sidecar.json"


def test_train_step_vector_parity_subprocess():
    if not _SIDE.is_file():
        pytest.skip("train_step_vector fixture missing")
    r = run_native_executable(
        "train_step_vector_parity",
        [_FIX],
        timeout=120,
        env_override="CYPHA_TRAIN_STEP_VECTOR_PARITY_BIN",
    )
    if r is None:
        pytest.skip(
            "train_step_vector_parity not built (cmake native/; set CYPHA_TRAIN_STEP_VECTOR_PARITY_BIN; "
            "Windows: WSL ELF under native/build-wsl or native/build)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
