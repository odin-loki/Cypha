"""
``memory_train_parity`` vs ``parity_fixtures/memory_train/`` (one ``DIFMemory.train``).

CTest: ``native_memory_train``. Override: ``CYPHA_MEMORY_TRAIN_PARITY_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures" / "memory_train"


def test_memory_train_parity_subprocess():
    for name in ("sidecar.json", "before.cypha", "after.cypha"):
        if not (_FIX / name).is_file():
            pytest.skip("memory_train fixtures missing — run scripts/generate_memory_train_parity.py")
    r = run_native_executable(
        "memory_train_parity",
        [_FIX],
        timeout=120,
        env_override="CYPHA_MEMORY_TRAIN_PARITY_BIN",
    )
    if r is None:
        pytest.skip(
            "memory_train_parity not built (cmake native/; set CYPHA_MEMORY_TRAIN_PARITY_BIN; "
            "Windows: WSL ELF under native/build-wsl or native/build)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
