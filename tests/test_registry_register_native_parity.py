"""
``registry_register``: install ``reference.cypha`` + ``registry_register/card.json`` into a temp registry root; ``--and-verify``.

Matches CTest ``native_registry_register``. Override: ``CYPHA_REGISTRY_REGISTER_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_REF = _ROOT / "parity_fixtures" / "reference.cypha"
_CARD = _ROOT / "parity_fixtures" / "registry_register" / "card.json"


def test_registry_register_subprocess(tmp_path: Path):
    if not _REF.is_file() or not _CARD.is_file():
        pytest.skip("parity_fixtures/reference.cypha or registry_register/card.json missing")
    reg_root = tmp_path / "registry_native_smoke_root"
    reg_root.mkdir(parents=True, exist_ok=True)
    r = run_native_executable(
        "registry_register",
        [
            reg_root,
            "native_reg_smoke",
            "0.0.1",
            _REF,
            _CARD,
            "--overwrite",
            "--and-verify",
        ],
        timeout=60,
        env_override="CYPHA_REGISTRY_REGISTER_BIN",
    )
    if r is None:
        pytest.skip(
            "registry_register not built (cmake native/; set CYPHA_REGISTRY_REGISTER_BIN; "
            "Windows: WSL ELF under native/build-wsl or native/build)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
