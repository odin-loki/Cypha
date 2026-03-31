"""
``memory_train_roundtrip``: native train → ``save_cypha_file`` → reload vs ``after.cypha``.

CTest: ``native_memory_train_roundtrip``. Override: ``CYPHA_MEMORY_TRAIN_ROUNDTRIP_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Cypha import cypha_load_binary

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures" / "memory_train"
_AFTER = _FIX / "after.cypha"


def _state_equal(a, b, path: str = "root") -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict), f"type {path}: dict vs {type(b)}"
        assert set(a) == set(b), f"keys {path}: {set(a) ^ set(b)}"
        for k in sorted(a):
            _state_equal(a[k], b[k], f"{path}/{k}")
        return
    if isinstance(a, (bool, np.bool_)):
        assert bool(a) == bool(b), f"bool {path}: {a} vs {b}"
        return
    if isinstance(a, (int, np.integer)):
        assert isinstance(b, (int, np.integer)), f"type {path}: int vs {type(b)}"
        assert int(a) == int(b), f"int {path}: {a} vs {b}"
        return
    if isinstance(a, (float, np.floating)):
        assert isinstance(b, (float, np.floating)), f"type {path}: float vs {type(b)}"
        assert float(a) == pytest.approx(float(b), abs=1e-12), f"float {path}: {a} vs {b}"
        return
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), f"type {path}: ndarray vs {type(b)}"
        assert a.shape == b.shape, f"shape {path}: {a.shape} vs {b.shape}"
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)
        return
    assert type(a) is type(b), f"type {path}: {type(a)} vs {type(b)}"
    assert a == b, f"val {path}: {a!r} vs {b!r}"


def test_memory_train_roundtrip_subprocess_matches_after_cypha(tmp_path):
    if not (_FIX / "before.cypha").is_file() or not _AFTER.is_file():
        pytest.skip("memory_train fixtures missing")
    out_cypha = tmp_path / "rt.cypha"
    r = run_native_executable(
        "memory_train_roundtrip",
        [_FIX, out_cypha],
        timeout=90,
        env_override="CYPHA_MEMORY_TRAIN_ROUNDTRIP_BIN",
    )
    if r is None:
        pytest.skip(
            "memory_train_roundtrip not built (cmake native/; set CYPHA_MEMORY_TRAIN_ROUNDTRIP_BIN; "
            "Windows: native/build*/memory_train_roundtrip.exe or WSL ELF under native/build-wsl)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "memory_train_roundtrip OK" in (r.stdout or "")
    assert out_cypha.is_file()
    rt = cypha_load_binary(str(out_cypha))
    ref = cypha_load_binary(str(_AFTER))
    _state_equal(rt, ref)
