"""``cypha_save_binary_to_bytes`` / ``cypha_load_binary_from_bytes`` mirror native buffer I/O (v3)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Cypha import cypha_load_binary, cypha_load_binary_from_bytes, cypha_save_binary_to_bytes

_REF = _ROOT / "parity_fixtures" / "reference.cypha"


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
        assert float(a) == pytest.approx(float(b), rel=0, abs=0), f"float {path}: {a} vs {b}"
        return
    if isinstance(a, str):
        assert a == b, f"str {path}: {a!r} vs {b!r}"
        return
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), f"type {path}: ndarray vs {type(b)}"
        np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg=path)
        return
    assert a == b, f"{path}: {a!r} vs {b!r}"


def test_load_reference_from_bytes_matches_path():
    if not _REF.is_file():
        pytest.skip("parity_fixtures/reference.cypha missing")
    raw = _REF.read_bytes()
    d_path = cypha_load_binary(str(_REF))
    d_bytes = cypha_load_binary_from_bytes(raw)
    _state_equal(d_path, d_bytes)


def test_save_to_bytes_roundtrip_matches_reference_bytes():
    if not _REF.is_file():
        pytest.skip("parity_fixtures/reference.cypha missing")
    d = cypha_load_binary(str(_REF))
    blob = cypha_save_binary_to_bytes(d)
    assert blob == _REF.read_bytes()
    d2 = cypha_load_binary_from_bytes(blob)
    _state_equal(d, d2)
