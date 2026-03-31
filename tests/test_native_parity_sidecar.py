"""Sanity-check native parity sidecar committed with parity_fixtures/."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BIN = _ROOT / "parity_fixtures" / "native_parity.bin"


def _require_bin():
    if not _BIN.is_file():
        pytest.skip("native_parity.bin missing — run: python scripts/generate_parity_fixtures.py")


def test_native_parity_bin_magic_and_sizes():
    _require_bin()
    raw = _BIN.read_bytes()
    assert raw[:8] == b"CYPHNP01"
    ver, n, d, k, field_dim = struct.unpack_from("<IIIII", raw, 8)
    assert ver in (1, 2)
    assert n >= 1 and d >= 1 and k >= 1 and field_dim >= 1
    off = 8 + 20 + 16
    need_v1 = off + d * field_dim * 8 + n * d * 8 + n * k * 8 * 2 + n * 8
    if ver == 1:
        assert len(raw) == need_v1
    else:
        assert len(raw) == need_v1 + 2 * n * 8
