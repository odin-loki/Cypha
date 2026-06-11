"""
``embed_table_parity`` vs ``parity_fixtures/embed_table/sidecar.json``.

Covers Izaac GF(2^n) permutation-polynomial token embeddings (Python ``IzaacEmbedding``).

CTest: ``native_embed_table``.
Override: ``CYPHA_EMBED_TABLE_PARITY_BIN``.
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

_SIDE = _ROOT / "parity_fixtures" / "embed_table" / "sidecar.json"


def test_embed_table_fixture_exists():
    assert _SIDE.is_file(), (
        f"Missing {_SIDE} — run scripts/generate_embed_table_fixture.py"
    )


@pytest.mark.skipif(not _SIDE.is_file(), reason="embed_table fixture missing")
def test_embed_table_native_parity():
    r = run_native_executable("embed_table_parity", [str(_SIDE)], env_override="CYPHA_EMBED_TABLE_PARITY_BIN")
    if r is None:
        pytest.skip("embed_table_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "embed_table_parity OK" in r.stdout
