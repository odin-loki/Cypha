"""
``retrieval_parity`` vs ``parity_fixtures/retrieval/sidecar.json``.

Covers Python ``CyphaDIF.retrieve`` (log-likelihood ranked retrieval).

CTest: ``native_retrieval``.
Override: ``CYPHA_RETRIEVAL_PARITY_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable  # noqa: E402

_SIDE = _ROOT / "parity_fixtures" / "retrieval" / "sidecar.json"


def test_retrieval_fixture_exists():
    assert _SIDE.is_file(), (
        f"Missing {_SIDE} — run scripts/generate_retrieval_fixture.py"
    )


@pytest.mark.skipif(not _SIDE.is_file(), reason="retrieval fixture missing")
def test_retrieval_native_parity():
    r = run_native_executable("retrieval_parity", [str(_SIDE)], env_override="CYPHA_RETRIEVAL_PARITY_BIN")
    if r is None:
        pytest.skip("retrieval_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "retrieval_parity OK" in r.stdout
