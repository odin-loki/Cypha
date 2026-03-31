"""
``csv_ingest_parity`` — native ``cypha::load_csv_dense`` vs Python ``CSVDataset.from_file`` goldens (name-based and index-based ``cases.json`` rows).

CTest: ``native_csv_ingest``. Override: ``CYPHA_CSV_INGEST_PARITY_BIN``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures" / "csv_ingest"


def test_csv_ingest_parity_subprocess():
    if not (_FIX / "cases.json").is_file():
        pytest.skip("run scripts/generate_csv_ingest_fixture.py")
    r = run_native_executable(
        "csv_ingest_parity",
        [_FIX],
        timeout=60,
        env_override="CYPHA_CSV_INGEST_PARITY_BIN",
    )
    if r is None:
        pytest.skip("csv_ingest_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
