"""
``experiment_db_smoke``: DDL + DML + optional on-disk reopen (matches CMake CTests).

- ``native_experiment_db_smoke``: one arg (in-memory).
- ``native_experiment_db_file``: DDL + sqlite path (round-trip file).

Override: ``CYPHA_EXPERIMENT_DB_SMOKE_BIN`` (same as ``tests/test_experiment_native_seed.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.experiment_schema_ddl import experiment_schema_sql
from tests.native_subprocess import run_native_executable


def test_experiment_db_smoke_memory_subprocess(tmp_path: Path):
    ddl_path = tmp_path / "experiment_ddl.sql"
    try:
        ddl_path.write_text(experiment_schema_sql(), encoding="utf-8")
    except FileNotFoundError:
        pytest.skip("cypha_studio/core/experiment.py missing")
    r = run_native_executable(
        "experiment_db_smoke",
        [ddl_path],
        timeout=90,
        env_override="CYPHA_EXPERIMENT_DB_SMOKE_BIN",
    )
    if r is None:
        pytest.skip(
            "experiment_db_smoke not built (cmake native/ with SQLite; set CYPHA_EXPERIMENT_DB_SMOKE_BIN; "
            "Windows: WSL ELF under native/build-wsl or native/build)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_experiment_db_smoke_file_roundtrip_subprocess(tmp_path: Path):
    ddl_path = tmp_path / "experiment_ddl.sql"
    db_path = tmp_path / "experiment_db_native_roundtrip.sqlite"
    try:
        ddl_path.write_text(experiment_schema_sql(), encoding="utf-8")
    except FileNotFoundError:
        pytest.skip("cypha_studio/core/experiment.py missing")
    r = run_native_executable(
        "experiment_db_smoke",
        [ddl_path, db_path],
        timeout=90,
        env_override="CYPHA_EXPERIMENT_DB_SMOKE_BIN",
    )
    if r is None:
        pytest.skip(
            "experiment_db_smoke not built (cmake native/ with SQLite; set CYPHA_EXPERIMENT_DB_SMOKE_BIN; "
            "Windows: WSL ELF under native/build-wsl or native/build)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
