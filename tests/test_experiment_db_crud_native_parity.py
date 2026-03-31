"""
``experiment_db_crud_parity`` vs canonical ExperimentDB DDL (in-memory SQLite harness).

CTest: ``native_experiment_db_crud``. Override: ``CYPHA_EXPERIMENT_DB_CRUD_PARITY_BIN``.

DDL from ``tests.experiment_schema_ddl`` (AST — no ``cypha_studio`` / ``numpy`` import).
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


def test_experiment_db_crud_parity_subprocess(tmp_path: Path):
    ddl_path = tmp_path / "experiment_ddl.sql"
    try:
        ddl_path.write_text(experiment_schema_sql(), encoding="utf-8")
    except FileNotFoundError:
        pytest.skip("cypha_studio/core/experiment.py missing")
    r = run_native_executable(
        "experiment_db_crud_parity",
        [ddl_path],
        timeout=120,
        env_override="CYPHA_EXPERIMENT_DB_CRUD_PARITY_BIN",
    )
    if r is None:
        pytest.skip(
            "experiment_db_crud_parity not built (cmake native/ with SQLite; "
            "set CYPHA_EXPERIMENT_DB_CRUD_PARITY_BIN; Windows: WSL ELF under native/build-wsl or native/build)"
        )
    assert r.returncode == 0, (r.stdout, r.stderr)
