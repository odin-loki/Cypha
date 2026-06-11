"""M6: keep ``experiment.py`` SQLite DDL aligned with ``docs/port/EXPERIMENTS_SCHEMA.md``."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DOC = _ROOT / "docs" / "port" / "EXPERIMENTS_SCHEMA.md"


def test_experiment_py_schema_tables_indexes_match_doc():
    from cypha_studio.core import experiment as exp_mod

    schema = getattr(exp_mod, "_SCHEMA", "")
    assert "CREATE TABLE IF NOT EXISTS experiments" in schema
    assert "CREATE TABLE IF NOT EXISTS runs" in schema
    tables = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", schema)
    assert tables == ["experiments", "runs"]
    indexes = re.findall(r"CREATE INDEX IF NOT EXISTS (\w+)", schema)
    assert indexes == ["idx_runs_experiment", "idx_runs_status"]

    assert _DOC.is_file(), "docs/port/EXPERIMENTS_SCHEMA.md missing"
    doc = _DOC.read_text(encoding="utf-8")
    assert "### `experiments`" in doc
    assert "### `runs`" in doc
    for name in indexes:
        assert name in doc, f"index {name!r} not mentioned in EXPERIMENTS_SCHEMA.md"
    for col in (
        "experiment_id",
        "run_id",
        "metrics_history",
        "checkpoint_path",
        "preprocessor_path",
    ):
        assert col in schema and col in doc, col


def test_experiment_schema_sqlite_pragma_matches_tables():
    """Apply ``_SCHEMA`` to SQLite and verify core columns (M6 / future native reader)."""
    import sqlite3
    import tempfile

    from cypha_studio.core.experiment import _SCHEMA

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = sqlite3.connect(path)
        conn.executescript(_SCHEMA)

        def cols(table: str) -> set[str]:
            cur = conn.execute(f"PRAGMA table_info({table})")
            return {str(r[1]) for r in cur.fetchall()}

        exp_c = cols("experiments")
        assert "experiment_id" in exp_c and "name" in exp_c and "tags" in exp_c
        run_c = cols("runs")
        for c in (
            "run_id",
            "experiment_id",
            "config",
            "metrics_history",
            "checkpoint_path",
            "preprocessor_path",
        ):
            assert c in run_c, c
        idx = {r[1] for r in conn.execute("PRAGMA index_list(runs)").fetchall()}
        assert "idx_runs_experiment" in idx
        assert "idx_runs_status" in idx
        conn.close()
    finally:
        Path(path).unlink(missing_ok=True)


def test_export_experiment_schema_sql_script_matches_schema():
    from cypha_studio.core import experiment as exp_mod

    script = _ROOT / "scripts" / "export_experiment_schema_sql.py"
    assert script.is_file()
    r = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == exp_mod._SCHEMA.strip()


def test_experiment_schema_ddl_ast_matches_module():
    """``tests/experiment_schema_ddl.py`` must track ``experiment._SCHEMA`` (subprocess tests avoid importing it)."""
    from cypha_studio.core import experiment as exp_mod

    from tests.experiment_schema_ddl import experiment_schema_sql

    assert experiment_schema_sql() == exp_mod._SCHEMA.strip() + "\n"


def test_native_experiment_schema_sql_matches_module():
    """Committed ``native/sql/experiment_schema.sql`` must track ``experiment._SCHEMA``."""
    from cypha_studio.core import experiment as exp_mod

    sql_path = _ROOT / "native" / "sql" / "experiment_schema.sql"
    assert sql_path.is_file(), "native/sql/experiment_schema.sql missing"
    assert sql_path.read_text(encoding="utf-8").strip() == exp_mod._SCHEMA.strip()
