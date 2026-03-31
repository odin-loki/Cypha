"""ExperimentDB enforces SQLite foreign keys (matches native ``experiment_db_smoke``)."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_create_run_rejects_unknown_experiment_id(tmp_path):
    from cypha_studio.core.experiment import ExperimentDB

    db = ExperimentDB(str(tmp_path / "exp.sqlite"))
    with pytest.raises(sqlite3.IntegrityError):
        db.create_run("nonexistent", "orphan-run", config=None)
