"""Python ExperimentDB can read a DB initialized by native ``experiment_db_smoke`` (M6 interchange)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.experiment_schema_ddl import experiment_schema_sql
from tests.native_subprocess import run_native_executable


def test_python_reads_native_seeded_file(tmp_path):
    ddl = tmp_path / "exp_ddl.sql"
    try:
        ddl.write_text(experiment_schema_sql(), encoding="utf-8")
    except FileNotFoundError:
        pytest.skip("cypha_studio/core/experiment.py missing")
    db_path = tmp_path / "native_seeded.sqlite"
    r = run_native_executable(
        "experiment_db_smoke",
        [ddl, db_path],
        env_override="CYPHA_EXPERIMENT_DB_SMOKE_BIN",
        timeout=90,
    )
    if r is None:
        pytest.skip(
            "experiment_db_smoke not built or not runnable (cmake native/; Python3 at configure; "
            "SQLite dev package or default amalgamation fetch; on Windows without .exe, use WSL "
            "build under native/build or native/build-exp; or set CYPHA_EXPERIMENT_DB_SMOKE_BIN)"
        )
    if r.returncode != 0:
        pytest.fail(f"experiment_db_smoke failed ({r.returncode}): {r.stderr or r.stdout}")

    from cypha_studio.core.experiment import ExperimentDB

    db = ExperimentDB(str(db_path))
    exp = db.get_experiment("exp_smoke")
    assert exp is not None
    assert exp.name == "smoke-suite"
    run = db.get_run("run_smoke01")
    assert run is not None
    assert run.status == "done"
    assert run.accuracy == pytest.approx(0.91)
    assert run.macro_f1 == pytest.approx(0.88)
    assert run.r2_score == pytest.approx(0.77)
    assert run.rmse == pytest.approx(0.12)
    assert run.n_steps == 100
    assert run.n_classes == 3
    assert run.duration_s == pytest.approx(42.5)
    assert run.checkpoint_path == "ckpt.cypha"
    assert run.preprocessor_path == "prep.json"
    assert run.metrics_history == [{"epoch": 0, "loss": 0.5}]
