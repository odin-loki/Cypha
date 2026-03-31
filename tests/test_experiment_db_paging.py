"""ExperimentDB list_runs limit/offset."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_list_runs_offset_paging(tmp_path):
    from cypha_studio.core.experiment import ExperimentDB
    from cypha_studio.core.trainer import TrainerConfig

    db_path = tmp_path / "exp.sqlite"
    db = ExperimentDB(str(db_path))
    exp = db.create_experiment("paging-test")
    cfg = TrainerConfig()
    for i in range(5):
        db.create_run(exp.experiment_id, f"run-{i}", cfg)

    all_rows = db.list_runs(limit=100, offset=0)
    assert len(all_rows) == 5
    p0 = db.list_runs(limit=2, offset=0)
    p1 = db.list_runs(limit=2, offset=2)
    p2 = db.list_runs(limit=2, offset=4)
    assert len(p0) == 2 and len(p1) == 2 and len(p2) == 1
    names = [r.name for r in all_rows]
    assert {r.name for r in p0} | {r.name for r in p1} | {r.name for r in p2} == set(names)
