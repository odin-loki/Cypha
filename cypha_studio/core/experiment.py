"""
cypha_studio.core.experiment
────────────────────────────
Local experiment tracking backed by SQLite. No external dependencies.

Every training run is a Row in the 'runs' table with a JSON blob of
metrics time-series and config. The ExperimentWidget queries this DB.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .trainer import EvalMetrics, TrainerConfig


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Run:
    """One training run inside an experiment."""
    run_id      : str
    experiment  : str
    name        : str
    config      : Dict[str, Any]
    status      : str           = 'pending'   # pending | running | done | failed
    created_at  : float         = field(default_factory=time.time)
    updated_at  : float         = field(default_factory=time.time)
    finished_at : Optional[float] = None
    duration_s  : float         = 0.0

    # Scalar summary metrics (best / final)
    accuracy    : float = 0.0
    macro_f1    : float = 0.0
    r2_score    : float = 0.0
    rmse        : float = 0.0
    n_steps     : int   = 0
    n_classes   : int   = 0

    # Paths
    checkpoint_path   : Optional[str] = None
    preprocessor_path : Optional[str] = None

    # Full metrics history — stored as JSON in DB
    metrics_history : List[Dict] = field(default_factory=list)

    # Tags for filtering
    tags : List[str] = field(default_factory=list)
    notes: str = ''


@dataclass
class Experiment:
    """Groups a set of runs under a shared name and dataset."""
    experiment_id : str
    name          : str
    description   : str  = ''
    dataset_name  : str  = ''
    task          : str  = 'classification'
    created_at    : float = field(default_factory=time.time)
    tags          : List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentDB
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT DEFAULT '',
    dataset_name  TEXT DEFAULT '',
    task          TEXT DEFAULT 'classification',
    created_at    REAL,
    tags          TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    experiment_id     TEXT NOT NULL,
    name              TEXT NOT NULL,
    config            TEXT,        -- JSON
    status            TEXT DEFAULT 'pending',
    created_at        REAL,
    updated_at        REAL,
    finished_at       REAL,
    duration_s        REAL DEFAULT 0,
    accuracy          REAL DEFAULT 0,
    macro_f1          REAL DEFAULT 0,
    r2_score          REAL DEFAULT 0,
    rmse              REAL DEFAULT 0,
    n_steps           INTEGER DEFAULT 0,
    n_classes         INTEGER DEFAULT 0,
    checkpoint_path   TEXT,
    preprocessor_path TEXT,
    metrics_history   TEXT DEFAULT '[]',  -- JSON list of EvalMetrics dicts
    tags              TEXT DEFAULT '[]',
    notes             TEXT DEFAULT '',
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""


class ExperimentDB:
    """
    SQLite-backed store for experiments and runs.

    db = ExperimentDB('~/.cypha/experiments.db')
    exp = db.create_experiment('iris-test', dataset_name='iris')
    run = db.create_run(exp.experiment_id, 'run-01', config)
    db.update_run(run.run_id, status='running')
    db.log_metrics(run.run_id, eval_metrics)
    db.finish_run(run.run_id, final_metrics)
    """

    def __init__(self, path: str = '~/.cypha/experiments.db'):
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        # Align with native M6 smoke (``experiment_db_smoke``) and enforce runs→experiments FK.
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ── Experiments ──────────────────────────────────────────────────────────

    def create_experiment(self, name: str,
                          description: str = '',
                          dataset_name: str = '',
                          task: str = 'classification',
                          tags: Optional[List[str]] = None) -> Experiment:
        exp = Experiment(
            experiment_id=str(uuid.uuid4())[:8],
            name=name, description=description,
            dataset_name=dataset_name, task=task,
            tags=tags or [],
        )
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO experiments VALUES (?,?,?,?,?,?,?)",
                (exp.experiment_id, exp.name, exp.description,
                 exp.dataset_name, exp.task, exp.created_at,
                 json.dumps(exp.tags))
            )
        return exp

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id=?",
                (experiment_id,)
            ).fetchone()
        if row is None:
            return None
        return Experiment(
            experiment_id=row['experiment_id'], name=row['name'],
            description=row['description'], dataset_name=row['dataset_name'],
            task=row['task'], created_at=row['created_at'],
            tags=json.loads(row['tags']),
        )

    def list_experiments(self) -> List[Experiment]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        return [Experiment(
            experiment_id=r['experiment_id'], name=r['name'],
            description=r['description'], dataset_name=r['dataset_name'],
            task=r['task'], created_at=r['created_at'],
            tags=json.loads(r['tags']),
        ) for r in rows]

    def delete_experiment(self, experiment_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM runs WHERE experiment_id=?", (experiment_id,))
            conn.execute("DELETE FROM experiments WHERE experiment_id=?", (experiment_id,))

    # ── Runs ─────────────────────────────────────────────────────────────────

    def create_run(self, experiment_id: str, name: str,
                   config: Optional[TrainerConfig] = None,
                   tags: Optional[List[str]] = None,
                   notes: str = '') -> Run:
        cfg_dict = {}
        if config is not None:
            try:
                from dataclasses import asdict as _asdict
                cfg_dict = _asdict(config)
            except Exception:
                cfg_dict = {}

        run = Run(
            run_id=str(uuid.uuid4())[:12],
            experiment=experiment_id,
            name=name, config=cfg_dict,
            tags=tags or [], notes=notes,
        )
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO runs
                   (run_id, experiment_id, name, config, status, created_at,
                    updated_at, tags, notes)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (run.run_id, experiment_id, name, json.dumps(cfg_dict),
                 'pending', run.created_at, run.updated_at,
                 json.dumps(run.tags), notes)
            )
        return run

    def get_run(self, run_id: str) -> Optional[Run]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def list_runs(self, experiment_id: Optional[str] = None,
                  status: Optional[str] = None,
                  limit: int = 100,
                  offset: int = 0) -> List[Run]:
        query = "SELECT * FROM runs"
        params: List[Any] = []
        conditions = []
        if experiment_id:
            conditions.append("experiment_id=?"); params.append(experiment_id)
        if status:
            conditions.append("status=?"); params.append(status)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, max(0, int(offset))])

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_run(r) for r in rows]

    def _row_to_run(self, row) -> Run:
        return Run(
            run_id=row['run_id'],
            experiment=row['experiment_id'],
            name=row['name'],
            config=json.loads(row['config'] or '{}'),
            status=row['status'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            finished_at=row['finished_at'],
            duration_s=row['duration_s'] or 0.0,
            accuracy=row['accuracy'] or 0.0,
            macro_f1=row['macro_f1'] or 0.0,
            r2_score=row['r2_score'] or 0.0,
            rmse=row['rmse'] or 0.0,
            n_steps=row['n_steps'] or 0,
            n_classes=row['n_classes'] or 0,
            checkpoint_path=row['checkpoint_path'],
            preprocessor_path=row['preprocessor_path'],
            metrics_history=json.loads(row['metrics_history'] or '[]'),
            tags=json.loads(row['tags'] or '[]'),
            notes=row['notes'] or '',
        )

    def update_run(self, run_id: str, **kwargs):
        """Update run fields by keyword."""
        kwargs['updated_at'] = time.time()
        set_clauses = ', '.join(f"{k}=?" for k in kwargs)
        values = list(kwargs.values()) + [run_id]
        with self._conn() as conn:
            conn.execute(
                f"UPDATE runs SET {set_clauses} WHERE run_id=?", values
            )

    def log_metrics(self, run_id: str, metrics: EvalMetrics):
        """Append one EvalMetrics snapshot to the run's history."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT metrics_history FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                return
            history = json.loads(row['metrics_history'] or '[]')
            # Convert dataclass to dict
            try:
                from dataclasses import asdict as _asdict
                m_dict = _asdict(metrics)
            except Exception:
                m_dict = {'accuracy': metrics.accuracy, 'step': metrics.step}
            history.append(m_dict)
            conn.execute(
                "UPDATE runs SET metrics_history=?, updated_at=? WHERE run_id=?",
                (json.dumps(history), time.time(), run_id)
            )

    def finish_run(self, run_id: str, final_metrics: EvalMetrics,
                   checkpoint_path: Optional[str] = None,
                   preprocessor_path: Optional[str] = None):
        """Mark run as done and store final metrics."""
        now = time.time()
        run = self.get_run(run_id)
        duration = now - run.created_at if run else 0.0

        self.log_metrics(run_id, final_metrics)
        self.update_run(
            run_id,
            status='done',
            finished_at=now,
            duration_s=duration,
            accuracy=final_metrics.accuracy,
            macro_f1=final_metrics.macro_f1,
            r2_score=final_metrics.r2_score,
            rmse=final_metrics.rmse,
            n_steps=final_metrics.step,
            n_classes=final_metrics.n_classes,
            checkpoint_path=checkpoint_path,
            preprocessor_path=preprocessor_path,
        )

    def fail_run(self, run_id: str, error: str):
        self.update_run(run_id, status='failed', notes=error, finished_at=time.time())

    def delete_run(self, run_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))

    # ── Comparisons ──────────────────────────────────────────────────────────

    def compare_runs(self, run_ids: List[str]) -> List[Dict]:
        """Return a summary table of multiple runs for side-by-side comparison."""
        rows = []
        for rid in run_ids:
            run = self.get_run(rid)
            if run:
                rows.append({
                    'run_id'   : run.run_id,
                    'name'     : run.name,
                    'status'   : run.status,
                    'accuracy' : run.accuracy,
                    'macro_f1' : run.macro_f1,
                    'r2'       : run.r2_score,
                    'rmse'     : run.rmse,
                    'n_steps'  : run.n_steps,
                    'duration' : run.duration_s,
                    'config'   : run.config,
                })
        return rows

    def best_run(self, experiment_id: str,
                 metric: str = 'accuracy') -> Optional[Run]:
        """Return the run with the highest value of metric."""
        runs = self.list_runs(experiment_id=experiment_id, status='done')
        if not runs:
            return None
        return max(runs, key=lambda r: getattr(r, metric, 0.0))

    def leaderboard(self, experiment_id: str,
                    metric: str = 'accuracy',
                    top_n: int = 10) -> List[Run]:
        runs = self.list_runs(experiment_id=experiment_id, status='done')
        return sorted(runs, key=lambda r: getattr(r, metric, 0.0),
                      reverse=True)[:top_n]
