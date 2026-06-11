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
