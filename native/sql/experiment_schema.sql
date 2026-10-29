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

-- S7 (DISCOVERY 7.2 step 1; invariants I5, I6): the pre-registration of hypothesis tests, mirrored in C++ by
-- PreregLog (native/include/cypha/eval/hypothesis.hpp). Rows are only ever inserted: a plan cannot be changed or
-- removed, and a result cannot be changed or removed. A plan may be registered again for the same hypothesis only
-- before any of its tests has a result; the newest registration is the plan.
CREATE TABLE IF NOT EXISTS hypothesis_preregistrations (
    seq            INTEGER PRIMARY KEY,   -- log order
    hypothesis_id  TEXT NOT NULL,
    plan           TEXT NOT NULL,         -- per test: kind, source, predicted, refuted_if, disconfirmation, independent, s, c
    has_disconfirmation INTEGER NOT NULL CHECK (has_disconfirmation = 1),  -- I6
    prev_hash      TEXT NOT NULL,
    hash           TEXT NOT NULL,
    created_at     REAL
);

CREATE TABLE IF NOT EXISTS hypothesis_test_results (
    seq            INTEGER PRIMARY KEY,
    hypothesis_id  TEXT NOT NULL,
    test_index     INTEGER NOT NULL,
    outcome        TEXT NOT NULL CHECK (outcome IN ('positive', 'negative', 'inconclusive')),
    prev_hash      TEXT NOT NULL,
    hash           TEXT NOT NULL,
    created_at     REAL,
    UNIQUE (hypothesis_id, test_index)
);

CREATE INDEX IF NOT EXISTS idx_prereg_hypothesis ON hypothesis_preregistrations(hypothesis_id);

CREATE TRIGGER IF NOT EXISTS prereg_no_update BEFORE UPDATE ON hypothesis_preregistrations
BEGIN SELECT RAISE(ABORT, 'I5: a registered plan cannot be edited'); END;
CREATE TRIGGER IF NOT EXISTS prereg_no_delete BEFORE DELETE ON hypothesis_preregistrations
BEGIN SELECT RAISE(ABORT, 'I5: a registered plan cannot be removed'); END;
CREATE TRIGGER IF NOT EXISTS prereg_no_replan BEFORE INSERT ON hypothesis_preregistrations
WHEN EXISTS (SELECT 1 FROM hypothesis_test_results WHERE hypothesis_id = NEW.hypothesis_id)
BEGIN SELECT RAISE(ABORT, 'I5: a test of this plan already has a result'); END;
CREATE TRIGGER IF NOT EXISTS result_needs_plan BEFORE INSERT ON hypothesis_test_results
WHEN NOT EXISTS (SELECT 1 FROM hypothesis_preregistrations WHERE hypothesis_id = NEW.hypothesis_id)
BEGIN SELECT RAISE(ABORT, 'I5: the test was not pre-registered'); END;
CREATE TRIGGER IF NOT EXISTS result_no_update BEFORE UPDATE ON hypothesis_test_results
BEGIN SELECT RAISE(ABORT, 'I5: a result cannot be edited'); END;
CREATE TRIGGER IF NOT EXISTS result_no_delete BEFORE DELETE ON hypothesis_test_results
BEGIN SELECT RAISE(ABORT, 'I5: a result cannot be removed'); END;
