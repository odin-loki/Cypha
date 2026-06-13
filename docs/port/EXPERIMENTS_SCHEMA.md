# Experiments DB schema (M6 reference)

Native ports that reimplement **`ExperimentDB`** (native **`experiment_db_crud`**) should keep the same SQLite layout so `experiments.db` files remain interchangeable.

## Tables

The authoritative DDL is CMake-generated **`native/experiment_schema.sql`** (see **EXPERIMENTS_SCHEMA.md**). **`ExperimentDB`** opens each connection with **`PRAGMA foreign_keys=ON`** so **`runs.experiment_id`** is enforced (same assumption as native **`experiment_db_smoke`**). Summary:

### `experiments`

| Column | Type | Notes |
|--------|------|--------|
| `experiment_id` | TEXT PK | |
| `name` | TEXT NOT NULL | |
| `description` | TEXT | default `''` |
| `dataset_name` | TEXT | default `''` |
| `task` | TEXT | default `'classification'` |
| `created_at` | REAL | |
| `tags` | TEXT | JSON array, default `'[]'` |

### `runs`

| Column | Type | Notes |
|--------|------|--------|
| `run_id` | TEXT PK | |
| `experiment_id` | TEXT FK → experiments | |
| `name` | TEXT NOT NULL | |
| `config` | TEXT | JSON |
| `status` | TEXT | default `'pending'` |
| `created_at`, `updated_at`, `finished_at` | REAL | |
| `duration_s` | REAL | default `0` |
| `accuracy`, `macro_f1`, `r2_score`, `rmse` | REAL | metrics |
| `n_steps`, `n_classes` | INTEGER | |
| `checkpoint_path`, `preprocessor_path` | TEXT | |
| `metrics_history` | TEXT | JSON list, default `'[]'` |
| `tags` | TEXT | default `'[]'` |
| `notes` | TEXT | default `''` |

### Indexes

- `idx_runs_experiment` on `runs(experiment_id)`
- `idx_runs_status` on `runs(status)`

## Tooling

- **CMake configure** on `native/` regenerates DDL for **`experiment_db_smoke`**.
- Drift guard: **`ctest -R native_experiment`** vs this document and SQLite **`PRAGMA`**.

## Native status

Native provides **`cypha::ExperimentDb`** (RAII session) and low-level **`experiment_sqlite_exec`** in **`native/include/cypha/experiment_db.hpp`**, plus CRUD-style helpers in **`native/include/cypha/experiment_db_crud.hpp`**: insert experiment/run, **`experiment_db_finish_run`**, **`experiment_db_append_metrics_json`** (Python **`log_metrics`**-style array append), **`experiment_db_fail_run`**, **`experiment_db_delete_run`**, **`experiment_db_delete_experiment`**, plus **`experiment_db_scalar_int_query`** (used by **`experiment_db_crud_parity`**; CTest **`native_experiment_db_crud`**). Native also exposes row structs, **`get_experiment`** / **`get_run`**, **`list_experiments`** / **`list_runs`**, **`update_run_status`**, **`update_run_notes`**, **`best_done_run`**, **`leaderboard`**, and **`compare_runs`** (see **`experiment_db_crud.hpp`** / **`ExperimentDbRunCompareRow`**). Still not a full clone of Python’s dynamic **`update_run`** for arbitrary column sets. Registry bundle copy remains **`registry_register`** / **`registry_register_bundle`**. In **`native/`**, optional targets **`experiment_db_smoke`** (with **libsqlite3** + **Python 3** at CMake configure time) applies the canonical DDL, checks **`experiments`** / **`runs`** and **`idx_runs_*`**, then (with **`PRAGMA foreign_keys=ON`**) inserts sample rows matching Python shapes, verifies a join, rejects a bad FK, runs a prepared **`UPDATE`** covering scalar metrics + paths + **`metrics_history`** JSON, **`SELECT`**-verifies, and (when given a second arg **db_path**) closes and reopens the file read-only to confirm persistence. CTests: **`native_experiment_db_smoke`** (in-memory), **`native_experiment_db_file`** (on-disk round-trip).
