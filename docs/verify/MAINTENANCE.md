# Maintenance — what to regen, rebuild, and align

*If you were looking for a **“do docs all”** checklist, this is it — plus fixtures, native, and schema.*

Use this when you change contracts or reference math. **Snapshot of automation:** [`VERIFICATION_STATUS.md`](VERIFICATION_STATUS.md). **Commands hub:** [`VERIFY_PLAN.md`](VERIFY_PLAN.md).

## Parity fixtures & native CTest

| You changed… | Do this |
|--------------|---------|
| **`CyphaDIF._save_state()` / `.cypha` v3 layout / inference / context (Tier-2+)** | `python scripts/generate_parity_fixtures.py` — updates `reference.cypha` (now includes **`field_a_eff`**), `expected.npz`, `native_parity.bin`, `manifest.json`, `train_hparams.json`, etc. |
| **Train-step vector sidecar** (`train_step_vector/`) | Same run as main parity: **`python scripts/generate_parity_fixtures.py`** (embeds `_write_train_step_vector_sidecar`). |
| **Memory train golden** | `python scripts/generate_memory_train_parity.py` → `parity_fixtures/memory_train/` ( **`after.cypha`** includes **`field_a_eff`** with **`CyphaDIF.save_state()`** ). |
| **Preprocessor fixture** | `python scripts/generate_preprocessor_parity.py`. |
| **Preprocessor fit (native PCA)** | `python scripts/generate_preprocessor_fit_fixture.py` → `parity_fixtures/preprocessor_fit/` + `preprocessor_fit_no_scale/` (CTest **`native_preprocessor_fit`**; scale on/off + PCA — RFF fit in Python). |
| **CSV ingest (dense load)** | `python scripts/generate_csv_ingest_fixture.py` → `parity_fixtures/csv_ingest/` (CTest **`native_csv_ingest`**; names/indices; multiline quoted fields; generator checks **`read_chunk_rows`**). |
| **DIFRegressor train-step slice** | `python scripts/generate_dif_regressor_train_step_fixture.py` → `parity_fixtures/dif_regressor_train_step/` (CTest **`native_dif_regressor_train_step`**; cold + warm **`infer`**-equivalent routing; **`replay_u01`** when **`replay_ratio>0`**). |
| **M4 regression_m4 sidecar** | `python scripts/generate_regression_m4_fixture.py` (or `py -3` on Windows) → `parity_fixtures/regression_m4/` (batch/EMA + RFF RLS + MKE RLS + two-stage + MKE routing). |
| **M5 two_stage_pipeline sidecar** | `python scripts/generate_two_stage_pipeline_fixture.py` (repo root on `PYTHONPATH`) → `parity_fixtures/two_stage_pipeline/`. |
| **M6 two_stage_ridge_fit sidecar** | `python scripts/generate_two_stage_ridge_fit_fixture.py` → `parity_fixtures/two_stage_ridge_fit/`. |
| **two_stage_e2e_ridge sidecar** | `python scripts/generate_two_stage_e2e_ridge_fixture.py` (PYTHONPATH=repo root) → `parity_fixtures/two_stage_e2e_ridge/`. |
| **batch_llr sidecar** | `python scripts/generate_batch_llr_fixture.py` (also runs at end of `generate_parity_fixtures.py`) → `parity_fixtures/batch_llr/`. |
| **quantile_dif_train** | `python scripts/generate_quantile_dif_train_fixture.py` (also runs at end of `generate_parity_fixtures.py`) → `parity_fixtures/quantile_dif_train/` (CTest **`native_quantile_dif_train`**; uses **`replay_ratio=0`** so replay never samples). |
| **dif_train_replay** | `python scripts/generate_dif_train_replay_fixture.py` → `parity_fixtures/dif_train_replay/` (CTest **`native_dif_train_replay`**; sidecar **`replay_u01`** for `replay_ratio>0`). |
| **studio_trainer_classify_hotpath** | `python scripts/generate_studio_trainer_classify_hotpath_fixture.py` → `parity_fixtures/studio_trainer_classify_hotpath/` (CTest **`native_studio_trainer_classify_hotpath`**; Studio **`Trainer.fit`** loop + **`enc_lr>0`** + **`replay_u01`**). |
| **studio_trainer_gh_classify_hotpath** | `python scripts/generate_studio_trainer_gh_classify_hotpath_fixture.py` → `parity_fixtures/studio_trainer_gh_classify_hotpath/` (CTest **`native_studio_trainer_gh_classify_hotpath`**; **`gh_train_step`** + **`chi`**/**`psi`**). |
| **studio_trainer_preprocess_classify_hotpath** | `python scripts/generate_studio_trainer_preprocess_classify_hotpath_fixture.py` → `parity_fixtures/studio_trainer_preprocess_classify_hotpath/` (CTest **`native_studio_trainer_preprocess_classify_hotpath`**; **`Preprocessor`** + train). |
| **studio_trainer_preprocess_gh_classify_hotpath** | `python scripts/generate_studio_trainer_preprocess_gh_classify_hotpath_fixture.py` → `parity_fixtures/studio_trainer_preprocess_gh_classify_hotpath/` (CTest **`native_studio_trainer_preprocess_gh_classify_hotpath`**; identity **`Preprocessor`** + **`x_raw`** steps; numeric goldens copied from **`studio_trainer_gh_classify_hotpath`**). Requires that GH fixture dir first. |
| **csv_preprocess_classify_hotpath** | `python scripts/generate_csv_preprocess_classify_hotpath_fixture.py` (after **`studio_trainer_preprocess_classify_hotpath`**) → `parity_fixtures/csv_preprocess_classify_hotpath/` (CTest **`native_csv_preprocess_classify_hotpath`**; same **`preprocess_train_classify_parity`** goldens via **`train.csv`** + **`load_csv_dense`**). |
| **mke_train_step** | `python scripts/generate_mke_train_step_fixture.py` (also runs at end of `generate_parity_fixtures.py`) → `parity_fixtures/mke_train_step/` (CTest **`native_mke_train_step`**). |
| **mke_train_extended** | `python scripts/generate_mke_train_extended_fixture.py` (also runs at end of `generate_parity_fixtures.py`) → `parity_fixtures/mke_train_extended/` (CTest **`native_mke_train_extended`**; **`replay_warmup`** + **`replay_u01`**, **`enc_lr>0`**, **`replay_ratio>0`**). |
| **RFF / ridge / MKE-dot sidecar** | `python scripts/generate_rff_regression_fixture.py` → `parity_fixtures/rff_regression/` (CTest **`native_regression_rff`**). |
| **MoE `regression_head.json`** (REST / native **`--regression-json`**) | `python scripts/export_regression_head.py` (see **`PORT_CONTRACT.md`** §3, **`CYPHA_REGRESSION_HEAD`** in **`CYPHA_ENV.md`**). |
| **Embedded `F_field` JSON** (`f_field.json` next to fixtures) | `python scripts/export_f_field_json.py` when world field tensor export changes. |
| **Native Bessel tables** (`K₂/K₁`, `K₀/K₁`) | `python scripts/gen_native_bessel_table.py` if SciPy **`kv`**-based grids change (`native/src/bessel_table_data.cpp`). |
| **SciPy-free Python Bessel ratios** (GH gates / `expected.npz` vs C++) | `python scripts/export_bessel_ratios_npz.py` (once; requires SciPy) → repo-root **`bessel_ratios.npz`** next to **`Cypha.py`**. Loaded when SciPy import fails so **`_BESSEL_TABLES_OK`** matches embedded C++ tables. |
| **`.cypha` v3 buffer I/O** (Python **`cypha_save_binary_to_bytes`** / **`cypha_load_binary_from_bytes`** ↔ native **`save_cypha_to_buffer`** / **`load_cypha_from_buffer`**) | **`pytest tests/test_cypha_binary_buffer_api.py`** — load **`parity_fixtures/reference.cypha`** from bytes vs path; **`save_binary_to_bytes`** byte-identical to on-disk file. Native **`memory_train_roundtrip`** also **`memcmp`**s file bytes vs **`save_cypha_to_buffer`** after train (CTest **`native_memory_train_roundtrip`**). |
| **Qt** (`cypha_qt_stub`, **`cypha_qt_shell`**) | **`sudo apt-get install qt6-base-dev`** (Debian/Ubuntu), then **`cmake … -DCYPHA_BUILD_QT=ON`** — CTests **`native_qt_stub_load_reference`**, **`native_qt_shell_smoke`**; pytest **`tests/test_qt_stub_native.py`**, **`tests/test_qt_shell_native.py`**. Local: **`CYPHA_BUILD_QT=1 bash scripts/ci_native_linux.sh`**. |
| **Any of the above** | Reconfigure/rebuild **`native/`** and run **`ctest --test-dir native/build --output-on-failure`**. When you add **`add_test(NAME native_…)`**, update **`tests/test_native_ctest_pytest_registry.py`** **`_NATIVE_CTEST_TO_PYTEST`**. Local one-liner **`bash scripts/ci_native_linux.sh`** runs CTest then **`pytest tests/test_native_ctest_pytest_registry.py`** when **`python3 -m pytest`** works ( **`SKIP_NATIVE_CTEST_REGISTRY_PYTEST=1`** to skip). |

**Tier-1:** committed **`reference.cypha`** includes **Tier-1** context; native loads it in **`CyphaInferModel::from_root`**. See [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md) §4 and **`tests/test_parity_fixtures.py::test_reference_fixture_restores_tier1_for_native_cypha_parity`**. **M6 / native pytest:** CMake can build **`experiment_db_smoke`** without **`libsqlite3-dev`** via default **`CYPHA_FETCH_SQLITE3_AMALGAMATION`** (network at configure). **`tests/native_subprocess.py`** runs ELF under **`native/build*`** via **`wsl -e`** from Windows when **`wsl.exe`** is available. Canonical experiment DDL for subprocess tests is parsed from **`cypha_studio/core/experiment.py`** in **`tests/experiment_schema_ddl.py`** (avoids importing **`cypha_studio`** / **`numpy`**). Full **`CYPHA_*_BIN`** override list: [`native/README.md`](../../native/README.md).

## Experiments SQLite (M6)

| You changed… | Do this |
|--------------|---------|
| **`cypha_studio/core/experiment.py`** **`_SCHEMA`** | Update [`EXPERIMENTS_SCHEMA.md`](../port/EXPERIMENTS_SCHEMA.md); run **`pytest tests/test_experiment_schema_contract.py`**. **`tests/experiment_schema_ddl.py`** mirrors **`_SCHEMA`** for native subprocess tests (AST parse — keep in sync). |
| **DDL export** | `python scripts/export_experiment_schema_sql.py` (optional **`-o path.sql`**). Pytest asserts stdout matches **`_SCHEMA`**. |
| **Native CMake** | Re-run **`cmake`** on **`native/`** so **`experiment_ddl.sql`** in the build dir is regenerated for **`experiment_db_smoke`**. |

Python **`ExperimentDB`** uses **`PRAGMA foreign_keys=ON`**. Native **`experiment_db_smoke`** checks DDL, inserts, join, and FK failure.

## REST / API contract

| You changed… | Do this |
|--------------|---------|
| **REST JSON shapes or routes** | [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md) §3 + **`tests/test_api_contract.py`** + **`tests/test_cypha_rest_smoke.py`**. |
| **Native `cypha_rest`** | Build binary; set **`CYPHA_REST_BIN`** so subprocess smokes are not skipped (matches CI). |

## CI parity (local)

GitHub Actions (`.github/workflows/ci.yml`): **`libsqlite3-dev`**, **`cmake`** build **`native/`**, **`ctest`**, then **`pytest tests/`** with **`CYPHA_REST_BIN`** and **`QT_QPA_PLATFORM=offscreen`**. Python: **`pip install -r requirements-verify.txt`**, then **`PySide6`**, **`pyqtgraph`**, and **`pytest-qt`** (CI runs GUI + qtbot tests; **`pytest-qt`** is intentionally not in **`requirements-verify.txt`** so head-only venvs do not load the plugin without Qt).

Locally: without **`CYPHA_REST_BIN`**, many **`test_cypha_rest_smoke`** cases **skip**. For a full bar, build **`cypha_rest`** and export the path (see [`native/README.md`](../../native/README.md), [`CONTRIBUTING.md`](../../CONTRIBUTING.md)).

## Doc index when editing

| Edit | Also update |
|------|-------------|
| Bump pins in **`requirements-verify.txt`** or **`cypha_studio/requirements.txt`** | Keep **`requirements-pip-merged.txt`** in sync (merged install / encoding fallback) |
| Bump **`.cypha`** version | `PORT_CONTRACT.md` §1; run **`tests/test_cypha_binary_buffer_api.py`** + **`native_memory_train_roundtrip`** if layout changes |
| Registry on-disk layout | `PORT_FULL_STACK.md` §4, `registry.py` |
| Preprocessor JSON | `PREPROCESSOR_CONTRACT.md`, `schemas/preprocessor.schema.json` |

## Makefile shortcuts (repo root)

- **`make test`** — `pytest tests/` + `test_cypha.py` + `cypha_studio/test_cypha_studio.py`
- **`make regen-parity`** — `generate_parity_fixtures.py`
- **`make experiment-ddl`** — writes `artifacts/experiment_schema.sql` (gitignored; canonical DDL remains `experiment._SCHEMA`)

## One-shot scripts (WSL / CI-style)

- **`bash scripts/run_all_regressions.sh`** / **`powershell -File scripts/run_all_regressions.ps1`** — **`test_cypha.py`** + regression pytest bundle: native subprocess parity (incl. **`native_regression_mixture`**), REST/API, schema/registry, engine + accel + fixture modules (see script list); default **`QT_QPA_PLATFORM=offscreen`**. Many **skip** without ELF/**`.exe`** or Qt. Optional **`--full`** / **`-Full`** → **`pytest tests/ -m "not slow"`**. Override interpreter with **`PY=…`** (shell); **`.venv-wsl`** / **`.venv-win`** when **`import pytest`** succeeds there. **`scripts/*.sh`** is **`eol=lf`** in **`.gitattributes`**. On Unix, **`chmod +x scripts/run_all_regressions.sh`** to run **`./scripts/run_all_regressions.sh`**.
- **`bash scripts/wsl_verify.sh`** — venv, **`requirements-verify.txt`** always, optional **`FULL_STUDIO_DEPS=1`** (adds **`cypha_studio/requirements.txt`** + **`pytest-qt`** — CI-like GUI/qtbot without dropping **httpx**/pytest from verify), optional **`RUN_NATIVE=1`** (native + **`ctest`** + REST smoke), optional **`CYPHA_BUILD_QT=1`** with **`RUN_NATIVE=1`** (matches CI: **`qt6-base-dev`** + **`native_qt_stub_load_reference`** + **`test_qt_stub_native.py`**), optional **`PYTEST_MARK='not slow'`**. Default **`QT_QPA_PLATFORM=offscreen`**.
- **`bash scripts/setup_and_test.sh`** / **`powershell … -File scripts/setup_and_test.ps1`** — bootstrap + pytest + legacy suites; optional **`FULL_STUDIO_DEPS=1`** / **`-Studio`** adds studio + **`pytest-qt`** (after **`requirements-verify.txt`**). Sets **`QT_QPA_PLATFORM=offscreen`** when unset.

## Large port backlog (not automated “maintenance”)

See [`PORT_FULL_STACK.md`](../port/PORT_FULL_STACK.md): **RFF/MKE/two-stage** in C++, **Qt** training shell, **C++ ExperimentDB** read/write beyond **`experiment_db_smoke`**.
