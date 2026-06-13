# Maintenance — what to regen, rebuild, and align

*If you were looking for a **“do docs all”** checklist, this is it — plus fixtures, native, and schema.*

Use this when you change contracts or reference math. **Snapshot of automation:** [`VERIFICATION_STATUS.md`](VERIFICATION_STATUS.md). **Commands hub:** [`VERIFY_PLAN.md`](VERIFY_PLAN.md).

## Parity fixtures & native CTest

**`fixtures/` are frozen.** Regenerate only when the contract intentionally changes, using native parity tools (`native/tools/*_parity`) and PR review — not removed Python generators.

| You changed… | Do this |
|--------------|---------|
| **`.cypha` v3 layout / inference / context (Tier-2+)** | Update committed `fixtures/` (`reference.cypha`, `expected.npz`, `native_parity.bin`, `manifest.json`, `train_hparams.json`, …) |
| **Train-step vector sidecar** (`train_step_vector/`) | Update sidecar + run **`ctest -R native_train_step_vector`** |
| **Memory train golden** | Update `fixtures/memory_train/` → **`ctest -R native_memory_train`** |
| **Preprocessor fixture** | Update `fixtures/preprocessor/` → **`ctest -R native_preprocessor`** |
| **Preprocessor fit (native PCA/RFF)** | Update `preprocessor_fit/` + `preprocessor_fit_no_scale/` (+ `preprocessor_fit_rff/`) → **`ctest -R native_preprocessor_fit`** |
| **CSV ingest** | Update `fixtures/csv_ingest/` → **`ctest -R native_csv_ingest`** |
| **DIFRegressor train-step slice** | Update `dif_regressor_train_step/` → **`ctest -R native_dif_regressor_train_step`** |
| **M4 regression_m4 sidecar** | Update `regression_m4/` → **`ctest -R native_regression_m4`** |
| **M5 two_stage_pipeline** | Update `two_stage_pipeline/` → **`ctest -R native_regression_two_stage_pipeline`** |
| **M6 two_stage_ridge_fit** | Update `two_stage_ridge_fit/` → **`ctest -R native_regression_two_stage_ridge_fit`** |
| **two_stage_e2e_ridge** | Update `two_stage_e2e_ridge/` → **`ctest -R native_regression_two_stage_e2e_ridge`** |
| **batch_llr** | Update `batch_llr/` → **`ctest -R native_batch_llr`** |
| **quantile_dif_train / dif_train_replay** | Update sidecars → **`ctest -R native_quantile_dif_train`** / **`native_dif_train_replay`** |
| **studio_trainer_* hotpaths** | Update matching dir → **`ctest -R native_studio_trainer_*`** |
| **csv_preprocess_classify_hotpath** | After preprocess hotpath → **`ctest -R native_csv_preprocess_classify_hotpath`** |
| **mke_train_step / mke_train_extended** | Update sidecars → **`ctest -R native_mke_train_*`** |
| **RFF / ridge sidecar** | Update `rff_regression/` → **`ctest -R native_regression_rff`** |
| **MoE `regression_head.json`** | Update root JSON → REST / **`--regression-json`** smoke |
| **Embedded `F_field` JSON** | Update `f_field.json` when world field export changes |
| **Native Bessel tables** | Regenerate `native/src/bessel_table_data.cpp` if GH grid changes |
| **`.cypha` v3 buffer I/O** | **`ctest -R native_memory_train_roundtrip`** + buffer load/save CTests |
| **Qt** (`cypha_qt_stub`, **`cypha_qt_shell`**) | **`qt6-base-dev`**, **`cmake … -DCYPHA_BUILD_QT=ON`**, **`ctest -R native_qt`** (headless CI excludes GUI exec) |
| **CyphaLM checkpoint sidecars** | Update `fixtures/cyphalm_checkpoint/` → **`ctest -R native_cyphalm_*`** |
| **Any of the above** | Rebuild **`native/`** and run **`ctest --test-dir native/build -R native_ --output-on-failure`**. New **`add_test(NAME native_…)`** must appear in **`ctest -N`**. Local one-liner: **`bash scripts/ci_native_linux.sh`**. |

**Tier-1:** committed **`reference.cypha`** includes context keys; native **`CyphaInferModel::from_root`** restores them. See [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md) §4 and **`ctest -R native_parity`**.

Full **`CYPHA_*_BIN`** override list: [`native/README.md`](../../native/README.md).

## Experiments SQLite (M6)

| You changed… | Do this |
|--------------|---------|
| **Experiment DB schema** | Update [`EXPERIMENTS_SCHEMA.md`](../port/EXPERIMENTS_SCHEMA.md); reconfigure **`native/`** (CMake regenerates DDL); **`ctest -R native_experiment`** |
| **Native CMake** | Re-run **`cmake`** on **`native/`** so build-dir DDL matches for **`experiment_db_smoke`** |

Native **`experiment_db_smoke`** checks DDL, inserts, join, and FK failure with **`PRAGMA foreign_keys=ON`**.

## REST / API contract

| You changed… | Do this |
|--------------|---------|
| **REST JSON shapes or routes** | [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md) §3 + **`ctest -R native_`** + **`native/scripts/smoke_cypha_rest_mingw.ps1`** |
| **Native `cypha_rest`** | Build binary; set **`CYPHA_REST_BIN`** for REST smokes (matches CI) |

## CI parity (local)

GitHub Actions (`.github/workflows/ci.yml`): **`libsqlite3-dev`**, **`cmake`** build **`native/`**, **`ctest -R native_`**, with **`CYPHA_REST_BIN`** and **`QT_QPA_PLATFORM=offscreen`**.

Locally: without **`CYPHA_REST_BIN`**, REST subprocess tests may **skip**. Full bar: build **`cypha_rest`** and export the path (see [`native/README.md`](../../native/README.md), [`CONTRIBUTING.md`](../../CONTRIBUTING.md)).

## Doc index when editing

| Edit | Also update |
|------|-------------|
| Bump native / Qt deps | **`native/CMakeLists.txt`**, [`native/qt/README.md`](../../native/qt/README.md) |
| Bump **`.cypha`** version | `PORT_CONTRACT.md` §1; **`ctest -R native_`** |
| Registry on-disk layout | `PORT_FULL_STACK.md` §4, `PORT_CONTRACT.md` §3 |
| Preprocessor JSON | `PREPROCESSOR_CONTRACT.md`, `schemas/preprocessor.schema.json` |

## Makefile shortcuts (repo root)

- **`make test`** — `ctest --test-dir native/build -R native_`
- **`make experiment-ddl`** — writes `artifacts/experiment_schema.sql` (gitignored; canonical DDL in CMake / **EXPERIMENTS_SCHEMA.md**)

## One-shot scripts

- **`bash scripts/ci_native_linux.sh`** — cmake + **`ctest -R native_`** (CI mirror)
- **`powershell -File scripts/cypha_native_validate_all.ps1`** — Windows full gate
- **`bash scripts/wsl_verify.sh`** with **`RUN_NATIVE=1`** — WSL build + CTest + optional REST smoke

## Large port backlog (not automated “maintenance”)

See [`PORT_FULL_STACK.md`](../port/PORT_FULL_STACK.md): optional Qt polish, CUDA tuning, packaged binaries.
