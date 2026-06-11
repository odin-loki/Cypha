# Verification status — how “debugged” is this?

Honest snapshot for **port planning**. “Debugged” here means *automated checks + known contracts*, not formal proof.

**Relationship to [`VERIFY_PLAN.md`](VERIFY_PLAN.md):** this file is the **snapshot** (what runs, counts, gaps). The verify plan is the **checklist** (commands, WSL, profiling workflow). **When to regen / rebuild:** [`MAINTENANCE.md`](MAINTENANCE.md).

## Automated coverage

| Layer | What runs | Count / notes |
|--------|-----------|----------------|
| `test_cypha.py` | `Cypha.py` math, encoders, regressors, save/load, batch↔serial parity | **54** checks (custom runner) |
| `cypha_studio/test_cypha_studio.py` | dataset → trainer → registry → inference → API | **48** checks |
| `tests/test_parity_fixtures.py` | Reload `.cypha`, numeric targets vs `expected.npz`, `train_hparams.json` (+ **`align_every`** / **`temp_recalib_every`**), `train_step_vector/sidecar.json`, **`batch_infer_full`** dict parity (M1), **`regression_head.json`** vs manifest labels, **Tier-1** present on `reference.cypha` for native **`cypha_parity`** | **11** pytest |
| `tests/test_memory_train_fixture.py` | `parity_fixtures/memory_train/` sidecar + before/after `.cypha` + **`field_a_eff`** on **`after.cypha`** when **`field_W_T`** present | **3** pytest |
| `tests/test_preprocessor_fixture.py` | `parity_fixtures/preprocessor/` JSON + sidecar vectors | **2** pytest |
| `tests/test_cypha_rest_smoke.py` | Subprocess `cypha_rest` vs parity fixtures (Windows: auto **`native/build-mingw-w64/cypha_rest.exe`** or MSVC **`native/build/`**; **`CYPHA_REST_BIN`** overrides; skips if none); **`/adapt_temperature`**, startup **without** `f_field.json` when embedded `world.F_field`, malformed JSON **400** on POST predict/update/adapt + **`POST /load`** (with registry), **`POST /load`** garbage body **503** when no registry, **wrong input dim** **400** on `/predict` (standalone + **FastAPI vs native** inside shape parity), **`POST /update`** exact keys **`loss`** / **`n_corrections`**, FastAPI vs native **JSON key parity** (health/ready/metrics/session/predict/**`POST /update`**/**`POST /adapt_temperature`**/**`DELETE /session`**/**`GET /models?summary=true`**/classes), **`POST /load`** 503 + **404** + **bad JSON** + **success** (+ **non-empty** `models?summary=true`) vs FastAPI, **`regression_head.json`** → `/predict` MoE + **FastAPI numeric parity** vs native | **16** pytest |
| `tests/test_export_regression_head_script.py` | Subprocess `scripts/export_regression_head.py` → valid `regression_head.json`; committed JSON Schema file parses | **2** pytest |
| `tests/test_regression_mixture_contract.py` | Scalar MoE `Σ p·μ` / `√(Σ p·var)`; batched `P @ μ` / `√(P·σ²)` vs `predict_batch` | **2** pytest |
| `tests/test_native_parity_sidecar.py` | `native_parity.bin` header + byte size vs geometry (**v1** / **v2** tail) | **1** pytest |
| `tests/test_api_contract.py` | Default `app`, routes, **`POST`** paths for predict/update/load/adapt, `/predict` & **`POST /update`** (**`loss`** + **`n_corrections`** keys only), **`POST /load`** success → **`loaded`** = full **`ModelCard`** keys, **`/adapt_temperature`**, **`400`** wrong input dim (same ``detail`` as native), **`503`** `{ "detail": "No model loaded" }` on predict/update/**adapt_temperature**/classes without engine, **`503`** **`POST /load`** with `registry=None`, **`422`** on malformed JSON **`/predict`** + **`/update`** + **`/adapt_temperature`** + **`POST /load`** (missing **`name`** or invalid JSON), **`GET /health`** **`n_predictions`** = **`/metrics`**, empty **`/models`** when `registry=None`, `/metrics` (full `session` + **`gh_*_session`** when `CyphaDIF`, **`session: null`** when no `InferenceSession`), **`DELETE /session`** no-op + **`POST /predict`** engine counters without session, **`DELETE /session`** with no engine, `/ready`, `/health`, **`GET /session`** with `session=None`, `/session`+`DELETE`, `/classes`, `/models?summary=`, **`regression_head_path`** MoE overlay, **`POST /register`** bundle copy + **`503`** without registry, default **`api:app`**: **`GET /models`**, **`GET /metrics.registry_model_count`**, root = **`CYPHA_REGISTRY_ROOT`**, **`create_app()`** **`registry=None`**, **`registry_model_count`** after register | **40** pytest |
| `tests/test_inference_engine.py` | `InferenceEngine` vs `CyphaDIF.infer`, LLR breakdown | **2** pytest |
| `tests/test_score_matrix_field_modes.py` | `use_field`, empty classes, batch parity | **3** pytest |
| `tests/test_accel_cross_gemm.py` | `cypha_accel.cross_r_dT` vs NumPy (any backend) | **2** pytest |
| `tests/test_accel_cypha_wired.py` | `score_matrix` `fused_score_llr` vs patched NumPy | **1** pytest |
| `tests/test_gig_vectorized.py` | `cypha_accel.nig_gh` vectorized `gig_e_inv_v_vec` / `nig_r_eff_vec` vs Cypha scalars | **2** pytest |
| `tests/test_env_config.py` | `CYPHA_*` env: registry root, API host/port, CORS, CSV chunk rows | **5** pytest |
| `tests/test_csv_chunked_parity.py` | `CSVDataset.from_file` chunked vs full-buffer | **1** pytest |
| `tests/test_trainer_registry_eval_wiring.py` | `Trainer.evaluate` after `_model` / `_preprocessor` assign (registry compare) | **1** pytest |
| `tests/test_trainer_regression_fit.py` | `Trainer.fit` smoke on diabetes: **`RFFRegressor`**, **`TwoStageDIF`**, **`MKE`** | **3** pytest |
| `tests/test_cypha_studio_runner.py` | Subprocess `cypha_studio/test_cypha_studio.py` (full 48 checks) | **1** pytest `@slow` |
| `tests/test_experiment_db_paging.py` | `ExperimentDB.list_runs` `LIMIT`/`OFFSET` | **1** pytest |
| `tests/test_experiment_foreign_key.py` | **`PRAGMA foreign_keys=ON`** — `create_run` rejects unknown `experiment_id` | **1** pytest |
| `tests/test_experiment_native_seed.py` | Native **`experiment_db_smoke`** seeds **`.sqlite`** → **`ExperimentDB`** reads metrics / **`metrics_history`** (**`tests/native_subprocess.py`**: **`.exe`** or **WSL** + ELF; **`CYPHA_EXPERIMENT_DB_SMOKE_BIN`**; DDL via **`tests/experiment_schema_ddl.py`**) | **1** pytest |
| `tests/test_experiment_schema_contract.py` | `experiment._SCHEMA` vs [`EXPERIMENTS_SCHEMA.md`](../port/EXPERIMENTS_SCHEMA.md) + SQLite **`PRAGMA`** + **`export_experiment_schema_sql.py`** stdout + AST helper vs **`_SCHEMA`** | **4** pytest |
| `tests/test_experiment_db_smoke_native_parity.py` | Subprocess **`experiment_db_smoke`** in-memory + file round-trip (**`CYPHA_EXPERIMENT_DB_SMOKE_BIN`**) | **2** pytest |
| `tests/test_experiment_db_crud_native_parity.py` | Subprocess **`experiment_db_crud_parity`** (**`CYPHA_EXPERIMENT_DB_CRUD_PARITY_BIN`**) | **1** pytest |
| `tests/test_registry_register_native_parity.py` | Subprocess **`registry_register`** **`--and-verify`** (**`CYPHA_REGISTRY_REGISTER_BIN`**) | **1** pytest |
| `tests/test_cypha_parity_native.py` | Subprocess **`cypha_parity`** (**`CYPHA_CYPHA_PARITY_BIN`**) | **1** pytest |
| `tests/test_preprocessor_native_parity.py` | Subprocess **`preprocessor_parity`** (**`CYPHA_PREPROCESSOR_PARITY_BIN`**) | **1** pytest |
| `tests/test_preprocessor_fit_native_parity.py` | Subprocess **`preprocessor_fit_parity`** vs **`preprocessor_fit/`** + **`preprocessor_fit_no_scale/`** (**`CYPHA_PREPROCESSOR_FIT_PARITY_BIN`**) | **1** pytest |
| `tests/test_csv_ingest_native_parity.py` | Subprocess **`csv_ingest_parity`** vs **`csv_ingest/`** (**`CYPHA_CSV_INGEST_PARITY_BIN`**) | **1** pytest |
| `tests/test_dif_regressor_train_step_native_parity.py` | Subprocess **`dif_regressor_train_step_parity`** vs **`dif_regressor_train_step/`** (**`CYPHA_DIF_REGRESSOR_TRAIN_STEP_PARITY_BIN`**) | **1** pytest |
| `tests/test_nig_adapt_native_parity.py` | Subprocess **`nig_adapt_parity`** (**`CYPHA_NIG_ADAPT_PARITY_BIN`**) | **1** pytest |
| `tests/test_regression_mixture_native_parity.py` | Subprocess **`regression_mixture_parity`** (**`CYPHA_REGRESSION_MIXTURE_PARITY_BIN`**) | **1** pytest |
| `tests/test_memory_train_native_parity.py` | Subprocess **`memory_train_parity`** (**`CYPHA_MEMORY_TRAIN_PARITY_BIN`**) | **1** pytest |
| `tests/test_train_step_vector_native_parity.py` | Subprocess **`train_step_vector_parity`** (**`CYPHA_TRAIN_STEP_VECTOR_PARITY_BIN`**) | **1** pytest |
| `tests/test_memory_train_roundtrip_native.py` | Subprocess **`memory_train_roundtrip`** + Python **`cypha_load_binary`** vs **`after.cypha`** (**`CYPHA_MEMORY_TRAIN_ROUNDTRIP_BIN`**) | **1** pytest |
| `tests/test_mke_train_step_native_parity.py` | Subprocess **`mke_train_step_parity`** vs **`mke_train_step/`** + **`mke_train_extended/`** (**`CYPHA_MKE_TRAIN_STEP_PARITY_BIN`**) + replay RNG wiring | **5** pytest |
| `tests/test_native_ctest_pytest_registry.py` | CMake **`NAME native_*`** set matches subprocess coverage map (add **`_NATIVE_CTEST_TO_PYTEST`** when you add CTests) | **2** pytest |
| `tests/test_cypha_binary_buffer_api.py` | Python v3 **bytes** I/O vs **`parity_fixtures/reference.cypha`** — path vs **`read_bytes()`** load; **`save_binary_to_bytes`** vs file bytes + round-trip | **2** pytest |
| `tests/test_qt_stub_native.py` | Subprocess **`cypha_qt_stub`** + **`reference.cypha`** (**`CYPHA_QT_STUB_BIN`**; **`QT_QPA_PLATFORM=offscreen`**) — mirrors CTest **`native_qt_stub_load_reference`** | **1** pytest |
| `tests/test_qt_shell_native.py` | Subprocess **`cypha_qt_shell --smoke`** (**`CYPHA_QT_SHELL_BIN`**) — headless load + zero-vector predict; `--help` output asserts PNG/SVG/CSV export, EMA overlay, `return_explanation`, Y-lock, training log | **2** pytest |
| `tests/test_regression_m4_native_parity.py` | Subprocess **`regression_m4_parity`** vs `parity_fixtures/regression_m4/` (batch/EMA + RFF RLS + MKE RLS + two-stage + MKE routing; **`tests/native_subprocess.py`**: **`.exe`** or **WSL**; **`CYPHA_REGRESSION_M4_PARITY_BIN`**) | **1** pytest |
| `tests/test_two_stage_pipeline_native_parity.py` | Subprocess **`regression_two_stage_pipeline_parity`** vs `parity_fixtures/two_stage_pipeline/` (**`CYPHA_TWO_STAGE_PIPELINE_PARITY_BIN`**) | **1** pytest |
| `tests/test_two_stage_ridge_fit_native_parity.py` | Sidecar NumPy self-check + subprocess **`regression_two_stage_ridge_fit_parity`** vs `parity_fixtures/two_stage_ridge_fit/` (**`CYPHA_TWO_STAGE_RIDGE_FIT_PARITY_BIN`**) | **2** pytest |
| `tests/test_two_stage_e2e_ridge_native_parity.py` | Subprocess same binary vs `parity_fixtures/two_stage_e2e_ridge/` (LLR from Python **`TwoStageDIFRegressor.fit`**) | **1** pytest |
| `tests/test_batch_llr_native_parity.py` | Sidecar vs **`expected.npz`** + subprocess **`batch_llr_parity`** (**`CYPHA_BATCH_LLR_PARITY_BIN`**) | **2** pytest |
| `tests/test_quantile_dif_train_native_parity.py` | Sidecar geometry + subprocess **`quantile_dif_train_parity`** vs `parity_fixtures/quantile_dif_train/` (**`CYPHA_QUANTILE_DIF_TRAIN_PARITY_BIN`**) | **2** pytest |
| `tests/test_dif_train_replay_native_parity.py` | **`replay_u01`** sidecar + subprocess vs `parity_fixtures/dif_train_replay/` (**`CYPHA_DIF_TRAIN_REPLAY_PARITY_BIN`**) | **2** pytest |
| `tests/test_generation_native_parity.py` | Subprocess **`generation_parity`** vs `parity_fixtures/generation/sidecar.json` (all generation methods) | **3** pytest |
| `tests/test_studio_trainer_classify_hotpath_native_parity.py` | Studio **`Trainer.fit`** loop fixture + **`quantile_dif_train_parity`** (**`CYPHA_STUDIO_TRAINER_CLASSIFY_HOTPATH_BIN`** or **`CYPHA_QUANTILE_DIF_TRAIN_PARITY_BIN`**) | **2** pytest |
| `tests/test_studio_trainer_gh_classify_hotpath_native_parity.py` | **`gh_train_step`** / **`chi`**/**`psi`** fixture + **`quantile_dif_train_parity`** (**`CYPHA_STUDIO_TRAINER_GH_CLASSIFY_HOTPATH_BIN`** or **`CYPHA_QUANTILE_DIF_TRAIN_PARITY_BIN`**) | **2** pytest |
| `tests/test_studio_trainer_preprocess_classify_hotpath_native_parity.py` | **`preprocess_train_classify_parity`** + **`CYPHA_PREPROCESS_TRAIN_CLASSIFY_PARITY_BIN`** | **2** pytest |
| `tests/test_studio_trainer_preprocess_gh_classify_hotpath_native_parity.py` | same binary vs **`studio_trainer_preprocess_gh_classify_hotpath/`** (**`use_gh`**) | **2** pytest |
| `tests/test_csv_preprocess_classify_hotpath_native_parity.py` | **`preprocess_train_classify_parity`** vs **`csv_preprocess_classify_hotpath/`** (**CSV** + **`CYPHA_PREPROCESS_TRAIN_CLASSIFY_PARITY_BIN`**) | **2** pytest |
| `tests/test_regression_rff_native_parity.py` | Sidecar **NumPy** self-check + subprocess **`regression_rff_parity`** vs `parity_fixtures/rff_regression/` (**WSL** / env **`CYPHA_REGRESSION_RFF_PARITY_BIN`**) | **2** pytest |
| `tests/test_training_plot_compress.py` | `TrainingWidget._compress_xy` for plot downsampling | **2** pytest |
| `tests/test_studio_data_registry.py` | `CSVDataset.from_file` + `ModelRegistry` register / list / load + `save_state` ndarray parity; **`DIFRegressor`** / **`RFFRegressor`** / **`TwoStageDIFRegressor`** / **`MKE`** registry round-trips | **6** pytest |
| `tests/test_cypha_load_state_context.py` | ``CyphaDIF`` Tier-1 context in ``save_state`` / ``load_state`` → ``score_matrix`` unchanged | **1** pytest |
| `tests/test_gui_training_dataset.py` | `TrainingWidget` step signals; `DatasetWidget.load_file` → `AppState` train/val split | **2** pytest |

**GUI window settings:** `pytest tests/test_gui_window_settings.py -v` (**1** test) — `closeEvent` writes `geometry` + `windowState` to `QSettings` (INI path isolated).

**GUI smoke:** `pytest tests/test_gui_smoke.py -v` (**6** tests; PySide6 + pyqtgraph) — `MainWindow`, `TrainConfigDialog`, chat (no model / with tiny classifier), `ConfidenceWidget` + `SignalBus`, `MessageBubble`, all under **`QT_QPA_PLATFORM=offscreen`**. **GUI qtbot:** `pytest tests/test_gui_qtbot.py -v` (**5** tests; **pytest-qt** + PySide6) — Send/Clear clicks, `TrainConfigDialog` OK, focus-chat shortcut handler, toolbar Train ( **`QMessageBox.information` mocked**). **Profile:** `python scripts/profile_gui_startup.py` (cold start); **`python scripts/profile_studio_hotpaths.py`** for training monitor, chat, dataset load, registry scan, API predict loops ([`docs/README.md`](../README.md)). Not a substitute for full manual UI coverage.

**Full studio script in pytest:** `pytest tests/test_cypha_studio_runner.py -m slow` (subprocess + UTF-8 env for Windows). Full `pytest tests/` includes one **`@slow`** plus native subprocess parity modules (many **skip** without ELF / **`.exe`** / fixtures); use **`pytest tests/ --collect-only -q`** for the exact count with your installed deps (CI: **`requirements-verify.txt`** + **PySide6** + **pyqtgraph** + **`pytest-qt`**). Headless dev machines without Qt: **`requirements-verify.txt`** only — GUI tests **skip**; do not install **`pytest-qt`** without Qt bindings (plugin aborts pytest). **`pytest tests/ -m "not slow"`** deselects **`@slow`**. On **Windows**, **`test_cypha_rest_smoke.py`** auto-finds **`native/build-mingw-w64/cypha_rest.exe`** (WSL MinGW cross-build per **`native/scripts/build_cypha_rest_mingw_wsl.ps1`**) or MSVC **`native/build/cypha_rest.exe`**; **`CYPHA_REST_BIN`** overrides. Without any **`cypha_rest`** binary, those subprocess tests **skip**. CI sets **`CYPHA_REST_BIN`** to the Linux ELF **`native/build/cypha_rest`**. **`scripts/wsl_verify.sh`** accepts **`PYTEST_MARK='not slow'`** for the main pytest step.

**Current totals (typical):** ~`189 passed, 1 skipped` (`test_cuda_bench` when no CUDA GPU). `pytest-qt` must be installed for `test_training_plot_compress.py`; the tests error with "fixture 'qapp' not found" when it is absent — install it: `pip install pytest-qt`.

GitHub Actions **CI** (`.github/workflows/ci.yml`): **four blocking jobs** — **`build_and_test`** (Ubuntu: Ruff, **`CYPHA_BUILD_QT=ON`**, **`ctest`** excluding headless Qt GUI exec, WikiText-2, **`pytest tests/`** + **`cypha_lm/model/tests/`** with **`CYPHA_REST_BIN`**), **`mingw_cross`** (MinGW Windows PE smoke), **`windows_cuda_msvc`** (MSVC + CUDA compile + **`native_cuda_smoke`** / **`native_score_batch`**), **`linux_cuda`** (GCC + nvcc compile smoke). **`native_cuda_bench`** is GPU-only and non-blocking when a device is present. Subprocess pytest mirror + drift guard: **`tests/test_native_ctest_pytest_registry.py`**.

**Not in CI** (run locally before LM/SOM changes):

| Package | Command | Notes |
|---------|---------|--------|
| **`cypha_som/`** | `pytest cypha_som/tests/ -v` | **7** tests — GNG/SOM/GRIA/Hebbian/temporal hooks; flags default OFF |
| **`cypha_lm/`** | `pytest cypha_lm/ -v` | **~70** tests across embeddings, experts, projection, temporal, model, integration |

**Not covered automatically today:** interactive GUI workflows, long-run memory leaks, multi-thread stress, real KDD-scale files (benchmark Section 9 uses synthetic unless you drop `.npy` in `/tmp`). Typical local runs have **no GPU**; `cypha_accel` falls back to NumPy. Use `scripts/gpu_fullbench.py` on a CUDA box for encode+LLR+softmax+gate timing and fp64 LLR parity vs CPU.

**D04 benchmark bug (not a CyphaLM failure):** domain D04 in `cypha_bench` runs **`CyphaDIF + CharNgramEncoder`**, not CyphaLM. The reported **33.2 bpc** comes from wrong probability indexing (`probs[next_idx]` uses char ID into a label-ordered array → **`1e-10` floor** → −log₂(1e-10) ≈ 33.2). Real CyphaLM held-out BPC is **D17 = 4.50** (bigram baseline 3.69). See [`CHANGELOG.md`](../../CHANGELOG.md) §1.0.0 and [`docs/RESEARCH_STATUS.md`](../RESEARCH_STATUS.md).

## Contracts frozen for the port

- **[`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md)** — `.cypha` v3, LLR/softmax/GH/temperature, REST JSON.
- **`parity_fixtures/`** — committed golden models + vectors; native code should reproduce within tolerance.

### Parity fixture inventory (24 directories + root artifacts)

**Root files:** `reference.cypha`, `native_parity.bin`, `expected.npz`, `train_hparams.json`, `regression_head.json`, `manifest.json`.

| Directory | Primary gate |
|-----------|----------------|
| `batch_llr` | `native_batch_llr`, `test_batch_llr_native_parity` |
| `csv_ingest` | `native_csv_ingest` |
| `csv_preprocess_classify_hotpath` | `native_csv_preprocess_classify_hotpath` |
| `dif_regressor_train_step` | `native_dif_regressor_train_step` |
| `dif_train_replay` | `native_dif_train_replay` |
| `generation` | `native_generation`, `test_generation_native_parity` |
| `memory_train` | `native_memory_train`, `native_memory_train_roundtrip` |
| `mke_train_extended` | `native_mke_train_extended` |
| `mke_train_step` | `native_mke_train_step` |
| `preprocessor` | `native_preprocessor`, `test_preprocessor_fixture` |
| `preprocessor_fit` | `native_preprocessor_fit` |
| `preprocessor_fit_no_scale` | `native_preprocessor_fit` |
| `quantile_dif_train` | `native_quantile_dif_train` |
| `registry_register` | `native_registry_register` |
| `regression_m4` | `native_regression_m4` |
| `rff_regression` | `native_regression_rff` |
| `studio_trainer_classify_hotpath` | `native_studio_trainer_classify_hotpath` |
| `studio_trainer_gh_classify_hotpath` | `native_studio_trainer_gh_classify_hotpath` |
| `studio_trainer_preprocess_classify_hotpath` | `native_studio_trainer_preprocess_classify_hotpath` |
| `studio_trainer_preprocess_gh_classify_hotpath` | `native_studio_trainer_preprocess_gh_classify_hotpath` |
| `train_step_vector` | `native_train_step_vector` |
| `two_stage_e2e_ridge` | `native_regression_two_stage_e2e_ridge` |
| `two_stage_pipeline` | `native_regression_two_stage_pipeline` |
| `two_stage_ridge_fit` | `native_regression_two_stage_ridge_fit` |

## Known gaps (before you trust production scale)

1. **Pytest vs studio script** — the full **48** studio checks run via `pytest tests/test_cypha_studio_runner.py -m slow` (subprocess). Default `pytest tests/` skips `@slow`; include them in release gates when convenient.
2. **InferenceEngine** used `score_matrix` with default `use_field=False` while `infer()` uses `use_field=True` — **fixed** so `all_scores` matches classification.
3. **Full GPU training** — not implemented; inference uses CuPy for fused LLR (`fused_score_llr`), encoder projection (`project_features`), and row softmax when **K>8** (K≤8 keeps CPU softmax for classify parity). See `scripts/gpu_microbench.py` (raw GEMM) and `scripts/gpu_fullbench.py` (pipeline).
4. **Real-data profiling** — use `scripts/profile_real_datasets.py` (sklearn) as a step toward your own CSV/KDD dumps.

## Green bar (achieved — keep this clean)

All items below are currently green. They form the CI gate:

- [x] GitHub Actions **CI green** on `main` — **four jobs**: Linux CTest + pytest, MinGW PE smoke, **MSVC + CUDA (Windows)**, optional Linux GCC CUDA
- [x] `pytest tests/` + `cypha_lm/model/tests/` — **~274 passed** on CI (headless; `test_gui_qtbot.py` ignored)
- [x] Native **`ctest`** green: **52** `native_*` tests (Qt GUI exec excluded on headless Linux CI); drift guard **`pytest tests/test_native_ctest_pytest_registry.py`**
- [x] `pytest tests/test_parity_fixtures.py` and `python test_cypha.py` clean on reference branch
- [x] `python scripts/profile_real_datasets.py` — no exceptions; top `cumtime` reviewed
- [ ] Optional: `python scripts/gpu_microbench.py` and `python scripts/gpu_fullbench.py` on a GPU box (see [`docs/FUTURE.md`](../FUTURE.md) §1)

## Full Python stack replacement (product port) — COMPLETE

All milestones M1–M6 are complete. See **[`PORT_FULL_STACK.md`](../port/PORT_FULL_STACK.md)** for the full per-milestone record and **[`docs/FUTURE.md`](../FUTURE.md)** for next engineering horizons (CUDA, Qt packaging, Web UI, multi-model serving).

**Still reference-only in Python (by design):** parity generation scripts, tuning/benchmark scripts — native CI consumes **`parity_fixtures/`** and JSON/binary artifacts, not `Cypha.py` at runtime. **[`PREPROCESSOR_CONTRACT.md`](../port/PREPROCESSOR_CONTRACT.md)** freezes `preprocessor.json` next to `model.cypha`.
