# Native runtime (C++ / CUDA / Qt)

Monorepo C++ core for [`docs/port/PORT_FULL_STACK.md`](../docs/port/PORT_FULL_STACK.md). **Vendored:** `third_party/nlohmann/json.hpp`, `third_party/httplib.h` (no OpenSSL — do not `#define CPPHTTPLIB_OPENSSL_SUPPORT` unless you link libssl).

**Accel** (`cypha/accel_backend.hpp`): optional **CUDA** (`-DCYPHA_ENABLE_CUDA=ON`, NVIDIA toolkit + driver); otherwise **ISO C++** parallel CPU via `std::thread`. **`cuda_smoke`** checks correctness vs a serial reference; **`cuda_smoke --bench`** compares CUDA vs CPU when a GPU is present (exit 2 skip otherwise).

## CMake presets (Windows + WSL trees)

| Preset | Binary directory | Use |
|--------|------------------|-----|
| **`windows-msvc-release`** | `native/build-windows-msvc/` | Native Windows: Visual Studio **2022** generator, x64. `cmake --preset windows-msvc-release` then `cmake --build build-windows-msvc --config Release`. |
| **`windows-vs2026-release`** | `native/build-windows-vs2026/` | Visual Studio **18 2026** / Build Tools 18 (MSVC 14.5x). Same workflow; use if preset **windows-msvc-release** fails (no VS 2022). |
| **`wsl-gcc-release`** | `native/build-wsl-gcc/` | WSL/Linux: **Ninja** + Release. Needs `ninja-build` (or use **`wsl-gcc-release-make`** for Makefiles). |
| **`mingw-w64-cross`** | `native/build-mingw-w64/` | Cross-compile `.exe` from Linux/WSL (no CUDA). |

**Qt shell on Windows:** after a Release build with **`-DCYPHA_BUILD_QT=ON`** and **`-DCMAKE_PREFIX_PATH=`** pointing at your Qt **msvc*_64** kit, run **`windeployqt`** on `qt/Release/cypha_qt_shell.exe`, then from the repo root: **`powershell -ExecutionPolicy Bypass -File scripts/run_cypha_qt_windows.ps1`** (starts **`cypha_rest`** with `parity_fixtures/reference.cypha` and opens **`cypha_qt_shell`**). Use **`-NoServer`** for GUI only.

```bash
cd native
cmake --list-presets
cmake --preset wsl-gcc-release
cmake --build --preset wsl-gcc-release-build
ctest --test-dir build-wsl-gcc --output-on-failure
```

**CUDA (Windows or WSL with NVIDIA):** add `-DCYPHA_ENABLE_CUDA=ON` at configure (requires `nvcc` + `CUDA::cudart`). Override arch: `-DCMAKE_CUDA_ARCHITECTURES=89` (Ada), `86` (Ampere), etc. Optional: `-DCYPHA_ACCEL_GPU_MIN_BATCH_ROWS=8` to dispatch smaller batches to the GPU (default **16** avoids tiny-batch launch + copy overhead).

Device memory: CUDA accel reuses one **growing device pool** plus a one-time **Bessel K₂/K₁ table** upload for the GH–NIG world gate (no per-call `cudaMalloc` for the main buffers once warmed up).

## Targets

**v2.2 production binaries** (release `bin/`): **`cypha_rest`**, **`cypha_bench_run`**, **`cypha_bench_report`**, **`cypha_tune_run`**, **`cypha_diagnostics_run`**. Dev parity tools ship under **`bin/dev/`** when packaged. See [`docs/native/NATIVE_QUICKSTART.md`](../docs/native/NATIVE_QUICKSTART.md).

### Libraries

| Lib | Role |
|-----|------|
| **`cypha_core`** | M1–M7 CyphaDIF runtime: `.cypha` v3 I/O, CPU infer, training, regression, generation/RAG, Branch A router, MultiLabel, merge_from, SimilarityIndex, kernel LLR, retrieval |
| **`cypha_lm_native`** | CyphaLM: Izaac embed, model/SSM, BPE, Hebbian/HierSSM, decode, checkpoint I/O — see [`CYPHALM_NATIVE_BUILD.md`](../docs/native/CYPHALM_NATIVE_BUILD.md) |
| **`cypha_bench_native`** | Bench engine: d01–d17 domains, encoders, sklearn-equivalent baselines, cross-domain analyses, figure data + PNG render, tune sweep helpers |

### Production binaries

| Binary | Role |
|--------|------|
| **`cypha_bench_run`** | Master bench runner (mirrors **`cypha_bench/run_all.py`**): domains **d01–d17**, cross-domain tables, **`BASELINE_REPORT.md`**, **`report/summary.json`**, figure JSON/PNG under **`cypha_bench/report/figures/`**. CLI: `--domain N`, `--from-domain N`, `--report-only`, `--list-domains`. Env: **`CYPHA_REPO_ROOT`**, **`CYPHA_BENCH_FAST=1`**. CTest **`native_bench_run_list_domains`**. Contract: [`PORT_CONTRACT.md`](../docs/port/PORT_CONTRACT.md) §6. |
| **`cypha_bench_report`** | Report-only rebuild (same helpers as **`--report-only`**). CLI: `--output PATH`, `--skip-cross-domain`. |
| **`cypha_tune_run`** | Native tuning/sweep orchestrator (mirrors **`cypha_bench/tuning/*.py`**). Loads sweep JSON, invokes **`cyphalm_bench_native`** or **`cypha_bench_run`** per cell. CLI: `--config PATH`, `--out PATH`, `--exe-dir DIR`, `--max-cells N`, `--write`, `--dry-run`. CTest **`native_tune_run_smoke`**. |
| **`cypha_diagnostics_run`** | Phases 1–4 validation orchestrator: runs parity exes + inline **`cypha_core`** checks (no sklearn). CLI: `--fixtures DIR`, `--out DIR`, `--exe-dir DIR`, `--phases 1,2,3,4`, `--list`, `--inline-only`. CTest **`native_diagnostics_run`**. |
| **`cyphalm_bench_native`** | Per-profile LM BPC bench (char-LSTM / hybrid / ablations). Used by **`cypha_bench_run`** d04/d17 and **`cypha_tune_run`**. |

### Core parity & smoke

| Binary / lib | Milestone | Role |
|--------------|-----------|------|
| **`cypha_core`** | M1–M7 | `.cypha` **load** + **`load_cypha_from_buffer`** / **`save_cypha_file`** / **`save_cypha_to_buffer`** / **`clone_cnode`** (Python **`cypha_save_binary`** / **`cypha_load_binary`** / **`cypha_save_binary_to_bytes`** / **`cypha_load_binary_from_bytes`** v3 layout), `mid_trans`, `llr_scale_*`, `field_W_T`, optional **`field_a_eff`**, `w_inject`, …, CPU infer, full **Tier-1+2** `context_prior` / `context_record_step`, preprocessor JSON, `memory_train` + **`merge_state_into_root_for_save`** + `dedup_check`, `sync_infer`, replay, contrastive + deliberate + **`encoder_align_to_offsets`**, NIG field, `dif_train_step_vector`, **`dif_train_classify_sequence`**, **`dif_gh_train_classify_sequence`** (GH online loop), **`mke_scalar_train_step`**, **`registry_scan`** + **`registry_register_bundle`**, **`create_fresh_model_root`**, **`generate_retrieval_augmented`**, **`SimilarityIndex`**, **`merge_from`**, **`MultiLabelDIF`**, kernel-memory LLR |
| **`create_model_smoke`** | M1 | Empty-model create → infer + memory load → save/reload roundtrip. CTest **`native_create_model`**. |
| **`cuda_smoke`** | Accel | **`cypha::accel`**: CUDA if `-DCYPHA_ENABLE_CUDA=ON` + GPU; else parallel CPU (`std::thread`). CTest **`native_cuda_smoke`** / **`native_cuda_bench`** (bench exit 2 without GPU). Pytest **`tests/test_cuda_smoke_native.py`**; **`CYPHA_CUDA_SMOKE_BIN`**. |
| **`cypha_parity`** | M1 | `reference.cypha` + `native_parity.bin` → LLR / probs / gates; **v2** sidecar tail checks **`batch_infer_full`** entropy + confidence. **`CyphaInferModel::from_root`** restores **Tier-1** from **`ctx_hist_packed`** / co-occurrence / last label; see [`PORT_CONTRACT.md`](../docs/port/PORT_CONTRACT.md) §4. CTest **`native_parity`**. Pytest **`tests/test_cypha_parity_native.py`**; env **`CYPHA_CYPHA_PARITY_BIN`**. |
| **`batch_llr_parity`** | M7 | **`batch_llr_from_x`** vs `parity_fixtures/batch_llr/sidecar.json` (same **X**/**LLR** as **`expected.npz`**) |
| **`score_batch_parity`** | Accel | **`cypha::accel::batch_encode`** + **`score_matrix`** vs `parity_fixtures/score_batch/sidecar.json` (`project_features` + `fused_score_llr` goldens). CTest **`native_score_batch`**. Pytest **`tests/test_score_batch_native_parity.py`**; **`CYPHA_SCORE_BATCH_PARITY_BIN`**. See [`docs/native/ACCEL_CUDA.md`](../docs/native/ACCEL_CUDA.md). |
| **`kernel_llr_parity`** | M1+ | Kernel-memory **`score_matrix`** vs `parity_fixtures/kernel_llr/sidecar.json`. CTest **`native_kernel_llr`**. Pytest **`tests/test_kernel_llr_native_parity.py`**; **`CYPHA_KERNEL_LLR_PARITY_BIN`**. |
| **`gh_infer_deliberation_parity`** | M1+ | GH infer + deliberation gate vs `parity_fixtures/gh_infer_deliberation/`. CTest **`native_gh_infer_deliberation`**. Pytest **`tests/test_gh_infer_deliberation_native_parity.py`**; **`CYPHA_GH_INFER_DELIBERATION_PARITY_BIN`**. |
| **`multilabel_dif_parity`** | M3+ | **`MultiLabelDIF`** train/infer vs `parity_fixtures/multilabel_dif/`. CTest **`native_multilabel_dif`**. Pytest **`tests/test_multilabel_dif_native_parity.py`**; **`CYPHA_MULTILABEL_DIF_PARITY_BIN`**. |
| **`merge_from_parity`** | M3+ | **`merge_from`** state fusion vs `parity_fixtures/merge_from/`. CTest **`native_merge_from`**. Pytest **`tests/test_merge_from_native_parity.py`**; **`CYPHA_MERGE_FROM_PARITY_BIN`**. |
| **`similarity_index_parity`** | M3+ | **`SimilarityIndex`** retrieval index vs `parity_fixtures/similarity_index/`. CTest **`native_similarity_index`**. Pytest **`tests/test_similarity_index_native_parity.py`**; **`CYPHA_SIMILARITY_INDEX_PARITY_BIN`**. |
| **`generation_parity`** | M5+ | RAG **`generate_retrieval_augmented`** vs `parity_fixtures/generation/sidecar.json`. CTest **`native_generation`**. Pytest **`tests/test_generation_native_parity.py`**; **`CYPHA_GENERATION_PARITY_BIN`**. |
| **`retrieval_parity`** | M5+ | Dense retrieval top-k vs `parity_fixtures/retrieval/sidecar.json`. CTest **`native_retrieval`** (disabled until fixture exists). Pytest **`tests/test_retrieval_native_parity.py`**; **`CYPHA_RETRIEVAL_PARITY_BIN`**. |
| **`memory_train_parity`** | M3 | `parity_fixtures/memory_train/` — one `DIFMemory.train` step vs `after.cypha`; CTest **`native_memory_train`**. Pytest **`tests/test_memory_train_native_parity.py`**; env **`CYPHA_MEMORY_TRAIN_PARITY_BIN`**. |
| **`memory_train_roundtrip`** | M3 | Same fixture: train → **`merge_state_into_root_for_save`** → **`patch_field_a_eff_into_root`** (aligns with Python **`field_a_eff`**) → **`save_cypha_to_buffer`** (bytes) + **`save_cypha_file`** → on-disk bytes **`memcmp`** with buffer → **`load_cypha_file`** / **`load_cypha_from_buffer`** vs file reload. Tree ≈ **`after.cypha`** (CTest **`native_memory_train_roundtrip`**). Pytest **`tests/test_memory_train_roundtrip_native.py`**; env **`CYPHA_MEMORY_TRAIN_ROUNDTRIP_BIN`**. Optional 2nd arg: output path (default `<dir>/roundtrip_native.cypha`). |
| **`preprocessor_parity`** | M2 | `parity_fixtures/preprocessor/` — `transform_one` vs Python; CTest **`native_preprocessor`**. Pytest **`tests/test_preprocessor_native_parity.py`**; env **`CYPHA_PREPROCESSOR_PARITY_BIN`**. |
| **`preprocessor_fit_parity`** | M2 | **`preprocessor_fit/`** + **`preprocessor_fit_no_scale/`** + **`preprocessor_fit_rff/`** — **`PreprocessorState::fit_from_design_matrix`** (scale on/off + PCA + RFF) vs Python **`Preprocessor.fit`** + probe **`transform_one`**; CTest **`native_preprocessor_fit`**. Pytest **`tests/test_preprocessor_fit_native_parity.py`**; env **`CYPHA_PREPROCESSOR_FIT_PARITY_BIN`**. |
| **`csv_ingest_parity`** | M2 | **`csv_ingest/`** — **`cypha::load_csv_dense`** vs **`CSVDataset.from_file`** (**`target_col_name`** / **`feature_col_names`** and/or indices; multiline quoted fields); CTest **`native_csv_ingest`**. Pytest **`tests/test_csv_ingest_native_parity.py`**; env **`CYPHA_CSV_INGEST_PARITY_BIN`**. |
| **`dif_regressor_train_step_parity`** | M4 | **`dif_regressor_train_step/`** — **`dif_train_step_vector`** + **`expert_target_ema_step`** + mixture predict vs Python **`DIFRegressor`** (cold hash then **`score_matrix_use_field`** argmax = **`infer()`** routing; **`replay_ratio>0`** + **`replay_u01`** / **`TrainStepExtras`**); CTest **`native_dif_regressor_train_step`**. Pytest **`tests/test_dif_regressor_train_step_native_parity.py`**; env **`CYPHA_DIF_REGRESSOR_TRAIN_STEP_PARITY_BIN`**. |
| **`preprocess_train_classify_parity`** | M3 | **`studio_trainer_preprocess_classify_hotpath/`** — `preprocessor.json` + raw **`x_raw`** (or **`csv_preprocess_classify_hotpath/`** + **`train.csv`** / **`csv_spec`** → **`load_csv_dense`**) → **`transform_one`** → **`dif_train_classify_sequence`** + **`batch_llr_from_x`**. CTests **`native_studio_trainer_preprocess_classify_hotpath`**, **`native_csv_preprocess_classify_hotpath`**. **`studio_trainer_preprocess_gh_classify_hotpath/`** — same tool, sidecar **`use_gh: true`** → **`dif_gh_train_classify_sequence`**; CTest **`native_studio_trainer_preprocess_gh_classify_hotpath`**. Pytest **`tests/test_studio_trainer_preprocess_classify_hotpath_native_parity.py`**, **`tests/test_csv_preprocess_classify_hotpath_native_parity.py`**, **`tests/test_studio_trainer_preprocess_gh_classify_hotpath_native_parity.py`**; env **`CYPHA_PREPROCESS_TRAIN_CLASSIFY_PARITY_BIN`**. |
| **`nig_adapt_parity`** | M5 | `nig_adapt_session_chi` vs Cypha `_nig_adapt` (3 fixed cases); CTest **`native_nig_adapt`**. Pytest **`tests/test_nig_adapt_native_parity.py`**; env **`CYPHA_NIG_ADAPT_PARITY_BIN`**. |
| **`train_step_vector_parity`** | M3 | `parity_fixtures/train_step_vector/` — one `dif_train_step_vector` loss vs Python `train_step`; CTest **`native_train_step_vector`**. Pytest **`tests/test_train_step_vector_native_parity.py`**; env **`CYPHA_TRAIN_STEP_VECTOR_PARITY_BIN`**. |
| **`quantile_dif_train_parity`** | M3 | `parity_fixtures/quantile_dif_train/` (`replay_ratio=0`), **`dif_train_replay/`** (`replay_ratio>0` + `replay_u01`), **`studio_trainer_classify_hotpath/`** (Studio **`Trainer.fit`** + `enc_lr>0` + `replay_u01`), **`studio_trainer_gh_classify_hotpath/`** (`use_gh` + **`dif_gh_train_classify_sequence`**) — multi-step train + `batch_llr_from_x` vs Python |
| **`mke_train_step_parity`** | M4–M5 | `parity_fixtures/mke_train_step/` — one **`MKERegressor.train_step`**: RFF φ, **`score_matrix_use_field(φ)`** (matches Python `_route`), expert RLS, **`dif_train_step_vector`** (`enc_lr=0`, `replay_ratio=0`); CTest **`native_mke_train_step`**. **`parity_fixtures/mke_train_extended/`** — multi-step (**`steps`**), **`replay_warmup`** + **`replay_u01`**, **`enc_lr>0`**, **`replay_ratio>0`**; CTest **`native_mke_train_extended`** |
| **`regression_mixture_parity`** | M4 | Fixed scalar mixture — `predict_mixture_scalar` vs reference values (`DIFRegressor.predict` d=1); CTest **`native_regression_mixture`**. Pytest **`tests/test_regression_mixture_native_parity.py`**; env **`CYPHA_REGRESSION_MIXTURE_PARITY_BIN`**. |
| **`regression_m4_parity`** | M4–M6 | MoE batch + EMA + RLS + two-stage combine + **`MKERegressor`** routing softmax / scalar predict vs `parity_fixtures/regression_m4/sidecar.json` (**`native_regression_milestone()` ≥ 5**; library may report **6**) |
| **`regression_two_stage_pipeline_parity`** | M5 | **`two_stage_dif_predict_with_clf`**: native LLR from **`reference.cypha`** + stage-2 RFF vs `parity_fixtures/two_stage_pipeline/sidecar.json` |
| **`regression_two_stage_ridge_fit_parity`** | M6–M7 | **`two_stage_dif_ridge_fit_from_llr`** + **`two_stage_dif_predict_batch`** vs `parity_fixtures/two_stage_ridge_fit/sidecar.json` or **`two_stage_e2e_ridge/`** (quantile-DIF LLR); CTests **`native_regression_two_stage_ridge_fit`**, **`native_regression_two_stage_e2e_ridge`** (**`k_native_regression_milestone` ≥ 7**) |
| **`regression_rff_parity`** | M4 | **`RFFRegressor` / `MKERegressor` math kernels:** `rff_encode_batch_rowmajor`, `ridge_fit_bias`, `linear_predict_with_bias`, `mke_expert_linear_dots` (+ mixture sanity) vs `parity_fixtures/rff_regression/sidecar.json` |
| **`registry_register`** | M5 | Copy **`model.cypha`** + **`card.json`** (+ optional **`--pre preprocessor.json`**) into `<root>/<name>/<version>/`; **`--and-verify`** runs **`registry_scan`**. CTest **`native_registry_register`**. Pytest **`tests/test_registry_register_native_parity.py`**; env **`CYPHA_REGISTRY_REGISTER_BIN`**. |
| **`cypha_rest`** | M5+ | **CyphaDIF** routes mirroring [`cypha_studio/server/api.py`](../cypha_studio/server/api.py): **`/health`**, **`/ready`**, **`/metrics`**, **`/predict`**, **`/update`** (native `dif_train_step_vector` / GH wrapper + sync), **`/adapt_temperature`**, **`/session`**, **`DELETE /session`**, **`/classes`**, **`/models`**, **`/load`**, **`/register`**. Optional **`--regression-json`** / `regression_head.json` → `/predict` **`regression_val`** + **`uncertainty`** (scalar MoE; optional **`mke`** block → RFF routing + **`POST /update`** with **`regression_y`**). **CyphaLM** routes (same process): **`POST /lm/load`**, **`GET /lm/metrics`**, **`POST /lm/predict_next`**, **`POST /generate`**, **`POST /generate/stream`**. **Branch A** router (optional **`--branch-a-json`** / **`CYPHA_BRANCH_A_CHECKPOINT`**): **`GET /route/health`**, **`POST /route/text`**, **`POST /route/generate`**, **`POST /route/save`**. See [`PORT_CONTRACT.md`](../docs/port/PORT_CONTRACT.md) §3. Smoke: **`tests/test_cypha_rest_smoke.py`**, **`tests/test_cyphalm_rest_lm_smoke.py`**. |
| **`experiment_db_smoke`** | M6 | **Optional** — needs **Python 3** at configure for DDL. **SQLite:** system **`find_package(SQLite3)`** or default **`CYPHA_FETCH_SQLITE3_AMALGAMATION=ON`** (downloads official amalgamation). Uses **`cypha/experiment_db.hpp`** (`ExperimentDb`, `experiment_sqlite_exec`, …). CTests **`native_experiment_db_smoke`** / **`native_experiment_db_file`**. Pytest **`tests/test_experiment_db_smoke_native_parity.py`** (subprocess parity for both modes; DDL via **`tests/experiment_schema_ddl.py`**), **`tests/test_experiment_native_seed.py`** (Python reads native-seeded file); env **`CYPHA_EXPERIMENT_DB_SMOKE_BIN`**. |
| **`experiment_db_crud_parity`** | M6 | **`cypha/experiment_db_crud.hpp`** — insert/finish, append metrics, fail/delete, get/list, best/leaderboard, **`compare_runs`**, **`update_run_notes`** vs canonical DDL; CTest **`native_experiment_db_crud`**. Pytest **`tests/test_experiment_db_crud_native_parity.py`** (DDL from **`experiment.py`** **`_SCHEMA`** via AST — no **`numpy`** import); env **`CYPHA_EXPERIMENT_DB_CRUD_PARITY_BIN`**. |
| **`cypha_qt_stub`** | M5 | Optional Qt6 **Core** + **`cypha_core`**: optional arg **`reference.cypha`** → **`QFile`** → **`load_cypha_from_buffer`**. **`${BUILD_DIR}/qt/cypha_qt_stub`**. CTest **`native_qt_stub_load_reference`**. Pytest **`tests/test_qt_stub_native.py`**; env **`CYPHA_QT_STUB_BIN`**. |
| **`cypha_qt_shell`** | M5–M6 | Qt6 **Widgets** + **Network** + **`cypha_core`**: **Dataset panel** — column picker (`QComboBox` target + `QListWidget` feature checkboxes), raw CSV preview table (first 8 rows), val-split % hold-out with post-train accuracy eval; **Fit preprocessor dialog** — scale on/off, PCA dim, fit via `fit_from_design_matrix`, save `preprocessor.json` (no Python needed); train CSV + REST/native bulk; **training progress panel** (per-class accuracy + rolling stats); loss chart REST vs native + optional EMA + **PNG/SVG/CSV** export (optional **`-DCYPHA_QT_CHARTS=ON`**); **`POST /predict`** **`return_explanation`**; **save `.cypha`** (merge + infer snapshot incl. **`feat_dim`**, context, **`mid_trans`**, **`field_W_T`**, **`field_a_eff`**); **train hparams** + auto **`train_hparams.json`**; **`replay_u01`**; MKE regressor loop + `regression_y` bulk; registry + **`POST /load`**; **`GET /health`**, **`/ready`**, **`/models`**; Experiments DB panel (M6: open `.db`, start/finish runs, list table); spawn **`cypha_rest`**; **`--smoke`**. CTest **`native_qt_shell_smoke`**. Pytest **`tests/test_qt_shell_native.py`**. See [`qt/README.md`](qt/README.md). |

### CyphaLM parity

| Binary | Role |
|--------|------|
| **`cyphalm_parity`** | Full CyphaLM parity suite. CTest **`native_cyphalm_parity_suite`**. **`CYPHALM_PARITY_BIN`**. |
| **`cyphalm_model_parity`** | Model forward vs reference. CTest **`native_cyphalm_model_parity`**. |
| **`cyphalm_ssm_parity`** | HierSSM / SSM kernels. CTest **`native_cyphalm_ssm`**, **`native_cyphalm_ssm_fixture`**. |
| **`cyphalm_hebbian_parity`** | Hebbian update step. CTest **`native_cyphalm_hebbian`**. |
| **`cyphalm_char_lstm_parity`** | Char-LSTM forward. CTest **`native_cyphalm_char_lstm`** (disabled until fixture). **`CYPHALM_CHAR_LSTM_PARITY_BIN`**. |
| **`cyphalm_checkpoint_parity`** | Checkpoint save/load roundtrip. CTest **`native_cyphalm_checkpoint_parity`** (disabled until fixture). **`CYPHALM_CHECKPOINT_PARITY_BIN`**. |
| **`embed_table_parity`** | Izaac embed table vs `parity_fixtures/embed_table/sidecar.json`. CTest **`native_embed_table`** (disabled until fixture). **`CYPHA_EMBED_TABLE_PARITY_BIN`**. |

**Qt:** [`cmake -DCYPHA_BUILD_QT=ON`](qt/README.md) builds **`cypha_qt_stub`** (Core) and **`cypha_qt_shell`** (Widgets). Optional **`-DCYPHA_QT_CHARTS=ON`** links Qt Charts for the shell loss widget when **`Qt6::Charts`** is installed. **GitHub CI** (four **blocking** jobs): **`build_and_test`** installs **`qt6-base-dev`** (Charts off), passes **`-DCYPHA_BUILD_QT=ON`**, and runs **`native_qt_stub_load_reference`**; headless Linux excludes GUI exec tests **`native_qt_shell_smoke`**. **`windows_cuda_msvc`** and **`linux_cuda`** compile with **`-DCYPHA_ENABLE_CUDA=ON`** and run **`native_cuda_smoke`** / **`native_score_batch`** (see [`ACCEL_CUDA.md`](../docs/native/ACCEL_CUDA.md)). Local **`scripts/ci_native_linux.sh`** defaults Qt OFF unless **`CYPHA_BUILD_QT=1`**; optional **`CYPHA_QT_CHARTS=1`** passes **`-DCYPHA_QT_CHARTS=ON`** (install **`qt6-charts-dev`** first).

## Build

```bash
cd native
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --output-on-failure   # includes native_train_step_vector, native_regression_mixture, …
```

From repo root, **`pytest tests/test_native_ctest_pytest_registry.py`** checks that every **`NAME native_*`** in **`native/CMakeLists.txt`** still maps to a subprocess pytest module (update **`_NATIVE_CTEST_TO_PYTEST`** when you add CTests).

Linux/WSL one-liner matching CI’s native step: **`bash scripts/ci_native_linux.sh`** (optional **`CYPHA_NATIVE_BUILD_DIR`** / **`CMAKE_BUILD_TYPE`**). When **`python3 -m pytest`** is available, the script also runs **`tests/test_native_ctest_pytest_registry.py`** (CMake vs subprocess drift guard); set **`SKIP_NATIVE_CTEST_REGISTRY_PYTEST=1`** to skip.

### Build and test from Windows via WSL (Linux ELF)

Use WSL’s GCC/CMake when the Windows host has no toolchain, or when you want Linux binaries on `/mnt/c/...`:

```powershell
# From repo root (adjust path if needed)
powershell -ExecutionPolicy Bypass -File scripts/build_native_wsl.ps1
```

Options: **`-SkipTests`**, **`-ConfigureOnly`**, **`-CtestRegex native_experiment_db`**, **`-BuildType Debug`**. Output goes to **`native/build-wsl/`** (ignored by git).

Run a built tool manually (ELF — must execute inside WSL):

```powershell
wsl -e bash -lc "/mnt/c/Users/you/path/to/Cypha/native/build-wsl/experiment_db_crud_parity /mnt/c/Users/you/path/to/Cypha/native/build-wsl/experiment_ddl.sql"
```

**Install SQLite (development files) — copy/paste**

- **WSL / Ubuntu / Debian:** `sudo apt-get update && sudo apt-get install -y libsqlite3-dev`
- **Fedora:** `sudo dnf install -y sqlite-devel`
- **macOS (Homebrew):** `brew install sqlite`
- **Windows (vcpkg, x64):** `vcpkg install sqlite3:x64-windows` then configure CMake with `-DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake`

*(There is no package named “Cslite” — you want **SQLite**.)*

### SQLite for **`experiment_db_smoke`** (optional)

CMake uses **`find_package(SQLite3)`** first. If it is missing and **`CYPHA_FETCH_SQLITE3_AMALGAMATION`** is **ON** (default), CMake **downloads** the official **SQLite 3.47.2 amalgamation** at configure time (needs network) and builds static **`cypha_sqlite3_amalg`** — no **`libsqlite3-dev`** / vcpkg required. Set **`-DCYPHA_FETCH_SQLITE3_AMALGAMATION=OFF`** to disable (then install a system SQLite3 dev package or skip the target).

The Python export script still runs at **configure** time, so you need **Python 3** on the PATH.

When using a **system** SQLite3, you still need the **library + headers** (not only the `sqlite3` CLI).

| Environment | Install |
|-------------|---------|
| **Ubuntu / Debian / WSL** | `sudo apt-get install -y libsqlite3-dev` |
| **Fedora / RHEL** | `sudo dnf install sqlite-devel` |
| **macOS (Homebrew)** | `brew install sqlite` — then re-run CMake; if not found, pass `-DCMAKE_PREFIX_PATH="$(brew --prefix sqlite)"` |
| **Windows (MSVC)** | **Default:** CMake can **fetch** the amalgamation (**network** at configure). **Or** **vcpkg** `sqlite3:x64-windows` + `-DCMAKE_TOOLCHAIN_FILE=.../vcpkg.cmake` for **`find_package(SQLite3)`**. |

With **no** system SQLite3 dev package, **`CYPHA_FETCH_SQLITE3_AMALGAMATION=ON`** (default) is enough. If you install **libsqlite3-dev** / vcpkg / Homebrew sqlite, CMake prefers **`SQLite::SQLite3`** and skips the download.

After a successful configure, the build includes **`experiment_db_smoke`** and CTests **`native_experiment_db_smoke`** + **`native_experiment_db_file`**.

**Pytest on Windows:** parity tools built as **Linux ELF** under **`native/build*/**` or **`native/build*/qt/**` are run via **`wsl -e`** from **`tests/native_subprocess.py`**, which chooses the **newest** matching binary by modification time when several trees contain the same tool. Full **`CYPHA_*_BIN`** / **`CYPHALM_*_BIN`** map: **`scripts/cypha_native_validate_all.ps1`** (`Set-ParityEnv`) and the target tables above. Production runners: **`CYPHA_BENCH_RUN_BIN`**, **`CYPHA_TUNE_RUN_BIN`**, **`CYPHA_DIAGNOSTICS_RUN_BIN`**, **`CYPHA_REST_BIN`**. New v2.2 parity overrides: **`CYPHA_KERNEL_LLR_PARITY_BIN`**, **`CYPHA_GH_INFER_DELIBERATION_PARITY_BIN`**, **`CYPHA_MULTILABEL_DIF_PARITY_BIN`**, **`CYPHA_MERGE_FROM_PARITY_BIN`**, **`CYPHA_SIMILARITY_INDEX_PARITY_BIN`**, **`CYPHA_GENERATION_PARITY_BIN`**, **`CYPHA_RETRIEVAL_PARITY_BIN`**, **`CYPHA_EMBED_TABLE_PARITY_BIN`**, **`CYPHA_CREATE_MODEL_SMOKE_BIN`**, **`CYPHALM_BENCH_NATIVE_BIN`**, **`CYPHALM_PARITY_BIN`**.

**Full validation (one command):** from repo root, **`powershell -File scripts/cypha_native_validate_all.ps1`** — Release build outside OneDrive, **`ctest -R native_`**, pytest parity + REST, bench smoke (d01/d04/d17), tune dry-run. See [`docs/native/NATIVE_QUICKSTART.md`](../docs/native/NATIVE_QUICKSTART.md).

Manual runs:

**Python smoke:** from repo root, after building `cypha_rest`, run `CYPHA_REST_BIN=native/build/cypha_rest pytest tests/test_cypha_rest_smoke.py -v` (on Windows set the path to your `cypha_rest.exe`).

```bash
./build/cypha_parity ../parity_fixtures/reference.cypha ../parity_fixtures/native_parity.bin
./build/memory_train_parity ../parity_fixtures/memory_train
./build/preprocessor_parity ../parity_fixtures/preprocessor
./build/cypha_rest --listen 127.0.0.1:8099 \
  --cypha ../parity_fixtures/reference.cypha \
  --f-field-json ../parity_fixtures/f_field.json
# Optional: --train-hparams path.json (else auto-loads train_hparams.json next to model.cypha).
# Optional keys: `align_every` (default 500), `temp_recalib_every` (default 0) — temperature auto-recal every N `/update` steps when > 0.
# curl -s http://127.0.0.1:8099/ready
# curl -s http://127.0.0.1:8099/metrics
# curl -s http://127.0.0.1:8099/predict -H 'Content-Type: application/json' \
#   -d '{"input":[0,0,0,0,0,0,0,0],"use_gh":true,"return_explanation":false}'
# Registry + hot load: copy f_field.json into each `<root>/<name>/<version>/` next to model.cypha, then:
# ./build/cypha_rest ... --registry ~/.cypha/models
# curl -s http://127.0.0.1:8099/models?summary=true
# curl -s http://127.0.0.1:8099/load -H 'Content-Type: application/json' -d '{"name":"my","version":"latest"}'
```

On Windows, link **`ws2_32`** for `cypha_rest` (already in CMake).

### Cross-compile Windows `.exe` from WSL (MinGW-w64)

Requires: `g++-mingw-w64-x86-64` (e.g. `sudo apt-get install -y g++-mingw-w64-x86-64`).

**CMake layout:** MinGW-specific options and link flags live in **`native/cmake/CyphaMinGW.cmake`** (included from **`native/CMakeLists.txt`**). Toolchain file: **`native/toolchains/mingw-w64-x86_64.cmake`** (cache **`CYPHA_MINGW_TOOLCHAIN_PREFIX`**, default **`x86_64-w64-mingw32`**, for non-Debian triplet layouts).

**Cache toggles (MinGW targets only):**

- **`CYPHA_MINGW_STATIC_CXX_RUNTIME`** (default **ON**) — **`-static-libgcc -static-libstdc++`**
- **`CYPHA_MINGW_FULLY_STATIC_EXECUTABLES`** (default **OFF**) — add **`-static`** (fully static where linking allows)

```bash
cd native
cmake --preset mingw-w64-cross
cmake --build --preset mingw-w64-cross-release
ctest --preset mingw-w64-cross
```

Equivalent manual configure (from repo root; use an absolute toolchain path if CMake cannot resolve this file):

```bash
cmake -S native -B native/build-mingw-w64 \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/native/toolchains/mingw-w64-x86_64.cmake" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build native/build-mingw-w64 -j$(nproc)
cmake --test-dir native/build-mingw-w64 --output-on-failure
```

Or: `bash native/scripts/build-windows-mingw.sh`

Outputs **`*.exe`** under `native/build-mingw-w64/`. With defaults, MinGW links **`-static-libgcc -static-libstdc++`** so binaries usually run on a stock Windows install without those runtime DLLs on **PATH**.

**Windows pytest** auto-discovers **`native/build-mingw-w64/cypha_rest.exe`** for **`tests/test_cypha_rest_smoke.py`** (before MSVC **`native/build/`** paths). Set **`CYPHA_REST_BIN`** only to override.

When the repo lives under **`/mnt/c/...`**, CMake rewrites parity fixture paths to **`C:/...`** for `ctest` so Windows can open them.

**Console check (Windows `cmd`):**

```bat
cd native\build-mingw-w64
cypha_parity.exe ..\..\parity_fixtures\reference.cypha ..\..\parity_fixtures\native_parity.bin
```

(`..\\..\\parity_fixtures` from `native\build-mingw-w64`; or use absolute `C:\...\parity_fixtures\...`.)

**REST smoke (Windows, MinGW binary):** `powershell -File native/scripts/smoke_cypha_rest_mingw.ps1` (add `-WithRegression` to assert `/predict` **`regression_val`** with `parity_fixtures/regression_head.json`).

**One-shot MinGW build from PowerShell + optional pytest:** `powershell -File native/scripts/build_cypha_rest_mingw_wsl.ps1` (add `-RunPytest` to set **`CYPHA_REST_BIN`** and run `tests/test_cypha_rest_smoke.py`).

## Regenerate fixtures

```bash
python scripts/generate_parity_fixtures.py
python scripts/generate_memory_train_parity.py
python scripts/generate_preprocessor_parity.py
python scripts/export_f_field_json.py
python scripts/export_regression_head.py   # demo DIFRegressor → regression_head.json (MoE sidecar)
python scripts/gen_native_bessel_table.py   # K₂/K₁ + K₀/K₁ grids if SciPy kv tables change
```

## Specs

- [`NATIVE_QUICKSTART.md`](../docs/native/NATIVE_QUICKSTART.md) — install, validate, bench, tune, REST (v2.2)  
- [`PORT_CONTRACT.md`](../docs/port/PORT_CONTRACT.md) — `.cypha` v3, inference math, REST JSON, native bench §6  
- [`PREPROCESSOR_CONTRACT.md`](../docs/port/PREPROCESSOR_CONTRACT.md) + [`schemas/preprocessor.schema.json`](../docs/port/schemas/preprocessor.schema.json)  
- [`schemas/regression_head.schema.json`](../docs/port/schemas/regression_head.schema.json) — optional MoE sidecar for `/predict` **`regression_val`**  
- [`EXPERIMENTS_SCHEMA.md`](../docs/port/EXPERIMENTS_SCHEMA.md) — SQLite (M6). DDL: `python scripts/export_experiment_schema_sql.py` (or `-o file.sql`). Native **`experiment_db_smoke`** + **`experiment_db_crud_parity`** (CTests **`native_experiment_db_smoke`** / **`native_experiment_db_crud`**) validate DDL and core run/metrics CRUD when SQLite is available; see [`PORT_FULL_STACK.md`](../docs/port/PORT_FULL_STACK.md) M6 for any remaining optional API surface.  
- [`regression_stub.hpp`](include/cypha/regression_stub.hpp) — M4 placeholder  

## Next engineering waves

- **CUDA accel** — `-DCYPHA_ENABLE_CUDA=ON` + NVIDIA toolkit/driver; see **`cuda_smoke`** and presets above. See [`docs/FUTURE.md`](../docs/FUTURE.md) §1.  
- **Qt shell streaming** — move bulk training to a `QThread`; emit per-step loss/accuracy signals; live loss chart update during training. See [`docs/FUTURE.md`](../docs/FUTURE.md) §2a.  
- **Packaged binary** — AppImage (Linux) or `windeployqt` folder / `.msi` (Windows) distributing Qt shell as a self-contained executable. See [`docs/FUTURE.md`](../docs/FUTURE.md) §3.  
- **REST multi-model** — `cypha_rest --registry <root>` serving N models; per-model mutex; LRU eviction. See [`docs/FUTURE.md`](../docs/FUTURE.md) §5.  
- **CyphaLM native inference** — C++ decode path for `.cypha` LM checkpoints (generation + REST parity with Python); training remains Python-only. See [`docs/port/PORT_CONTRACT.md`](../docs/port/PORT_CONTRACT.md) §4.  
- **Full future directions** — Web UI, curriculum/active learning, ONNX export, federated training: [`docs/FUTURE.md`](../docs/FUTURE.md).  
