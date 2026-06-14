# Native runtime (C++ / CUDA / Qt)

Monorepo C++ core for [`docs/port/PORT_FULL_STACK.md`](../docs/port/PORT_FULL_STACK.md). **Vendored:** `third_party/nlohmann/json.hpp`, `third_party/httplib.h` (OpenSSL is optional — see **Federated TLS** below).

**Accel** (`cypha/accel_backend.hpp`): optional **CUDA** (`-DCYPHA_ENABLE_CUDA=ON`, NVIDIA toolkit + driver); otherwise **ISO C++** parallel CPU via `std::thread`. **`cuda_smoke`** checks correctness vs a serial reference; **`cuda_smoke --bench`** compares CUDA vs CPU when a GPU is present (exit 2 skip otherwise).

## Repo layout (paths native code resolves)

| Path | Role |
|------|------|
| **`fixtures/`** | Committed parity goldens (`.cypha`, `sidecar.json`, etc.) for CTest and **`cypha_diagnostics_run`**. |
| **`bench/`** | Native benchmark tree: **`config/`** (profiles + sweep JSON), **`data/`**, **`report/`** (tables, figures, **`BASELINE_REPORT.md`**), **`artifacts/`**. |
| **`native/`** | This C++ tree (CMake source root). |

**Repo root discovery:** `cypha::bench::find_repo_root()` walks upward from the current directory (or **`CYPHA_REPO_ROOT`** if set) until **`fixtures/`**, **`bench/`**, and **`native/`** all exist as directories; otherwise it falls back to the compile-time path derived from `native/src/bench/bench_paths.cpp`. Bench helpers (`bench_root()`, `config_dir()`, …) resolve under **`bench/`**.

## Native source layout (Phase B)

| Path | Role |
|------|------|
| **`include/`** | Public headers (`cypha/…`, `cypha/cyphalm/…`) — unchanged install / include paths |
| **`src/`** | Library implementation (`cypha_core`, `cyphalm/*`, `bench/*`, …) |
| **`apps/`** | Production entrypoint sources (binary names unchanged) |
| **`tests/parity/`** | Python-vs-native parity harness sources (`*_parity.cpp`, `parity_main.cpp`) |
| **`tools/`** | Dev smoke, roundtrip, fixture-gen, and LM bench CLIs not tied to CTest parity |
| **`cmake/`** | `CyphaMinGW.cmake`, `CyphaParity.cmake`, `CyphaApps.cmake` |

### Libraries (`add_library`)

| Target | Sources |
|--------|---------|
| **`cypha_core`** | `src/*.cpp`, `src/som/*`, `src/intelligence/*` — CyphaDIF runtime |
| **`cypha_lm_native`** | `src/cyphalm/*.cpp` — CyphaLM |
| **`cypha_bench_native`** | `src/bench/*.cpp` — benchmark engine |

### Apps (`apps/` → executables)

| Binary | Sources |
|--------|---------|
| **`cypha_rest`** | `apps/cypha_rest.cpp` + `cyphalm_rest_routes.cpp` + `branch_a_rest_routes.cpp` + `dif_rest_routes.cpp` |
| **`cypha_bench_run`** | `apps/cypha_bench_run.cpp` |
| **`cypha_tune_run`** | `apps/cypha_tune_run.cpp` |
| **`cypha_diagnostics_run`** | `apps/cypha_diagnostics_run.cpp` |
| **`cyphalm_train`** | `apps/cyphalm_train.cpp` |

(`cypha_bench_report` also lives under **`apps/`**; same CMake grouping as the runners above.)

### Parity harness (`tests/parity/`)

One executable per `*_parity.cpp` (plus **`parity_main.cpp`** → **`cypha_parity`**). Built via **`cmake/CyphaParity.cmake`**; CTest names and output binary names are unchanged. Remaining dev tools (roundtrip, smoke, **`cypha_fixture_gen`**, **`cyphalm_bench_native`**, …) stay under **`tools/`**.

## CMake presets (Windows + WSL trees)

| Preset | Binary directory | Use |
|--------|------------------|-----|
| **`windows-msvc-release`** | `native/build-windows-msvc/` | Native Windows: Visual Studio **2022** generator, x64. `cmake --preset windows-msvc-release` then `cmake --build build-windows-msvc --config Release`. |
| **`windows-vs2026-release`** | `native/build-windows-vs2026/` | Visual Studio **18 2026** / Build Tools 18 (MSVC 14.5x). Same workflow; use if preset **windows-msvc-release** fails (no VS 2022). |
| **`wsl-gcc-release`** | `native/build-wsl-gcc/` | WSL/Linux: **Ninja** + Release. Needs `ninja-build` (or use **`wsl-gcc-release-make`** for Makefiles). |
| **`mingw-w64-cross`** | `native/build-mingw-w64/` | Cross-compile `.exe` from Linux/WSL (no CUDA). |

**Qt shell on Windows:** after a Release build with **`-DCYPHA_BUILD_QT=ON`** and **`-DCMAKE_PREFIX_PATH=`** pointing at your Qt **msvc*_64** kit, run **`windeployqt`** on `qt/Release/cypha_qt_shell.exe`, then from the repo root: **`powershell -ExecutionPolicy Bypass -File scripts/run_cypha_qt_windows.ps1`** (starts **`cypha_rest`** with `fixtures/reference.cypha` and opens **`cypha_qt_shell`**). Use **`-NoServer`** for GUI only.

```bash
cd native
cmake --list-presets
cmake --preset wsl-gcc-release
cmake --build --preset wsl-gcc-release-build
ctest --test-dir build-wsl-gcc --output-on-failure
```

**CUDA (Windows or WSL with NVIDIA):** add `-DCYPHA_ENABLE_CUDA=ON` at configure (requires `nvcc` + `CUDA::cudart`). Override arch: `-DCMAKE_CUDA_ARCHITECTURES=89` (Ada), `86` (Ampere), etc. Default **`CYPHA_ACCEL_GPU_MIN_BATCH_ROWS=1`** (CUDA for all batch sizes n≥1 when a GPU is available). Optional: `-DCYPHA_ACCEL_GPU_MIN_BATCH_ROWS=N` to raise the threshold so smaller batches stay on CPU threads.

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
| **`cypha_bench_run`** | Master bench runner: domains **d01–d17** + **`d03_xor`** (kernel LLR), cross-domain tables, **`BASELINE_REPORT.md`**, **`report/summary.json`**, figure JSON/PNG under **`bench/report/figures/`**. CLI: `--domain N`, **`--domain-tag TAG`** (e.g. `d03_xor`), `--from-domain N`, `--report-only`, `--list-domains`. Env: **`CYPHA_REPO_ROOT`**, **`CYPHA_BENCH_FAST=1`**. CTest **`native_bench_run_list_domains`**. Contract: [`PORT_CONTRACT.md`](../docs/port/PORT_CONTRACT.md) §6. |
| **`cypha_bench_report`** | Report-only rebuild (same helpers as **`--report-only`**). CLI: `--output PATH`, `--skip-cross-domain`. |
| **`cypha_tune_run`** | Native tuning/sweep orchestrator. Loads sweep JSON, invokes **`cyphalm_bench_native`** or **`cypha_bench_run`** per cell. CLI: `--config PATH`, `--out PATH`, `--exe-dir DIR`, `--max-cells N`, `--write`, `--dry-run`. CTest **`native_tune_run_smoke`**. |
| **`cypha_diagnostics_run`** | Phases 1–4 validation orchestrator: runs parity exes + inline **`cypha_core`** checks (no sklearn). CLI: `--fixtures DIR`, `--out DIR`, `--exe-dir DIR`, `--phases 1,2,3,4`, `--list`, `--inline-only`. CTest **`native_diagnostics_run`**. |
| **`cyphalm_bench_native`** | Per-profile LM BPC bench (char-LSTM / hybrid / ablations). Used by **`cypha_bench_run`** d04/d17 and **`cypha_tune_run`**. |
| **`cyphalm_train`** | Train CyphaLM from corpus text via native **`train_sequence`**, save Python-compatible checkpoint (`checkpoint.json` + `.npz`). CLI: `--profile {d17,d04}`, `--corpus bench/data/...`, `--epochs N`, `--out checkpoint_dir/`, optional `--synthetic-tokens N` (smoke), `--max-train-steps S`, `--threads T`. CTest **`native_cyphalm_train_smoke`**. |

### Core parity & smoke

| Binary / lib | Milestone | Role |
|--------------|-----------|------|
| **`cypha_core`** | M1–M7 | `.cypha` **load** + **`load_cypha_from_buffer`** / **`save_cypha_file`** / **`save_cypha_to_buffer`** / **`clone_cnode`** (Python **`cypha_save_binary`** / **`cypha_load_binary`** / **`cypha_save_binary_to_bytes`** / **`cypha_load_binary_from_bytes`** v3 layout), `mid_trans`, `llr_scale_*`, `field_W_T`, optional **`field_a_eff`**, `w_inject`, …, CPU infer, full **Tier-1+2** `context_prior` / `context_record_step`, preprocessor JSON, `memory_train` + **`merge_state_into_root_for_save`** + `dedup_check`, `sync_infer`, replay, contrastive + deliberate + **`encoder_align_to_offsets`**, NIG field, `dif_train_step_vector`, **`dif_train_classify_sequence`**, **`dif_gh_train_classify_sequence`** (GH online loop), **`mke_scalar_train_step`**, **`registry_scan`** + **`registry_register_bundle`**, **`create_fresh_model_root`**, **`generate_retrieval_augmented`**, **`SimilarityIndex`**, **`merge_from`**, **`MultiLabelDIF`**, kernel-memory LLR |
| **`create_model_smoke`** | M1 | Empty-model create → infer + memory load → save/reload roundtrip. CTest **`native_create_model`**. |
| **`cuda_smoke`** | Accel | **`cypha::accel`**: CUDA if `-DCYPHA_ENABLE_CUDA=ON` + GPU; else parallel CPU (`std::thread`). CTest **`native_cuda_smoke`** / **`native_cuda_bench`** (bench exit 2 without GPU). |
| **`cypha_parity`** | M1 | `reference.cypha` + `native_parity.bin` → LLR / probs / gates; **v2** sidecar tail checks **`batch_infer_full`** entropy + confidence. **`CyphaInferModel::from_root`** restores **Tier-1** from **`ctx_hist_packed`** / co-occurrence / last label; see [`PORT_CONTRACT.md`](../docs/port/PORT_CONTRACT.md) §4. CTest **`native_parity`**. |
| **`batch_llr_parity`** | M7 | **`batch_llr_from_x`** vs `fixtures/batch_llr/sidecar.json` (same **X**/**LLR** as **`expected.npz`**) |
| **`score_batch_parity`** | Accel | **`cypha::accel::batch_encode`** + **`score_matrix`** vs `fixtures/score_batch/sidecar.json` (`project_features` + `fused_score_llr` goldens). CTest **`native_score_batch`**. |
| **`kernel_llr_parity`** | M1+ | Whitened Nyström **`KernelMemory`** vs `fixtures/kernel_llr/sidecar.json`. CTest **`native_kernel_llr`**. |
| **`kernel_snapshot_roundtrip`** | P2 | C++ `export_snapshot` / `import_snapshot` smoke. CTest **`native_kernel_snapshot_roundtrip`**. |
| **`xor_kernel_bench`** | P2 | XOR linear vs kernel LLR; Python-matched hyperparams (`world_lr=0.008`, `delta_lr=0.05`, `ood_sigma=15`, `temperature=1.15`, NumPy `permutation`). CTest **`native_xor_kernel_bench_smoke`**. |
| **`gh_infer_deliberation_parity`** | M1+ | GH infer + deliberation gate vs `fixtures/gh_infer_deliberation/`. CTest **`native_gh_infer_deliberation`**. |
| **`multilabel_dif_parity`** | M3+ | **`MultiLabelDIF`** train/infer vs `fixtures/multilabel_dif/`. CTest **`native_multilabel_dif`**. |
| **`merge_from_parity`** | M3+ | **`merge_from`** state fusion vs `fixtures/merge_from/`. CTest **`native_merge_from`**. |
| **`similarity_index_parity`** | M3+ | **`SimilarityIndex`** retrieval index vs `fixtures/similarity_index/`. CTest **`native_similarity_index`**. |
| **`generation_parity`** | M5+ | RAG **`generate_retrieval_augmented`** vs `fixtures/generation/sidecar.json`. CTest **`native_generation`**. |
| **`retrieval_parity`** | M5+ | Dense retrieval top-k vs `fixtures/retrieval/sidecar.json`. CTest **`native_retrieval`** (disabled until fixture exists). |
| **`memory_train_parity`** | M3 | `fixtures/memory_train/` — one `DIFMemory.train` step vs `after.cypha`; CTest **`native_memory_train`**. |
| **`memory_train_roundtrip`** | M3 | Same fixture: train → **`merge_state_into_root_for_save`** → **`patch_field_a_eff_into_root`** (aligns with Python **`field_a_eff`**) → **`save_cypha_to_buffer`** (bytes) + **`save_cypha_file`** → on-disk bytes **`memcmp`** with buffer → **`load_cypha_file`** / **`load_cypha_from_buffer`** vs file reload. Tree ≈ **`after.cypha`** (CTest **`native_memory_train_roundtrip`**). |
| **`preprocessor_parity`** | M2 | `fixtures/preprocessor/` — `transform_one` vs Python; CTest **`native_preprocessor`**. |
| **`preprocessor_fit_parity`** | M2 | **`preprocessor_fit/`** + **`preprocessor_fit_no_scale/`** + **`preprocessor_fit_rff/`** — **`PreprocessorState::fit_from_design_matrix`** (scale on/off + PCA + RFF) vs Python **`Preprocessor.fit`** + probe **`transform_one`**; CTest **`native_preprocessor_fit`**. |
| **`csv_ingest_parity`** | M2 | **`csv_ingest/`** — **`cypha::load_csv_dense`** vs **`CSVDataset.from_file`** (**`target_col_name`** / **`feature_col_names`** and/or indices; multiline quoted fields); CTest **`native_csv_ingest`**. |
| **`dif_regressor_train_step_parity`** | M4 | **`dif_regressor_train_step/`** — **`dif_train_step_vector`** + **`expert_target_ema_step`** + mixture predict vs Python **`DIFRegressor`** (cold hash then **`score_matrix_use_field`** argmax = **`infer()`** routing; **`replay_ratio>0`** + **`replay_u01`** / **`TrainStepExtras`**); CTest **`native_dif_regressor_train_step`**. |
| **`preprocess_train_classify_parity`** | M3 | **`studio_trainer_preprocess_classify_hotpath/`** — `preprocessor.json` + raw **`x_raw`** (or **`csv_preprocess_classify_hotpath/`** + **`train.csv`** / **`csv_spec`** → **`load_csv_dense`**) → **`transform_one`** → **`dif_train_classify_sequence`** + **`batch_llr_from_x`**. CTests **`native_studio_trainer_preprocess_classify_hotpath`**, **`native_csv_preprocess_classify_hotpath`**. **`studio_trainer_preprocess_gh_classify_hotpath/`** — same tool, sidecar **`use_gh: true`** → **`dif_gh_train_classify_sequence`**; CTest **`native_studio_trainer_preprocess_gh_classify_hotpath`**. |
| **`nig_adapt_parity`** | M5 | `nig_adapt_session_chi` vs Cypha `_nig_adapt` (3 fixed cases); CTest **`native_nig_adapt`**. |
| **`train_step_vector_parity`** | M3 | `fixtures/train_step_vector/` — one `dif_train_step_vector` loss vs Python `train_step`; CTest **`native_train_step_vector`**. |
| **`quantile_dif_train_parity`** | M3 | `fixtures/quantile_dif_train/` (`replay_ratio=0`), **`dif_train_replay/`** (`replay_ratio>0` + `replay_u01`), **`studio_trainer_classify_hotpath/`** (Studio **`Trainer.fit`** + `enc_lr>0` + `replay_u01`), **`studio_trainer_gh_classify_hotpath/`** (`use_gh` + **`dif_gh_train_classify_sequence`**) — multi-step train + `batch_llr_from_x` vs Python |
| **`mke_train_step_parity`** | M4–M5 | `fixtures/mke_train_step/` — one **`MKERegressor.train_step`**: RFF φ, **`score_matrix_use_field(φ)`** (matches Python `_route`), expert RLS, **`dif_train_step_vector`** (`enc_lr=0`, `replay_ratio=0`); CTest **`native_mke_train_step`**. **`fixtures/mke_train_extended/`** — multi-step (**`steps`**), **`replay_warmup`** + **`replay_u01`**, **`enc_lr>0`**, **`replay_ratio>0`**; CTest **`native_mke_train_extended`** |
| **`regression_mixture_parity`** | M4 | Fixed scalar mixture — `predict_mixture_scalar` vs reference values (`DIFRegressor.predict` d=1); CTest **`native_regression_mixture`**. |
| **`regression_m4_parity`** | M4–M6 | MoE batch + EMA + RLS + two-stage combine + **`MKERegressor`** routing softmax / scalar predict vs `fixtures/regression_m4/sidecar.json` (**`native_regression_milestone()` ≥ 5**; library may report **6**) |
| **`regression_two_stage_pipeline_parity`** | M5 | **`two_stage_dif_predict_with_clf`**: native LLR from **`reference.cypha`** + stage-2 RFF vs `fixtures/two_stage_pipeline/sidecar.json` |
| **`regression_two_stage_ridge_fit_parity`** | M6–M7 | **`two_stage_dif_ridge_fit_from_llr`** + **`two_stage_dif_predict_batch`** vs `fixtures/two_stage_ridge_fit/sidecar.json` or **`two_stage_e2e_ridge/`** (quantile-DIF LLR); CTests **`native_regression_two_stage_ridge_fit`**, **`native_regression_two_stage_e2e_ridge`** (**`k_native_regression_milestone` ≥ 7**) |
| **`regression_rff_parity`** | M4 | **`RFFRegressor` / `MKERegressor` math kernels:** `rff_encode_batch_rowmajor`, `ridge_fit_bias`, `linear_predict_with_bias`, `mke_expert_linear_dots` (+ mixture sanity) vs `fixtures/rff_regression/sidecar.json` |
| **`registry_register`** | M5 | Copy **`model.cypha`** + **`card.json`** (+ optional **`--pre preprocessor.json`**) into `<root>/<name>/<version>/`; **`--and-verify`** runs **`registry_scan`**. CTest **`native_registry_register`**. |
| **`cypha_rest`** | M5+ | **CyphaDIF** REST routes (see [`PORT_CONTRACT.md`](../docs/port/PORT_CONTRACT.md) §3): **`/health`**, **`/ready`**, **`/metrics`**, **`/predict`**, **`/update`**, **`/adapt_temperature`**, **`/session`**, **`DELETE /session`**, **`/classes`**, **`/models`**, **`/load`**, **`/register`**. Optional **`--regression-json`**. **CyphaLM** routes: **`POST /lm/load`**, **`GET /lm/metrics`**, **`POST /lm/predict_next`**, **`POST /generate`**, **`POST /generate/stream`**. **Branch A** router (optional **`--branch-a-json`**): **`GET /route/health`**, **`POST /route/text`**, **`POST /route/generate`**, **`POST /route/save`**. Smoke: **`native/scripts/smoke_cypha_rest_mingw.ps1`**. |
| **`experiment_db_smoke`** | M6 | **Optional** — uses embedded experiment DDL at configure time. **SQLite:** system **`find_package(SQLite3)`** or default **`CYPHA_FETCH_SQLITE3_AMALGAMATION=ON`** (downloads official amalgamation). Uses **`cypha/experiment_db.hpp`** (`ExperimentDb`, `experiment_sqlite_exec`, …). CTests **`native_experiment_db_smoke`** / **`native_experiment_db_file`**. |
| **`experiment_db_crud_parity`** | M6 | **`cypha/experiment_db_crud.hpp`** — insert/finish, append metrics, fail/delete, get/list, best/leaderboard, **`compare_runs`**, **`update_run_notes`** vs canonical DDL; CTest **`native_experiment_db_crud`**. |
| **`cypha_qt_stub`** | M5 | Optional Qt6 **Core** + **`cypha_core`**: optional arg **`reference.cypha`** → **`QFile`** → **`load_cypha_from_buffer`**. **`${BUILD_DIR}/qt/cypha_qt_stub`**. CTest **`native_qt_stub_load_reference`**. |
| **`cypha_qt_shell`** | M5–M6 | Qt6 **Widgets** + **Network** + **`cypha_core`**: **Dataset panel** — column picker (`QComboBox` target + `QListWidget` feature checkboxes), raw CSV preview table (first 8 rows), val-split % hold-out with post-train accuracy eval; **Fit preprocessor dialog** — scale on/off, PCA dim, fit via `fit_from_design_matrix`, save `preprocessor.json` (no Python needed); train CSV + REST/native bulk; **training progress panel** (per-class accuracy + rolling stats); loss chart REST vs native + optional EMA + **PNG/SVG/CSV** export (optional **`-DCYPHA_QT_CHARTS=ON`**); **`POST /predict`** **`return_explanation`**; **save `.cypha`** (merge + infer snapshot incl. **`feat_dim`**, context, **`mid_trans`**, **`field_W_T`**, **`field_a_eff`**); **train hparams** + auto **`train_hparams.json`**; **`replay_u01`**; MKE regressor loop + `regression_y` bulk; registry + **`POST /load`**; **`GET /health`**, **`/ready`**, **`/models`**; Experiments DB panel (M6: open `.db`, start/finish runs, list table); spawn **`cypha_rest`**; **`--smoke`**. CTest **`native_qt_shell_smoke`**. |

### CyphaLM parity

| Binary | Role |
|--------|------|
| **`cyphalm_parity`** | Full CyphaLM parity suite. CTest **`native_cyphalm_parity_suite`**. **`CYPHALM_PARITY_BIN`**. |
| **`cyphalm_model_parity`** | Model forward vs reference. CTest **`native_cyphalm_model_parity`**. |
| **`cyphalm_ssm_parity`** | HierSSM / SSM kernels. CTest **`native_cyphalm_ssm`**, **`native_cyphalm_ssm_fixture`**. |
| **`cyphalm_hebbian_parity`** | Hebbian update step. CTest **`native_cyphalm_hebbian`**. |
| **`cyphalm_char_lstm_parity`** | Char-LSTM forward. CTest **`native_cyphalm_char_lstm`** (disabled until fixture). **`CYPHALM_CHAR_LSTM_PARITY_BIN`**. |
| **`cyphalm_checkpoint_parity`** | Checkpoint save/load roundtrip. CTest **`native_cyphalm_checkpoint_parity`** (disabled until fixture). **`CYPHALM_CHECKPOINT_PARITY_BIN`**. |
| **`embed_table_parity`** | Izaac embed table vs `fixtures/embed_table/sidecar.json`. CTest **`native_embed_table`** (disabled until fixture). **`CYPHA_EMBED_TABLE_PARITY_BIN`**. |

### Federated merge & TLS

| Binary | Role |
|--------|------|
| **`cypha_federated_merge`** | Offline merge of worker JSON payloads via **`federated_average_payloads`**. CTest **`native_federated_merge_smoke`**. |
| **`cypha_federated_coordinator`** | Watch-dir or HTTP **`/submit`** coordinator; optional **`--listen <port> --tls-cert <pem> --tls-key <pem>`**. CTest **`native_federated_coordinator_smoke`**. |
| **`cypha_federated_worker`** | POST worker JSON to coordinator (**`--scheme http|https`**). Used with coordinator listen mode. |
| **`federated_worker_smoke`** | In-process HTTP loopback (no TLS). CTest **`native_federated_worker_smoke`**. |
| **`federated_tls_smoke`** | HTTPS loopback with self-signed cert (requires OpenSSL build + **`openssl`** CLI on PATH). CTest **`native_federated_tls_smoke`** (exit **2** = skip). |
| **`ewc_cyphalm_smoke`** | CyphaLM char-LSTM EWC on embed + recurrent + lm_head. CTest **`native_ewc_cyphalm_smoke`**. |
| **`ewc_hybrid_smoke`** | Hybrid EWC on SSM multiscale α + GRIA per-token α (B0 ngram count path when `ngram_context > 0`). CTest **`native_ewc_hybrid_smoke`**. |
| **`ewc_weights_smoke`** | Hybrid EWC weight Fisher on GRIA **U**/**V** + SSM **W_fast**; EWC anchor/Fisher checkpoint round-trip. CTest **`native_ewc_weights_smoke`**. *(Phase 9 shipped.)* |
| **`ewc_weights_smoke`** *(Phase 9–10)* | Hybrid EWC Fisher on GRIA **U/V/bias** + SSM **W_fast/W_slow**. CTest **`native_ewc_weights_smoke`**. |
| **`cypha_baseline_lock`** | Merge D17 / d21 / cell-sweep BPC into **`bench/BASELINE_LOCK.json`**; **`--run all`** chains all three lock runs; **`--medium`** for 5k train tier (`status=medium_smoke`); **`--production`** for 300k train tier (`status=production`). CTest **`native_baseline_lock_smoke`**. Wrapper: **`scripts/update_baseline_lock.ps1`**. Validator: **`scripts/validate_baseline_lock.ps1`** (`-Strict`, **`-Production`**), **`baseline_lock_validate`**. |
| **`d23_overnight_lock_smoke`** | Bench **d23** overnight lock validation wiring (`cypha_bench_run --domain-tag d23`). CTest **`native_d23_overnight_lock_smoke`**. *(Phase 9 shipped.)* |
| **`d24_production_lock_smoke`** *(Phase 10)* | Bench **d24** production lock validation wiring (`cypha_bench_run --domain-tag d24`). CTest **`native_d24_production_lock_smoke`**. |
| **`corpus_smoke`** *(Phase 11)* | Direct d17/d21 **`load_bench_corpus`** probe (WikiText or gutenberg fallback). CTest **`native_corpus_smoke`**. |
| **`d25_corpus_readiness`** *(Phase 11)* | Bench **d25** corpus readiness validation (`cypha_bench_run --domain-tag d25`); invokes **`corpus_smoke`** when built. CTest **`native_d25_corpus_smoke`**. |
| **`d26_medium_overnight`** *(Phase 12)* | Bench **d26** medium overnight lock validation (`cypha_bench_run --domain-tag d26`); runs **`cypha_baseline_lock --run d17 --medium`**, checks **`status=medium_smoke`**. CTest **`native_d26_medium_overnight_smoke`**. |
| **`d27_production_lock_smoke`** *(Phase 13)* | Bench **d27** production overnight lock validation (`cypha_bench_run --domain-tag d27`); if **`overnight_results.n_train < 300000`**, reports **`pending_production`** (smoke pass); if **≥ 300k**, validates BPC within **0.05** of d17 hybrid **2.873** pin. CTest **`native_d27_production_lock_smoke`**. |
| **`d28_overnight_complete`** *(Phase 14)* | Bench **d28** unified overnight completion validation (`cypha_bench_run --domain-tag d28`); validates schema + production tier + matching **`n_train`** / **`n_eval`** across **`overnight_results`**, **`rpsm_results`**, and **`cell_sweep_results`**; **`pending_overnight_complete`** when **< 300k** (smoke pass). CTest **`native_d28_overnight_complete_smoke`**. |
| **`baseline_lock_validate`** *(Phase 12–14)* | Validate **`bench/BASELINE_LOCK.json`** schema + d17 hybrid pin; accepts **`medium_smoke`** / **`production`** statuses; **`--production`** enforces production/completed status @ 300k. CTest **`native_baseline_lock_validate_smoke`**. PS wrapper: **`scripts/validate_baseline_lock.ps1`** (`-Strict`, **`-Production`**). Post-overnight: **`scripts/finalize_production_overnight.ps1`**. |

**Federated TLS (optional OpenSSL):** default build uses plain HTTP only. Enable HTTPS with **`-DCYPHA_ENABLE_OPENSSL=ON`** at configure (requires OpenSSL dev libs; links **`OpenSSL::SSL`** + **`OpenSSL::Crypto`** and defines **`CPPHTTPLIB_OPENSSL_SUPPORT`** on **`cypha_federated_coordinator`**, **`cypha_federated_worker`**, and **`federated_tls_smoke`**).

```bash
cmake -S native -B native/build -DCYPHA_ENABLE_OPENSSL=ON
cmake --build native/build --target cypha_federated_coordinator cypha_federated_worker federated_tls_smoke
```

Coordinator TLS listen example:

```bash
cypha_federated_coordinator --listen 8443 --tls-cert cert.pem --tls-key key.pem \
  --min-workers 2 --once --out merged.json
```

Worker HTTPS submit:

```bash
cypha_federated_worker --payload worker_a.json --coordinator 127.0.0.1:8443 --scheme https
```

Without OpenSSL at build time, **`--tls-cert`** / **`https`** fall back only when **`CYPHA_FEDERATED_INSECURE=1`** is set (plain HTTP with a warning). Run **`ctest -R native_federated_tls_smoke`** — skipped when OpenSSL is off or the **`openssl`** CLI is missing; passes when TLS loopback merge succeeds.

**Overnight runners (maintainer):** per-domain scripts **`scripts/run_d17_overnight.ps1`**, **`scripts/run_rpsm_overnight.ps1`**; Phase 9 unified **`scripts/run_overnight_all.ps1`** chains all overnight jobs + baseline-lock refresh. Phase 10 adds **`cypha_baseline_lock --run all`** and Windows TLS mirror **`scripts/ci_federated_tls_windows.ps1`**. Phase 11 adds WikiText download (**`scripts/download_wikitext2.ps1`**, **`scripts/download_wikitext2.sh`**), gutenberg fallback for d17/d21, **`corpus_smoke`**, bench **d25**, and **`-Fast`** propagation on overnight scripts. Phase 12 adds **`-Medium`** tier (5k train, real corpus), bench **d26**, **`validate_baseline_lock.ps1`** / **`baseline_lock_validate`**, and **`publish_release.ps1 -DryRun`** *(shipped everywhere)*. Phase 13 adds **`-Production`** tier (300k train / 2000 eval), dedicated **`scripts/run_production_overnight.ps1`**, bench **d27**, and **`validate_baseline_lock.ps1 -Production`**. Phase 14 adds status validator fix (**`medium_smoke`** / **`production`**), cell-sweep output **`bench/results/cell_sweep`**, bench **d28**, and **`finalize_production_overnight.ps1`**. Phase 15-23 add release readiness (**d29**), artifact hygiene (**d30**), post-overnight pipeline (**d31**), production complete (**d32**), release publish (**d33**), repo smoke hygiene (**d34**), lock commit pipeline (**d35**), production pipeline E2E (**d36**), and overnight lock refresh (**d37**, prep) gates. See [`docs/reports/INTELLIGENCE_STATS_IMPLEMENTATION.md`](../docs/reports/INTELLIGENCE_STATS_IMPLEMENTATION.md) Phase 9-23.

**Corpus data (D17/D21):** WikiText-2 lives under **`bench/data/wikitext2/wikitext-2/`** (not committed). Download with the scripts above; when absent, **`load_bench_corpus`** falls back to **`bench/data/gutenberg/*.txt`** (`gutenberg_fallback`). See **`bench/data/wikitext2/README.md`**.

**Qt:** [`cmake -DCYPHA_BUILD_QT=ON`](qt/README.md) builds **`cypha_qt_stub`** (Core) and **`cypha_qt_shell`** (Widgets). Optional **`-DCYPHA_QT_CHARTS=ON`** links Qt Charts for the shell loss widget when **`Qt6::Charts`** is installed. **GitHub CI:** two **blocking** jobs — **`build_and_test`** installs **`qt6-base-dev`** (Charts off), passes **`-DCYPHA_BUILD_QT=ON`**, and runs **`native_qt_stub_load_reference`**; headless Linux excludes GUI exec tests **`native_qt_shell_smoke`**. **`mingw_cross`** verifies MinGW Windows PE artifacts. Optional third job **`federated_tls`** (**`-DCYPHA_ENABLE_OPENSSL=ON`**, **`native_federated_tls_smoke`**, `continue-on-error`; local mirrors **`bash scripts/ci_federated_tls_linux.sh`**, **`scripts/ci_federated_tls_windows.ps1`** *(Phase 10)*). Optional fourth job **`corpus_and_d25`** (WikiText fetch + **`native_corpus_smoke`** / **`native_d25_corpus_smoke`**, `continue-on-error`; Phase 12). **114 CTests** in the blocking gate (`ctest -R native_`; Phase 22 shipped; **115** when d37 merges). Production overnight (300k) is **not** run in CI — use **`scripts/run_production_overnight.ps1`** (+ **`finalize_production_overnight.ps1`**, **`validate_production_complete.ps1`**, **`verify_release_publish.ps1`**, **`verify_production_pipeline.ps1`**, **`run_post_overnight.ps1`**) locally. CUDA is not in CI — build locally with **`-DCYPHA_ENABLE_CUDA=ON`** and run **`native_cuda_smoke`** / **`native_score_batch`** (see [`ACCEL_CUDA.md`](../docs/native/ACCEL_CUDA.md)). Local **`scripts/ci_native_linux.sh`** defaults Qt OFF unless **`CYPHA_BUILD_QT=1`**; optional **`CYPHA_QT_CHARTS=1`** passes **`-DCYPHA_QT_CHARTS=ON`** (install **`qt6-charts-dev`** first).

## Build

```bash
cd native
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --output-on-failure   # includes native_train_step_vector, native_regression_mixture, …
```

From repo root, **`ctest --test-dir native/build -N`** lists every **`NAME native_*`** registered in **`native/CMakeLists.txt`**.

Linux/WSL one-liner matching CI’s native step: **`bash scripts/ci_native_linux.sh`** (optional **`CYPHA_NATIVE_BUILD_DIR`** / **`CMAKE_BUILD_TYPE`**). Set **`SKIP_NATIVE_CTEST_REGISTRY=1`** to skip the post-CTest registry check.

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
- **Windows (vcpkg, x64):** `vcpkg install sqlite3:x64-windows` then configure CMake with `-DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake`

*(There is no package named “Cslite” — you want **SQLite**.)*

### SQLite for **`experiment_db_smoke`** (optional)

CMake uses **`find_package(SQLite3)`** first. If it is missing and **`CYPHA_FETCH_SQLITE3_AMALGAMATION`** is **ON** (default), CMake **downloads** the official **SQLite 3.47.2 amalgamation** at configure time (needs network) and builds static **`cypha_sqlite3_amalg`** — no **`libsqlite3-dev`** / vcpkg required. Set **`-DCYPHA_FETCH_SQLITE3_AMALGAMATION=OFF`** to disable (then install a system SQLite3 dev package or skip the target).

Experiment DDL is embedded at configure time (no Python required).

When using a **system** SQLite3, you still need the **library + headers** (not only the `sqlite3` CLI).

| Environment | Install |
|-------------|---------|
| **Ubuntu / Debian / WSL** | `sudo apt-get install -y libsqlite3-dev` |
| **Fedora / RHEL** | `sudo dnf install sqlite-devel` |
| **Windows (MSVC)** | **Default:** CMake can **fetch** the amalgamation (**network** at configure). **Or** **vcpkg** `sqlite3:x64-windows` + `-DCMAKE_TOOLCHAIN_FILE=.../vcpkg.cmake` for **`find_package(SQLite3)`**. |

With **no** system SQLite3 dev package, **`CYPHA_FETCH_SQLITE3_AMALGAMATION=ON`** (default) is enough. If you install **libsqlite3-dev** / vcpkg / Homebrew sqlite, CMake prefers **`SQLite::SQLite3`** and skips the download.

After a successful configure, the build includes **`experiment_db_smoke`** and CTests **`native_experiment_db_smoke`** + **`native_experiment_db_file`**.

**ctest on Windows:** ELF parity tools under **`native/build-wsl/`** run inside WSL when the repo is on **`/mnt/c/...`**. Full validation: **`powershell -File scripts/cypha_native_validate_all.ps1`**.

**Full validation (one command):** from repo root, **`powershell -File scripts/cypha_native_validate_all.ps1`** — Release build outside OneDrive, **`ctest -R native_`**, REST curl smoke, bench smoke (d01/d04/d17), tune dry-run. See [`docs/native/NATIVE_QUICKSTART.md`](../docs/native/NATIVE_QUICKSTART.md).

Manual runs:

**REST smoke:** after building `cypha_rest`, run **`powershell -File native/scripts/smoke_cypha_rest_mingw.ps1`** (or curl **`/health`** + **`/predict`** against your binary).

```bash
./build/cypha_parity ../fixtures/reference.cypha ../fixtures/native_parity.bin
./build/memory_train_parity ../fixtures/memory_train
./build/preprocessor_parity ../fixtures/preprocessor
./build/cypha_rest --listen 127.0.0.1:8099 \
  --cypha ../fixtures/reference.cypha \
  --f-field-json ../fixtures/f_field.json
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

**CMake layout:** MinGW-specific options and link flags live in **`native/cmake/CyphaMinGW.cmake`** (included from **`native/CMakeLists.txt`**). Parity and app targets are grouped in **`native/cmake/CyphaParity.cmake`** and **`native/cmake/CyphaApps.cmake`**. Toolchain file: **`native/toolchains/mingw-w64-x86_64.cmake`** (cache **`CYPHA_MINGW_TOOLCHAIN_PREFIX`**, default **`x86_64-w64-mingw32`**, for non-Debian triplet layouts).

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

**Windows ctest** auto-discovers **`native/build-mingw-w64/cypha_rest.exe`** for **`native/scripts/smoke_cypha_rest_mingw.ps1`** (before MSVC **`native/build/`** paths). Set **`CYPHA_REST_BIN`** only to override.

When the repo lives under **`/mnt/c/...`**, CMake rewrites parity fixture paths to **`C:/...`** for `ctest` so Windows can open them.

**Console check (Windows `cmd`):**

```bat
cd native\build-mingw-w64
cypha_parity.exe ..\..\fixtures\reference.cypha ..\..\fixtures\native_parity.bin
```

(`..\\..\\fixtures` from `native\build-mingw-w64`; or use absolute `C:\...\fixtures\...`.)

**REST smoke (Windows, MinGW binary):** `powershell -File native/scripts/smoke_cypha_rest_mingw.ps1` (add `-WithRegression` to assert `/predict` **`regression_val`** with `fixtures/regression_head.json`).

**One-shot MinGW build from PowerShell + optional ctest:** `powershell -File native/scripts/build_cypha_rest_mingw_wsl.ps1` (add `-RunSmoke` to set **`CYPHA_REST_BIN`** and run `native/scripts/smoke_cypha_rest_mingw.ps1`).

## Regenerate fixtures

Committed goldens live under **`fixtures/`**. After intentional **`cypha_core`** math or sidecar layout changes, update fixtures and re-run **`ctest -R native_ --output-on-failure`**. See [`docs/verify/MAINTENANCE.md`](../docs/verify/MAINTENANCE.md).

## Specs

- [`NATIVE_QUICKSTART.md`](../docs/native/NATIVE_QUICKSTART.md) — install, validate, bench, tune, REST (v2.2)  
- [`PORT_CONTRACT.md`](../docs/port/PORT_CONTRACT.md) — `.cypha` v3, inference math, REST JSON, native bench §6  
- [`PREPROCESSOR_CONTRACT.md`](../docs/port/PREPROCESSOR_CONTRACT.md) + [`schemas/preprocessor.schema.json`](../docs/port/schemas/preprocessor.schema.json)  
- [`schemas/regression_head.schema.json`](../docs/port/schemas/regression_head.schema.json) — optional MoE sidecar for `/predict` **`regression_val`**  
- [`EXPERIMENTS_SCHEMA.md`](../docs/port/EXPERIMENTS_SCHEMA.md) — SQLite (M6). DDL is embedded at CMake configure time. Native **`experiment_db_smoke`** + **`experiment_db_crud_parity`** (CTests **`native_experiment_db_smoke`** / **`native_experiment_db_crud`**) validate DDL and core run/metrics CRUD when SQLite is available.  
- [`regression_stub.hpp`](include/cypha/regression_stub.hpp) — M4 placeholder  

## Next engineering waves

- **CUDA accel** — `-DCYPHA_ENABLE_CUDA=ON` + NVIDIA toolkit/driver; see **`cuda_smoke`** and presets above. See [`docs/FUTURE.md`](../docs/FUTURE.md) §1.  
- **Qt shell streaming** — move bulk training to a `QThread`; emit per-step loss/accuracy signals; live loss chart update during training. See [`docs/FUTURE.md`](../docs/FUTURE.md) §2a.  
- **Packaged binary** — AppImage (Linux) or `windeployqt` folder / `.msi` (Windows) distributing Qt shell as a self-contained executable. See [`docs/FUTURE.md`](../docs/FUTURE.md) §3.  
- **REST multi-model** — `cypha_rest --registry <root>` serving N models; per-model mutex; LRU eviction. See [`docs/FUTURE.md`](../docs/FUTURE.md) §5.  
- **CyphaLM native inference** — C++ decode path for `.cypha` LM checkpoints (generation + REST parity). See [`docs/port/PORT_CONTRACT.md`](../docs/port/PORT_CONTRACT.md) §4.  
- **Full future directions** — Web UI, curriculum/active learning, ONNX export, federated training: [`docs/FUTURE.md`](../docs/FUTURE.md).  
