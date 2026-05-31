# Scripts index

Utility and verification scripts for the Cypha repo. Narrative “when to run what” lives in **[`docs/README.md`](../docs/README.md)**. **Fixtures / native / schema cadence:** **[`docs/verify/MAINTENANCE.md`](../docs/verify/MAINTENANCE.md)**. **Pip encoding fallback:** repo-root **`requirements-pip-merged.txt`** (see **[`CONTRIBUTING.md`](../CONTRIBUTING.md)**).

| Script | Purpose | Typical output |
|--------|---------|----------------|
| `setup_and_test.sh` / `setup_and_test.ps1` | Bootstrap venv + **`requirements-verify.txt`** + pytest + legacy suites; optional **`FULL_STUDIO_DEPS=1`** / **`-Studio`** adds studio + **`pytest-qt`**; **`QT_QPA_PLATFORM=offscreen`** when unset | console |
| `run_all_regressions.ps1` / `run_all_regressions.sh` | **`test_cypha.py`** + pytest bundle: API + REST smoke, experiment/schema/registry, **mixture contract + native mixture parity**, M4·RFF + trainer + fixtures, studio trainer hotpaths, **full native subprocess parity set** (many **skip** without binary), engine tests (GIG vectorized, fused encode, inference, score-matrix modes, **`CYPHA_*` env), accel GEMM wiring, CSV chunked parity, memory/preprocessor fixture modules, native sidecar, registry eval wiring, training plot compress (Qt **skip** without PySide6); default **`QT_QPA_PLATFORM=offscreen`** when unset; **`--full`** / **`-Full`** → **`pytest tests/ -m "not slow"`** (adds e.g. **`@slow`** studio runner, still skips GUI without Qt) | console |
| `wsl_verify.sh` | WSL verify + optional benchmark; default **`QT_QPA_PLATFORM=offscreen`**; always **`requirements-verify.txt`**, then **`FULL_STUDIO_DEPS=1`** adds **`cypha_studio/requirements.txt`** + **`pytest-qt`** (CI-like GUI/qtbot); **`PYTEST_MARK='not slow'`** → `pytest -m …`; **`RUN_NATIVE=1`** → `native/build`, ctest, `CYPHA_REST_BIN` + `test_cypha_rest_smoke`; without **`RUN_NATIVE`**, tip for **`native/scripts/build_cypha_rest_mingw_wsl.ps1`** | console; `BENCHMARK_LOG` → `artifacts/profiles/benchmark_baseline.txt` (default) |
| `export_experiment_schema_sql.py` | Print `experiment._SCHEMA` DDL (M6); **`-o path.sql`** to write file; used by CMake when building **`experiment_db_smoke`** | stdout / disk |
| `generate_parity_fixtures.py` | Regenerate `parity_fixtures/` (incl. `train_hparams.json` for `cypha_rest`) | disk |
| `generate_demo_lm_checkpoint.py` | Train tiny CyphaLM → `examples/demo_cyphalm/demo.{json,npz}` for REST/Studio smoke | disk |
| `bench_cyphalm.py` | CyphaLM CPU vs CUDA train/predict timing + optional stage breakdown | stdout |
| `cypha_bench/tuning/cyphalm_sweep.py` | Grid search CyphaLM configs on WikiText valid BPC | `cypha_bench/config/cyphalm_profile_sweep.json` |
| `generate_regression_m4_fixture.py` | `parity_fixtures/regression_m4/sidecar.json` for CTest `native_regression_m4` | disk |
| `generate_rff_regression_fixture.py` | `parity_fixtures/rff_regression/sidecar.json` — RFF encode + ridge + expert dots (CTest **`native_regression_rff`**) | disk |
| `generate_memory_train_parity.py` | Regenerate `parity_fixtures/memory_train/` (native `DIFMemory.train`) | disk |
| `generate_preprocessor_parity.py` | Regenerate `parity_fixtures/preprocessor/` | disk |
| `generate_preprocessor_fit_fixture.py` | `parity_fixtures/preprocessor_fit/` + `preprocessor_fit_no_scale/` — native `fit_from_design_matrix` (scale on/off + PCA) vs Python | disk |
| `generate_csv_ingest_fixture.py` | `parity_fixtures/csv_ingest/` — native `load_csv_dense` vs `CSVDataset.from_file` | disk |
| `generate_dif_regressor_train_step_fixture.py` | `parity_fixtures/dif_regressor_train_step/` — Python `DIFRegressor` (cold + warm `infer` routing + `replay_u01` for `replay_ratio>0`) vs native | disk |
| `generate_studio_trainer_classify_hotpath_fixture.py`, `generate_studio_trainer_gh_classify_hotpath_fixture.py`, `generate_studio_trainer_preprocess_classify_hotpath_fixture.py`, `generate_studio_trainer_preprocess_gh_classify_hotpath_fixture.py`, `generate_csv_preprocess_classify_hotpath_fixture.py` | `parity_fixtures/studio_trainer_*` + **`csv_preprocess_classify_hotpath/`** — CTests `native_studio_trainer_*`, **`native_csv_preprocess_classify_hotpath`**, + `preprocess_train_classify_parity`; **`preprocess_gh`** copies goldens from **`studio_trainer_gh_classify_hotpath/`** (generate GH fixture first); CSV fixture clones preprocess hotpath + **`train.csv`** | disk |
| `export_f_field_json.py` | Write `parity_fixtures/f_field.json` for `cypha_rest` demos | disk |
| `export_regression_head.py` | Demo `DIFRegressor` train → `regression_head.json` (MoE sidecar for `cypha_rest` / `CYPHA_REGRESSION_HEAD`) | disk (`artifacts/regression_head_demo.json` default) |
| `gen_native_bessel_table.py` | Regenerate `native/src/bessel_table_data.cpp` — **K₂/K₁** and **K₀/K₁** grids (needs SciPy) | disk |
| `export_bessel_ratios_npz.py` | Write repo-root **`bessel_ratios.npz`** (same grid as above) for SciPy-free **`Cypha.py`** / **`cypha_accel/nig_gh.py`** GH gates | disk |
| `profile_gui_startup.py` | cProfile CyphaStudio `MainWindow` cold start | stdout; optional `-o` (e.g. `artifacts/profiles/gui_startup_cprofile.txt`) |
| `profile_studio_hotpaths.py` | Training/chat/dataset/registry/API hot paths | stdout or `-o` |
| `profile_studio_memory.py` | tracemalloc diff around training widget loop | stdout |
| `profile_real_datasets.py` | sklearn tabular cProfile | `artifacts/profiles/profile_real_cumtime.txt` (default) |
| `download_profile_e2e.py` | OpenML + CA housing; class/reg/gen cProfile | `artifacts/profiles/profile_e2e_download.txt` (default) |
| `gpu_microbench.py` / `gpu_fullbench.py` | GPU / CuPy micro and pipeline timing | stdout |
| `bench_gpu_production.py` | Bundle micro + full + tuning-style load | `artifacts/bench/*.json` |
| `tune_quality_performance.py` | Hyperparameter grid + metrics | `artifacts/tuning/tuning_*.{csv,json,txt}` |
| `tune_quality_performance.py` (presets) | Invoked via `Makefile` `tune-*` targets | same |
| `loadtest_ab_predict_example.sh` / `.ps1` | Example `ab` against live `/predict` | console |
| `print_profile_hotspots.py` | Helper to summarize cProfile dumps | stdout |
| `download_profile_e2e.py` | (see above) | |
| `wsl_bench_gpu.sh` | WSL GPU bench helper | console |

## Parity fixture generators (supplementary)

All are also invoked by `generate_parity_fixtures.py`. Full regen docs: [`docs/verify/MAINTENANCE.md`](../docs/verify/MAINTENANCE.md).

| Script | Fixture produced | CTest name |
|--------|-----------------|------------|
| `generate_batch_llr_fixture.py` | `parity_fixtures/batch_llr/` | `native_batch_llr` |
| `generate_quantile_dif_train_fixture.py` | `parity_fixtures/quantile_dif_train/` | `native_quantile_dif_train` |
| `generate_dif_train_replay_fixture.py` | `parity_fixtures/dif_train_replay/` | `native_dif_train_replay` |
| `generate_mke_train_step_fixture.py` | `parity_fixtures/mke_train_step/` | `native_mke_train_step` |
| `generate_mke_train_extended_fixture.py` | `parity_fixtures/mke_train_extended/` | `native_mke_train_extended` |
| `generate_two_stage_pipeline_fixture.py` | `parity_fixtures/two_stage_pipeline/` | `native_two_stage_pipeline` |
| `generate_two_stage_ridge_fit_fixture.py` | `parity_fixtures/two_stage_ridge_fit/` | `native_two_stage_ridge_fit` |
| `generate_two_stage_e2e_ridge_fixture.py` | `parity_fixtures/two_stage_e2e_ridge/` | `native_two_stage_e2e_ridge` |
| `generate_generation_fixture.py` | `parity_fixtures/generation/sidecar.json` | `native_generation` |

## CyphaLM report and SOM evaluation

| Script | Purpose | Output |
|--------|---------|--------|
| `run_cypha_lm_report.py` | Run all `experiments/` + `benchmarks/` to regenerate `cypha_lm/REPORT.md` + `paper/figures/` | `cypha_lm/REPORT.md` |
| `run_som_upgrade_eval.py` | Batch SOM upgrade eval (U1-U6 + all) via `benchmark_baseline.py` | `artifacts/` |

## Analysis utilities

| Script | Purpose | Output |
|--------|---------|--------|
| `diff_cypha_state.py` | Shallow diff of two `.cypha` binary states (key / shape / mean) | stdout |
| `merge_final_profile.py` | Merge `cypha_bench/config/everyday_profile.json` with `profiled_medium.json` regression | disk |

## Windows PowerShell extras

| Script | Purpose |
|--------|---------|
| `build_native_wsl.ps1` | WSL CMake build in `native/build-wsl` + optional ctest |
| `run_cypha_qt_windows.ps1` | Launch `cypha_rest` + Qt shell on Windows |
| `setup_and_test.ps1` | Windows equivalent of `setup_and_test.sh` |
| `run_all_regressions.ps1` | Windows equivalent of `run_all_regressions.sh` |
| `loadtest_ab_predict_example.ps1` | Windows equivalent of `loadtest_ab_predict_example.sh` |

> **Note on .sh scripts:** `setup_and_test.sh`, `run_all_regressions.sh`, `wsl_verify.sh`,
> `wsl_bench_gpu.sh`, `loadtest_ab_predict_example.sh`, and `ci_native_linux.sh` were
> historically gitignored by a blanket `*.sh` rule (removed May 2026). The `.sh` files now
> exist on disk and are tracked. PowerShell equivalents exist for all Windows workflows.
