# Contributing

**Documentation hub:** [`docs/README.md`](docs/README.md) (organized by use / verify / port). **Research journal:** [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md). **Script index:** [`scripts/README.md`](scripts/README.md). **Regen / native / schema upkeep:** [`docs/verify/MAINTENANCE.md`](docs/verify/MAINTENANCE.md).

## Setup

**Quick install (recommended):** platform scripts under [`install/`](install/) set up a venv and dependencies. See [`install/README.md`](install/README.md).

```powershell
# Windows (headless verify deps)
powershell -ExecutionPolicy Bypass -File install\install_windows.ps1
```

```bash
# Linux / WSL
bash install/install_linux.sh
```

**Manual setup:**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-verify.txt   # pytest, pytest-cov, sklearn, scipy, httpx, fastapi (headless parity + studio script)
# GUI + full studio:
pip install -r cypha_studio/requirements.txt
# Optional one-shot (verify + studio + pytest-qt, same order as CI):
#   FULL_STUDIO_DEPS=1 bash scripts/setup_and_test.sh
#   powershell -ExecutionPolicy Bypass -File scripts/setup_and_test.ps1 -Studio
```

**Headless only (no PySide6 / pyqtgraph / pytest-qt):** use `requirements-verify.txt` alone.

**If `pip install -r` fails** (encoding, BOM, or a broken editor copy): save the requirement file as **UTF-8 without BOM** with **LF** line endings, or install from the merged file (no comments, ASCII-only pins): `pip install -r requirements-pip-merged.txt`. Same packages as verify + studio + **pytest-qt** in one line:

```bash
pip install "numpy>=1.24" "scipy>=1.10" "scikit-learn>=1.3" "fastapi>=0.100" "uvicorn[standard]>=0.23" "httpx>=0.24" "pydantic>=2.0" "pytest>=7.0" "pytest-cov>=4.0" "PySide6>=6.5" "pyqtgraph>=0.13" pytest-qt
```

When you bump pins in `requirements-verify.txt` or `cypha_studio/requirements.txt`, update **`requirements-pip-merged.txt`** to match.

## Before you share or archive changes

```bash
python scripts/generate_parity_fixtures.py   # only if you changed inference/state format
pytest tests/ -v
python test_cypha.py
python cypha_studio/test_cypha_studio.py
pytest tests/test_gui_smoke.py -v   # PySide6 + pyqtgraph; QT_QPA_PLATFORM=offscreen
pip install pytest-qt   # after studio requirements — qtbot clicks + TrainConfig OK
pytest tests/test_gui_qtbot.py -v
pytest tests/test_gui_training_dataset.py -v   # training monitor + dataset panel CSV
pytest tests/test_gui_window_settings.py -v   # QSettings window geometry
pytest tests/test_studio_data_registry.py -v   # CSV loader + registry (no Qt)
pytest tests/test_preprocessor_fit_native_parity.py -v   # native Preprocessor fit (scale on/off + PCA) vs Python
pytest tests/test_csv_ingest_native_parity.py -v   # native CSV dense load vs CSVDataset.from_file
pytest tests/test_dif_regressor_train_step_native_parity.py -v   # DIFRegressor-shaped train + MoE predict vs native
pytest tests/test_cypha_binary_buffer_api.py -v   # v3 .cypha bytes API vs reference.cypha
pytest tests/test_qt_stub_native.py -v   # cypha_qt_stub + reference.cypha (needs -DCYPHA_BUILD_QT=ON + Qt6)
pytest tests/test_studio_trainer_classify_hotpath_native_parity.py -v   # native online loop vs Trainer.fit-shaped fixture
pytest tests/test_studio_trainer_gh_classify_hotpath_native_parity.py -v   # gh_train_step + chi/psi vs native
pytest tests/test_studio_trainer_preprocess_classify_hotpath_native_parity.py tests/test_studio_trainer_preprocess_gh_classify_hotpath_native_parity.py -v   # preprocessor + train / GH native
# Studio environment: docs/studio/CYPHA_ENV.md, docs/studio/STUDIO_THREADING.md
# Default ASGI app ``cypha_studio.server.api:app`` uses ``CYPHA_REGISTRY_ROOT`` for ``/models``, ``/load``, ``/register``.
```

**Research packages (not in CI — run before LM/SOM changes):**

```bash
pytest cypha_som/tests/ -v    # SOM/GNG/GRIA hooks (7 tests; flags default OFF)
pytest cypha_lm/ -v           # CyphaLM stack (~70 tests across subpackages)
pytest tests/test_lm_api.py -v  # CyphaLM generation + FastAPI /generate routes
```

Optional profiling (local):

```bash
python scripts/profile_real_datasets.py --fast   # writes artifacts/profiles/profile_real_cumtime.txt (default)
python scripts/gpu_microbench.py                 # raw GEMM; CuPy optional
python scripts/gpu_fullbench.py                  # full infer path timing + LLR CPU/GPU diff
python scripts/download_profile_e2e.py --fast    # real data + class/reg/gen + cProfile → artifacts/profiles/profile_e2e_download.txt (default)
python scripts/tune_quality_performance.py --preset coarse --include-generation   # hyperparameter grid → artifacts/tuning/
make cov                                           # pytest-cov on cypha_studio + cypha_accel
pytest tests/test_accel_cross_gemm.py tests/test_accel_cypha_wired.py -v   # GEMM backend parity
```

Or: `make test` (Unix/WSL; Makefile sets **`QT_QPA_PLATFORM=offscreen`** for pytest), **`bash scripts/setup_and_test.sh`**, or **`FULL_STUDIO_DEPS=1 bash scripts/setup_and_test.sh`** / **`-Studio`** on Windows for CI-like GUI deps.

## Native `cypha_rest` (optional)

CI runs **two jobs** (`.github/workflows/ci.yml`): **`build_and_test`** (Ubuntu native build + CTest + `pytest tests/` with **`CYPHA_REST_BIN`**) and **`mingw_cross`** (MinGW Windows PE cross-build). The Linux binary gates REST smokes so they are not skipped.

**Local (Linux / WSL ELF):** from repo root, install **`sudo apt-get install -y libsqlite3-dev`** (optional M6 CTest **`native_experiment_db_smoke`**), then either **`bash scripts/ci_native_linux.sh`** (CTest + optional drift pytest when **`python3 -m pytest`** is available) or manually: `cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release && cmake --build native/build -j$(nproc) && ctest --test-dir native/build --output-on-failure`.

After adding **`add_test(NAME native_…)`**, extend **`_NATIVE_CTEST_TO_PYTEST`** in **`tests/test_native_ctest_pytest_registry.py`** ( **`pytest tests/test_native_ctest_pytest_registry.py`** ).

Without **`libsqlite3-dev`**, CMake skips **`experiment_db_smoke`**; other CTest targets still build.

**Windows `.exe` from WSL (MinGW):** `powershell -File native/scripts/build_cypha_rest_mingw_wsl.ps1` (add **`-AllTargets`** for every `*.exe`; **`-RunPytest`** runs REST smoke). **`tests/test_cypha_rest_smoke.py`** picks up **`native\build-mingw-w64\cypha_rest.exe`** automatically — set **`CYPHA_REST_BIN`** only to override. Optional:

```powershell
$env:CYPHA_REST_BIN = "$PWD\native\build-mingw-w64\cypha_rest.exe"
pytest tests/test_cypha_rest_smoke.py -v
```

**MoE sidecar:** `parity_fixtures/regression_head.json` or `python scripts/export_regression_head.py -o path.json`. See **`docs/port/PORT_CONTRACT.md`** §3 and **`docs/studio/CYPHA_ENV.md`**.

## Principles (pre-port)

- Prefer **one** code path for inference math (`classify` ↔ `score_matrix` ↔ `batch_infer`).
- Prefer **`cypha_save_binary`** over ad-hoc pickle for anything that must load in native code.
- If you add a feature, add a **test** and, when it affects numbers, refresh **`parity_fixtures/`** and note it in the PR.

## Diagnostics and research scripts

**Re-running the empirical diagnostic** (phase-by-phase accuracy/calibration/encoder analysis):
```bash
python cypha_diagnostics/run_diagnostics.py
```
Output: JSON phases written to `cypha_diagnostics/results/` (gitignored). The permanent
summary is [`docs/reports/DIAGNOSTIC_REPORT.md`](docs/reports/DIAGNOSTIC_REPORT.md).

**Re-running the SOM upgrade evaluation:**
```bash
python scripts/run_som_upgrade_eval.py
```
Output: JSON upgrade results in `results/` (gitignored). The permanent summary is
[`docs/reports/SOM_UPGRADE_REPORT.md`](docs/reports/SOM_UPGRADE_REPORT.md).

**Applying diagnostic-confirmed upgrades** (deliberation fix, delta_lr, auto-RFF):
```bash
python cypha_diagnostics/apply_upgrades.py
```
These are already applied in the current `cypha_bench/config/everyday_profile.json`.

## Changelog

Update [`CHANGELOG.md`](CHANGELOG.md) when:
- A bug is fixed that affects numeric outputs.
- A parity fixture is regenerated.
- A new domain / encoder / regressor is added.
- A milestone benchmark is completed.

## Layout

- `Cypha.py` — engine (keep changes focused; native port will mirror this file first).
- `cypha_studio/` — studio only (no duplicate copies at repo root).
- `cypha_diagnostics/` — diagnostic investigation scripts (`run_diagnostics.py`,
  `apply_upgrades.py`) and their results (`results/` — gitignored).
- `docs/reports/` — permanent experiment records; committed, not regenerated.
- `docs/RESEARCH_STATUS.md` — benchmark journal and research priorities (Kernel LLR, CyphaLM D17).
