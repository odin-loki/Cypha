# Cypha prototype — debug, profile, verify

This document is the **master checklist** for proving the Python reference implementation and the **native C++ production stack** stay aligned.

For a **living snapshot** of automated tests and known gaps, see [`VERIFICATION_STATUS.md`](VERIFICATION_STATUS.md). For **what to regen and rebuild** after contract changes, see [`MAINTENANCE.md`](MAINTENANCE.md).

**Remote CI:** [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) — **four jobs** on push to `main`:
1. **`build_and_test`** (Ubuntu) — Ruff; `cmake -DCYPHA_BUILD_QT=ON` + CTest (GUI exec tests excluded on headless); WikiText-2 corpus; **`pytest tests/ cypha_lm/model/tests/`** with `CYPHA_REST_BIN` and `QT_QPA_PLATFORM=offscreen`.
2. **`mingw_cross`** — MinGW Windows PE artifact smoke (`cypha_rest.exe`, `cypha_bench_run.exe`, …).
3. **`windows_cuda_msvc`** (optional, non-blocking) — MSVC + CUDA compile smoke.
4. **`linux_cuda`** (optional, non-blocking) — GCC + nvcc smoke.

**Release:** tag `v*` → [`.github/workflows/release.yml`](../../.github/workflows/release.yml) publishes Linux + Windows installer archives (latest: **v2.2.6**).

## 1. Scope

| Layer | Path | Role |
|--------|------|------|
| Core engine | `Cypha.py` | DIF classifier/regressors, encoders, save/load |
| Studio core | `cypha_studio/core/` | Dataset, trainer, experiment, registry, inference |
| Studio API | `cypha_studio/server/` | FastAPI + local server |
| Studio GUI | `cypha_studio/gui/` | PySide6 desktop (same surface area as future Qt) |

## 2. Environment (WSL)

- **Why WSL**: Linux matches typical CI and server targets; avoids Windows-specific path and optional native-build quirks for scientific stacks.
- **Repo in WSL**: `/mnt/c/Users/<you>/OneDrive/Desktop/Cypha` (adjust if you clone inside `~/` instead).
- **PEP 668**: Ubuntu’s system Python blocks `pip install` globally — always use a **venv** (the script below creates `.venv-wsl` automatically).

### One-shot verification

```bash
cd /path/to/Cypha
bash scripts/wsl_verify.sh
# Optional: full sklearn benchmark (slow)
RUN_BENCHMARK=1 bash scripts/wsl_verify.sh
# Optional: CI-like PySide6 + qtbot (still headless): FULL_STUDIO_DEPS=1 bash scripts/wsl_verify.sh
# (installs requirements-verify.txt, then studio + pytest-qt — same layering as .github/workflows/ci.yml)
```

### Native `cypha_rest` on Windows (MinGW cross-build in WSL)

From **PowerShell** at repo root: `powershell -File native/scripts/build_cypha_rest_mingw_wsl.ps1` (add **`-AllTargets`** for every MinGW **`.exe`**). **`tests/test_cypha_rest_smoke.py`** discovers **`native/build-mingw-w64/cypha_rest.exe`** without **`CYPHA_REST_BIN`**. **`-RunPytest`** runs that file with **`CYPHA_REST_BIN`** set explicitly. Quick check: `powershell -File native/scripts/smoke_cypha_rest_mingw.ps1` or **`-WithRegression`**.

Linux ELF build + CTest (inside WSL): `bash scripts/ci_native_linux.sh` (or `cmake -S native -B native/build && cmake --build native/build -j$(nproc) && ctest --test-dir native/build --output-on-failure`). With **`qt6-base-dev`** installed, **`CYPHA_BUILD_QT=1 bash scripts/ci_native_linux.sh`** enables **`cypha_qt_stub`** and CTest **`native_qt_stub_load_reference`**. The script optionally runs **`pytest tests/test_native_ctest_pytest_registry.py`** after CTest when pytest is installed. If you pull commits that add **`add_test(NAME native_…)`** lines, re-run **`cmake`** on your existing build directory once so **`ctest -N`** lists the new tests (otherwise **`ctest -R new_name`** can report *No tests were found*).

**Full WSL verify including `cypha_rest`:** after the usual `bash scripts/wsl_verify.sh`, run again with **`RUN_NATIVE=1`** so the script configures/builds **`native/build`**, runs **CTest**, sets **`CYPHA_REST_BIN`**, and re-runs **`tests/test_cypha_rest_smoke.py`** (the first pytest pass still skips those without the binary). With **`qt6-base-dev`** installed, **`CYPHA_BUILD_QT=1 RUN_NATIVE=1 bash scripts/wsl_verify.sh`** also enables **`cypha_qt_stub`** and runs **`tests/test_qt_stub_native.py`** after CTest.

**M1 / `cypha_parity`:** **`reference.cypha`** includes **Tier-1**; C++ **`from_root`** restores **`ctx_*`** ([`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md) §4; pytest **`test_reference_fixture_restores_tier1_for_native_cypha_parity`**).

### Dependencies

- **Headless (tests + API)**: `pip install -r requirements-verify.txt` — covers `test_cypha.py` and `cypha_studio/test_cypha_studio.py` without PySide6.
- **Full studio + GUI**: same venv — `pip install -r cypha_studio/requirements.txt`, then **`pip install pytest-qt`** for **`test_gui_qtbot`** (**`pytest-qt`** is not in **`requirements-verify.txt`**). One-shot: **`FULL_STUDIO_DEPS=1 bash scripts/wsl_verify.sh`** or **`FULL_STUDIO_DEPS=1 bash scripts/setup_and_test.sh`** (always installs verify first, then studio + **`pytest-qt`**).
- **GUI on WSL**: requires a display (`WSLg` on Windows 11, or X forwarding). Headless CI still installs **`requirements-verify.txt`** + **PySide6** + **pyqtgraph** + **`pytest-qt`** so `pytest tests/` runs GUI modules offscreen (see `.github/workflows/ci.yml`).
- **Pip / `install -r` encoding issues** (e.g. UTF-8 BOM or broken copies on Windows): **`pip install -r requirements-pip-merged.txt`** at repo root (comment-free merged pins), or the single-line **`pip install …`** in [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## 3. Testing matrix

### 3.1 Automated (must pass)

1. **`python3 test_cypha.py`** — formal unit tests for `Cypha.py` (GIG/NIG, training, inference, distillation, etc.).
2. **`python3 cypha_studio/test_cypha_studio.py`** — dataset, preprocessor, trainer, search, experiment, registry, inference, API wiring.
3. **`pytest tests/test_gui_smoke.py -v`** (with **`pip install -r cypha_studio/requirements.txt`**) — Qt offscreen: `MainWindow`, `TrainConfigDialog`, chat send (no model / with tiny `CyphaDIF`), `ConfidenceWidget` + bus, `MessageBubble`. Not a substitute for manual UI testing.
4. **`pytest tests/test_gui_qtbot.py -v`** — studio venv plus **`pip install pytest-qt`**: `qtbot` clicks Send/Clear, training config OK, toolbar Train (message box mocked).

### 3.2 Real data + full profile (local)

**`python scripts/download_profile_e2e.py`** (or **`make e2e-profile`**) downloads/caches OpenML 1464 (blood transfusion, classification) and California housing (regression), runs **classification**, **DIFRegressor regression**, and **`CyphaDIF.generate`** (rejection + simple), and writes **`artifacts/profiles/profile_e2e_download.txt`** (default) with three **cProfile** sections (cumtime, top 50 each) plus JSON metrics in the header. Use **`--fast`** / **`make e2e-profile-fast`** for a shorter run. First OpenML fetch needs network; data lives under **`data_cache/`** (large cache; omit from hand-tight bundles if needed).

**`python scripts/tune_quality_performance.py`** (or **`make tune-coarse`** / **`make tune-medium`**) runs a **preset grid** (coarse / medium / fine) over `CyphaDIF` and `DIFRegressor` hyperparameters, optional **`--include-generation`** brute-force on `generate()` temperature × `max_candidates`, records **val accuracy / R² / MAE / generation match-rate**, **wall-clock** breakdowns, **GPU** probe + optional CuPy warmup, and writes **`artifacts/tuning/tuning_*_{results.csv,summary.json,_profile.txt}`** (large output). Cap cost with **`--max-combos`**, parallelize with **`--jobs`** (joblib).

**GPU production bundle** (CuPy + CUDA, Python 3.10–3.12): **`make bench-gpu-prod`** or **`python scripts/bench_gpu_production.py`** — runs **`gpu_microbench`**, **`gpu_fullbench`**, then **`make tune-gpu-heavy`**-style tuning with **`--jobs 1`**. See [`BENCHMARK_GPU.md`](../benchmarks/BENCHMARK_GPU.md).

### 3.2 Benchmark / regression (should pass before port)

3. **`python3 benchmark.py`** — accuracy/latency vs sklearn baselines on standard datasets. Use as a **regression oracle**: after any change, metrics should not collapse without explanation.

- **Section 9 (streaming intrusion)** loads `/tmp/X_intrusion.npy` and `/tmp/y_intrusion.npy` when both exist; otherwise it uses a **fixed synthetic stream** (same *N*, *d*, and class proportions) so the benchmark always completes in CI/WSL.
- A captured run lives in [`artifacts/profiles/benchmark_baseline.txt`](../../artifacts/profiles/benchmark_baseline.txt) (stdout from `tee`). Re-run with `RUN_BENCHMARK=1 bash scripts/wsl_verify.sh` or `python3 benchmark.py 2>&1 | tee artifacts/profiles/benchmark_baseline.txt` after **material changes** (e.g. `score_matrix`, softmax, GH gate, or training loop).

### 3.3 Manual / integration

4. **Studio GUI**: `python3 cypha_studio/main.py` — train a small run, save model, load from registry, inference tab. **Automated smoke:** `pytest tests/test_gui_smoke.py -v` (offscreen). **Startup profile:** `python scripts/profile_gui_startup.py` (optionally `-o artifacts/profiles/gui_startup_cprofile.txt`) — expect import-time dominance (PySide6, pyqtgraph); `MainWindow.__init__` should be a smaller slice after caches warm.
5. **API**: start `uvicorn` (or studio server entry) and hit health/train/predict routes per `api.py`.
6. **Binary round-trip**: save with `cypha_save_binary` / registry, load, compare predictions on fixed seeds. **Buffer parity:** `pytest tests/test_cypha_binary_buffer_api.py` — **`cypha_load_binary_from_bytes(Path.read_bytes())`** vs **`cypha_load_binary(path)`**, and **`cypha_save_binary_to_bytes`** vs on-disk **`reference.cypha`** bytes (same v3 layout as native **`load_cypha_from_buffer`** / **`save_cypha_to_buffer`**).

## 4. Profiling (what to measure before native GPU)

Goal: know **which kernels** to port first (hot loops in NumPy/Python).

| Tool | Command / use |
|------|----------------|
| **cProfile** | `python3 -m cProfile -o profile_stats.cprof test_cypha.py` then `python3 scripts/print_profile_hotspots.py` — top cumulative time. |
| **GUI cold start** | `python scripts/profile_gui_startup.py` — one `MainWindow` build + `processEvents` under `QT_QPA_PLATFORM=offscreen`; use `-o` to save pstats text. |
| **Studio hot paths** | `python scripts/profile_studio_hotpaths.py --help` — subcommands `training`, `chat`, `dataset`, `registry`, `api`; historical checklist in [`CYPHA_STUDIO_MASTER_PLAN.md`](../studio/CYPHA_STUDIO_MASTER_PLAN.md) (superseded — see [`STUDIO_THREADING.md`](../studio/STUDIO_THREADING.md)). |
| **line_profiler** (optional) | Decorate hottest functions in `Cypha.py`, run `kernprof -l -v script.py` for line-level hotspots. |
| **memory** (optional) | `tracemalloc` or `memory_profiler` on long training loops if RAM growth is suspected. |

**Example (WSL, after `test_cypha.py`)** — dominant `cumtime` sources included:

- `numpy.linalg.solve` (RFF / ARD / kernel fitting)
- `RFFEncoder.auto_gamma_cv`, `auto_ard`, `from_data`, `fit`
- `CyphaDIF.train_step`, `RFFRegressor.train_step`, `WorldPrior.update`

**Porting hint**: put `score_matrix`, batched softmax/GH gate, and RFF feature maps on the CUDA/C++ short list; keep orchestration in Python until parity tests pass.

### Batch vs serial inference (parity)

`batch_infer` must use the **same** field-conditioned μ₀ and **GH–NIG** world gate as `infer()`. A prior mismatch (legacy sigmoid gate + `use_field=False` default) caused flaky confidence tests; that path is aligned in the reference implementation so the Qt/C++ port can treat `batch_infer` as the vectorised spec for `infer`.

## 5. Debugging checklist

- [ ] All `test_cypha.py` tests green (fixed seeds).
- [ ] All `cypha_studio` tests green.
- [ ] Benchmark within expected bands vs last known good (store a short log in `artifacts/profiles/benchmark_baseline.txt` when satisfied).
- [ ] No silent `sys.path` hacks to non-existent dirs (repo root only).
- [ ] Registry save/load identical inference on sample batch.
- [ ] Document any **known limitations** (e.g. headless GUI, optional sentence-transformers).

## 6. Definition of “ready to port”

1. **Correctness**: test suites + benchmark regression satisfied.
2. **Observability**: at least one saved cProfile report and a short list of top functions.
3. **Contract**: documented public APIs you will freeze for the C++/Qt side (tensor shapes, `save_state` / binary format, REST JSON schemas).
4. **Parity tests**: plan (or scripts) to compare Python vs future native implementation on the **same** saved weights and inputs.

## 7. Suggested order of port

1. **Numerics core** (`Cypha.py` hot paths) → C++/CUDA with Python bindings or subprocess, validated by shared tests.
2. **Training loop** — optional second phase once inference parity is proven.
3. **Qt shell** — replace PySide6 UI while keeping the same core/API contracts.

---

*Maintainers: re-run `scripts/wsl_verify.sh` after substantive changes to `Cypha.py` or `cypha_studio/core/`.*

## 8. Pytest + parity fixtures + port contract

- **Doc hub**: [`docs/README.md`](../README.md)
- **Roadmap**: [`ROADMAP.md`](ROADMAP.md)
- **Contract (normative for C++/Qt)**: [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md)
- **Contributing / PR checklist**: [`../CONTRIBUTING.md`](../CONTRIBUTING.md)
- **Parity data**: `parity_fixtures/` — regenerate with `python scripts/generate_parity_fixtures.py` after intentional state-format or inference changes.
- **Tests**: `pytest tests/ -v` (includes `test_parity_fixtures.py`, `test_api_contract.py`, `test_trainer_regression_fit.py` for RFF / two-stage / MKE trainers).
- **M6 DDL**: `python scripts/export_experiment_schema_sql.py` (also run at end of `scripts/wsl_verify.sh` as a smoke step).
- **One-shot (WSL/Linux)**: `bash scripts/setup_and_test.sh` (optional **`FULL_STUDIO_DEPS=1`** for studio + **`pytest-qt`** after verify deps)
- **One-shot (Windows)**: `powershell -ExecutionPolicy Bypass -File scripts/setup_and_test.ps1` (uses `.venv-win` and `py -3` if available). Add **`-Studio`** for **`cypha_studio/requirements.txt`** + **`pytest-qt`** on top of verify deps.
