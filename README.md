# Cypha

**CyphaDIF** — an online differential information field classifier/regressor.  
**CyphaStudio** — dataset → train → registry → API → Qt desktop, all running natively.

The Python tree (`Cypha.py`, `cypha_studio/`) is the **golden reference** for math and parity generation. The **product hot path** lives entirely in native C++ (`native/`) — no Python at inference or training time.

---

## Current state

| Layer | Status |
|-------|--------|
| **Core inference** (encode → LLR → GH gate → softmax) | Native C++ (`cypha_core`), CI-gated parity vs `parity_fixtures/` |
| **Online training** (`dif_train_step`, GH, replay, NIG, context) | Native C++ |
| **Regression stack** (MKE/RFF/two-stage/ridge) | Native C++ |
| **REST server** (`cypha_rest`) | Native C++ (`cpp-httplib`), JSON-compatible with FastAPI `cypha_studio/server/api.py` |
| **Qt desktop shell** (`cypha_qt_shell`) | Native Qt 6 Widgets — dataset panel, column picker, CSV preview, preprocessor fit, val split, training progress, loss charts, MKE loop, Experiments DB (SQLite), registry, bulk train, REST client |
| **Experiments DB** | SQLite via `ExperimentDb` + `experiment_db_crud`; Qt M6 panel |
| **Preprocessor fit** | Native `fit_from_design_matrix` (scale + PCA); save `preprocessor.json` from Qt shell — no Python needed |
| **Parity** | 33 CTests + 188 pytest (WSL + Windows) |

---

## Quick start

```bash
# Python reference + tests
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-verify.txt
pytest tests/ -q                         # 188 tests, 2 opencl skips
python test_cypha.py                     # 54 Cypha.py checks
python cypha_studio/test_cypha_studio.py # 48 Studio checks

# Qt GUI (PySide6)
pip install -r cypha_studio/requirements.txt
python cypha_studio/main.py

# GUI tests (needs pytest-qt)
pip install pytest-qt
QT_QPA_PLATFORM=offscreen pytest tests/test_gui_smoke.py tests/test_gui_qtbot.py -v
```

**Windows:** `powershell -ExecutionPolicy Bypass -File scripts/setup_and_test.ps1` (add `-Studio` for PySide6 + pytest-qt)

**WSL one-shot:** `bash scripts/setup_and_test.sh` or `bash scripts/wsl_verify.sh`

---

## Native C++ build

```bash
cd native
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure   # 33 tests (2 opencl skips = expected)
```

**Qt shell** (needs Qt 6):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_QT=ON
cmake --build build --target cypha_qt_shell
./build/qt/cypha_qt_shell
```

**Windows cross-build from WSL:**
```bash
cd native && cmake --preset mingw-w64-cross && cmake --build --preset mingw-w64-cross-release
```
Or: `powershell -File native/scripts/build_cypha_rest_mingw_wsl.ps1 -RunPytest`

**Native REST server:**
```bash
./build/cypha_rest --cypha ../parity_fixtures/reference.cypha \
                   --f-field-json ../parity_fixtures/f_field.json \
                   --listen 127.0.0.1:8099
```

Full native docs: [`native/README.md`](native/README.md).

---

## Layout

```
Cypha/
├── Cypha.py              # Python reference engine (CyphaDIF, DIFRegressor, …)
├── cypha_studio/         # PySide6 GUI + FastAPI server + dataset/trainer/registry
├── cypha_accel/          # Fused GEMM / GPU acceleration (CuPy optional, NumPy fallback)
├── native/               # C++ core: cypha_core lib, parity tools, cypha_rest, Qt shell
│   ├── include/cypha/    # Public headers
│   ├── src/              # Implementation
│   ├── tools/            # CLI parity / smoke binaries
│   ├── qt/               # Qt 6 shell (cypha_qt_stub + cypha_qt_shell)
│   └── scripts/          # Build helpers (MinGW, CI, smoke runners)
├── docs/                 # Documentation hub — see docs/README.md
├── parity_fixtures/      # Committed .cypha + expected.npz golden assets
├── tests/                # pytest suite (parity + API + native subprocess)
├── scripts/              # Fixture generators, benchmarks, profiling
└── artifacts/            # Generated: profiles/, bench/, tuning/ (gitignored outputs)
```

---

## Docs

**Start here:** [`docs/README.md`](docs/README.md)

| Area | Entry |
|------|-------|
| Verify / status | [`docs/verify/ROADMAP.md`](docs/verify/ROADMAP.md), [`docs/verify/VERIFICATION_STATUS.md`](docs/verify/VERIFICATION_STATUS.md) |
| Port contracts | [`docs/port/PORT_CONTRACT.md`](docs/port/PORT_CONTRACT.md), [`docs/port/PORT_FULL_STACK.md`](docs/port/PORT_FULL_STACK.md) |
| Future directions | [`docs/FUTURE.md`](docs/FUTURE.md) |
| CyphaStudio | [`docs/studio/CYPHA_ENV.md`](docs/studio/CYPHA_ENV.md) |
| Contributing | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Scripts | [`scripts/README.md`](scripts/README.md) |

---

## Upkeep

```bash
make regen-parity         # regenerate parity_fixtures/ after Python math changes
make experiment-ddl       # sync SQLite DDL to artifacts/experiment_schema.sql
bash scripts/run_all_regressions.sh   # regression gate (Linux / WSL)
```

Maintenance checklist: [`docs/verify/MAINTENANCE.md`](docs/verify/MAINTENANCE.md).
