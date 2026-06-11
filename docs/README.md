# Cypha — documentation hub

Start here, then open the section that matches what you need.

---

## Quick start

```bash
# Python reference stack (headless, no GUI)
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-verify.txt

# Full Studio (GUI + REST)
pip install -r cypha_studio/requirements.txt
python cypha_studio/main.py

# Native C++ core (Linux / WSL)
bash scripts/ci_native_linux.sh          # builds + runs CTest + optional pytest drift check

# Windows installer (builds Qt shell + cypha_rest)
# See install/install_windows.ps1
```

For platform-specific setup see [CONTRIBUTING.md](../CONTRIBUTING.md) and [`install/`](../install/).

---

## Use / run

| Doc | What it covers |
|-----|----------------|
| [Environment variables](studio/CYPHA_ENV.md) | `CYPHA_*` registry root, API host/port, CORS, **`CYPHA_LM_CHECKPOINT`**, REST routes |
| [GUI threading](studio/STUDIO_THREADING.md) | `QThread` + `SignalBus` rules for the PySide6 Studio |
| [Optional memory & load testing](studio/OPTIONAL_MEMORY_AND_LOAD.md) | tracemalloc, memray, `ab` / Locust notes |

**Run GUI:** `python cypha_studio/main.py`  
**Run headless REST:** `python -m uvicorn cypha_studio.server.api:app --host 0.0.0.0 --port 8765`  
**Run native REST:** `./native/build/cypha_rest --model parity_fixtures/reference.cypha`

---

## Develop / verify

| Doc | What it covers |
|-----|----------------|
| [Verification status](verify/VERIFICATION_STATUS.md) | Snapshot: test counts (~274 pytest on CI / 52 CTest), per-fixture inventory, known gaps |
| [Roadmap](verify/ROADMAP.md) | Milestones M1–M6 complete; current engineering horizon (Phase 5) |
| [Maintenance](verify/MAINTENANCE.md) | When to regen fixtures / rebuild native / sync DDL |
| [Verify plan](verify/VERIFY_PLAN.md) | Debug / profile / benchmark / WSL workflow checklist |
| [Future directions](FUTURE.md) | CUDA GPU, Qt UX polish, packaged binary, Web UI, multi-model serving, ONNX |
| [Contributing](../CONTRIBUTING.md) | Setup, PR checklist, full test command reference |
| [CHANGELOG](../CHANGELOG.md) | Release history and what changed in each milestone |

**Quick gate (matches GitHub Actions CI):**
```bash
bash scripts/ci_native_linux.sh                    # native CTest (+ optional drift pytest)
pytest tests/ cypha_lm/model/tests/ -q             # full CI pytest (ignore test_gui_qtbot.py on headless)
python test_cypha.py                               # 54 deterministic checks on Cypha.py math
python cypha_studio/test_cypha_studio.py           # 48 pipeline checks
make test                                          # Unix: QT_QPA_PLATFORM=offscreen + pytest
```

**Full native production gate (Windows):**
```powershell
powershell -File scripts\cypha_native_validate_all.ps1
```

---

## Port / native

| Doc | What it covers |
|-----|----------------|
| [Port contract](port/PORT_CONTRACT.md) | Normative: `.cypha` v3, LLR/softmax/GH, CyphaDIF REST, **CyphaLM `/generate` §4** |
| [Full stack port](port/PORT_FULL_STACK.md) | Per-milestone record M1–M6 (complete) |
| [Preprocessor contract](port/PREPROCESSOR_CONTRACT.md) | `preprocessor.json` format next to `model.cypha` |
| [Experiments schema](port/EXPERIMENTS_SCHEMA.md) | SQLite layout for `ExperimentDB` |
| [parity_fixtures/README.md](../parity_fixtures/README.md) | Committed parity assets — inputs + expected outputs |
| [native/README.md](../native/README.md) | C++ build guide: CMake presets, CTest inventory, CUDA, Qt |

---

## Benchmarks

| Doc | What it covers |
|-----|----------------|
| [BENCHMARK_GPU.md](benchmarks/BENCHMARK_GPU.md) | GPU bench bundle, CuPy notes |
| [Profile improvements (2026-03-21 WSL GPU)](benchmarks/PROFILE_IMPROVEMENTS_20260321_WSL_GPU.md) | Example captured run analysis |

**Run benchmarks:**
```bash
python benchmark_baseline.py                     # 17-domain benchmark (writes cypha_bench/BASELINE_REPORT.md)
python benchmark.py                              # full sklearn regression oracle
python scripts/profile_real_datasets.py --fast   # sklearn tabular profiling
python scripts/gpu_fullbench.py                  # encode+LLR+softmax GPU vs CPU timing
python scripts/run_cypha_lm_report.py            # CyphaLM experiments + figures
```

See [RESEARCH_STATUS.md](RESEARCH_STATUS.md) for the interpretation of all benchmark numbers.

---

## Research status and experimental reports

For the full picture of where the project stands — benchmark numbers, confirmed properties, hard limits, hypothesis ledger, and next priorities — read:

**[`docs/RESEARCH_STATUS.md`](RESEARCH_STATUS.md)** — the canonical research journal.

**[`docs/MULTI_VIEW_TRAINING_PLAN.md`](MULTI_VIEW_TRAINING_PLAN.md)** — planned multi-view online training (CyphaLM Phase 1, CyphaDIF Phase 2): structure-preserving reorderings, view schedules, execution roadmap.

Permanent investigation reports:

| Report | What happened |
|--------|---------------|
| [DIAGNOSTIC_REPORT.md](reports/DIAGNOSTIC_REPORT.md) | Full diagnostic run (2026-05-30): XOR ceiling confirmed, bugs found (+23.5 pp fixes). |
| [SOM_UPGRADE_REPORT.md](reports/SOM_UPGRADE_REPORT.md) | SOM/GNG/GRIA/Hebbian/temporal upgrade evaluation: all six benchmarked. Default OFF. |
| [BENCH_TUNING_REPORT.md](reports/BENCH_TUNING_REPORT.md) | Before/after tuning deltas across all 17 benchmark domains. |
| [BENCH_ARCH_TUNING_REPORT.md](reports/BENCH_ARCH_TUNING_REPORT.md) | Architecture hyperparameter grid search results. |
| [BENCH_ARCH_RESCORE_REPORT.md](reports/BENCH_ARCH_RESCORE_REPORT.md) | Post-architecture rescore with best config. |
| [BENCH_UPGRADE_REPORT.md](reports/BENCH_UPGRADE_REPORT.md) | Deliberation and SOM upgrade combined benchmark effects. |
| [BENCH_PAPER.md](reports/BENCH_PAPER.md) | Short paper: regression-competent DIF routing, variant-aware profiles, arch search fixes. |

---

## Generated output (repo layout)

| Path | Purpose |
|------|---------|
| `artifacts/profiles/` | cProfile / tracemalloc text (output from `scripts/profile_*.py`) |
| `artifacts/bench/` | JSON timing reports (e.g. `bench_gpu_production`) |
| `artifacts/tuning/` | Tuning grid CSV / JSON from `tune_quality_performance.py` |

See [scripts/README.md](../scripts/README.md) for the full script index.
