# Cypha documentation

Start here, then open the section that matches what you need.

## Use / run

| Doc | What it is |
|-----|------------|
| [Studio master plan](studio/CYPHA_STUDIO_MASTER_PLAN.md) | Profiling, tests, UX backlog, Studio checklist |
| [Environment variables](studio/CYPHA_ENV.md) | `CYPHA_*` registry, API host/port, CORS, CSV chunking, optional `CYPHA_REGRESSION_HEAD`, REST routes overview |
| [Optional memory & load testing](studio/OPTIONAL_MEMORY_AND_LOAD.md) | tracemalloc, memray, `ab`/Locust notes |
| [GUI threading](studio/STUDIO_THREADING.md) | `QThread` + `SignalBus` rules |

**Run GUI:** `pip install -r cypha_studio/requirements.txt` → `python cypha_studio/main.py`  
**Run headless API:** see [CYPHA_ENV.md](studio/CYPHA_ENV.md) and `cypha_studio/main.py --headless`.

## Develop / verify

| Doc | What it is |
|-----|------------|
| [Verification status](verify/VERIFICATION_STATUS.md) | **Snapshot**: test counts (188 pytest / 33 CTest), known gaps |
| [Roadmap](verify/ROADMAP.md) | All milestones M1–M6 complete; current engineering horizon |
| [Future directions](FUTURE.md) | **Depth**: OpenCL GPU, Qt packaging, Web UI, multi-model, ONNX |
| [Maintenance](verify/MAINTENANCE.md) | **When to regen fixtures / rebuild native / align schema & REST** |
| [Verify plan](verify/VERIFY_PLAN.md) | **Checklist**: debug, profile, benchmark, WSL flows |
| [Repo layout plan](verify/REPO_LAYOUT_PLAN.md) | How docs + `artifacts/` were organized (maintenance reference) |
| [Contributing](../CONTRIBUTING.md) | Setup, PR checklist, extended verify commands |

**Quick tests:** `pytest tests/` (set **`QT_QPA_PLATFORM=offscreen`** when running GUI modules without a display; **`make test`** does this on Unix), `python test_cypha.py`, `python cypha_studio/test_cypha_studio.py` (see Contributing for GUI subsets). **Regression gate:** `scripts/run_all_regressions.sh` / `.ps1`. **Pip:** prefer **`requirements-verify.txt`** + **`cypha_studio/requirements.txt`**; if **`pip install -r`** fails on encoding, use **`requirements-pip-merged.txt`** or Contributing’s one-liner. **After contract or parity changes:** [Maintenance checklist](verify/MAINTENANCE.md).

## Port / native

| Doc | What it is |
|-----|------------|
| [Port contract](port/PORT_CONTRACT.md) | Normative: `.cypha` v3, LLR/softmax/GH, REST JSON |
| [Full stack port](port/PORT_FULL_STACK.md) | Replacing Python core + Studio + REST + Qt (milestones; **§7** = cutover to all-native hot path) |
| [Preprocessor contract](port/PREPROCESSOR_CONTRACT.md) | `preprocessor.json` next to `model.cypha` |
| [parity_fixtures/README.md](../parity_fixtures/README.md) | Committed parity assets |
| [native/README.md](../native/README.md) | Native tree pointer (incl. optional SQLite **`experiment_db_smoke`**) |
| [Experiments schema](port/EXPERIMENTS_SCHEMA.md) | SQLite layout for `ExperimentDB`; DDL via `scripts/export_experiment_schema_sql.py` |

## Benchmarks & GPU

| Doc | What it is |
|-----|------------|
| [BENCHMARK_GPU.md](benchmarks/BENCHMARK_GPU.md) | GPU bench bundle, CuPy notes |
| [Profile improvements (WSL GPU)](benchmarks/PROFILE_IMPROVEMENTS_20260321_WSL_GPU.md) | Example captured run analysis |

## Generated output (repo layout)

| Path | Purpose |
|------|---------|
| `artifacts/profiles/` | cProfile / tracemalloc text (default output for several `scripts/profile_*.py`) |
| `artifacts/bench/` | JSON timing reports (e.g. `bench_gpu_production`) |
| `artifacts/tuning/` | Tuning grid CSV / JSON / profile txt from `tune_quality_performance.py` |

See [scripts/README.md](../scripts/README.md) for script index.
