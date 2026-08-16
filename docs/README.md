# Cypha -- documentation hub

**One product:** `cypha::Cypha` -- classify + regress + sample latents + next-token / text generate. Native C++ only (`cypha_rest`, `cypha_qt_shell`, `cypha_bench_run`).

Living sequence default: **Hybrid GRIA+LSTM L2+Wave2 BPTT (2.664 BPC)** @ 300k lock, with predictive arithmetic coding. See [`reports/ONE_CYPHA_CUTOVER.md`](reports/ONE_CYPHA_CUTOVER.md) and [`research/upgrades/PREDICTIVE_ARITHMETIC_CODING.md`](research/upgrades/PREDICTIVE_ARITHMETIC_CODING.md).

Start here, then open the section that matches what you need.

---

## Quick start

**Native only.** See [`docs/native/NATIVE_QUICKSTART.md`](native/NATIVE_QUICKSTART.md) for install -> validate -> bench -> tune -> REST.

```powershell
# Windows -- build + validate
cmake -S native -B C:\Temp\cypha_build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build C:\Temp\cypha_build --parallel
ctest --test-dir C:\Temp\cypha_build -R native_ --output-on-failure
powershell -File scripts\cypha_native_validate_all.ps1
```

```bash
# Linux / WSL
bash scripts/ci_native_linux.sh
# With Qt compile-check: CYPHA_BUILD_QT=1 bash scripts/ci_native_linux.sh
```

For platform-specific setup see [CONTRIBUTING.md](../CONTRIBUTING.md) and [`packaging/`](../packaging/).

---

## Use / run

| Doc | What it covers |
|-----|----------------|
| [Native quick start](native/NATIVE_QUICKSTART.md) | Build, CTest, `cypha_rest`, `cypha_qt_shell`, `cypha_bench_run` |
| [Environment variables](studio/CYPHA_ENV.md) | `CYPHA_*` registry root, API host/port, REST routes |
| [Optional memory & load testing](studio/OPTIONAL_MEMORY_AND_LOAD.md) | Load-testing notes for REST |
| [One Cypha cutover](reports/ONE_CYPHA_CUTOVER.md) | `cypha::Cypha` ownership, routes, Hybrid default (U06 PGM→Wy opt-in) |

**Run native REST:** `./native/build/cypha_rest --model fixtures/reference.cypha`  
**Run Qt shell:** build with `-DCYPHA_BUILD_QT=ON`, then `cypha_qt_shell` (see [`native/qt/README.md`](../native/qt/README.md))

---

## Develop / verify

| Doc | What it covers |
|-----|----------------|
| [Verification status](verify/VERIFICATION_STATUS.md) | Snapshot: CTest tally via `scripts/cypha_native_validate_all.ps1` / `ctest -N -R native_`, per-fixture status, known gaps |
| [Roadmap](verify/ROADMAP.md) | Milestones M1-M6 complete; current engineering horizon |
| [Maintenance](verify/MAINTENANCE.md) | When to regen fixtures / rebuild native / sync DDL |
| [Verify plan](verify/VERIFY_PLAN.md) | Debug / profile / benchmark / MSVC-Linux workflow checklist |
| [Intelligence statistics](research/intelligence_stats/README.md) | P-space profiler papers (I-V); C++ in `native/include/cypha/intelligence/` |
| [Possible upgrades](research/upgrades/README.md) | Closed / opt-in tracks (default OFF) |
| [Archive](archive/README.md) | Historical reports, studies, plans (not product spine) |
| [C++2023 migration](native/migration/CPLUSPLUS_2023_MASTER_PLAN.md) | Python decommission phases (P7 complete) |
| [Contributing](../CONTRIBUTING.md) | Setup, PR checklist, CTest gate reference |
| [CHANGELOG](../CHANGELOG.md) | Release history |

**Quick gate (matches GitHub Actions CI -- two blocking jobs: `build_and_test`, `windows_msvc`):**
```bash
bash scripts/ci_native_linux.sh
ctest --test-dir native/build -R native_ --output-on-failure
```

**Full native production gate (Windows):**
```powershell
powershell -File scripts\cypha_native_validate_all.ps1
```

---

## Port / native

| Doc | What it covers |
|-----|----------------|
| [Port contract](port/PORT_CONTRACT.md) | Normative: `.cypha` v3, LLR/softmax/GH, Cypha REST, sequence `/generate` |
| [Full stack port](port/PORT_FULL_STACK.md) | Per-milestone record M1-M6 (complete); Python runtime decommissioned |
| [Preprocessor contract](port/PREPROCESSOR_CONTRACT.md) | `preprocessor.json` format next to `model.cypha` |
| [Experiments schema](port/EXPERIMENTS_SCHEMA.md) | SQLite layout for experiments DB |
| [fixtures/README.md](../fixtures/README.md) | Committed parity assets -- inputs + expected outputs |
| [native/README.md](../native/README.md) | C++ build guide: CMake presets, CTest inventory, CUDA, Qt |

---

## Benchmarks

| Doc | What it covers |
|-----|----------------|
| [BENCHMARK_GPU.md](benchmarks/BENCHMARK_GPU.md) | GPU bench bundle notes |
| [Profile improvements (2026-03-21 WSL GPU)](benchmarks/PROFILE_IMPROVEMENTS_20260321_WSL_GPU.md) | Example captured run analysis |

**Run benchmarks (native):**
```bash
cypha_bench_run --list-domains
cypha_bench_run --from-domain 1          # d01 ... d17
cypha_bench_run --report-only            # cross-domain + BASELINE_REPORT.md + figures
native/build/xor_kernel_bench --seeds 3 --passes 8 --kernel-blend 1.0
```

See [RESEARCH_STATUS.md](RESEARCH_STATUS.md) for interpretation of benchmark numbers.

---

## Research status

Canonical journal:

**[`docs/RESEARCH_STATUS.md`](RESEARCH_STATUS.md)** -- current pins, confirmed properties, priorities.

**[`CYPHA_BILL_OF_WORK.md`](../CYPHA_BILL_OF_WORK.md)** -- open items only (historical BoW archived).

**[`docs/research/upgrades/README.md`](research/upgrades/README.md)** -- upgrade index (closed tracks + opt-ins).

Dated investigation reports, CyphaLM studies, and mega-plans live under **[`docs/archive/`](archive/README.md)** (e.g. diagnostic, SOM, continuum closeout, PGM/unified-context notes).

---

## Generated output (repo layout)

| Path | Purpose |
|------|---------|
| `artifacts/profiles/` | Timing / profiling JSON (e.g. XOR kernel LLR sweeps) |
| `artifacts/bench/` | JSON timing reports |
| `artifacts/tuning/` | Tuning grid output from `cypha_tune_run` |

See [scripts/README.md](../scripts/README.md) for the script index (fixture generators, packaging, validation).
