# Cypha prototype — debug, profile, verify

This document is the **master checklist** for proving the **native C++ production stack** passes CI and local validation.

For a **living snapshot** of automated tests and known gaps, see [`VERIFICATION_STATUS.md`](VERIFICATION_STATUS.md). For **what to regen and rebuild** after contract changes, see [`MAINTENANCE.md`](MAINTENANCE.md).

**Remote CI:** [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) — **four blocking jobs** on push to `main`:
1. **`build_and_test`** (Ubuntu) — `cmake -DCYPHA_BUILD_QT=ON` + **`ctest -R native_`** (GUI exec tests excluded on headless); WikiText-2 corpus; **`CYPHA_REST_BIN`** and `QT_QPA_PLATFORM=offscreen`.
2. **`mingw_cross`** — MinGW Windows PE artifact smoke (`cypha_rest.exe`, `cypha_bench_run.exe`, …).
3. **`windows_cuda_msvc`** — MSVC + CUDA compile smoke.
4. **`linux_cuda`** — GCC + nvcc compile smoke.

**Release:** tag `v*` → [`.github/workflows/release.yml`](../../.github/workflows/release.yml) publishes Linux + Windows installer archives.

## 1. Scope

| Layer | Path | Role |
|--------|------|------|
| Core engine | `cypha_core` | DIF classifier/regressors, encoders, save/load |
| REST | `cypha_rest` | JSON inference/training server |
| GUI | `cypha_qt_shell` | Qt 6 desktop shell |
| Bench / tune | `cypha_bench_run`, `cypha_tune_run` | Multi-domain benchmark and sweeps |

## 2. Environment (WSL / Windows)

- **Build:** CMake + Ninja — see [`native/README.md`](../../native/README.md).
- **Repo path (WSL example):** `/mnt/c/Users/<you>/OneDrive/Desktop/Cypha`

### One-shot verification

```bash
cd /path/to/Cypha
bash scripts/ci_native_linux.sh
# Optional: REST smoke after build
RUN_NATIVE=1 bash scripts/wsl_verify.sh
```

```powershell
powershell -File scripts\cypha_native_validate_all.ps1
```

### Native `cypha_rest` on Windows (MinGW cross-build in WSL)

From **PowerShell** at repo root: `powershell -File native/scripts/build_cypha_rest_mingw_wsl.ps1`. Smoke: `powershell -File native/scripts/smoke_cypha_rest_mingw.ps1`.

**M1 / `cypha_parity`:** **`reference.cypha`** includes **Tier-1**; C++ **`from_root`** restores **`ctx_*`** ([`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md) §4; **`ctest -R native_parity`**).

### Dependencies

- **Headless gate:** cmake build + **`ctest -R native_`**
- **Full Qt:** **`-DCYPHA_BUILD_QT=ON`** + `qt6-base-dev`; headless CI compiles Qt but excludes GUI exec CTests

## 3. Testing matrix

### 3.1 Automated (must pass)

1. **`ctest --test-dir native/build -R native_ --output-on-failure`** — core parity, registry, REST contract smokes.
2. **`ctest -R native_qt`** (optional Qt build) — offscreen Qt smoke tests.
3. **`native/scripts/smoke_cypha_rest_mingw.ps1`** — subprocess REST vs fixtures when **`CYPHA_REST_BIN`** is set.

### 3.2 Benchmark / regression (should pass before release)

**`cypha_bench_run --from-domain 1`** — accuracy/latency vs sklearn baselines on standard datasets.

- Section 9 (streaming intrusion) uses `/tmp/X_intrusion.npy` when present; otherwise a **fixed synthetic stream**.
- Baseline capture: `cypha_bench_run --from-domain 1 2>&1 | tee artifacts/profiles/benchmark_baseline.txt`

### 3.3 Manual / integration

4. **Studio GUI:** `cypha_qt_shell` — train, save, registry, inference. Automated: **`ctest -R native_qt`** offscreen.
5. **API:** start **`cypha_rest`** and hit `/health`, `/predict`, `/update` per [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md) §3.
6. **Binary round-trip:** save/load `.cypha`; **`ctest -R native_memory_train_roundtrip`** for buffer parity.

## 4. Profiling (native)

| Tool | Use |
|------|-----|
| **`cypha_bench_run`** | Domain timing and accuracy regression |
| **`xor_kernel_bench`** | Kernel LLR XOR profile (JSON stdout) |
| **`cypha_diagnostics_run`** | Parity orchestration + inline checks |
| **CUDA builds** | `-DCYPHA_ENABLE_CUDA=ON` — see [`BENCHMARK_GPU.md`](../benchmarks/BENCHMARK_GPU.md) |

Porting hint: hot paths are **encode → score_matrix → softmax/GH gate**; native **`cypha::accel`** covers fused LLR.

## 5. Debugging checklist

- [x] **`ctest -R native_`** green — CI + local gate
- [ ] Benchmark within expected bands vs `artifacts/profiles/benchmark_baseline.txt`
- [x] Registry save/load identical inference on sample batch — native registry CTests
- [x] Document limitations — [`VERIFICATION_STATUS.md`](VERIFICATION_STATUS.md)

## 6. Definition of “ready to ship”

1. **Correctness:** **`ctest -R native_`** + benchmark regression satisfied.
2. **Contract:** [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md) frozen for `.cypha`, REST, bench §6.
3. **Parity:** committed **`fixtures/`** match native tools.

## 7. Doc index

- [`docs/README.md`](../README.md) — hub
- [`ROADMAP.md`](ROADMAP.md) — phases
- [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md) — normative contracts
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — PR checklist
- **`fixtures/`** — frozen goldens; see [`MAINTENANCE.md`](MAINTENANCE.md)

---

*Maintainers: run **`scripts/cypha_native_validate_all.ps1`** or **`scripts/ci_native_linux.sh`** after substantive native changes.*
