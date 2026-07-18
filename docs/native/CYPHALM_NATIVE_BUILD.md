# CyphaLM Native — Build Guide

How to configure and build **`cypha_lm_native`** and CyphaLM tooling (`cyphalm_bench_native`, parity executables) on Windows and Linux/WSL.

**Tracker:** [`CYPHALM_NATIVE_UPGRADE_MASTER.md`](CYPHALM_NATIVE_UPGRADE_MASTER.md)  
**Port contract:** [`docs/port/PORT_CONTRACT.md`](../port/PORT_CONTRACT.md) §4b

## Quick start (recommended)

**Windows: MSVC (`windows-vs2026-release` / `windows-msvc-release` presets) is the primary, recommended toolchain** — see [`MSVC_TOOLCHAIN_MIGRATION_2026-07-12.md`](../reports/MSVC_TOOLCHAIN_MIGRATION_2026-07-12.md). MSVC is at least as correct on the full CTest suite, faster on this workload (~28% on D17), and the only Windows toolchain that can build the CUDA path at all (MinGW + nvcc is unsupported). MinGW/Ninja is now a secondary path — see "Cross-compile / CI (MinGW)" below.

```powershell
# Windows — MSVC preset (from native/)
cd native
cmake --preset windows-vs2026-release   # or windows-msvc-release on VS 2022
cmake --build build-windows-vs2026 --config Release --target cyphalm_bench_native cyphalm_parity
```

```bash
# Linux / WSL — from repo root
cmake -S native -B /tmp/cypha_native_build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCYPHA_BUILD_EXPERIMENT_DB=OFF \
  -G Ninja
cmake --build /tmp/cypha_native_build --parallel \
  --target cypha_lm_native cyphalm_bench_native cyphalm_parity
```

**Smoke:**

```powershell
native\build-windows-vs2026\Release\cyphalm_bench_native.exe --mode hybrid --profile d17 --n-train 500 --n-eval 100
```

```bash
/tmp/cypha_native_build/cyphalm_bench_native --mode hybrid --profile d17 --n-train 500 --n-eval 100
```

### Optional local MinGW (not CI / not release)

MinGW is **not** a CI or release gate — those use **`windows_msvc`**. Keep MinGW only for optional local/WSL experiments. It cannot build the CUDA path (`CYPHA_ENABLE_CUDA` is unsupported for MinGW targets). Prefer MSVC presets above for Windows work.

Use a build directory **outside OneDrive** (or any cloud-synced folder). Sync tools lock object files and slow Ninja; links can fail with `Permission denied` when relinking a running `cyphalm_bench_native.exe`.

## Why avoid OneDrive build dirs

| Issue | Cause | Fix |
|-------|-------|-----|
| Slow incremental builds | Cloud sync scans every `.obj` / `.o` | Build under `C:\Temp\`, `%LOCALAPPDATA%\cypha_build\`, or WSL `~/build/` |
| Intermittent link failures | File lock during sync | Same — out-of-sync path |
| Huge git noise | Accidental `native/build-*` under synced Desktop | Add local build dirs to global gitignore; never commit build trees |

Source can stay in OneDrive; only the **binary directory** (`-B`) should be local and unsynced.

## CMake options (CyphaLM-relevant)

| Option | Default | CyphaLM note |
|--------|---------|--------------|
| `CYPHA_BUILD_EXPERIMENT_DB` | OFF (unset) | Set **OFF** to skip SQLite experiment targets — not needed for LM bench/parity |
| `CYPHA_FETCH_SQLITE3_AMALGAMATION` | ON | Set **OFF** on Windows when you have no network or want faster configure; LM targets do not need SQLite |
| `CMAKE_BUILD_TYPE` | — | Use **Release** for bench timings |
| `CYPHA_ENABLE_CUDA` | OFF | Optional GPU path for **`cypha::accel`** (encode / fused LLR / softmax / gate). **Windows: MSVC only** — see [`ACCEL_CUDA.md`](ACCEL_CUDA.md). CyphaLM (`cypha_lm_native`) remains CPU-only. |
| `CYPHA_BUILD_QT` | OFF | Unrelated to CyphaLM |

`cypha_lm_native` is always built when configuring `native/`; sources are collected via `file(GLOB … src/cyphalm/*.cpp)`.

## OpenMP

CMake runs `find_package(OpenMP)` and links **`OpenMP::OpenMP_CXX`** into `cypha_lm_native` when found.

- **Configure log:** `cypha_lm_native: OpenMP enabled` or `OpenMP not found; batch uses std::thread`
- **Runtime:** `cyphalm_bench_native --threads N` calls `set_thread_count(N)`; `0` leaves the OpenMP default (usually all cores)
- **Windows MinGW:** install OpenMP support with your GCC toolchain (Strawberry Perl/MinGW or MSYS2 `mingw-w64-x86_64-gcc` with `libgomp`)
- **Windows MSVC:** OpenMP is part of the C++ workload; `/openmp` is applied via the imported target
- **Linux:** `sudo apt install libomp-dev` (LLVM) or rely on GCC `libgomp` (usually present with `build-essential`)

Batch matmul in `cyphalm_batch.cpp` uses `#ifdef _OPENMP` with a `std::thread` fallback when OpenMP is absent.

## Windows toolchains

### MSVC (Visual Studio 2022+) — recommended, primary

```powershell
cd native
cmake --preset windows-msvc-release
cmake --build build-windows-msvc --config Release --target cyphalm_bench_native
```

Or VS 2026: `windows-vs2026-release` preset (binaries under `build-windows-vs2026/Release/`). Multi-config generators place binaries under `build-windows-msvc/Release/`.

### Optional: Ninja + MinGW (Strawberry / MSYS2)

Not used by CI or releases. See [`MSVC_TOOLCHAIN_MIGRATION_2026-07-12.md`](../reports/MSVC_TOOLCHAIN_MIGRATION_2026-07-12.md). Cannot build the CUDA path.

```powershell
cmake -S native -B C:\Temp\cypha_native_build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCYPHA_BUILD_EXPERIMENT_DB=OFF -DCYPHA_FETCH_SQLITE3_AMALGAMATION=OFF
cmake --build C:\Temp\cypha_native_build --parallel
```

Ensure `ninja` and `g++` are on `PATH` (Strawberry: `C:\Strawberry\c\bin`).

## Linux / WSL

```bash
cd native
cmake --preset wsl-gcc-release    # or: cmake -S . -B /tmp/cypha_native_build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build --preset wsl-gcc-release-build --target cyphalm_bench_native
```

Install deps (Debian/Ubuntu): `sudo apt install cmake ninja-build g++ libomp-dev`

WSL builds against a repo on `/mnt/c/...` work but are slower than a clone under `~/Cypha`. Prefer `~/` for the build dir even when source is on `/mnt/c`.

## CyphaLM targets

| Target | Type | Role |
|--------|------|------|
| `cypha_lm_native` | STATIC lib | All `src/cyphalm/*.cpp` |
| `cyphalm_bench_native` | executable | BPC bench CLI |
| `cyphalm_parity` | executable | Meta parity runner |
| `cyphalm_char_lstm_parity` | executable | Char LSTM fixture |
| `cyphalm_ssm_parity` | executable | SSM one-step golden |
| `cyphalm_model_parity` | executable | 10-token model scaffold |
| `cyphalm_hebbian_parity` | executable | Hebbian hooks golden |

## Test

```powershell
ctest --test-dir C:\Temp\cypha_native_build -R native_cyphalm --output-on-failure
ctest --test-dir native/build -R native_qt --output-on-failure
```

Override binary paths: `CYPHALM_BENCH_NATIVE_BIN`, `CYPHALM_PARITY_BIN`, `CYPHALM_CHAR_LSTM_PARITY_BIN`.

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `OpenMP not found` | Install OpenMP runtime/dev package for your compiler; rebuild |
| Configure hangs on SQLite download | `-DCYPHA_FETCH_SQLITE3_AMALGAMATION=OFF` |
| `ninja: error: loading build.ninja` | Re-run `cmake -S native -B <dir>` after moving the build tree |
| Bench prints `"synthetic": true` | Expected when bench corpus files are missing; BPC is still finite for smoke |
| Parity CTest disabled | Run native parity fixture workflow (see MAINTENANCE.md) for char_lstm sidecar |
