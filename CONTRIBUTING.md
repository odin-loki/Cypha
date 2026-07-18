# Contributing

**Documentation hub:** [`docs/README.md`](docs/README.md) (organized by use / verify / port). **Research journal:** [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md). **Script index:** [`scripts/README.md`](scripts/README.md). **Regen / native / schema upkeep:** [`docs/verify/MAINTENANCE.md`](docs/verify/MAINTENANCE.md).

## Setup

**Release install (recommended for users):** prebuilt bundles via [`packaging/README.md`](packaging/README.md) (`install_release_linux.sh` / `install_release_windows.ps1`).

**Build from source:** see [`docs/native/NATIVE_QUICKSTART.md`](docs/native/NATIVE_QUICKSTART.md).

**Manual build:**

```powershell
# Windows — MSVC Release (preferred; matches CI windows_msvc)
cmake --preset windows-msvc-release
cmake --build native/build-windows-msvc --config Release --parallel

# Or Ninja + MSVC outside OneDrive
cmake -S native -B C:\Temp\cypha_build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build C:\Temp\cypha_build --parallel
```

```bash
# Linux / WSL
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build -j$(nproc)
```

Optional CMake flags: **`-DCYPHA_ENABLE_CUDA=ON`**, **`-DCYPHA_BUILD_QT=ON`**, **`-DCYPHA_BUILD_EXPERIMENT_DB=ON`**.

## Native production gate (required after C++ changes)

```powershell
# Windows — full gate: rebuild + ~160 CTests + bench smoke + tune dry-run + REST smoke
powershell -File scripts\cypha_native_validate_all.ps1
```

```bash
# Linux / WSL mirror of CI native step
bash scripts/ci_native_linux.sh
# With Qt compile-check: CYPHA_BUILD_QT=1 bash scripts/ci_native_linux.sh
```

**Manual subset:**

```bash
ctest --test-dir native/build -R native_ --output-on-failure
```

After adding **`add_test(NAME native_…)`**, ensure it appears in **`ctest -N`** under the `native_` filter.

## Before you share or archive changes

```bash
# CTest gate (required after C++ or fixture changes)
ctest --test-dir native/build -R native_ --output-on-failure
```

Update **`fixtures/`** only when the frozen contract intentionally changes — see [`docs/verify/MAINTENANCE.md`](docs/verify/MAINTENANCE.md).

Set **`CYPHA_*_BIN`** env vars to point smoke tests at your build tree (see `scripts/cypha_native_validate_all.ps1`).

## Native `cypha_rest`

CI runs **two blocking jobs** (`.github/workflows/ci.yml`): **`build_and_test`** (Ubuntu native build + CTest) and **`windows_msvc`** (native MSVC Release). The Linux **`build_and_test`** job gates REST smokes via **`CYPHA_REST_BIN`**. CUDA is optional for local builds only — configure with `-DCYPHA_ENABLE_CUDA=ON` and run **`native_cuda_smoke`** / **`native_score_batch`** before merging accel changes (see [`docs/native/ACCEL_CUDA.md`](docs/native/ACCEL_CUDA.md)).

**Local (Linux / WSL ELF):** install **`sudo apt-get install -y libsqlite3-dev`** (optional M6 CTest **`native_experiment_db_smoke`**), then either **`bash scripts/ci_native_linux.sh`** or manually:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build -j$(nproc)
ctest --test-dir native/build --output-on-failure
```

Without **`libsqlite3-dev`**, CMake skips **`experiment_db_smoke`**; other CTest targets still build.

**Windows (optional — not CI/release gate):** MinGW cross-build from WSL: `powershell -File native/scripts/build_cypha_rest_mingw_wsl.ps1` (add **`-AllTargets`** for every `*.exe`). Prefer **MSVC** presets above for local validation and release parity.

**MoE sidecar:** `fixtures/regression_head.json`. See **`docs/port/PORT_CONTRACT.md`** §3.

## Principles

- Prefer **one** code path for inference math (`score_matrix` ↔ `batch_infer`).
- Use **`.cypha` v3** binary format for anything that must persist.
- If you add a feature, add a **CTest** and, when it affects numbers, refresh **`fixtures/`** and note it in the PR.

## Changelog

Update [`CHANGELOG.md`](CHANGELOG.md) when:
- A bug is fixed that affects numeric outputs.
- A parity fixture is regenerated.
- A new domain / encoder / regressor is added.
- A milestone benchmark is completed.

## Layout

- `native/` — authoritative runtime (Cypha, REST, Qt shell, bench/tune/diagnostics).
- `fixtures/` — committed goldens for CTest parity.
- `docs/reports/` — permanent experiment records; committed, not regenerated.
- `docs/RESEARCH_STATUS.md` — benchmark journal and research priorities.
