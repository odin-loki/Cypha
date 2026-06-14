# Scripts index

Native build, validation, and release helpers. Narrative “when to run what” lives in **[`docs/README.md`](../docs/README.md)**. **Fixtures / native / schema cadence:** **[`docs/verify/MAINTENANCE.md`](../docs/verify/MAINTENANCE.md)**.

## Validation gates (primary)

| Script | Purpose | Typical output |
|--------|---------|----------------|
| `cypha_native_validate_all.ps1` | Windows full gate: rebuild + CTest (`-R native_`) + bench fig01–09 + tune dry-run + REST smoke (`-SkipBuild` after rebuild) | console |
| `ci_native_linux.sh` | Linux/WSL mirror of CI **`build_and_test`**: cmake + `ctest -R native_` | console |
| `ci_federated_tls_linux.sh` | Linux/WSL mirror of optional CI **`federated_tls`**: `-DCYPHA_ENABLE_OPENSSL=ON` + `ctest -R native_federated_tls` | console |
| `cyphalm_native_validate.ps1` | CyphaLM native CTest subset + checkpoint smoke | console |
| `wsl_verify.sh` | WSL: native build + CTest + optional REST smoke (`RUN_NATIVE=1`) | console |
| `build_native_wsl.ps1` | WSL CMake build in `native/build-wsl` + optional ctest | console |

```powershell
# Windows — full production gate
powershell -File scripts\cypha_native_validate_all.ps1
```

```bash
# Linux / WSL — CI mirror
bash scripts/ci_native_linux.sh
ctest --test-dir native/build -R native_ --output-on-failure
```

## Release packaging

| Script | Purpose |
|--------|---------|
| `package_release_linux.sh` | Bundle Linux release tarball |
| `package_release_windows.sh` | Bundle Windows release zip |
| `native/scripts/package_windows_qt.ps1` | Windows Qt deployment helper |

## Bench / tune / diagnostics

| Script / binary | Purpose | Output |
|-----------------|---------|--------|
| **`cypha_bench_run`** (native) | Multi-domain benchmark (d01–d17 + cross-domain) | stdout / JSON |
| **`cypha_bench_report`** (native) | Regenerate bench report figures from JSON artifacts | disk |
| **`cypha_tune_run`** (native) | Sweep JSON → per-cell native bench | disk |
| **`cypha_diagnostics_run`** (native) | Phases 1–4 validation orchestrator (parity exes + inline checks) | stdout |
| `cypha_bench_full_baseline.ps1` | Capture baseline bench profile | `artifacts/profiles/` |
| `cypha_tune_smoke.ps1` | Dry-run tune sweep smoke | console |
| `cyphalm_native_run_modes.ps1` | CyphaLM native mode matrix | console |
| `cyphalm_native_sweep.ps1` / `cyphalm_native_sweep_safe.ps1` | CyphaLM config sweeps | disk |
| `wsl_bench_gpu.sh` | WSL GPU bench helper | console |

## Kernel LLR / XOR profiling

| Binary | Purpose | Output |
|--------|---------|--------|
| **`xor_kernel_bench`** (native) | XOR linear vs Nyström kernel via `dif_train_step_vector` + kernel update | stdout JSON; CTest `native_xor_kernel_bench_smoke` |
| **`kernel_llr_parity`** (native) | Kernel LLR vs `fixtures/kernel_llr/` | CTest `native_kernel_llr` |

## REST / Qt helpers

| Script | Purpose |
|--------|---------|
| `run_cypha_qt_windows.ps1` | Launch `cypha_rest` + Qt shell on Windows |
| `native/scripts/build_cypha_rest_mingw_wsl.ps1` | MinGW cross-build `cypha_rest.exe` in WSL |
| `native/scripts/smoke_cypha_rest_mingw.ps1` | Subprocess REST smoke vs parity fixtures |
| `loadtest_ab_predict_example.sh` / `.ps1` | Example `ab` against live `/predict` |

## Parity fixtures

**`fixtures/`** are **frozen committed goldens**. Do not regenerate from removed Python generators. After intentional contract changes, update sidecars manually or extend native parity tools under `native/tools/`, then run **`ctest -R native_<name>`**. See [`fixtures/README.md`](../fixtures/README.md) and [`docs/verify/MAINTENANCE.md`](../docs/verify/MAINTENANCE.md).

| Native parity binary | Fixture dir | CTest |
|---------------------|-------------|-------|
| `cypha_parity` | root (`reference.cypha`, `native_parity.bin`) | `native_parity` |
| `batch_llr_parity` | `batch_llr/` | `native_batch_llr` |
| `quantile_dif_train_parity` | `quantile_dif_train/` | `native_quantile_dif_train` |
| `memory_train_parity` | `memory_train/` | `native_memory_train` |
| `preprocessor_parity` | `preprocessor/` | `native_preprocessor` |
| `preprocessor_fit_parity` | `preprocessor_fit/`, `preprocessor_fit_no_scale/` | `native_preprocessor_fit` |
| `regression_m4_parity` | `regression_m4/` | `native_regression_m4` |
| *(see `native/README.md`)* | *(24+ dirs)* | `ctest -R native_` |

Override binary paths with **`CYPHA_*_BIN`** env vars (see [`native/README.md`](../native/README.md)).

> **Note on `.sh` scripts:** `wsl_verify.sh`, `ci_native_linux.sh`, and release packagers use **`eol=lf`** in **`.gitattributes`**. On Unix, **`chmod +x scripts/*.sh`** if needed.
