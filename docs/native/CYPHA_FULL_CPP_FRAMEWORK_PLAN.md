# Cypha Full C++ Framework — Master Port Plan

**Goal:** **P7 complete** — all production runtime, bench, REST, and validation runs in native C++ only.

**Status:** **v2.5 complete — SHIPPED** ([`v2.2.8` release](https://github.com/odin-loki/Cypha/releases/tag/v2.2.8) — Linux + Windows installers; **CI fully green** — two blocking jobs: **`build_and_test`**, **`mingw_cross`**).

Normative contracts: [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md).  
Quick start: [`NATIVE_QUICKSTART.md`](NATIVE_QUICKSTART.md).  
Changelog: [`CHANGELOG.md`](../../CHANGELOG.md).

---

## Validation gate (all green)

```
scripts/cypha_native_validate_all.ps1  → OK
  106 CTests │ fig01–fig09 PNG │ /dif/retrieve │ tune dry-run
```

GitHub Actions **CI** (`a3b48c4`+): Linux CTest + ctest, MinGW PE smoke. CUDA jobs removed — validate locally. Release **v2.2.8** installers published.

Build: `C:\Temp\cypha_full_cpp_build`

---

## Full baseline lock (300k d17) ✅

| Native hybrid @ 300k | **2.897 BPC** (Python 2.873, Δ +0.024) |

---

## Shipped (git)

| **`v2.2.8`** tag | **Latest release** — two CI jobs blocking (`build_and_test`, `mingw_cross`); CUDA local-only |
| **`v2.2.7`** tag | MSVC CUDA CI green |
| **`v2.2.4`** tag | CI green; portable checkpoint fixtures; Studio GUI import shims |
| **`v2.2.3`** tag | Release packaging hardening — Linux + Windows native installers on GitHub Releases |
| **`v2.2.0`** tag | Core framework release (236 files) |

**Published:** `git push origin main --tags` ✅

---

## Architecture

```
cypha_qt_shell (9 tabs + Settings/Confusion dialogs)
cypha_rest (/dif/*, /lm/*, /route/*)
cypha_bench_run │ cypha_bench_report │ cypha_tune_run │ cypha_diagnostics_run
cyphalm_bench_native │ cypha_lm_native │ cypha_core (C++ library)
```

Bench configs, data, and reports live under **`bench/`**; parity goldens under **`fixtures/`**.

---

## Phase completion — ALL ✅

Phases 0–14 complete including Qt Studio parity (Settings, Confusion Matrix), full figure set, tune configs, 300k baseline, release packaging.

---

## Optional future work

- **RPSM Option A — CyphaDIF matrix refactor:** unified Ψ_mu / Ψ_var state, batched LLR/GEMM, parity-validated. Spec: [`docs/research/upgrades/RPSM_COMBINED_SPEC.md`](../research/upgrades/RPSM_COMBINED_SPEC.md). Leads to Option B (CyphaLM sequence layer) in [`docs/FUTURE.md`](../FUTURE.md) §10.
- PySide6 chat widget (Qt Predict tab is feature-vector, not chat)
- Local GPU microbench on a CUDA box (`cuda_smoke --bench`)

---

## Quick commands

```powershell
powershell -File scripts\cypha_native_validate_all.ps1
powershell -File scripts\cypha_bench_full_baseline.ps1
git push origin main --tags   # publish release (see v2.2.8 on GitHub Releases)
```
