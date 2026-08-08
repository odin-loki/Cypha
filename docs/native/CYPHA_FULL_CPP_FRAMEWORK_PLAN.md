# Cypha Full C++ Framework — Master Port Plan

**Goal:** **P7 complete** — all production runtime, bench, REST, and validation runs in native C++ only.

**Status:** **v2.5 complete — SHIPPED** ([`v2.3.25` release](https://github.com/odin-loki/Cypha/releases/tag/v2.3.25) — One Cypha cutover; Linux + Windows MSVC installers, AppImage, arxiv bundle; **CI fully green** — two blocking jobs: **`build_and_test`**, **`windows_msvc`**).

Normative contracts: [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md).  
Quick start: [`NATIVE_QUICKSTART.md`](NATIVE_QUICKSTART.md).  
Changelog: [`CHANGELOG.md`](../../CHANGELOG.md).

---

## Validation gate (all green)

```
scripts/cypha_native_validate_all.ps1  → OK
  214 CTests | fig01-fig09 PNG | /retrieve | tune dry-run
```

GitHub Actions **CI**: Linux **`build_and_test`** + **`windows_msvc`** (MSVC Release). MinGW is optional local only — not a CI/release gate. CUDA jobs removed — validate locally. Release **v2.3.25** installers published.

Build: `C:\Temp\cypha_full_cpp_build`

---

## Full baseline lock (300k d17) ✅

**Canonical pin:** **2.873 BPC** (`bench/BASELINE_LOCK.json` → `d17_hybrid_baseline`). Full reconciliation: [`docs/reports/BASELINE_PIN_CANONICAL_2026-07-17.md`](../archive/reports/BASELINE_PIN_CANONICAL_2026-07-17.md).

| Figure | BPC | Role |
|--------|-----|------|
| **Pinned hybrid (canonical)** | **2.873** | Regression / `-Production` validator (±0.05) |
| Native build6 sweep (2026-06-10) | 2.892 | Historical sweep — not an alternate lock |
| Early v2.5 framework note | 2.897 | Historical release-time figure — superseded |
| Latest overnight native run | 2.864 | `overnight_results` in BASELINE_LOCK.json |

---

## Shipped (git)

| **`v2.3.25`** tag | **Latest release** — One Cypha cutover; two CI jobs blocking (`build_and_test`, `windows_msvc`); CUDA local-only |
| **`v2.3.24`** tag | Prior release — same CI layout |
| **`v2.2.8`** tag | Prior release — same CI layout |
| **`v2.2.7`** tag | MSVC CUDA CI green |
| **`v2.2.4`** tag | CI green; portable checkpoint fixtures; Studio GUI import shims |
| **`v2.2.3`** tag | Release packaging hardening — Linux + Windows native installers on GitHub Releases |
| **`v2.2.0`** tag | Core framework release (236 files) |

**Published:** `git push origin main --tags` ✅

---

## Architecture

```
cypha_qt_shell (9 tabs + Settings/Confusion dialogs)
cypha_rest (/sample, /retrieve, /sequence/*, /route/*)
cypha_bench_run │ cypha_bench_report │ cypha_tune_run │ cypha_diagnostics_run
cyphalm_bench_native │ cypha_core (C++ library; cypha_lm_native INTERFACE alias)
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
git push origin main --tags   # publish release (see v2.3.25 on GitHub Releases)
```
