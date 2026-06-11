# Cypha Full C++ Framework — Master Port Plan

**Goal:** Python becomes prototyping-only; **all production runtime, bench, REST, and validation** runs in native C++.

**Status:** **v2.5 complete — SHIPPED** (`v2.2.0` tag + v2.5 follow-up commit).

Normative contracts: [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md).  
Quick start: [`NATIVE_QUICKSTART.md`](NATIVE_QUICKSTART.md).  
Changelog: [`CHANGELOG.md`](../../CHANGELOG.md).

---

## Validation gate (all green)

```
scripts/cypha_native_validate_all.ps1  → OK
  52 CTests │ 155 pytest │ fig01–fig09 PNG │ /dif/retrieve │ tune dry-run
```

Build: `C:\Temp\cypha_full_cpp_build`

---

## Full baseline lock (300k d17) ✅

| Native hybrid @ 300k | **2.897 BPC** (Python 2.873, Δ +0.024) |

---

## Shipped (git)

| Tag / commit | Contents |
|--------------|----------|
| **`v2.2.0`** (`ee9b46e`) | Full C++ framework — 236 files |
| **v2.5 follow-up** | Qt Settings + Confusion Matrix dialogs, doc sync |

**Push:** `git push origin main --tags` when ready for GitHub Release.

---

## Architecture

```
cypha_qt_shell (9 tabs + Settings/Confusion dialogs)
cypha_rest (/dif/*, /lm/*, /route/*)
cypha_bench_run │ cypha_tune_run │ cypha_diagnostics_run
cypha_bench_native │ cypha_lm_native │ cypha_som │ cypha_core
```

---

## Phase completion — ALL ✅

Phases 0–14 complete including Qt Studio parity (Settings, Confusion Matrix), full figure set, tune configs, 300k baseline, release packaging.

---

## Optional future work

- CUDA GPU runtime on self-hosted runner
- PySide6 chat widget (Qt Predict tab is feature-vector, not chat)

---

## Quick commands

```powershell
powershell -File scripts\cypha_native_validate_all.ps1
powershell -File scripts\cypha_bench_full_baseline.ps1
git push origin main --tags   # publish v2.2.0 release
```
