# Cypha Full C++ Framework — Master Port Plan

**Goal:** Python becomes prototyping-only; **all production runtime, bench, REST, and validation** runs in native C++.

**Status:** **v2.5 complete** — full framework + Studio Qt shell + 300k baseline + release docs (June 2026).

Normative contracts: [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md).  
Quick start: [`NATIVE_QUICKSTART.md`](NATIVE_QUICKSTART.md).

---

## Validation gate (all green)

```
scripts/cypha_native_validate_all.ps1
  CTest native_*     → 52/52 passed
  pytest             → 155+ passed
  rest_dif_smoke     → POST /dif/retrieve
  bench PNG          → fig01–fig09 (manifest-driven)
  cypha_tune_smoke   → 3 configs dry-run (live with -TuneSmoke)
```

Build: `C:\Temp\cypha_full_cpp_build`

---

## Full baseline lock (300k d17) ✅

| Metric | Value |
|--------|-------|
| **Native hybrid @ 300k** | **2.897 BPC** |
| Python reference | 2.873 BPC |
| Δ | **+0.024 (~0.8%)** |

---

## Architecture (shipped)

```
cypha_qt_shell (9 tabs: Data, Model, Train, Predict, Registry, Server, Experiments, CyphaLM, Help)
cypha_rest (/dif/*, /lm/*, /route/*) │ cypha_bench_run │ cypha_tune_run │ cypha_diagnostics_run
        cypha_bench_native (d01–d17, fig01–fig09 PNG, baseline regression)
        cypha_lm_native + cypha_som (full SOM stack)
        cypha_core (CyphaDIF complete)
```

---

## Phase completion

| Phase | Scope | Status |
|-------|-------|--------|
| **0–11** v2.0–v2.2 | Core framework, REST, baseline, release | ✅ |
| **12** v2.3 | DiscriminativeFeedback, Qt Charts, 300k lock | ✅ |
| **13** v2.4 | Qt Registry/Help/Confidence, fig04–09, tune configs | ✅ |
| **14** v2.5 | Release docs, CHANGELOG v2.2.0, validate gate lock | ✅ |

---

## Qt shell tab map (PySide6 parity)

| Tab | Studio equivalent |
|-----|-------------------|
| 1 · Data | dataset_widget |
| 2 · Model | model_widget |
| 3 · Train | training_widget + loss charts |
| 4 · Predict | chat + confidence_widget |
| 5 · Registry | model registry browser |
| 6 · Server | REST server |
| 7 · Experiments | experiment_widget |
| 8 · CyphaLM | lm_generation_worker |
| 9 · Help | help_widget |

Not ported: settings_dialog, confusion_dialog (deferred)

---

## Remaining (future)

| Item | Notes |
|------|-------|
| Settings/confusion dialogs | Qt optional dialogs |
| CUDA GPU runtime validation | CI compile-smoke only |

---

## Production binaries (v2.2.0 release)

**`bin/` (12):** cypha_rest, cypha_bench_run, cypha_bench_report, cypha_tune_run, cypha_diagnostics_run, cyphalm_bench_native, cyphalm_parity, cyphalm_checkpoint_parity, gh_infer_deliberation_parity, kernel_llr_parity, registry_register, create_model_smoke

**`bin/dev/` (7):** score_batch, multilabel, merge_from, similarity_index, embed_table, retrieval, som_parity

---

## Quick commands

```powershell
powershell -File scripts\cypha_native_validate_all.ps1
powershell -File scripts\cypha_tune_smoke.ps1 -DryRun
powershell -File scripts\cypha_bench_full_baseline.ps1
cmake -S native -B build_qt -DCYPHA_BUILD_QT=ON -DCYPHA_QT_CHARTS=ON -DCYPHA_BUILD_EXPERIMENT_DB=ON
```

**Tag:** `v2.2.0`
