# General metrics MC1 (balanced accuracy / macro-F1) — 2026-07-17

**Scope:** Bill of Work Addendum 2, MC1.

## What shipped

| ID | Metric | Helper / path | Report field |
|----|--------|---------------|--------------|
| MC1 | Macro-averaged F1 | `cypha::bench::f1_macro` in `bench_metrics.hpp` | `cypha_scores.macro_f1` |
| MC1 | Balanced accuracy (mean per-class recall) | `cypha::bench::balanced_accuracy` in `bench_metrics.hpp` | `cypha_scores.balanced_accuracy` |

Qt compare already surfaces **Macro F1** from experiment DB (`runs.macro_f1`). Macro-F1 existed for offline baselines as `f1_macro` in `bench_baselines`, but was **missing** from `clf_metrics_native` / `cypha_scores` JSON. MC1 wires both metrics into that path; baselines continue to emit `f1_macro` (unchanged key).

## Where metrics appear

- **Per-task JSON:** `bench/report/tables/<domain>.json` → `experiments.tasks[].cypha_scores` (any path using `clf_metrics_native`, including D01/D03 via `train_eval_vectors`).
- **Baselines:** `baselines.*.f1_macro` unchanged (same formula via shared `f1_macro` helper).

## Sample numbers (unit smoke)

Hand-checked labels `y_true=["a","a","b","b"]`, `y_pred=["a","b","b","b"]`:

| Metric | Value |
|--------|-------|
| macro_f1 | 0.7333 (11/15) |
| balanced_accuracy | 0.7500 |

(Class `a`: P=1.0, R=0.5 → F1=2/3; class `b`: P=2/3, R=1.0 → F1=0.8; macro mean 11/15. Per-class recall mean: (0.5+1.0)/2 = 0.75.)

## Build & test

```powershell
cmake -S native -B native/build_mc1 -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_mc1 --target bench_metrics_smoke
ctest --test-dir native/build_mc1 -R native_bench_metrics_smoke
```

Smoke asserts `macro_f1` / `balanced_accuracy` ∈ [0, 1] and exact values on the fixture above.
