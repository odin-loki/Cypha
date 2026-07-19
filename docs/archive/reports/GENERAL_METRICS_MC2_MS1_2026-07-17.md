# General metrics MC2 (ECE) + MS1 (train/held-out gap) — 2026-07-17

**Scope:** Bill of Work Addendum 2, build-order item 1.

## What shipped

| ID | Metric | Helper / path | Report field |
|----|--------|---------------|--------------|
| MC2 | Expected Calibration Error | `cypha::bench::expected_calibration_error` in `bench_metrics.hpp` | `cypha_scores.ece`, `cypha_scores.mean_confidence` |
| MS1 | Train vs held-out accuracy gap | `train_eval_vectors` dual eval via `clf_metrics_native` | `cypha_scores.train_accuracy`, `cypha_scores.generalization_gap` |

Confidence for ECE uses max softmax probability from batched LLR (same signal as curriculum ordering), not NIG `disc × world_gate`.

## Where metrics appear

- **Per-task JSON:** `bench/report/tables/<domain>.json` → `experiments.tasks[].cypha_scores` (D01, D03 tabular paths via `train_eval_vectors`).
- **Direct clf bench paths:** any domain calling `clf_metrics_native` (e.g. D07) now emits `ece` / `mean_confidence` on the test split only.
- **Cross-domain calibration table:** `bench/report/tables/cross_uncertainty_calibration.json` prefers real `ece` when present; falls back to `ece_proxy` otherwise. MS1 `generalization_gap` forwarded when present.

## Sample numbers (D01, `CYPHA_BENCH_FAST=1`)

| Task | test acc | train acc | gap (train−test) | ECE | mean conf |
|------|----------|-----------|------------------|-----|-----------|
| linearly_separable_2class | 0.9875 | 0.9313 | −0.0563 | 0.0088 | 0.995 |
| 4_gaussian_blobs | 0.8875 | 0.8625 | −0.0250 | 0.1116 | 0.999 |

Negative gap here means held-out accuracy exceeds train (online single-pass training, no epoch replay).

## Build & test

```powershell
cmake -S native -B native/build_metrics -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_metrics --target bench_metrics_smoke cypha_bench_run
ctest --test-dir native/build_metrics -R native_bench_metrics_smoke
native/build_metrics/cypha_bench_run.exe --domain-tag d01
```

Smoke asserts ECE ∈ [0, 1] (finite) and generalization gap is finite.
