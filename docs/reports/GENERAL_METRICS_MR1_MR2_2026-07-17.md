# General metrics MR1 (CRPS) + MR2 (90% predictive interval coverage) — 2026-07-17

**Scope:** Bill of Work Addendum 2, build-order item 2.

## What shipped

| ID | Metric | Helper / path | Report field |
|----|--------|---------------|--------------|
| MR1 | Continuous Ranked Probability Score | `cypha::bench::crps_gaussian` / `crps_gaussian_mean` in `bench_metrics.hpp` | `cypha_scores.crps` |
| MR2 | 90% predictive interval coverage | `cypha::bench::predictive_interval_coverage` (z=1.645) | `cypha_scores.interval_coverage_90` |

CRPS and coverage use the moment-matched Gaussian implied by mixture outputs: per-row mean `ŷ = Σ p_k μ_k` and std `σ = √(Σ p_k var_k)` from `regression::predict_mixture_scalar` (same NIG/mixture head as `DIFRegressor.predict`).

## Where metrics appear

- **Regression bench paths:** any domain calling `reg_metrics_native` (D05 chess, D06 go regression, D14 Feynman, D11A cartpole, …) now emits `crps` and `interval_coverage_90` on the test split.
- **Per-equation JSON (D14):** each Feynman equation block under `experiments.per_equation.*` includes the new fields alongside `rmse` / `r2`.

## Sample numbers (D05 chess, `CYPHA_BENCH_FAST=1`)

| Metric | Value |
|--------|-------|
| RMSE | 0.868 |
| CRPS | 0.510 |
| interval_coverage_90 | 0.995 |
| R² | 0.190 |

Run:

```powershell
cmake -S native -B native/build_metrics_reg -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_metrics_reg --target bench_metrics_smoke cypha_bench_run
ctest --test-dir native/build_metrics_reg -R native_bench_metrics_smoke
$env:CYPHA_BENCH_FAST=1; native/build_metrics_reg/cypha_bench_run.exe --domain-tag d05
```

Smoke asserts finite CRPS ≥ 0 and `interval_coverage_90` ∈ [0, 1].
