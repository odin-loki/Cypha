# General metrics MR3 (residual autocorrelation + spectral flatness) — 2026-07-17

**Scope:** Bill of Work Addendum 2, build-order item MR3.

## What shipped

| ID | Metric | Helper / path | Report field |
|----|--------|---------------|--------------|
| MR3 | Lag-1 residual autocorrelation | `cypha::bench::residual_autocorr_lag1` in `bench_metrics.hpp` | `cypha_scores.residual_autocorr_lag1` |
| MR3 | Residual spectral flatness | `cypha::bench::residual_spectral_flatness` in `bench_metrics.hpp` | `cypha_scores.residual_spectral_flatness` |

Both metrics operate on test-split regression residuals `r_i = y_i − ŷ_i` from the same mixture head used by MR1/MR2 (`reg_metrics_native`).

- **Lag-1 autocorrelation:** sample Pearson correlation between consecutive residuals; returns `0` when variance vanishes.
- **Spectral flatness:** geometric / arithmetic mean ratio of DFT power bins on the residual series (values in `[0, 1]` for typical spectra; white noise → 1).

## Where metrics appear

- **Regression bench paths:** any domain calling `reg_metrics_native` (D05 chess, D06 go regression, D14 Feynman, D11A cartpole, …) now emits the MR3 fields alongside MR1/MR2.
- **Per-equation JSON (D14):** each Feynman equation block under `experiments.per_equation.*` includes the new fields alongside `rmse` / `r2`.

## Sample numbers (D05 chess, `CYPHA_BENCH_FAST=1`)

| Metric | Value |
|--------|-------|
| RMSE | 0.868 |
| CRPS | 0.510 |
| interval_coverage_90 | 0.995 |
| residual_autocorr_lag1 | −0.027 |
| residual_spectral_flatness | 0.473 |
| R² | 0.190 |

Run:

```powershell
cmake -S native -B native/build_mr3 -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_mr3 --target bench_metrics_smoke cypha_bench_run
ctest --test-dir native/build_mr3 -R native_bench_metrics_smoke
$env:CYPHA_BENCH_FAST=1; native/build_mr3/cypha_bench_run.exe --domain-tag d05
```

Smoke asserts finite lag-1 autocorr (fixture: alternating residuals → −0.75) and spectral flatness ∈ [0, 1].
