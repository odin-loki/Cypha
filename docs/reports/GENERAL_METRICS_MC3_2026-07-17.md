# General metrics MC3 — FGSM robustness curve — 2026-07-17

**Scope:** Bill of Work Addendum 2, MC3 — accuracy vs perturbation ε (not a single FGSM point).

## What shipped

| ID | Deliverable | Path |
|----|-------------|------|
| MC3 | FGSM-proxy ε-sweep helper | `run_d15_fgsm_robustness_curve` in `bench_domains.hpp` |
| MC3 | Curve runner | `native/tools/cyphalm_robustness_curve.cpp` |
| — | Profile config | `bench/config/robustness_curve_profile.json` |
| — | CTest smoke (≥3 ε) | `native_robustness_curve_smoke` |
| — | D15C extension | `15C_adversarial_fgsm_proxy` emits full curve + legacy `accuracy_*` fields |

Perturbation is the existing D15C finite-difference FGSM proxy: per-feature sign from ±1e-4 predict flips, then step by ε (clamp ≥0). Signs are computed once; the curve reuses them across ε.

## JSON curve format

```json
{
  "curve_id": "adversarial_robustness",
  "metric": "accuracy",
  "perturbation": "fgsm_proxy",
  "dataset": "digits_hog",
  "runner": "cyphalm_robustness_curve",
  "profile": "d15",
  "epsilons_requested": [0.0, 0.05, 0.1],
  "points": [
    {"epsilon": 0.0, "accuracy": 0.85, "mean_epistemic": 0.4, "n_eval": 40},
    {"epsilon": 0.05, "accuracy": 0.84, "mean_epistemic": 0.4, "n_eval": 40},
    {"epsilon": 0.1, "accuracy": 0.83, "mean_epistemic": 0.4, "n_eval": 40}
  ]
}
```

Default ε: **0 / 0.05 / 0.1 / 0.2 / 0.5**. `CYPHA_BENCH_FAST=1` → **0 / 0.05 / 0.1** (≥3 points). Smoke uses `--max-eval 40`.

## Build & test

```powershell
cmake -S native -B native/build_mc3 -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_mc3 --target cyphalm_robustness_curve robustness_curve_smoke
ctest --test-dir native/build_mc3 -R native_robustness_curve_smoke --output-on-failure
$env:CYPHA_BENCH_FAST=1; native/build_mc3/cyphalm_robustness_curve.exe --write-table
```

Smoke asserts **≥3** ε points and accuracy ∈ [0, 1] (finite).

## Curve summary

| ε | role |
|---|------|
| 0.0 | natural (clean) accuracy |
| 0.05 | mid-ε |
| 0.1 | legacy D15C single-point ε (also `accuracy_adversarial`) |
| 0.2 / 0.5 | full-profile tail (non-FAST) |

Historical D15C single-point (ε=0.1) was ~86.7% adv vs ~85.8% natural; the curve replaces that one-point claim with a multi-ε table under `bench/results/robustness_curve.json`.
