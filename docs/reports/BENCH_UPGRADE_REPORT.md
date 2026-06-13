# Cypha Upgrade Summary (Code + Profile)

Generated: 2026-05-26

This report summarizes the upgrades applied across the core model (`cypha_core`), bench adapters, tuning/scoring, and the selected tuned profile (`bench/config/everyday_profile.json`). Metrics are taken from the latest `bench/tuning/validate_profile.py` run.

## What changed (high-impact)

- **Regression is now competitive** via `DIFRegressor` upgrades:
  - **Y-quantile routing** when `reg_hash_routing=false` (routes using target bins instead of step-hash).
  - **Soft mixture-consistent training** (`use_soft_mixture`) so training updates match softmax mixture inference.
  - **LLR + X ridge head** (`use_linear_head` + `fit_linear_head`) to add an actual \(x \to y\) readout, closing most of the RMSE gap vs ridge/SGD.
- **Deliberation band wired** (`deliberation_lo/hi`) into `CyphaDIF.infer`/`infer_full` to abstain on mid-confidence predictions (reduces brittle routing in edge cases).
- **Profile system made variant-aware** (`load_profile()` applies algorithm variants), so sweeps and production runs are consistent.
- **Architecture search scoring fixed**:
  - Swarm baseline is now **swarm-domain + primary-metric only** (prevents min-ratio collapsing to 0 because of unrelated metrics).
  - Arch swarms store **all eligible** combos, enabling post-hoc rescoring and reproducible selection.
- **Binary/registry stability fix**:
  - `DIFRegressor.save_state()` now stores `linear_labels` as a dict (lists are not a native binary dtype). Registry round-trip parity is restored.

## Current tuned profile (everyday)

Active: `bench/config/everyday_profile.json` (regime-based: tabular / vision / regression) + `algorithm_variants` (including `reg_hash_routing=false`).

## Measured improvements (validate_profile)

Source: `bench/TUNING_REPORT.md` and `bench/artifacts/tuning/validation_compare.json`

### Regression (d02)

- **Diabetes RMSE**: 72.58 → **53.38**  (R² **0.46**, near ridge)
- **California housing RMSE**: 1.18 → **0.79** (R² **0.52**)

### Classification (d03)

- **Iris accuracy**: 0.00 → **0.87**
- **Wine accuracy**: 0.00 → **0.92**
- **Breast cancer accuracy**: 0.00 → **0.95**
- **Digits accuracy**: 0.00 → **0.92**
- **20newsgroups subset accuracy**: 0.00 → **0.17** (still a weak spot)

### Vision (d08, MNIST)

- **Raw accuracy**: 0.00 → **0.74**
- **HOG accuracy**: 0.00 → **0.89**

## Known remaining gaps

- **Text/doc classification (20newsgroups subset)** remains low and likely needs representation/routing changes beyond hyperparameters.
- **MNIST HOG** is improved but still below classic baselines (kNN/logreg/RF) on this harness.

