# CyphaDIF Upgrades: Regression-Competent DIF Routing, Variant-Aware Profiles, and Reproducible Architecture Search

**Authors:** Cypha project contributors  
**Date:** 2026-05-26  
**Repo:** `Cypha` (Windows)  

## Abstract

We describe a set of upgrades to the CyphaDIF system that substantially improve end-to-end benchmark performance, with the largest gains occurring in streaming regression where prior DIFRegressor behavior was structurally limited. The core changes add (i) target-informed routing for regression via online y-quantile binning, (ii) mixture-consistent expert updates aligned with inference-time soft routing, and (iii) a closed-form ridge readout using DIF log-likelihood ratio (LLR) features concatenated with raw inputs. In addition, we make tuned profiles variant-aware at load time and repair architecture-search scoring to prevent degenerate comparisons. On the CyphaDIF testbench validation suite, regression improves from “worse than dummy mean” to ridge-competitive (diabetes \(R^2 \approx 0.46\), California housing \(R^2 \approx 0.52\)), while tabular and vision classification remain strong under a regime-specific everyday profile.

## 1. Background and motivation

CyphaDIF is a differentiable information-field classifier operating on encoded latent vectors. The bench harness evaluates multiple regimes (tabular, vision, regression) and a set of cross-domain analyses. Early tuning improved several classification and vision tasks but consistently failed to improve regression: DIFRegressor stored per-expert *target means* and used y-agnostic routing, which fundamentally limits predictive capacity compared to ridge/SGD or tree ensembles.

The upgrade goals were:

- Make regression competitive without rewriting the full system.
- Ensure that tuning and production runs use the same semantics (variant application).
- Ensure architecture search produces meaningful comparisons (stable scoring and baselines).

## 2. Methods: algorithmic upgrades

### 2.1 Regression routing with target information (y-quantile routing)

When `reg_hash_routing=false`, DIFRegressor uses a bounded buffer of recent targets and assigns router labels by y-quantile bin rather than step-hash cold-start. This injects target information into the routing objective with minimal additional machinery.

### 2.2 Mixture-consistent training updates (soft mixture EMA)

Inference uses a softmax over LLRs to form a mixture prediction. Previously, training updated only a single hard-routed expert mean, creating a train/infer mismatch. We add an optional responsibility-weighted update: each expert’s EMA target mean is updated proportionally to its routing probability.

### 2.3 Linear readout on LLR + X (closed-form ridge head)

Per-expert target means alone cannot represent continuous \(x \to y\) structure. We add an optional stage-1 head analogous to the first stage of `TwoStageDIFRegressor`: compute LLR features for inputs, concatenate `[LLR | X]`, and fit ridge in closed form. This head dramatically improves regression while retaining DIF uncertainty and routing dynamics.

## 3. Systems upgrades

### 3.1 Variant-aware profiles

The profile loader applies `algorithm_variants` at load time so both tuning and runtime inherit consistent scaled parameters (`temperature_scale`, `mdl_lambda_scale`, `replay_ratio_scale`, `target_lr_scale`, `online_passes_extra`, and deliberation thresholds).

### 3.2 Architecture search fixes (comparability and reproducibility)

The original architecture search used a min-ratio objective across an overly broad metric set; the presence of “baseline = 0” keys made the min-ratio collapse to 0 for all candidates, destroying comparability.

We introduced:

- A **swarm baseline** restricted to swarm domains and primary task metrics (accuracy, RMSE, \(R^2\), AUROC, etc.).
- A stable **mean-ratio** aggregator for swarm scoring.
- Leaderboards that include **all eligible candidate rows**, enabling post-hoc rescoring and robust selection.

### 3.3 Registry/binary parity fix

`cypha_save_binary` does not natively encode lists; they fall back to string repr. We changed `DIFRegressor.save_state()` to store `linear_labels` as a dict, restoring `ModelRegistry` round-trip state parity.

## 4. Experimental setup

We report results from the project’s validation script:

- Script: `bench/tuning/validate_profile.py`
- Artifacts:
  - `bench/artifacts/tuning/validation_compare.json`
  - `bench/TUNING_REPORT.md`

Baseline is library defaults (`CYPHA_BENCH_USE_PROFILE=0`). Tuned uses the everyday profile (`bench/config/everyday_profile.json`) with regime routing enabled.

## 5. Results

All values below are from the latest validation run (2026-05-26).

### 5.1 Regression (d02)

- Diabetes: RMSE 72.58 → **53.38**, \(R^2\) **0.46**  
- California housing: RMSE 1.18 → **0.79**, \(R^2\) **0.52**

These results are near ridge/online-SGD baselines and represent the largest qualitative improvement relative to earlier tuned attempts.

### 5.2 Tabular classification (d03)

- Iris accuracy: 0.00 → **0.87**
- Wine accuracy: 0.00 → **0.92**
- Breast cancer accuracy: 0.00 → **0.95**
- Digits accuracy: 0.00 → **0.92**
- 20newsgroups subset accuracy: 0.00 → **0.17**

The remaining outlier is text classification (20newsgroups subset), likely requiring representation improvements rather than further LR/temperature sweeps.

### 5.3 Vision (d08)

- MNIST raw accuracy: 0.00 → **0.74**
- MNIST HOG accuracy: 0.00 → **0.89**

While improved, HOG remains below classic baselines on this harness (kNN/logreg/RF), suggesting remaining headroom in feature/routing choices.

## 6. Discussion and limitations

1. The regression upgrade is primarily an **expressivity fix**: adding a linear head on LLR+X bridges the gap between clustering-style routing and function approximation.
2. The architecture search improvements make results **comparable**, but searching for “near-100% on everything” remains unrealistic without deeper representation and domain-specific heads.
3. Some cross-domain assertions (e.g., aleatoric dominance under contradictory labels) can still fail, which is useful signal but indicates uncertainty decomposition is not perfect yet.

## 7. Reproducibility

- Validate current profile:

```bash
cypha_tune_run --config bench/config/validate_profile.py
```

- Full bench run:

```bash
cypha_bench_run
```

Artifacts are written under `bench/artifacts/tuning/`.

