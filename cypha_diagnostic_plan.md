# Cypha — Full Diagnostic & Upgrade Analysis Plan

**Purpose:** Systematic empirical diagnosis of CyphaDIF + CellAI performance ceilings  
**Executor:** Cursor on local machine  
**Constraint:** No architectural changes until diagnostics point at a specific bottleneck  
**Rule:** Every diagnostic produces a number. Every number points at or rules out a candidate upgrade.

---

## Table of Contents

1. [Philosophy & Ground Rules](#1-philosophy--ground-rules)
2. [Environment & Prerequisites](#2-environment--prerequisites)
3. [Benchmark Design](#3-benchmark-design)
4. [Phase 1 — Baseline Establishment](#4-phase-1--baseline-establishment)
5. [Phase 2 — Encoder Quality Analysis](#5-phase-2--encoder-quality-analysis)
6. [Phase 3 — NIG Posterior Fit Analysis](#6-phase-3--nig-posterior-fit-analysis)
7. [Phase 4 — Online Learning Dynamics](#7-phase-4--online-learning-dynamics)
8. [Phase 5 — CellAI / SSM Quality](#8-phase-5--cellai--ssm-quality)
9. [Phase 6 — GH Gate & Adversarial Hardening](#9-phase-6--gh-gate--adversarial-hardening)
10. [Phase 7 — Full Stack Integration](#10-phase-7--full-stack-integration)
11. [Phase 8 — Scaling & Stress Tests](#11-phase-8--scaling--stress-tests)
12. [Results Interpretation Matrix](#12-results-interpretation-matrix)
13. [Upgrade Decision Tree](#13-upgrade-decision-tree)
14. [Candidate Upgrades Ranked by Evidence](#14-candidate-upgrades-ranked-by-evidence)
15. [Reporting Format](#15-reporting-format)

---

## 1. Philosophy & Ground Rules

### 1.1 Why This Exists

Six self-organising upgrades produced zero movement in measured metrics. A learnable encoder was proposed next, but was also identified as likely conflicting with online sufficient statistics. The correct response to two failed hypotheses is not a third hypothesis — it is a structured investigation that rules possibilities in and out with evidence.

This document defines that investigation. Every phase answers a specific diagnostic question. Every diagnostic question, once answered, closes some doors and opens others. No upgrade gets built until this process has pointed at the bottleneck with evidence.

### 1.2 Core Diagnostic Questions

These are the questions the entire plan is built around. Every test traces back to at least one of them.

```
Q1  Is the test benchmark already saturated? 
    (Are we measuring nothing because there is nothing to measure?)

Q2  Is the RFF encoder producing well-separated features in the encoded space?
    (Is the bottleneck at the front door?)

Q3  Are the class-conditional distributions in encoded space approximately Gaussian?
    (Is the NIG posterior the right model for this data?)

Q4  Is the LLR decision boundary geometrically appropriate for the data?
    (Is linear discrimination in RFF space sufficient?)

Q5  Is the online learning schedule well-calibrated?
    (Is the system converging, forgetting, or oscillating?)

Q6  Does CellAI produce a hidden state h that captures temporal structure?
    (Is the SSM actually doing useful work?)

Q7  Does the GH gate fire at the right times?
    (Is uncertainty quantification calibrated?)

Q8  Does the system degrade gracefully under class imbalance, 
    distribution shift, and noise?
    (Are there regime-specific failures hidden by average metrics?)

Q9  Where does performance fall off as dimensionality and class count scale?
    (What is the scaling law for this architecture?)
```

### 1.3 Non-Negotiable Rules

- Every test runs with at least 5 random seeds. Report mean ± standard deviation.
- Every result gets saved to a structured results file before the next test runs.
- No test modifies Cypha source code. Diagnostics wrap the existing API.
- If a test crashes or produces NaN/Inf, that is itself a diagnostic result. Log it, do not skip it.
- Comparisons are made against the locked baseline from Phase 1. Never against a remembered number.

---

## 2. Environment & Prerequisites

### 2.1 Hardware Assumptions

Target machine: ThinkStation P920 with RTX 3090. Most tests run on CPU. GPU tests are explicitly labelled.

### 2.2 Software Requirements

```
Python 3.10+
numpy
scipy
scikit-learn
matplotlib
seaborn
pandas
psutil
tqdm
```

No additional ML frameworks required. All diagnostics run against the existing Cypha API. CellAI tests additionally require the CellAI module.

### 2.3 Directory Structure

Before running anything, create this structure:

```
cypha_diagnostics/
├── data/                    # Generated and cached datasets
│   ├── synthetic/
│   ├── real/
│   └── adversarial/
├── results/
│   ├── phase1_baseline/
│   ├── phase2_encoder/
│   ├── phase3_nig/
│   ├── phase4_online/
│   ├── phase5_cellai/
│   ├── phase6_gh/
│   ├── phase7_integration/
│   └── phase8_scaling/
├── plots/
│   ├── phase1/
│   ├── phase2/
│   └── ...
├── logs/
│   └── crash_reports/
└── scripts/
    └── (diagnostic scripts go here)
```

All results files are JSON. All plots are PNG at 300 DPI. Log every test's start time, end time, seed, and whether it completed or crashed.

### 2.4 Cypha Version Lock

Before Phase 1: record the exact git commit hash of Cypha.py and CellAI. All diagnostics run against this exact version. If bugs are found and fixed during diagnostics, re-run the affected phases from scratch against the fixed version and record both sets of results.

---

## 3. Benchmark Design

### 3.1 The Saturation Problem

The prior benchmarks (sklearn `make_classification`, standard 10-class) likely have insufficient headroom to detect improvement. A system at 94% accuracy on a 10-class problem has 6% headroom. If the upgrade moves it to 95%, that is a 17% relative improvement in error rate but only 1 percentage point absolute — easy to miss in noise.

All diagnostic benchmarks are designed with known difficulty levels, known theoretical ceilings, and headroom to detect improvement.

### 3.2 Synthetic Benchmark Suite

#### Benchmark S1 — Linear Separable (Sanity Check)
- 2 classes, 20 features, perfect linear separability
- Any reasonable classifier should reach 99%+
- **Purpose:** Verify Cypha is not broken. If it fails here, stop everything.
- **Theoretical ceiling:** 100%

#### Benchmark S2 — Gaussian Clusters, Increasing Classes
- 5, 10, 20, 50 classes. Equal covariance, well-separated means.
- Features: 50. Samples: 10,000.
- **Purpose:** Measure how performance scales with class count. Identify where Cypha's multi-class LLR degrades.
- **Theoretical ceiling:** ~98% (limited by overlap in high-class-count variants)

#### Benchmark S3 — Non-Gaussian Class Conditionals
- 10 classes, each class drawn from a mixture of 3 Gaussians (multimodal).
- Features: 50. Samples: 10,000.
- **Purpose:** Directly test NIG assumption violation. If S3 << S2 at matched difficulty, NIG misspecification is confirmed.
- **Theoretical ceiling:** ~90% with kernel SVM

#### Benchmark S4 — Non-Linear Boundary (XOR-style)
- 2 classes, 20 features, class label = XOR of two dominant features.
- Linear classifier theoretical ceiling: ~50% (random). Kernel SVM: ~99%.
- **Purpose:** Test whether LLR linear discriminant is the bottleneck.
- **If Cypha gets ~50%:** LLR linearity is confirmed as a hard ceiling.

#### Benchmark S5 — Distribution Drift
- 10 classes. Phase 1: 5,000 samples, class means at location A.
- Phase 2: 5,000 samples, class means shift to location B (+3σ in first 5 features).
- **Purpose:** Measure how fast accuracy recovers after distribution shift.
- **Metrics:** Accuracy at 100, 500, 1000, 2000 samples into Phase 2.

#### Benchmark S6 — Severe Class Imbalance
- 10 classes. Majority class: 80% of samples. Each minority class: ~2.2%.
- **Purpose:** Identify whether the Fisher-Rao update has implicit majority bias.
- **Metrics:** Per-class recall, macro-F1, weighted-F1.

#### Benchmark S7 — High Dimensionality
- 10 classes. Features: 10, 50, 100, 500, 1000.
- Samples: 10,000 each.
- **Purpose:** Identify where RFF dimensionality bottleneck appears. At high input dim, random projections have poor coverage — where does accuracy drop?

#### Benchmark S8 — Temporal Sequence (CellAI/CyphaLM Only)
- Sequences of length 100, 500, 1000, 5000.
- Class label depends on token at position 0 (short-range dependency test).
- Class label depends on token at position -50 (long-range dependency test).
- **Purpose:** Measure SSM effective memory depth. Identify at what sequence length recall degrades.

### 3.3 Real-World Benchmarks

#### Benchmark R1 — UCI Iris (3 classes, 4 features)
Near-linearly-separable. Ceiling ~97%. Warm-up calibration check.

#### Benchmark R2 — UCI Wine (3 classes, 13 features)
Well-behaved multi-class. Ceiling ~99% with good classifier.

#### Benchmark R3 — UCI Digits (10 classes, 64 features)
Moderate difficulty. Ceiling ~99% with SVM, ~95% with simple online learner. Known challenge for online systems.

#### Benchmark R4 — UCI Adult Income (2 classes, mixed features, imbalanced)
Real imbalance (~75%/25%). Feature interactions matter. Ceiling ~87%.

#### Benchmark R5 — MNIST (10 classes, 784 features)
Hard test for an online system. Ceiling ~99% deep learning, ~97% SVM.
**Note:** This is a stress test, not a pass/fail. Cypha is not expected to match deep learning. The question is where it lands and why.

#### Benchmark R6 — WikiText-2 (CellAI/CyphaLM Only)
Character-level perplexity baseline. Establish this number once; all language-related upgrades are measured against it.

### 3.4 Competitor Baselines

For every benchmark, run these competitor baselines alongside Cypha and record their scores. This establishes what is theoretically achievable on each dataset.

```
Competitors to run:
  - SGD online linear classifier (sklearn SGDClassifier)
  - Passive-Aggressive classifier (sklearn, online)
  - Gaussian Naive Bayes (sklearn, online with partial_fit)
  - Hoeffding Tree (river library, truly online)
  - Linear SVM (sklearn, batch, ceiling reference)
  - RBF SVM (sklearn, batch, ceiling reference)
  - Random Forest 100 trees (sklearn, batch, ceiling reference)
```

The batch SVM and Random Forest are ceiling references only — they see all data at once. The online competitors are the fair comparison. Every Cypha result should be reported with both the ceiling (batch SVM) and the best online competitor score on the same dataset. This immediately shows whether the gap is from the online constraint or from something specific to Cypha.

---

## 4. Phase 1 — Baseline Establishment

**Answers:** Q1 (saturation), and establishes the locked comparison point for all subsequent phases.

**Duration estimate:** 2–4 hours on target hardware.

### 4.1 Run Full Benchmark Suite

Run all synthetic (S1–S7) and real-world (R1–R5) benchmarks against both Cypha and all competitors. Record for each:

```
Per benchmark, per system:
  - Final accuracy (in-distribution)
  - Accuracy at N = 100, 500, 1000, 5000, 10000 samples
  - Steps to 80% accuracy (convergence speed)
  - Steps to 95% of asymptotic accuracy
  - Accuracy std deviation across 5 seeds
  - Wall time per train step (median over 1000 steps)
  - Wall time per inference (median over 1000 calls)
  - Memory footprint at end of run (MB)
  - Number of prototypes/experts at end of run
```

### 4.2 Saturation Detection

For each benchmark, compute the **Cypha headroom**:

```
headroom = ceiling_accuracy - cypha_accuracy
```

where `ceiling_accuracy` is the RBF SVM score.

Classify each benchmark:
```
headroom < 2%   → SATURATED. Remove from further analysis. Not useful.
headroom 2–10%  → TIGHT. Useful but requires careful experiment design.
headroom > 10%  → OPEN. Primary diagnostic target.
```

Any benchmark with headroom > 10% is a primary diagnostic target. If all benchmarks are saturated (headroom < 2% everywhere), this is itself a critical finding — the system is already near-optimal for these tasks and more complex real-world tasks are needed.

### 4.3 Gap Analysis

For each open benchmark, compute:

```
online_gap = best_online_competitor_accuracy - cypha_accuracy
batch_gap  = ceiling_accuracy - cypha_accuracy
```

If `online_gap` is large, Cypha underperforms even online competitors — this is a fundamental algorithmic problem, not a theoretical ceiling issue.

If `online_gap` is small but `batch_gap` is large, Cypha is competitive online but the online learning constraint itself is the main limiter — no amount of architectural change fixes this.

### 4.4 Deliverable

A locked file `results/phase1_baseline/BASELINE.json` containing all numbers from 4.1, the saturation classification for each benchmark, and the gap analysis. This file is read-only after Phase 1 completes. Every subsequent phase compares against it.

---

## 5. Phase 2 — Encoder Quality Analysis

**Answers:** Q2 (encoder separability), Q4 (LLR geometry sufficiency)  
**Run on:** Open benchmarks from Phase 1 only.

### 5.1 Feature Space Geometry

After training Cypha on each open benchmark, extract all encoded vectors `h = encode(x)` for the test set. Compute:

**Class separation metrics:**
- Fisher Discriminant Ratio (FDR) per dimension: `FDR_i = μ_between,i² / σ_within,i²`
- Mean FDR across all dimensions
- Distribution of FDR values (histogram): are most dimensions useless? Are a few doing all the work?
- Davies-Bouldin Index on the encoded space (cluster cohesion/separation quality)
- Silhouette score on the encoded space

**What the numbers mean:**
```
Mean FDR < 0.5:  Encoded space has very poor class separation. Encoder is the bottleneck.
Mean FDR 0.5–2:  Moderate separation. Encoder contributes but is not the only issue.
Mean FDR > 2:    Good separation. Encoder is not the primary bottleneck.

FDR distribution heavily skewed (few high, many near-zero): 
  Most RFF dimensions are wasted. Dimensionality reduction would help.

Silhouette < 0.3:  Clusters are poorly formed in encoded space.
Silhouette > 0.6:  Clusters are well-formed. LLR should work well here.
```

### 5.2 Linearity Test

This directly answers whether the LLR linear discriminant is the hard ceiling.

For each open benchmark:
- Train a **linear** classifier on the encoded space `h`
- Train a **kernel SVM (RBF)** on the encoded space `h`
- Train a **kernel SVM (RBF)** on the **raw input** `x`

Record accuracy for all three.

```
Analysis:
If linear(h) ≈ kernel(h):
  The encoded space is already well-linearised. LLR linearity is NOT the bottleneck.

If linear(h) << kernel(h):
  The encoded space has non-linear structure that LLR cannot capture.
  This is strong evidence that a non-linear extension of LLR would help.

If kernel(h) ≈ kernel(x):
  RFF encoding adds nothing useful over raw input. Encoder design is the problem.

If kernel(h) >> kernel(x):
  RFF encoding is genuinely helpful. Build on it.
```

### 5.3 Projection Dimensionality Sweep

Current Cypha uses a fixed number of RFF dimensions `D_rff`. Test whether this is optimal.

Run the full Cypha pipeline with `D_rff ∈ {16, 32, 64, 128, 256, 512, 1024}` on each open benchmark.

Plot accuracy vs `D_rff`. Look for:
```
Monotonically increasing → Current D_rff is undersized. Increasing it helps.
Plateau then decline    → Overfitting from noise in high-dimensional projections.
Early plateau           → D_rff is not the bottleneck; other factors dominate.
```

Also plot `wall_time_per_step` vs `D_rff` on the same chart. Identify the point where accuracy gain per unit of compute is no longer worth it.

### 5.4 Kernel Comparison

Run Cypha with three different RFF kernel approximations:
- RBF (current default)
- Laplace kernel (`p(ω) ∝ 1/(1+||ω||²)`)
- Arc-cosine kernel (approximates a deep network's feature map)

Record accuracy for each on all open benchmarks.

```
If one kernel dominates across benchmarks:
  Strong evidence that kernel choice is a major bottleneck.
  Upgrade direction: switch to better-matched kernel, or learn kernel mixture.

If all kernels perform similarly:
  Kernel family is not the bottleneck. Do not invest here.
```

### 5.5 Encoder vs Raw Input Comparison

For every benchmark, run Cypha's LLR head directly on raw features (no RFF). Compare to Cypha with RFF.

```
If raw > RFF:   The encoder is actively hurting. Remove it and rebuild from scratch.
If raw ≈ RFF:   The encoder adds no value. Redesign or replace.
If RFF > raw:   The encoder is earning its cost. Focus elsewhere.
```

---

## 6. Phase 3 — NIG Posterior Fit Analysis

**Answers:** Q3 (Gaussian assumption validity), Q7 (GH gate calibration)

### 6.1 Class-Conditional Distribution Tests

For each open benchmark, after training, extract per-class distributions of encoded vectors `h` for each class k.

For each class k, test whether `h | class=k` is Gaussian:

**Tests to run:**
- Shapiro-Wilk normality test on each feature dimension independently
- Mardia's multivariate normality test (multivariate skewness and kurtosis)
- Q-Q plots for the 5 most important RFF dimensions per class (those with highest FDR)
- Henze-Zirkler test for multivariate normality

**Record per class:**
```
- Fraction of dimensions passing Shapiro-Wilk (p > 0.05)
- Mardia skewness statistic and its p-value
- Mardia kurtosis statistic and its p-value
- Henze-Zirkler test statistic
```

**Interpretation:**
```
< 50% dimensions Gaussian:   
  NIG assumption is severely violated. 
  Posterior is misspecified for this class.
  Upgrade direction: mixture NIG, or non-parametric posterior.

50–80% dimensions Gaussian:
  Moderate violation. NIG is approximate.
  Upgrade direction: robust covariance estimation, outlier-robust prior.

> 80% dimensions Gaussian:
  NIG assumption is approximately valid. Posterior design is not the bottleneck.
```

### 6.2 Posterior Calibration Test

A well-calibrated uncertainty estimate means: when the system says it is 90% confident, it should be correct 90% of the time.

For each open benchmark, compute a calibration curve:

- Run Cypha inference on the test set, recording confidence scores and correctness.
- Bin predictions by confidence: [0–10%], [10–20%], ..., [90–100%].
- For each bin, compute the fraction that were actually correct.
- Plot confidence vs actual accuracy (calibration curve).
- Compute Expected Calibration Error (ECE).

```
ECE < 0.05:   Well-calibrated. Uncertainty estimates are trustworthy.
ECE 0.05–0.15: Moderately miscalibrated. Upgrade direction: temperature scaling, Platt scaling.
ECE > 0.15:   Severely miscalibrated. The NIG posterior is not providing reliable uncertainty.
```

### 6.3 OOD Detection Quality

The GH gate should suppress incorrect confident predictions for out-of-distribution inputs.

For each open benchmark:
- Generate OOD data: Gaussian noise, inputs from a held-out class, and adversarial perturbations.
- Run Cypha inference on OOD and in-distribution data.
- Compute AUROC for the task of distinguishing OOD from in-distribution using the GH gate score alone.
- Compute false positive rate at 95% true positive rate (FPR95).

```
AUROC < 0.7:   GH gate is not separating OOD from in-distribution. Gate is ineffective.
AUROC 0.7–0.9: Moderate OOD detection. Improvement possible.
AUROC > 0.9:   GH gate is working well. Do not prioritise this for upgrade.

FPR95 > 30%:   Gate has too many false alarms. Flag for calibration work.
```

### 6.4 NIG Hyperparameter Sensitivity

The NIG prior (κ₀, ν₀, α₀, β₀) governs how fast confidence builds and how strongly the prior resists updating. These are rarely tuned and have large effects.

Run a grid sweep over:
```
κ₀ ∈ {0.01, 0.1, 1.0, 10.0}
ν₀ ∈ {d_rff + 1, d_rff + 5, d_rff + 20}   (where d_rff = RFF output dim)
α₀ ∈ {1.5, 2.0, 3.0, 5.0}
β₀ ∈ {0.01, 0.1, 1.0}
```

Record final accuracy, convergence speed, and ECE for each configuration on the two highest-headroom benchmarks.

**What to look for:**
```
Accuracy varies > 5% across κ₀ values:
  Prior strength is a critical parameter. Current default is not optimal.
  Upgrade direction: empirical Bayes — fit κ₀ from first M samples.

ECE varies > 0.05 across ν₀ values:
  Degrees-of-freedom parameter controls calibration quality.
  Upgrade direction: fit ν₀ to match observed spread.

Narrow region of (α₀, β₀) performs well:
  Prior is over-constraining in the default setting.
  Upgrade direction: cross-validate NIG hyperparameters per-task.
```

---

## 7. Phase 4 — Online Learning Dynamics

**Answers:** Q5 (convergence and forgetting)

### 7.1 Learning Rate Schedule Analysis

Run Cypha on the standard benchmark with learning rate schedules:
```
world_lr ∈ {0.001, 0.01, 0.05, 0.1, 0.5}
delta_lr ∈ {same range, crossed with world_lr}
enc_lr   ∈ {same range}
```

For each combination, plot accuracy vs training step at steps: 100, 500, 1000, 5000, 10000, 50000.

**What to look for:**
```
Accuracy peaks then declines:         Catastrophic forgetting. LR too high or no decay.
Accuracy never converges:             LR too high or unstable.
Accuracy converges early then flat:   LR decayed too fast. System stopped learning.
Accuracy grows monotonically:         Well-calibrated schedule. Current default may be fine.
```

Also test common decay schedules: constant, 1/t decay, step decay, cosine annealing. Record which gives fastest convergence to asymptotic accuracy without forgetting.

### 7.2 Catastrophic Forgetting Test

This is the most important single test for any online learning system.

Protocol:
- Phase 1: Train on classes 1–5, 5,000 samples.
- Checkpoint: Record per-class accuracy on a held-out test set for classes 1–5.
- Phase 2: Train on classes 6–10 only, 5,000 samples. No samples from classes 1–5.
- Re-test: Record per-class accuracy on the same test set for classes 1–5.
- Compute: `forgetting_ratio = (phase1_accuracy - retest_accuracy) / phase1_accuracy`

```
forgetting_ratio < 0.05:   Excellent retention. No forgetting problem.
forgetting_ratio 0.05–0.2: Moderate forgetting. Upgrade direction: elastic weight consolidation signal.
forgetting_ratio > 0.2:    Severe forgetting. This is a critical flaw. Major upgrade priority.
```

Repeat this test with three class introduction orders to confirm results are not order-dependent.

### 7.3 Sample Efficiency

How many samples does Cypha need to reach 80%, 90%, and 95% of its asymptotic accuracy?

For each open benchmark:
- Plot accuracy vs number of training samples (learning curve).
- Fit a learning curve model: `accuracy(n) = ceiling - A · n^(-B)`
- Record the fitted parameters: `ceiling`, `A`, `B`.
- Compare to competitor online methods.

```
B < 0.3:   Very slow learning. Sample-inefficient.
B 0.3–0.6: Normal online learning curve.
B > 0.6:   Fast learning. Good sample efficiency.

Compare ceiling across methods:
If Cypha ceiling << competitor ceiling on same data:
  Architecture is limiting, not just sample count.
```

### 7.4 Noise Robustness

Run each benchmark with label noise at rates: 0%, 5%, 10%, 20%, 30%.

For each noise level, record accuracy after 10,000 samples. Plot accuracy degradation curve.

```
If accuracy at 10% noise << accuracy at 0% noise (gap > 10%):
  System is highly sensitive to label noise. 
  Upgrade direction: noise-robust loss, label smoothing at the NIG prior level.

If accuracy at 30% noise > 50% accuracy (5-class problem):
  System has robust performance. Noise handling is acceptable.
```

### 7.5 Correction / Update Responsiveness

When Cypha receives a correction (a mis-classified sample re-labelled), how quickly does accuracy on similar samples improve?

Protocol:
- Train until convergence.
- Identify the 20 most-confused pairs (most common misclassification pairs from confusion matrix).
- Feed 50 correction samples for the most confused pair.
- Measure: how many corrections needed before accuracy on that pair crosses 80%?

```
< 5 corrections:  Highly responsive. Update rule is effective.
5–20 corrections: Moderate. Acceptable for most use cases.
> 20 corrections: Sluggish. Update rule may be over-smoothing. Investigate world_lr vs delta_lr ratio.
```

---

## 8. Phase 5 — CellAI / SSM Quality

**Answers:** Q6 (SSM temporal quality)  
**Note:** Skip if CellAI is not integrated in current working build. Mark results as pending.

### 8.1 State Informativeness

The fundamental question: does the SSM state `h(t)` actually contain more useful information than the current input `x(t)` alone?

Protocol:
- For each sequence benchmark (S8): train CyphaDIF on `h(t)` (with CellAI) vs `x(t)` directly (without CellAI).
- Record accuracy delta: `Δ_cellai = accuracy(h(t)) - accuracy(x(t))`

```
Δ_cellai < 0:    CellAI is actively hurting performance. It is injecting noise.
Δ_cellai ≈ 0:   CellAI is not contributing. SSM state carries no extra information.
Δ_cellai > 2%:  CellAI is genuinely contributing to performance.
```

This is the most important CellAI diagnostic. If Δ_cellai ≤ 0, all further CellAI optimisation is pointless until this is fixed.

### 8.2 Effective Memory Depth

How far back does the SSM state actually carry information?

Protocol:
- Construct sequences where the correct label depends on a token at lag L (L = 1, 5, 10, 50, 100, 500).
- For each lag L, measure classification accuracy.
- Plot accuracy vs lag L.

```
Accuracy falls sharply at lag L_0:
  Effective memory depth = L_0.
  For sequence lengths > L_0, the SSM is forgetting.
  Upgrade direction: add more SSM scales, reduce decay rates.

Accuracy flat across all tested lags:
  Either the SSM has enough memory, or it is ignoring temporal structure entirely.
  Discriminate by testing Δ_cellai (8.1) — if Δ_cellai ≈ 0, it is ignoring structure.
```

### 8.3 Scale Contribution Analysis

CellAI uses L scales with decay rates α_l. Not all scales may contribute equally.

For each scale l, compute:
- Variance of `m_l(t)` across a 1,000-step window (low variance = scale is not updating)
- Correlation of `m_l(t)` with the final classification result

```
Scale l has near-zero variance:
  That scale is effectively frozen. It contributes nothing.
  Upgrade direction: remove it or reinitialise its decay rate.

Scale l has high variance but low correlation with output:
  That scale is noisy but not predictive.
  Upgrade direction: prune or regularise this scale.

Scale l has moderate variance and high correlation:
  That scale is genuinely useful. Protect it.
```

Also check: is there redundancy? If scales l=2 and l=3 have correlation > 0.95 with each other, they are duplicates — you are paying the cost of two scales for one unit of information.

### 8.4 Hebbian Update Quality

Is the Hebbian update actually producing useful weight changes?

Protocol:
- Record the angle between consecutive weight updates `ΔW(t)` and `ΔW(t+1)`.
- If updates are consistently pointing in opposite directions (angle > 90°), the update rule is oscillating.
- Compute the magnitude distribution of weight updates over 10,000 steps.

```
Mean update magnitude near zero after 1,000 steps:
  CellAI has converged. It is no longer adapting. Check if this is appropriate.

Update magnitude oscillates or grows:
  CellAI is unstable. Learning rate is too high.

Update angle consistently > 90° (anti-correlated updates):
  Oscillation. Reduce learning rate or add momentum.
```

### 8.5 CellAI Ring Diffusion Effectiveness

Is the spatial diffusion step adding information or blurring it?

Protocol:
- Run two versions: full CellAI with diffusion, and CellAI with diffusion coefficient γ = 0.
- Compare accuracy and convergence speed.

```
No diffusion performs better or equal:
  Diffusion is hurting or neutral. Remove it or reduce γ.

With diffusion performs better:
  Diffusion is contributing. The spatial coupling is useful.
  Upgrade direction: allow γ to adapt per region.
```

---

## 9. Phase 6 — GH Gate & Adversarial Hardening

**Answers:** Q7 (gate calibration), robustness verification

### 9.1 GH Gate Threshold Sensitivity

The GH gate uses a chi-squared threshold to flag OOD inputs. This threshold is a critical hyperparameter.

Sweep: `chi_threshold ∈ {0.01, 0.05, 0.1, 0.2, 0.5}` (significance levels)

For each threshold, on a dataset with 30% OOD contamination, record:
- True positive rate (OOD correctly flagged)
- False positive rate (in-distribution incorrectly flagged)
- Accuracy on in-distribution samples after contamination

Plot an ROC curve. Identify the optimal threshold per benchmark.

```
Optimal threshold is far from current default:
  Current threshold is wrong. Calibrate per-task or use an adaptive scheme.

ROC AUROC < 0.7:
  GH gate cannot separate OOD from in-distribution regardless of threshold.
  The chi-squared test is not the right test for this data geometry.
```

### 9.2 Adversarial Injection Scaling

The existing benchmark tests 15 adversarial injections. This may not be sufficient to find the breaking point.

Sweep: injections ∈ {5, 10, 15, 20, 30, 50, 100}

For each injection count, record dos_recall (fraction of legitimate samples correctly classified despite poisoning).

```
Plot dos_recall vs injection count.
Find the injection count where dos_recall drops below 0.5 (majority failure).
This is the adversarial capacity of the system.

If capacity < 20 injections:   Brittle. Major hardening needed.
If capacity 20–100 injections: Moderate. Improvements worth pursuing.
If capacity > 100 injections:  Robust. Adversarial hardening is not a priority.
```

### 9.3 Encoder Norm Monitoring

The known adversarial bypass involves encoder weights exploding in norm. Verify the fix holds.

Protocol:
- Run 15 adversarial injections.
- Record encoder weight norm at steps: 0, after injection 1, 5, 10, 15, and 500 steps after injection.
- Verify norm stays bounded within 2× initial value throughout.

If norm exceeds 2× at any point, the adversarial bypass fix is incomplete. This is a stop-everything issue.

### 9.4 Chi Session Contamination Regression Test

The known bug: OOD events in one session suppressing in-distribution inferences in subsequent sessions.

Protocol:
- Run a session with 5 OOD inputs.
- Immediately start a new session.
- Run 20 in-distribution inputs.
- Verify that accuracy on the in-distribution inputs is not degraded vs a clean session.

If in-distribution accuracy is more than 2% lower after the OOD session: chi contamination has regressed. Stop and fix before continuing.

---

## 10. Phase 7 — Full Stack Integration

**Answers:** Q8 (regime-specific failures), confirms all phases together

### 10.1 Cross-Phase Correlation

After Phases 2–6 complete, look for correlations across the diagnostic results:

- Benchmarks where encoder quality (FDR) is low — do they also have high ECE?
- Benchmarks where NIG is violated — do they also have poor AUROC on OOD detection?
- Benchmarks with slow convergence — do they also show high forgetting ratio?

These correlations reveal whether problems cluster together (a single root cause producing multiple symptoms) or are independent (multiple separate issues).

### 10.2 Confusion Matrix Deep Dive

For the two highest-headroom benchmarks, compute the full confusion matrix after training.

Identify:
- The 5 most confused class pairs
- Whether confusion is symmetric (A→B ≈ B→A) or asymmetric (A→B >> B→A)
- Whether confused classes have nearby encoded-space prototypes

```
Symmetric confusion between classes A and B:
  Their encoded-space distributions overlap.
  Upgrade direction: improve class-specific feature separation.

Asymmetric confusion (A frequently misclassified as B, not vice versa):
  Class A is underrepresented or its prototype is displaced.
  Upgrade direction: check class imbalance effects, improve delta_lr for class A.

Confusion between many classes via a hub class:
  One class is acting as a "garbage bin." 
  Upgrade direction: investigate that class's prototype quality and NIG calibration.
```

### 10.3 Timing Profile

Profile wall time breakdown of a full train_step and a full inference call.

For each: record what fraction of time is spent in each component:
```
train_step breakdown:
  - Encoding (RFF forward)
  - World prior update
  - Class differential update
  - Fisher-Rao gradient computation
  - GH gate computation
  - State save/buffer management

inference breakdown:
  - Encoding
  - LLR computation
  - GH gate check
  - Confidence score computation
```

**Purpose:** Identify whether the 25% overhead budget from the original upgrade plan was correctly allocated. If encoding takes 60% of inference time, that is the obvious target for optimisation. If LLR computation takes 60%, that is a different target.

### 10.4 Memory Scaling

Record memory footprint at:
- 100 samples trained
- 1,000 samples trained
- 10,000 samples trained
- 100,000 samples trained (if feasible)

Plot memory vs sample count. Expected: sub-linear (sufficient statistics should compress data). If memory grows linearly with sample count, the buffer management has a leak.

---

## 11. Phase 8 — Scaling & Stress Tests

**Answers:** Q9 (scaling laws), identifies architecture limits

### 11.1 Class Count Scaling

Run Cypha on Benchmark S2 (Gaussian clusters) with class counts: 2, 5, 10, 20, 50, 100, 200.

Record accuracy, convergence speed, memory, and wall time per step for each class count.

Plot: accuracy vs class count; wall time per step vs class count.

```
Accuracy degrades sharply beyond N_critical classes:
  The LLR multi-class formulation has a capacity limit.
  Identify N_critical. This is a hard constraint on system applicability.

Wall time grows super-linearly with class count:
  Algorithmic complexity is suboptimal.
  Identify which component scales badly (likely the K-class LLR sum).
```

### 11.2 Feature Dimension Scaling

From Benchmark S7 results: plot accuracy vs input dimensionality.

Find `d_critical` where accuracy starts declining with increasing input dim.

```
d_critical exists at low dimensionality (< 100):
  RFF dimensionality is not compensating for high input dim.
  Upgrade direction: adaptive D_rff scaling with input dim.

Accuracy flat with increasing input dim:
  Dimensionality handling is robust.
```

### 11.3 Throughput Under Load

On the target hardware (ThinkStation P920):
- Measure inference throughput: samples per second at batch size 1 (pure online mode)
- Measure train throughput: updates per second
- Measure peak memory under continuous training for 1 hour

**Purpose:** Establish operational limits for the C++ port planning. The Python prototype does not need to be fast, but these numbers set the floor that the C++/OpenCL implementation must beat by at least 10×.

### 11.4 Concurrent Session Stress Test

Run 10 simultaneous inference sessions, each with different class distributions.

Verify:
- No cross-session contamination (one session's OOD does not affect another's chi gate)
- Per-session accuracy identical to single-session accuracy
- Memory grows linearly with session count (no shared state leaking)

---

## 12. Results Interpretation Matrix

Use this table after all phases complete. For each finding, identify what it points to.

| Finding | Implied Bottleneck | Ruled Out | Next Action |
|---------|-------------------|-----------|-------------|
| Low mean FDR (< 0.5) in encoded space | Encoder quality | NIG, GH gate | Investigate kernel, D_rff, encoder architecture |
| linear(h) ≈ kernel(h) | Not LLR linearity | Boundary geometry | Focus on posterior or encoder |
| linear(h) << kernel(h) | LLR linearity ceiling | Encoder adequacy | Design non-linear discriminant extension |
| Normality tests fail > 50% of classes | NIG misspecification | LLR linearity | Design mixture posterior or non-parametric alternative |
| ECE > 0.15 | Posterior calibration | Structural issues | Tune NIG hyperparameters, add calibration layer |
| OOD AUROC < 0.7 | GH gate design | Everything else | Redesign OOD detection criterion |
| Forgetting ratio > 0.2 | Online update rule | Architecture | Investigate EWC-style forgetting prevention |
| Δ_cellai ≤ 0 | CellAI not contributing | SSM design | Fix CellAI state informativeness before any SSM upgrade |
| Effective memory depth < 50 for long sequences | SSM decay rates | Encoder | Add scales, reduce decay rates |
| Scale variance near-zero for multiple scales | Wasted capacity | Decay design | Prune scales, consolidate |
| Accuracy saturated on all benchmarks (headroom < 2%) | Benchmark choice | Architecture | Need harder benchmarks; current tasks too easy |
| Large online_gap (Cypha << online competitors) | Core algorithm | Benchmark design | Fundamental algorithmic review needed |
| LLR confusion asymmetric between specific classes | Class imbalance or prototype displacement | Global architecture | Per-class lr tuning, imbalance handling |
| Memory grows linearly with sample count | Buffer management | Inference logic | Audit _D_buf and sufficient statistics accumulation |
| NIG hyperparameter sweep: accuracy varies > 5% | Prior calibration | Architecture | Empirical Bayes warm-start for NIG hyperparameters |
| Encoder norm explodes during adversarial test | Adversarial bypass regression | OOD detection | Stop. Fix enc_lr scaling before anything else |
| Wall time super-linear with class count | LLR complexity | Memory | Algorithmic optimisation of K-class sum |

---

## 13. Upgrade Decision Tree

After Phase 1–8 results are in, follow this tree to determine what to build next.

```
START
│
├─ All benchmarks saturated (headroom < 2% everywhere)?
│   YES → Stop. Design harder benchmarks with more real-world tasks. Return to START.
│   NO  → Continue.
│
├─ Large online_gap on any open benchmark?
│   YES → Core algorithmic review. Compare train step logic against Hoeffding Tree 
│         and Passive-Aggressive. Identify structural difference.
│   NO  → Continue.
│
├─ Forgetting ratio > 0.2?
│   YES → PRIORITY: Online learning stability before anything else.
│         Upgrade direction: normalised delta_lr, decoupled class statistics,
│         EWC-style importance weighting.
│   NO  → Continue.
│
├─ Encoder norm explodes in adversarial test?
│   YES → STOP EVERYTHING. Fix enc_lr scaling bug regression before any upgrade.
│   NO  → Continue.
│
├─ Mean FDR < 0.5 AND kernel(h) ≈ linear(h)?
│   YES → Encoder is the bottleneck AND the encoded space is already nearly linear.
│         The problem is kernel quality/coverage, not the discriminant.
│         Upgrade direction: D_rff sweep to find optimal size,
│         kernel comparison to find best-matched kernel family,
│         deterministic quasi-random frequency sampling (Halton/Sobol sequences).
│   PARTIAL (FDR low, kernel >> linear) →
│         Encoder AND non-linear structure both bottlenecks.
│         Upgrade direction: two-step — fix encoder first, then revisit linearity.
│   NO  → Continue.
│
├─ linear(h) << kernel(h) on open benchmarks?
│   YES → LLR linearity is a hard ceiling for these tasks.
│         Upgrade direction: kernel LLR (apply kernel trick to the LLR discriminant),
│         or two-layer encoding to pre-linearise.
│   NO  → Continue.
│
├─ Normality tests fail > 50% of classes?
│   YES → NIG misspecification.
│         Upgrade direction: mixture NIG posterior (2–3 component mixture per class),
│         or Student-t posterior (heavier tails), or empirical covariance.
│   NO  → Continue.
│
├─ ECE > 0.15?
│   YES → Calibration problem. Does NIG hyperparameter sweep improve ECE?
│         YES → Run empirical Bayes NIG warm-start upgrade.
│         NO  → Investigate GH gate design for calibration role.
│   NO  → Continue.
│
├─ Δ_cellai ≤ 0 (CellAI not contributing)?
│   YES → CellAI produces no useful state. Fix SSM before any CellAI upgrade.
│         Run 8.2 (effective memory depth) and 8.3 (scale contribution) to find why.
│   NO  → Continue.
│
├─ Effective memory depth < target_depth?
│   YES → SSM timescale problem.
│         Upgrade direction: add scales with lower decay rates,
│         initialise decay rates from autocorrelation analysis of training data.
│   NO  → Continue.
│
└─ All above: NO
    → System is performing near its theoretical ceiling for current task set.
      Introduce harder benchmarks (R3 Digits, R5 MNIST, R6 WikiText-2).
      Repeat from START with new benchmarks.
```

---

## 14. Candidate Upgrades Ranked by Evidence

This section lists all plausible upgrades and the diagnostic evidence required before building each one. Nothing in this list gets implemented until the corresponding diagnostic finding is confirmed.

### Tier 1 — Only Build If Diagnostic Confirms It

**U-A: Empirical Bayes NIG Warm-Start**  
Evidence required: NIG hyperparameter sweep shows > 5% accuracy variation, AND phase-1 ECE > 0.1.  
What it does: Fits κ₀ and ν₀ from the first M samples of each run using method-of-moments. Removes the need for hand-tuned priors.  
Risk: Low. Only changes initialisation, not the update rule.  
Complexity: Low.

**U-B: Mixture NIG Posterior (2-component per class)**  
Evidence required: Normality tests fail on > 50% of classes on at least one open benchmark.  
What it does: Each class maintains two NIG components (a mixture). Online EM assigns new samples to the better-matching component.  
Risk: Medium. Changes the core posterior representation.  
Complexity: Medium.

**U-C: D_rff Increase**  
Evidence required: D_rff sweep shows monotonically increasing accuracy with no plateau within feasible compute budget.  
What it does: Simply increase the number of random Fourier features. No algorithmic change.  
Risk: Near-zero. Purely additive.  
Complexity: Trivial.

**U-D: Quasi-Random Frequency Sampling (Halton / Sobol)**  
Evidence required: Mean FDR is low AND D_rff sweep shows diminishing returns from random projection (plateau).  
What it does: Replaces random ω sampling with a low-discrepancy quasi-random sequence (Halton or Sobol), giving better coverage of frequency space with the same D_rff.  
Risk: Low. Compatible with all existing downstream logic.  
Complexity: Low.

**U-E: Kernel LLR (Non-Linear Discriminant)**  
Evidence required: linear(h) << kernel(h) confirmed on at least two open benchmarks.  
What it does: Replaces the linear LLR `δμ_k^T · Σ⁻¹ · h` with a kernel version using an RBF kernel on the encoded space. Computable online via Nyström approximation of the kernel matrix.  
Risk: High. Changes the core inference math. Requires careful validation.  
Complexity: High.

**U-F: Additional SSM Scales with Data-Driven Decay Rates**  
Evidence required: Effective memory depth < target on sequence benchmarks.  
What it does: Adds L_extra scales initialised with decay rates derived from the autocorrelogram of training data. Scale decay rates are not fixed but initialised to match the observed temporal statistics.  
Risk: Low for SSM stability. Medium for CellAI-CyphaDIF interface.  
Complexity: Medium.

**U-G: Forgetting Prevention (Decoupled Class Statistics)**  
Evidence required: Forgetting ratio > 0.2.  
What it does: Maintains a per-class sample count and uses it to compute an importance-weighted learning rate — classes with many examples resist rapid change. New classes can update freely; well-established classes require more evidence to shift.  
Risk: Low. Only modifies effective learning rate per class.  
Complexity: Low to medium.

**U-H: Kernel Mixture (RBF + Arc-Cosine)**  
Evidence required: Kernel comparison in Phase 2 shows no single kernel dominates across benchmarks.  
What it does: Runs two RFF encoders in parallel (RBF and arc-cosine kernels), concatenates outputs, and allows CyphaDIF to weight the concatenated space via a learned mixing vector updated online by the LLR gradient.  
Risk: Medium. Doubles encoder dimensionality.  
Complexity: Medium.

### Tier 2 — Consider After Tier 1 Is Stable

**U-I: Adaptive GH Gate Threshold**  
Evidence required: OOD AUROC varies significantly with threshold in Phase 6 sweep.  
What it does: Replaces the static chi-squared threshold with one that adapts based on recent false positive rate — if the gate is triggering too often on in-distribution samples, relax it; if it is missing OOD samples, tighten it.  
Risk: Low.  
Complexity: Low.

**U-J: Robust Covariance Estimation**  
Evidence required: Moderate normality violation (50–80% of dimensions Gaussian).  
What it does: Replaces the standard NIG covariance update with Ledoit-Wolf shrinkage estimator, which is more robust to non-Gaussianity and small sample counts. Drop-in replacement for the covariance component.  
Risk: Low.  
Complexity: Low.

---

## 15. Reporting Format

### 15.1 Per-Phase Report Structure

After each phase completes, produce a brief report with this structure:

```
Phase N Report
==============
Date / time completed:
Git commit of Cypha tested:
Seeds used:

FINDINGS:
  [Bullet list of concrete numerical findings]

INTERPRETATION:
  [What each finding implies per the interpretation matrix]

DECISIONS:
  [Which upgrade candidates are now confirmed, ruled out, or still pending]

OPEN QUESTIONS:
  [What this phase revealed that needs follow-up investigation]

NEXT PHASE:
  [Any modifications to Phase N+1 based on these results]
```

### 15.2 Master Results Dashboard

Maintain a single `MASTER_RESULTS.md` that is updated after every phase. It contains one row per benchmark per system, updated as results come in. This is the source of truth for all comparisons.

### 15.3 Crash and Anomaly Log

Any NaN, Inf, division by zero, or unexpected exception gets logged to `logs/crash_reports/` with:
- Phase and test number
- Benchmark and seed
- Full traceback
- Cypha config at time of crash

Do not silently swallow exceptions. Crashes are data.

### 15.4 Decision Gate

After Phases 1–4 complete, pause and review. Do not proceed to Phases 5–8 if:
- Any known bug regression is confirmed (encoder norm explode, chi contamination)
- Forgetting ratio > 0.4 (system is too unstable to reliably benchmark)
- All benchmarks are saturated (need new benchmarks first)

These are stop conditions, not warnings.

---

*Document version: 1.0*  
*Prepared for Cursor execution on ThinkStation P920*  
*No source code modifications permitted during diagnostic phases*
