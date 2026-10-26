# M1: ECS and LCI library

**Date:** 2026-10-27
**Author:** Odin Loch
**Scope:** PLAN.md section 7 (ECS) and section 8 (LCI). SCRIBE is not part of this item.

## What is implemented

Two header-only files, standard library only, namespace `cypha::eval`:

| File | Contents |
|---|---|
| `native/include/cypha/eval/ecs.hpp` | Beta helpers and Wilson bounds; evidence items and the factor helpers (7.1); fusion into an opinion (7.2); grades (7.3); answer confidence, risk-coverage and abstain threshold (7.4); ECE, Brier, reliability bins, PAV and an isotonic calibrator (7.5); admission rules (7.6) |
| `native/include/cypha/eval/lci.hpp` | Bayesian bootstrap with optional clusters; canary non-inferiority; gamma, c, sigma, kappa; `compute_lci`; alpha spending, e-process and Benjamini-Hochberg (8.4); Thresholdout (8.5); BWT and learning-curve fit (8.6) |
| `native/tools/ecs_lci_test.cpp` | 242 known-answer checks, nonzero exit on failure |
| `native/CMakeLists.txt` | One commented block appended at the end: target `ecs_lci_test`, CTest `native_ecs_lci_test` |

## API in brief

ECS:

- `reg_inc_beta(x,a,b)`, `beta_quantile(p,a,b)`, `beta_credible_interval(a,b,level)`, `wilson_lower/upper(k,n,z)`, `BetaPosterior` (with `observe`) and the four source priors.
- `EvidenceItem{stance, omega, tau, pi, theta, delta}.mass()`. Also `independence_weight(n)`, `independence_weights(cluster_ids)`, `time_decay(dt,h)`, `derivation_factor(premise_ecs, rule_reliability, depth)`.
- `fuse(items, FusionConfig)` and `opinion_from_counts(r, s, cfg)` return an `Opinion` with r, s, b, d, u, P, the Beta(r+aW, s+(1-a)W) parameters, the interval and the `Grade`.
- `answer_confidence`, `risk_coverage_curve`, `abstain_threshold`.
- `brier_score`, `reliability_bins`, `expected_calibration_error`, `isotonic_pav`, `IsotonicCalibrator`.
- `admission_decision`, `usable_as_premise`, `use_in_answer`, `pattern_admissible`, `weekly_audit`.

LCI:

- `bayesian_bootstrap(diffs, threshold, cluster_ids, {draws=4000, seed})` returns `BootstrapResult` with P(delta > threshold), mean, 90% range, n_items and n_eff.
- `canary_noninferiority`, `canary_r`, `generalisation_gamma`, `calibration_factor`, `stability_sigma`, `cost_kappa`.
- `compute_lci(LciInputs)` returns `LciDecision` with every component, `product`, capped `lci`, `meets_threshold`, and `components()` for logging. `promote(decision, multiplicity_ok)` applies the 0.95 rule together with the multiplicity result.
- `alpha_spend(i)`, `AlphaSpender`, `EProcess`, `multiplicity_check`, `e_to_p`, `benjamini_hochberg`, `bh_adjusted`.
- `Thresholdout::query(dev, test)`, `backward_transfer(R)`, `fit_learning_curve(n, L)` (with `predict` and `n_for`).

## Test results

- Standalone `g++ 13.3 -O2`, C++20 and C++23, `-Wall -Wextra -Wpedantic`: no warnings, 242 checks, 0 failed, about 0.6 s.
- CMake (Ninja, Release): configure 1.7 s, `ecs_lci_test` built in about 5 s, `ctest -R native_ecs_lci_test` passed.
- Known answers covered: Beta(2,5) mean and median 0.26445; Wilson 90/100 = 0.8256; no evidence gives P = a and u = 1; 40 supporting items give High; a contradicted functional fact gives Contradicted (and is not contradicted when phi = 0); PAV on [1,3,2,3,5,4,6]; ECE 0 on a calibrated set; BH on a known p-vector; bootstrap P near 1 for a clear gain and near 0.5 for zero mean; n_eff cap; Thresholdout; BWT = -0.15; learning-curve recovery of Linf=0.2, a=2, b=0.5 (error below 1e-5).
- Extra checks: empirical anytime validity of the e-process (10 of 400 null streams ever reached e >= 20, i.e. 2.5% against the 5% bound; the test asserts at most 5%), and a cluster-level bootstrap that is clearly wider than the item-level one on duplicated items.
- Two of my first test expectations were wrong, not the library. A mixture e-process under a loss settles near 0.33 rather than going to 0, because the small-lambda components lose little. That is inherent and harmless, since it stays below 1. A 150-item sample was too noisy for a 0.999 threshold.

## Spec ambiguities and what I chose

1. **Wilson z.** Not given. Default z = 1.96 (95% two-sided); this reproduces 0.8256 for 90/100. Used for the audit and for pattern admission too. `z` is a parameter.
2. **Time decay.** The plan says "half-life" but writes exp(-dt/h), which is an e-folding time. Implemented as written. For a true half-life, pass h/ln 2.
3. **Derivation depth cap 3.** Observed facts have depth 0 (delta = 1). A derived item has depth 1 + max premise depth. Depth above 3 gives delta = 0.
4. **Constraint penalties.** Added per violation (5 hard, 1 soft).
5. **Grade order.** Contradicted is checked first, then High, Moderate, Low, Very low. The table's rows only overlap at the edge d = 0.5.
6. **Admission gaps.** The plan leaves two cases open. P >= 0.8 with a grade below Moderate is treated as Candidate. P < 0.5 is Reject. "Above the threshold" is read as >=.
7. **Abstain threshold.** The highest-coverage point on the risk-coverage curve whose accuracy is at least the target precision. Ties in confidence are grouped.
8. **Isotonic map.** Tied scores are merged, PAV is run, and the map is interpolated linearly between points. ECE uses 10 equal-width bins.
9. **Clusters.** Resampling is over clusters, each valued at the mean of its items, with equal Dirichlet weight per cluster. n_eff is the number of clusters; no further correlation correction is applied.
10. **Units and sign.** epsilon and delta_k are in the same units as the differences (1 point = 0.01 on a 0-1 scale). All differences are oriented higher-is-better, so the caller negates BPC.
11. **Gamma draws.** Gamma(1,1) is exactly Exp(1), so it is drawn as -ln U from raw `mt19937_64` output. `std::gamma_distribution` gives different streams on different standard libraries.
12. **gamma with no dev gain.** If gain_D <= 0 then G = 0 (conservative).
13. **sigma.** 1 only if the gain has the same strict sign in at least 3 seeds. Fewer than 3 seeds, or any zero gain, gives 0.5.
14. **Empty canary.** A canary with no data cannot show non-inferiority, so it fails.
15. **e-process.** The plan only says "sequential e-values". I used an equal-weight mixture of 8 fixed-lambda betting martingales for bounded differences in [-B, B]. It needs no tuning, is valid under Ville's inequality, and costs at most ln 8 nats against the best single lambda. The null is mean <= `null_mean` (default 0; pass epsilon to test "at least epsilon"). Feed items in random order, or one cluster mean per cluster. Rejected candidates must still call `AlphaSpender::next()`.
16. **BH p-values.** The plan does not say where they come from. The function takes any p-vector. `e_to_p` gives min(1, 1/e). The bootstrap's P(delta <= eps) is a posterior probability, not a p-value, so I did not use it.
17. **Thresholdout.** Follows Dwork et al. Algorithm 1, with dev in the "training" role and T as the holdout. Budget is spent only when T is released. Defaults T = 0.04, sigma = 0.01, budget = 100 are my choice, not from the plan. Rotation every 20 promotions is the caller's job (`rotate()`).
18. **BWT.** Mean over the first T-1 tasks (Lopez-Paz and Ranzato). A vector form averages over whatever tasks are passed.

## What is left

- Wire LCI to the experiment DB and the referee flow. Thresholdout answers scalar queries, so how it gates the per-item differences the bootstrap needs on T is not decided.
- Ingest-side pieces of 7.1: MinHash clustering, per-relation half-life and phi learning, and Beta updates of source, pattern and rule posteriors from verification.
- The audit rollback (7.5) and the dashboards in 8.6 other than the BWT and curve primitives.
- Optional: e-BH for e-values, uncertainty bands on the learning-curve fit.
- SCRIBE (a separate item).
- Not tested on MinGW or Windows; the target uses the standard `cypha_mingw_apply_exe_link_options` call.
