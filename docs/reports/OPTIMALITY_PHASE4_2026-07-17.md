# Optimality Phase 4 — Bayesian model averaging over Δk (2026-07-17)

**Build:** `native/build_opt_p4` (Ninja, Release, MinGW 13.2.0)  
**Scope:** Opt-in analytic BMA at inference; did not touch `build_math`, `build_deff`, `BASELINE_*`, overnight. Left `CYPHA_*.md` for docs agent.

## Verdict

| Acceptance | Result |
|------------|--------|
| Analytic BMA over NIG posterior of Δk | **Shipped (bounded)** — closed-form LLR correction + epistemic variance |
| Credible interval as `confidence` | **Shipped** — midpoint of [credible lower, MAP point] |
| ECE not worsen vs MAP | **Pass within tol** — ΔECE = +0.0288 ≤ 0.05 |
| Coverage smoke | **Pass** — correct-row conf≥0.5 coverage = 0.592 |
| Default path / goldens | **Unchanged** — `use_nig_bma` default **OFF** |

## ECE (MC2) before / after — 4-class blobs canary

| Mode | ECE | Accuracy | Coverage proxy |
|------|-----|----------|----------------|
| MAP (flag OFF) | **0.2335** | 0.76 | — |
| BMA (flag ON) | **0.2624** | 0.76 | 0.592 |
| Δ | **+0.0288** | 0.00 | — |

Domain: synthetic 4-Gaussian blobs (n=800, d=12, seed=7), trained 10 passes, held-out 25%. Measured via `cypha::bench::expected_calibration_error` (MC2).

ECE is slightly higher under BMA on this canary (still within smoke tolerance). Accuracy unchanged. Flag stays **default OFF** until a stronger calibration win is shown on R1–R4.

## What shipped (opt-in, default OFF)

| Component | API | Default | Notes |
|-----------|-----|---------|-------|
| Posterior scale | `nig_delta_posterior_scale` | — | `τ = v_mean / (n_obs+1)` |
| BMA LLR correction | `nig_delta_bma_llr_correction` | — | `0.5·τ·(d + Σ r²/inv_v)` subtracted from MAP LLR |
| Epistemic var | `nig_delta_bma_epistemic_var` | — | `τ · Σ r²/inv_v` |
| Credible lower | `nig_delta_credible_lower` | z=1.28 | Clamped to [0,1] |
| Infer flag | `CyphaInferModel::use_nig_bma` | **Off** | Also `CYPHA_USE_NIG_BMA=1` / root bool |
| CTest | `native_nig_bma_p4_smoke` | — | ECE + coverage |

## Bounded approximation (remaining work)

Full conjugacy integral of the class score over the NIG posterior in high-d is deferred. This pass ships the **diagonal-Gaussian Laplace / moment** average:

- Prior/posterior for Δk treated as `N(Δ̂_k, diag(τ / inv_v))`.
- `E[LLR] ≈ LLR(Δ̂) − ½ τ (d + Σ rⱼ²/inv_vⱼ)`.
- Confidence = midpoint of MAP point estimate and Gaussian credible lower bound.

**Remaining for full Tier-1 BMA:** exact Student-t / NIG predictive integral (or quadrature) over the joint NIG posterior; regenerate goldens before default-on; re-measure ECE on Wine/Digits (R1–R4).

## Goldens

**Not regenerated.** Default numerics unchanged (`use_nig_bma=false`).

## Files touched

- `native/include/cypha/nig_gig_math.hpp`, `native/src/nig_gig_math.cpp`
- `native/include/cypha/infer_cpu.hpp`, `native/src/infer_cpu.cpp`
- `native/tools/nig_bma_p4_smoke.cpp` (new)
- `native/CMakeLists.txt`

## Build / test

```
cmake -S native -B native/build_opt_p4 -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_opt_p4 -j8 --target nig_bma_p4_smoke
ctest --test-dir native/build_opt_p4 -R native_nig_bma_p4_smoke --output-on-failure
```
