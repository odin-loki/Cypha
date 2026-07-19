# Optimality Phase 7 — Score Matching GH/NIG Gate (2026-07-17)

**Build:** `native/build_opt_p7` (Ninja, Release, MinGW 13.2.0)  
**Scope:** Independent of `build_math`, `build_deff`, `BASELINE_*`, overnight scripts.

## Shipped

| Component | API | Default | Notes |
|-----------|-----|---------|-------|
| **Score-match GIG norm** | `GigNormalisationMode::ScoreMatch`, `gig_k2k1_score_match`, `gig_k0k1_score_match` | **Off** (LUT preserved) | Hyvärinen score-matching rational fit of `x·K₂/K₁`; `K₀/K₁` via exact recurrence |
| **Env opt-in** | `CYPHA_GIG_SCORE_MATCH=1` | — | CPU `gig_e_inv_v_lam_neg1` / `gig_e_v_lam_neg1` dispatch |
| **Test override** | `set_gig_normalisation_mode_override(mode, active)` | — | CTest isolation |
| **Held-out metric** | `nig_gate_predictive_loglik` | — | Tier-1 NIG gate predictive log-likelihood proxy |
| **CTest** | `native_gate_score_match_p7_smoke` | — | Asserts held-out loglik ≥ LUT |

## Measurements (`gate_score_match_p7_smoke`, seed=42424242, holdout_n=512)

| Quantity | LUT (default) | Score-match | Δ |
|----------|---------------|-------------|---|
| Held-out predictive loglik | **−25.3173** | **−25.1582** | **+0.159** (SM ≥ LUT) |
| Max abs world-gate delta | — | — | **0.00120** |

Score-match beats LUT on held-out log-likelihood with sub-0.2% gate deviation on the operational `(χ, ψ, mp, r)` grid.

## LUT deletion status

**Not deleted.** `bessel_table.hpp`, `bessel_table_data.cpp`, and CUDA Bessel upload remain. Default numerics unchanged; goldens not regenerated.

**Follow-up (safe deletion gate):**

1. Promote `CYPHA_GIG_SCORE_MATCH=1` to default after extended bench / GH deliberation sweep.
2. Port score-match rational to CUDA `d_gig_e_inv_v` (currently LUT-only on GPU).
3. Regenerate `gh_infer_deliberation` + quantile-DIF goldens once default flips.
4. Remove `bessel_table_data.cpp` (~300 KB) and CUDA Bessel table upload.

## Goldens

**Not regenerated.** Default path is still `GigNormalisationMode::Lut`. Verified: `nig_adapt_golden` green on `build_opt_p7`.

## Files touched

- `native/include/cypha/nig_gig_score_match.hpp`, `native/src/nig_gig_score_match.cpp` (new)
- `native/include/cypha/nig_gig_math.hpp`, `native/src/nig_gig_math.cpp` (dispatch + `gig_k2k1_lut`)
- `native/tools/gate_score_match_p7_smoke.cpp` (new)
- `native/CMakeLists.txt`

## Promotion path

1. **Opt-in bench** — run GH/NIG workloads with `CYPHA_GIG_SCORE_MATCH=1`; confirm no regression on deliberation / quantile-DIF fixtures.
2. **CUDA parity** — device-side rational approx so GPU path can drop Bessel upload.
3. **Default flip + LUT delete** — only after (1)–(2) and golden regen.
