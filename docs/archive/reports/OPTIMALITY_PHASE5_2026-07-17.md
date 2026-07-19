# Optimality Phase 5 — Orthogonal / Leverage-Score Features (2026-07-17)

**Build:** `native/build_opt_p5` (Ninja, Release, MinGW 13.2.0)  
**Scope:** Independent of Phase 2; did not touch `build_math`, `build_deff`, `BASELINE_*`, overnight scripts.

## Shipped

| Component | API | Default | Notes |
|-----------|-----|---------|-------|
| **SORF RFF** | `RffProjectionKind::Sorf`, `KernelMemory::make_orthogonal_rff`, `PreprocessorState::rff_sorf` | **Off** (iid Gaussian preserved) | QR-block orthogonal rows (Yu et al. style); opt-in only |
| **Leverage Nyström** | `LandmarkSamplingKind::LeverageScore`, `KernelMemory::init_leverage_landmarks_from_samples`, `select_leverage_landmark_indices` | **Off** (uniform reservoir preserved) | Ridge-leverage batch + online replacement |
| **Shared math** | `native/include/cypha/rff_features.hpp`, `native/src/rff_features.cpp` | — | `‖K−K̂‖_F` estimators, FWHT helpers (for future Fastfood blocks) |
| **CTest** | `native_kernel_approx_p5_smoke` | — | Tier-1 leverage bound asserted; SORF measured |

## Measurements (`kernel_approx_p5_smoke`, seed=42424242, n=48, d=8)

| Quantity | Baseline | Optimized | Δ |
|----------|----------|-----------|---|
| RFF `‖K−K̂‖_F` @ D=256 | iid **2.554** (mean over 8 seeds) | SORF **2.219** | **−13.1%** |
| Nyström `‖K−K̂‖_F` @ m=32 | uniform **0.648** | leverage **0.554** | **−14.5%** |

## XOR kernel LLR (default path unchanged)

`xor_kernel_bench --kernel-xor-features` (3 seeds, nystrom M=256, xor_pair features):

- **kernel_mean_acc: 0.982** (linear 0.512) — D03 xor_pair default not regressed
- Leverage/SORF not wired into D03 default; remain opt-in on `KernelMemory` / bench env flags

## Goldens

**Not regenerated.** Default numerics unchanged (`IidGaussian`, `Uniform` reservoir). Existing tests green:

- `native_regression_rff`, `native_kernel_llr`, `native_xor_kernel_bench_smoke`, `native_kernel_snapshot_roundtrip`

## Promotion path

1. **Leverage Nyström** — ready for opt-in bench sweep (`set_landmark_sampling(LeverageScore)` or `init_leverage_landmarks_from_samples` on calib set).
2. **SORF** — opt-in via `make_orthogonal_rff` / `rff_sorf=true`; promote after D03 RFF-basis sweep confirms XOR lift.
3. **Fastfood FWHT blocks** — `fwht_inplace` stubbed in `rff_features`; full Walsh pipeline deferred (current SORF uses QR blocks).

## Files touched

- `native/include/cypha/rff_features.hpp`, `native/src/rff_features.cpp` (new)
- `native/include/cypha/kernel_memory.hpp`, `native/src/kernel_memory.cpp`
- `native/include/cypha/preprocessor.hpp`, `native/src/preprocessor_fit.cpp`
- `native/tools/kernel_approx_p5_smoke.cpp` (new)
- `native/CMakeLists.txt`
