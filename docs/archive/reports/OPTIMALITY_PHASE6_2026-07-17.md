# Optimality Phase 6 — Variational IB Encoder (2026-07-17)

**Build:** `native/build_opt_p6` (Ninja, Release, MinGW 13.2.0)  
**Scope:** Did not touch `native/build_math`, `native/build_deff`, `BASELINE_*`, overnight scripts, or D17 hybrid default training.

## Shipped (partial — honest scope)

| Component | API | Default | Notes |
|-----------|-----|---------|-------|
| **Variational IB update** | `variational_ib_update_encoder_w` in `encoder_contrastive.hpp/.cpp` | **Off** | Alemi-style bound in deterministic `h=Wf` limit: `grad_h = h/σ² − β·(r_k − r_j)` |
| **Fisher–Rao contrastive** | `contrastive_update_encoder_w` | **On** (unchanged) | Default DIF / D17 path byte-identical when IB flag off |
| **Train opt-in** | `TrainStepExtras::use_variational_ib_encoder`, `ib_beta` | `false`, `β=1.0` | Wired in `dif_train_step_vector` misclassification + deliberate branches only |
| **Frobenius cap** | `‖W‖_F ≤ 8.0` | unchanged | Shared `frobenius_cap` every 50 encoder steps |
| **MI proxy** | `latent_class_mi_proxy(h, labels)` | — | Between / (within + between) centroid ratio in `[0,1]` |
| **CTest** | `native_encoder_ib_p6_smoke` | — | Direct update smoke + 2-class blob train/compare |

## Deferred

- **Full Alemi stochastic encoder** `q(t|x)=N(f(x), σ_e²I)` with reparameterization and analytic KL — current path uses deterministic limit + isotropic prior compression term.
- **REST / `.cypha` / train_hparams.json** exposure of `use_variational_ib_encoder` and `ib_beta` (train harness only today).
- **Encoder golden regeneration** — not needed while default OFF; required before default-on promotion.
- **R1–R4 bench sweep** with IB ON — smoke uses synthetic 2-class blobs only.
- **β sweep / OOD separation metrics** — single `β=1.0` in CTest.

## Measurements (`encoder_ib_p6_smoke`, 3 seeds, 2-class blobs, d=12, β=1.0)

| Quantity | Fisher–Rao (default) | Variational IB (opt-in) | Δ |
|----------|----------------------|-------------------------|---|
| Test accuracy | **1.000** | **1.000** | **0.000** |
| Latent MI proxy | **0.999** | **0.999** | **+8.4e-05** |
| Max `‖W‖_F` | 1.74 | 1.74 | within cap |

Task is linearly separable after encoder co-training; both paths saturate. No regression on default path: `train_step_vector_golden` passes unchanged.

## Bound semantics

- **Tier-1 target:** minimize variational upper bound on `I(X;T) − β·I(T;Y)` (Alemi et al.).
- **Shipped approximation:** compression via `‖h‖²/(2σ²)` prior + class-conditional Fisher residuals for the prediction term — same trigger points as contrastive (misclass + deliberate ambiguity).

## Promotion path

1. Expose `use_variational_ib_encoder` / `ib_beta` on REST + Qt hparams form.
2. β sweep on Wine / Digits (R1/R3) with IB ON vs Fisher–Rao baseline.
3. If lift holds, regenerate `train_step_vector` golden under IB or keep dual goldens.
4. Optional: stochastic encoder noise + exact KL for tighter Tier-1 saturation.

## Files touched

- `native/include/cypha/encoder_contrastive.hpp`, `native/src/encoder_contrastive.cpp`
- `native/include/cypha/train_step_vector.hpp`, `native/src/train_step_vector.cpp`
- `native/tools/encoder_ib_p6_smoke.cpp` (new)
- `native/CMakeLists.txt`
