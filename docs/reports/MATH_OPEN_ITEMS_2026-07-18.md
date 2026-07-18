# Math §0-bis open items — mid-tier closeout (2026-07-18)

**Binary:** `native/build_ewc_d16/cyphalm_bench_native.exe`  
**Artifacts:** `bench/results/math_open/` (summarize: `python scripts/summarize_math_open.py`)  
**Seed / corpus:** `bench_seed=42`, WikiText official split (`CYPHA_BENCH_FULL_CORPUS=1`)

## 1. Eigenvalue `D_eff` (+0.096 claim)

| Arm @ 5k | BPC | κ | ΔBPC vs base |
|----------|-----|---|----------------|
| base (no math) | 4.0007 | 0.873 | — |
| math + variance-proxy `D_eff` | **3.9837** | 0.869 | **−0.017** |
| math + eigenvalue `D_eff` | 4.0951 | 0.874 | **+0.094** |

**Verdict:** d51 claim **reproduced** (~+0.096). Eigenvalue estimator remains **joint-fail / do not promote**; keep variance-proxy default in `apply_math_integration_preset`.

## 2. κ-target vs held-out BPC @ 5k

| `kappa_lambda_target` | BPC | κ |
|-----------------------|-----|---|
| 0.70 | 3.9909 | 0.8690 |
| 0.83 (preset) | 3.9837 | 0.8694 |
| 0.95 | 3.9845 | 0.8691 |

**Verdict:** Held-out BPC barely moves; achieved κ is flat across targets. κ-targeting is **not** a clear held-out generalization lever at 5k (and remains **harmful @ 300k** per production report +0.209 BPC). No further short-budget κ sweeps.

## 3. Scale-dependent sign flip (math − base ΔBPC)

| n_train | base BPC | math BPC | ΔBPC | Sign |
|---------|----------|----------|------|------|
| 500 | 5.468 | 5.650 | **+0.182** | worse |
| 2 000 | 4.629 | 4.666 | **+0.037** | worse |
| 5 000 | 4.001 | 3.984 | **−0.017** | better |
| 20 000 | 3.776 | 3.659 | **−0.117** | better |
| 300 000 (lock) | 2.864 | 3.073 | **+0.209** | worse |

**Verdict:** Sign flips **twice**: harmful → helpful (~2k–5k) → harmful again by 300k. Mid-tier (20k) is the **largest help** seen (−0.117). Production preset still fails the 300k joint lock; do not default-on overnight without a new recipe.

## 4. Ablation flatness @ 20k (`hybrid_blend_lr`)

| `--hybrid-blend-lr` | BPC @ 20k math |
|---------------------|----------------|
| 0.005 | 3.6598 |
| 0.02 | 3.6586 |

Δ ≈ **0.001** ≪ 0.01 falsify threshold → **still flat** at mid-tier for the only non-flat 5k knob. Production-tier recheck does **not** revive hyperparameter sensitivity here.

## Disposition (BoW §0-bis)

| Item | Status after this pass |
|------|------------------------|
| Eigenvalue `D_eff` +0.096 | **Closed** — reproduced; keep OFF |
| κ held-out transfer | **Closed at short/mid** — no lever; 300k already negative |
| Scale sign flip | **Characterized** — mid help / overnight hurt; open for recipe redesign only |
| Ablation grids flat | **Confirmed @ 20k** — not a FAST-only artifact for `hybrid_blend_lr` |

## Human-gated (unchanged)

- `gh auth login` → `publish_release.ps1`
- Paper venue upload
