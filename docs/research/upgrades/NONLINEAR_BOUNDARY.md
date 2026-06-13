# Nonlinear boundary fix — CyphaDIF discriminant

**Author:** Odin Loch  
**Problem:** Linear LLR in latent space hard-ceils XOR at ~48.2% vs kernel SVM ~83.5% (**32.3 pp gap**). Same limit affects nonlinear regression (Feynman R²≈0) and weakens CyphaLM token routing.

**Status:** **Fix 1 (Nyström kernel LLR) partially SHIPPED** — native C++ in `native/src/kernel_memory.cpp`; CTests `native_kernel_llr`, `native_xor_kernel_bench_smoke`. Tuning continues (M=512 profile ~71% XOR; ~18 pp gap vs sklearn RBF ceiling). See [`docs/FUTURE.md`](../../FUTURE.md) §0a.

---

## Root cause

Pipeline: `x → W_enc (linear) → h → LLR (linear in h) → y*`

Diagonal Gaussian LLR with tied covariances is linear in h. No nonlinear transform in the discriminant path — structural, not tuning.

---

## Fix taxonomy (ordered by cost / fit)

| # | Fix | Status | Notes |
|---|-----|--------|-------|
| **1** | **Nyström kernel LLR** | **Partially shipped** | Whitened landmarks, median-γ RBF; `use_kernel_llr` in score path |
| 2 | RFF kernel LLR | Partial (RFF encoder exists) | Extend to LLR path; preferred for streaming LM |
| 3 | Learned nonlinear encoder (2-layer MLP) | Planned | Diagnostic — if XOR closes, bottleneck is encoder only |
| 4 | GRIA-α kernel | Planned | Theoretically coherent; build on Fix 1 Nyström |
| 5 | Spectral mixture kernel | Long-term | σ_k ∝ 1/k harmonic structure |

---

## Fix 1 — Nyström kernel LLR (shipped baseline)

Nyström features:

```
φ(h) = K(h, Z) · K(Z,Z)^{−1/2}     Z = m landmarks
```

LLR in RKHS: `log p(φ(h) | N(μ_k^φ, diag v_k^φ))`

**Native (shipped):** Reservoir landmarks (M=256 default), Cholesky whitening, online softmax gradient on φ(h), blended into `score_matrix` when kernel enabled.

**Measured (2026-06):** Native linear 49.9% → kernel **59.2%** (+9.3 pp, 3 seeds); XOR bench +9–10 pp vs linear; sklearn RBF ceiling ~79% on same splits.

**Hyperparameters to sweep:** m ∈ {64,128,256,512}; σ median heuristic; kernel blend; landmark diversity.

**Remaining work:** Close ~18 pp sklearn gap; wire into Option A batched LLR; optional GRIA-α kernel (Fix 4).

---

## Fix 2 — RFF kernel LLR

Random Fourier Features approximate RBF without landmark eigendecomp. Faster init; lower accuracy per feature. **Use for CyphaLM streaming** (no runtime landmark refit). Auto-gamma RFF shipped in preprocessor ([`FUTURE.md`](../../FUTURE.md) §0b).

---

## Diagnostic protocol

| Step | Test | Expected |
|------|------|----------|
| D1 | XOR baseline | ~48% CyphaDIF, ~83% kernel SVM |
| D2 | Nonlinear encoder alone | >80% if encoder is bottleneck |
| D3 | Nyström LLR | >80% (native: ~60–71% today) |
| D4 | Encoder + Nyström | Match/exceed SVM |
| D5 | Full diagnostic suite | All tasks ≥ SVM ceiling |

Domains: S1 linear, S3 XOR, R1 Iris, R3 digits — see [`DIAGNOSTIC_REPORT.md`](../../reports/DIAGNOSTIC_REPORT.md).

---

## Parity requirements

| CTest | Status |
|-------|--------|
| `native_kernel_llr` | **Shipped** |
| `native_kernel_snapshot_roundtrip` | **Shipped** |
| `native_kernel_cypha_roundtrip` | **Shipped** |
| `rff_kernel_parity` | Planned |
| `nonlinear_enc_parity` | Planned |
| `alpha_kernel_parity` | Planned |

Extend [`PORT_CONTRACT.md`](../../port/PORT_CONTRACT.md) when new fixes land in native.

---

## Connection to CyphaLM / RPSM

- Nyström in CyphaDIF router → nonlinear token boundaries  
- RFF variant for autoregressive streaming  
- Feeds **Option A** in [RPSM_COMBINED_SPEC.md](RPSM_COMBINED_SPEC.md) Step 2 before Option B sequence layer  

Zero catastrophic forgetting (per isolated model) must be verified after each fix — forgetting ratio probe on D16F-style saves.
