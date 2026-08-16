# RPSM combined spec — CyphaDIF matrix refactor + CyphaLM sequence layer

**Author:** Odin Loch  
**Status:** STOP / closed (2026-07-18) — historical spec. Living pin is Hybrid **2.664 BPC**.  
**Historical target:** Beat `hybrid_gria_lstm` D17 BPC **2.873** @ 300k train (pre-Wave2 L1 pin; not met)

See also: [RPSM_IMPLEMENTATION.md](RPSM_IMPLEMENTATION.md) (Option B core), [NONLINEAR_BOUNDARY.md](NONLINEAR_BOUNDARY.md) (Step 2), [README.md](README.md) (index).

---

## Overview

Two parallel workstreams compose into one architecture:

| Track | What | Outcome |
|-------|------|---------|
| **Option A** | Refactor CyphaDIF internals into matrix form (Ψ_mu, Ψ_var) | Same behaviour, batched LLR/GEMM, faster CUDA, parity-validated |
| **Option B** | RPSM sequence layer in CyphaLM | CyphaDIF (post-A) at level 0; hierarchy handles long context |

Neither replaces the other. Nyström kernel LLR (partially **shipped** in `native/src/kernel_memory.cpp`) slots into Option A and benefits Option B automatically.

---

## Execution order

1. **Option A** — CyphaDIF matrix refactor  
2. **Nonlinear boundary** — Nyström kernel LLR into A ([NONLINEAR_BOUNDARY.md](NONLINEAR_BOUNDARY.md); native path shipped, tuning continues)  
3. **Option B** — RPSM sequence layer in CyphaLM  
4. **Global memory** — Izaac episodic store + working memory M_slots + Gaussian mixture world model  
5. **Benchmark** — D17 vs `hybrid_gria_lstm` baseline  

---

## Option A — CyphaDIF matrix refactor

CyphaDIF already implements RPSM maths; the refactor maps components onto unified state matrices without changing behaviour.

### Component mapping

| Current | Post-refactor |
|---------|---------------|
| `WorldPrior θ₀` (μ, v) | Row 0 of Ψ_mu, Ψ_var |
| `ClassDifferential Δk` | Rows 1..K of Ψ_mu, Ψ_var |
| Per-class LLR loop | Batched matmul: `mu_eff = Ψ_mu[0:1] + Ψ_mu[1:]` |
| `EncoderProjection W_enc` | Unchanged |
| `TieredContextBuffer` | Separate context matrix C (3 rows: short/mid/long) |
| `NIGField` posterior | Ψ_var (already diagonal) |
| `PriorityReplayBuffer` | M_slots working-memory matrix |
| MDL `‖Δk‖_F ≤ C` | Frobenius projection on Ψ rows 1..K |
| Fisher-Rao contrastive | Batched natural gradient on Ψ |

### Native target

```cpp
struct CyphaDIFState {
    Eigen::MatrixXf Psi_mu;   // (1 + n_classes, feat_dim)
    Eigen::MatrixXf Psi_var;
    Eigen::VectorXf counts;
};
// LLR: single batched GEMM call — replaces per-class loops
```

### Parity

All existing CTests must pass. Add:

- `native_batched_llr_parity` — batched vs per-class loop, numerically identical  
- `native_batched_update_parity` — batched update vs original  

Expected: same BPC/accuracy, lower wall time on batch infer.

---

## Option B — RPSM sequence layer

CyphaDIF (post-A) plugs into **level 0** as token router. Levels 1..L handle word/chunk/sentence/document hierarchy.

```
Input tokens → Level 0: CyphaDIF (LLR + context prior)
            → Level 1..L: RPSM rows (prediction error hierarchy)
            → Global: M_slots, Izaac store, GMM world model
            → vocab logits
```

State matrix Ψ ∈ ℝ^(L × D):

- Row 0: CyphaDIF world prior compressed to D dims  
- Rows 1..L−1: chunk / sentence / document states  

Core update (see [RPSM_IMPLEMENTATION.md](RPSM_IMPLEMENTATION.md)):

```
dΨ/dt = η(H_up − H_down)(1−A) W_u + α(Ψ_past − Ψ) + βM − λΨ/‖Ψ‖_F
```

CyphaDIF output projects to row 0; multi-level input injection at diminishing scale 1/(l+1).

### Global memory (three layers)

1. **Izaac episodic store** — VRF-keyed Ψ snapshots (stub hash until native Izaac wired)  
2. **Working memory M_slots** — surprise-gated write (NMP > 0.85), soft-attention read  
3. **Gaussian mixture world model** — slow semantic anchor injected into top level (weight ~0.05)  

### Success criteria (Option B)

| Check | Criterion |
|-------|-----------|
| Stability | No NaN after 1000 steps @ Tiny config |
| α range | `gria_alpha_spectral` ∈ [0.3, 0.6] after warmup |
| Forgetting | Ratio < 0.01 after normalised-update fix |
| D17 BPC | **< 2.873** @ Small config (main target) |

---

## Recommended configs

| Config | feat_dim | D | L | K_mem | Use case |
|--------|----------|---|---|-------|----------|
| Tiny | 64 | 128 | 4 | 32 | Unit tests |
| Small | 128 | 256 | 8 | 64 | BPC benchmarks |
| Medium | 256 | 512 | 16 | 128 | CyphaLM integration |
| Large | 512 | 1024 | 32 | 256 | Full LM |

Start Tiny for parity/stability; Small for D17 beat-bigram runs.

---

## What does not change

- External CyphaDIF REST / `.cypha` v3 contract ([`PORT_CONTRACT.md`](../../port/PORT_CONTRACT.md))  
- Existing bench domains and profiles  
- Parity fixtures — extend, do not replace  

---

## Verification checklist (summary)

**Option A:** Ψ row 0 = WorldPrior; all parity CTests green; batched LLR/update fixtures; BPC unchanged pre/post refactor.

**Option B:** Forward shapes; α + forgetting metrics; BPC < hybrid @ Small; Izaac read/write; world-model component spawn/stabilise.

**Integration:** CyphaDIF LLR → RPSM level 0; full `CyphaLM_RPSM` generates coherent samples @ T=1.0.
