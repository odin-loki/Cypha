# RPSM implementation spec

**Author:** Odin Loch  
**Status:** Planned — implements Option B in [RPSM_COMBINED_SPEC.md](RPSM_COMBINED_SPEC.md)

---

## What RPSM is

**Recursive Predictive State Matrix.** Entire system state is one matrix Ψ ∈ ℝ^(L × D). Every Cypha component maps to operations on Ψ — no parallel codepaths.

Core dynamics:

```
dΨ/dt = η(H_up − H_down)(1−A) W_u + α(Ψ_past − Ψ) + βM − λΨ/‖Ψ‖_F
```

Four terms: prediction-error update, Izaac memory blend, working-memory injection, MDL regularisation.

---

## Learned parameters

| Matrix | Shape | Role |
|--------|-------|------|
| W_enc[l] | input_dim × D | Per-level input encoder (Fix 4) |
| W_up | D × D | Bottom-up compression |
| W_down | W_up^T | Top-down prediction (**derived**, not learned) |
| W_update | D × D | State update rule |

W_down = W_up^T enforces symmetric hierarchy (Fix 5); halves core matrix params and controls spectral radius.

---

## Five critical fixes

### Fix 1 — Spectral α (critical)

Softmax GRIA α collapses to ~0 at init over D=256. **Use singular-value spectrum of Ψ** instead:

- `gria_alpha_spectral(Psi)` — system-level diagnostics, NMP write threshold  
- `gria_alpha_per_level(Psi)` — per-row error gating matrix A  

Target α ∈ [0.3, 0.6] at edge-of-chaos init (std=1.0).

### Fix 2 — Normalised learning rate (critical)

Fixed η caused **82% forgetting ratio**. Scale η inversely with ‖E_gated‖_F:

```python
eta = eta_base / (E_gated.norm(p='fro') + 1e-8)
Psi_new = Psi + eta * (E_gated @ W_update)
```

Expected forgetting ratio **< 0.01** after fix.

### Fix 3 — Orthogonal init (high)

Xavier on square matrices → κ ≈ 450. **QR orthogonal init** for W_up and W_update → κ = 1.0.

### Fix 4 — Multi-level input injection (high)

Input at level 0 only → 87.5% error at bottom. Inject at every level with scale **1/(l+1)**.

Cost: L × input_dim × D extra params (~131K for L=8, D=256, input_dim=64).

### Fix 5 — Symmetric W_down (medium)

Recompute W_down as property `W_up.T` after every optimiser step — never a separate parameter.

---

## Forward pass (11 steps)

1. Multi-level input injection (diminishing 1/(l+1))  
2. Bottom-up: `H_up = σ(Ψ @ W_up)`  
3. Top-down: `H_down = σ(Ψ @ W_down)`  
4. Error: `E = H_up − H_down`  
5. GRIA gate: `E_gated = E * (1 − A)`  
6. Izaac retrieve: `Psi_past` from VRF-keyed store  
7. Working memory read: soft attention over M_slots  
8. Normalised update + Izaac blend + M_read + MDL decay  
9. MDL Frobenius projection (C=8.0)  
10. Surprise-gated memory write (nmp > 0.85)  
11. Izaac store write; output logits from top row  

---

## Izaac VRF stub

Until native Izaac C++ is wired, use deterministic SHA-256 of quantised hidden state (precision=3). Same interface; swap for real VRF later.

---

## Config sizes

| Config | L | D | K | Params | Use case |
|--------|---|---|---|--------|----------|
| Tiny | 4 | 128 | 32 | ~62K | Unit tests |
| Small | 8 | 256 | 64 | ~233K | BPC vs D17 |
| Medium | 16 | 512 | 128 | ~901K | CyphaLM |
| Large | 32 | 1024 | 256 | ~3.5M | Full LM |

Init Ψ at edge-of-chaos: `randn(L,D)*1.0`, then scale to ‖Ψ‖_F = mdl_C.

---

## Verify first (ordered)

1. `gria_alpha_spectral` ∈ [0.3, 0.6] on random Ψ  
2. Forgetting ratio < 0.01 over 100 steps  
3. Spectral radius of W_up @ W_down < 1.0  
4. Forward pass — shapes, no NaN  
5. Tiny BPC > random baseline  
6. Small BPC < char-LSTM **2.979**  
7. Small BPC < hybrid_gria_lstm **2.873** ← stop/go gate  

If step 5 fails, diagnose α collapse or forgetting before scaling up.

---

## Training notes

- Auxiliary α loss: `0.01 * |mean_alpha − 0.485|` drives edge-of-chaos  
- M_slots / M_ages are buffers (no gradients) — surprise-gated writes only  
- Grad clip norm 1.0  

Native port: new module under `native/src/cyphalm/` or `native/src/rpsm/` with CTest smoke before bench integration.
