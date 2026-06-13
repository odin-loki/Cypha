# CyphaDIF → RPSM: Combined Refactor and Integration Spec
**Author:** Odin Loch  
**Status:** Ready for implementation in Cursor  
**Target:** Beat hybrid_gria_lstm D17 BPC 2.873

---

## Overview

Two parallel workstreams that compose into a single unified architecture:

- **Option A:** Refactor CyphaDIF internals into matrix form. Same behaviour, cleaner maths, faster CUDA, easier to extend. Validated by existing parity test suite.
- **Option B:** RPSM sequence layer in CyphaLM. CyphaDIF (post-A) plugs in at level 0 as the token router. Hierarchy handles long context, CyphaDIF handles token decisions.

Neither replaces the other. A makes CyphaDIF faster and cleaner. B builds new capability on top of A. The nonlinear boundary fix (Nyström) goes into A and automatically benefits B.

---

## Execution Order

```
Step 1 — Option A: Matrix refactor of CyphaDIF
Step 2 — Nonlinear boundary fix into A (Nyström kernel LLR)
Step 3 — Option B: RPSM sequence layer in CyphaLM
Step 4 — Wire Izaac episodic store + working memory into B
Step 5 — Full benchmark against D17 baseline
```

---

# OPTION A: CyphaDIF Matrix Refactor

## What Changes

CyphaDIF already implements RPSM maths — just not in matrix form. The refactor maps existing components onto a single state matrix Ψ without changing behaviour.

### Component Mapping

| Current CyphaDIF | Post-Refactor |
|---|---|
| `WorldPrior θ₀` (μ, v vectors) | Row 0 of Ψ_mu, Row 0 of Ψ_var |
| `ClassDifferential Δk` per class | Rows 1..K of Ψ_mu, Ψ_var |
| `DIFMemory` LLR computation | Batched matmul over Ψ rows |
| `EncoderProjection W_enc` | W_enc matrix, unchanged |
| `TieredContextBuffer` short/mid/long | Rows of a separate context matrix C |
| `NIGField` posterior | Ψ_var matrix (already there) |
| `PriorityReplayBuffer` | M_slots working memory matrix |
| `MDL decay ‖Δk‖_F ≤ C` | Frobenius norm projection on Ψ rows |
| `Fisher-Rao contrastive update` | Batched natural gradient on Ψ |

### New State Representation

```python
class CyphaDIFState:
    """
    Unified matrix state replacing separate WorldPrior + ClassDifferential objects.
    
    Ψ_mu  ∈ ℝ^((1 + n_classes) × feat_dim)
      Row 0:     WorldPrior μ (θ₀)
      Rows 1..K: ClassDifferential μ (Δk) per class
    
    Ψ_var ∈ ℝ^((1 + n_classes) × feat_dim)
      Same structure for variances.
    """
    def __init__(self, n_classes: int, feat_dim: int):
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        # Row 0 = WorldPrior, rows 1..K = ClassDifferentials
        self.Psi_mu  = torch.zeros(1 + n_classes, feat_dim)
        self.Psi_var = torch.ones(1 + n_classes, feat_dim) * 0.1
        # Counts per row for Welford update
        self.counts  = torch.zeros(1 + n_classes)

    @property
    def world_prior_mu(self):
        return self.Psi_mu[0]

    @property
    def world_prior_var(self):
        return self.Psi_var[0]

    def class_mu(self, k: int):
        return self.Psi_mu[0] + self.Psi_mu[k + 1]   # θ₀ ⊕ Δk

    def class_var(self, k: int):
        return self.Psi_var[0] + self.Psi_var[k + 1]
```

### Batched LLR (replaces per-class loop)

Current CyphaDIF computes LLR for each class in a Python loop. Post-refactor this is one batched matmul.

```python
def batched_llr(h: torch.Tensor, Psi_mu: torch.Tensor,
                Psi_var: torch.Tensor) -> torch.Tensor:
    """
    Compute LLR for all classes simultaneously.
    
    h:        (batch, feat_dim)
    Psi_mu:   (1 + n_classes, feat_dim)  — row 0 = world prior
    Psi_var:  (1 + n_classes, feat_dim)
    
    Returns:  (batch, n_classes) LLR scores
    """
    # Effective class parameters: θ₀ ⊕ Δk for all k simultaneously
    mu_eff  = Psi_mu[0:1]  + Psi_mu[1:]    # (n_classes, feat_dim)
    var_eff = Psi_var[0:1] + Psi_var[1:]   # (n_classes, feat_dim)
    var_eff = var_eff.clamp(min=1e-8)

    # Batched diagonal Gaussian log-likelihood
    # h: (batch, feat_dim) → (batch, 1, feat_dim)
    # mu_eff: (n_classes, feat_dim) → (1, n_classes, feat_dim)
    h_exp    = h.unsqueeze(1)
    mu_exp   = mu_eff.unsqueeze(0)
    var_exp  = var_eff.unsqueeze(0)

    llr = -0.5 * ((h_exp - mu_exp).pow(2) / var_exp + var_exp.log()).sum(-1)
    return llr  # (batch, n_classes)
```

### Batched State Update (replaces per-class update loop)

```python
def batched_update(state: CyphaDIFState, h: torch.Tensor,
                   labels: torch.Tensor, lr_world=0.008,
                   lr_delta=0.05, mdl_lambda=0.001, mdl_C=8.0):
    """
    Update WorldPrior and all matched ClassDifferentials in one pass.
    
    h:      (batch, feat_dim)
    labels: (batch,) integer class indices
    """
    batch_size = h.shape[0]

    # WorldPrior update — Welford EMA on row 0
    state.counts[0] += batch_size
    alpha_world = lr_world
    state.Psi_mu[0]  += alpha_world * (h.mean(0) - state.Psi_mu[0])
    state.Psi_var[0] += alpha_world * (
        (h - state.Psi_mu[0]).pow(2).mean(0) - state.Psi_var[0]
    )

    # ClassDifferential updates — scatter to matched rows
    for k in labels.unique():
        mask = (labels == k)
        h_k = h[mask]
        row = k.item() + 1  # offset by 1 (row 0 = world prior)

        state.counts[row] += mask.sum()
        state.Psi_mu[row]  += lr_delta * (h_k.mean(0) - state.Psi_mu[row])
        state.Psi_var[row] += lr_delta * (
            (h_k - state.Psi_mu[row]).pow(2).mean(0) - state.Psi_var[row]
        )

        # MDL norm constraint on Δk rows (not on world prior row 0)
        delta_norm = state.Psi_mu[row].norm()
        if delta_norm > mdl_C:
            state.Psi_mu[row] *= mdl_C / delta_norm

    # MDL decay across all Δk rows
    state.Psi_mu[1:] *= (1.0 - mdl_lambda)
```

### TieredContext as Matrix

```python
class TieredContextMatrix:
    """
    Replaces TieredContextBuffer.
    C ∈ ℝ^(3 × feat_dim) — three rows: short, mid, long.
    """
    def __init__(self, feat_dim: int, short_win=32, mid_alpha=0.98):
        self.C = torch.zeros(3, feat_dim)
        self.short_win = short_win
        self.mid_alpha = mid_alpha
        self.short_buffer = []

    def update(self, h: torch.Tensor):
        # Row 0: short — rolling window mean
        self.short_buffer.append(h.detach())
        if len(self.short_buffer) > self.short_win:
            self.short_buffer.pop(0)
        self.C[0] = torch.stack(self.short_buffer).mean(0)

        # Row 1: mid — EMA
        self.C[1] = self.mid_alpha * self.C[1] + (1 - self.mid_alpha) * h.detach()

        # Row 2: long — Welford (never resets)
        n = len(self.short_buffer)
        self.C[2] += (h.detach() - self.C[2]) / (n + 1)

    def context_prior(self, confidence: torch.Tensor) -> torch.Tensor:
        """
        Confidence-weighted blend of three tiers.
        confidence: (3,) tensor of per-tier confidence scores.
        Returns: (feat_dim,) blended context vector.
        """
        w = torch.softmax(confidence, dim=0)
        return (w.unsqueeze(1) * self.C).sum(0)
```

### Parity Validation

After refactor, run the full existing parity suite. All fixtures must pass:

```
cypha_parity
memory_train_parity
quantile_dif_train_parity
mke_train_step_parity
regression_m4_parity
```

Add two new parity fixtures:
- `batched_llr_parity` — batched matmul LLR vs original per-class loop, must be numerically identical
- `batched_update_parity` — batched update vs original per-class update, must be numerically identical

Expected outcome: same BPC, faster wall time (batched matmul vs Python loop).

### C++ Native Implications

The matrix refactor simplifies the native port significantly. Current native core has per-class C++ objects. Post-refactor it becomes:

```cpp
// Entire CyphaDIF state
struct CyphaDIFState {
    Eigen::MatrixXf Psi_mu;   // (1 + n_classes, feat_dim)
    Eigen::MatrixXf Psi_var;  // (1 + n_classes, feat_dim)
    Eigen::VectorXf counts;   // (1 + n_classes)
};

// LLR: one GEMM call
Eigen::MatrixXf llr = batched_llr(h, state.Psi_mu, state.Psi_var);
```

Replace all per-class loops in the native core with Eigen GEMM. Add to parity matrix as `native_batched_llr_parity`.

---

# OPTION B: RPSM Sequence Layer

## Architecture

CyphaDIF (post-A) plugs into level 0 of RPSM as the token router. The hierarchy handles sequence-level long context. CyphaDIF handles token-level decisions.

```
Input tokens
     ↓
Level 0: CyphaDIF token router
     ↓ LLR scores + context prior
Level 1: RPSM row — word/chunk level
     ↓ prediction error
Level 2: RPSM row — sentence level
     ↓ prediction error
Level N: RPSM row — document level
     ↓
Global state:
  Working memory M_slots (K × D)
  Izaac episodic store (VRF-keyed)
  Gaussian mixture world model
```

## The State Matrix

```
Ψ ∈ ℝ^(L × D)

Row 0:    CyphaDIF Ψ_mu top row (world prior + class offsets compressed to D dims)
Row 1:    chunk/word level state
Row 2:    sentence level state
...
Row L-1:  document/session level state
```

## RPSM Core (builds on RPSM implementation spec)

```python
class RPSMSequenceLayer(nn.Module):
    """
    Wraps RPSM for use as CyphaLM sequence layer.
    CyphaDIF handles level 0 token routing.
    RPSM rows 1..L handle hierarchy above token level.
    """
    def __init__(self, cypha_dif: CyphaDIF, D: int, L: int, K: int,
                 vocab_size: int):
        super().__init__()
        self.cypha = cypha_dif      # level 0 — token router
        self.D = D
        self.L = L                  # hierarchy depth above token level

        # Project CyphaDIF output to RPSM dim
        self.cypha_proj = nn.Linear(cypha_dif.feat_dim, D)

        # RPSM matrices (orthogonal init)
        W_up_init, _ = torch.linalg.qr(torch.randn(D, D))
        W_upd_init, _ = torch.linalg.qr(torch.randn(D, D))
        self.W_up     = nn.Parameter(W_up_init)
        self.W_update = nn.Parameter(W_upd_init)
        # W_down = W_up.T (property)

        # Output head
        self.out = nn.Linear(D, vocab_size)

        # Working memory
        self.register_buffer('M_slots', torch.zeros(K, D))
        self.register_buffer('M_ages',  torch.zeros(K))

        # Izaac episodic store (CPU dict — swap for real Izaac later)
        self.izaac_store = {}

    @property
    def W_down(self):
        return self.W_up.T

    def forward(self, token_ids: torch.Tensor, Psi: torch.Tensor):
        """
        token_ids: (batch,) current token
        Psi:       (L, D) current RPSM state
        Returns:   logits (batch, vocab_size), Psi_new (L, D)
        """
        # ── Level 0: CyphaDIF token routing ──────────────────────
        # Get LLR scores from CyphaDIF
        h_cypha = self.cypha.encode(token_ids)       # (batch, feat_dim)
        llr     = self.cypha.llr(h_cypha)            # (batch, n_classes)
        ctx     = self.cypha.context_prior()         # (feat_dim,)

        # Project to RPSM dim and inject into row 0
        h0 = self.cypha_proj(h_cypha)               # (batch, D)
        Psi = Psi.clone()
        Psi[0] = Psi[0] + h0.mean(0)

        # ── Levels 1..L: RPSM hierarchy ──────────────────────────
        # Multi-level input injection (diminishing scale)
        for l in range(1, self.L):
            Psi[l] = Psi[l] + (1.0 / l) * h0.mean(0)

        # Bottom-up pass
        H_up = torch.sigmoid(Psi @ self.W_up)        # (L, D)

        # Top-down prediction (symmetric)
        H_down = torch.sigmoid(Psi @ self.W_down)    # (L, D)

        # Prediction error
        E = H_up - H_down

        # GRIA α gating
        A = gria_alpha_per_level(Psi)
        E_gated = E * (1.0 - A)

        # Izaac retrieval
        key = izaac_vrf(Psi[-1])
        Psi_past = self.izaac_store.get(key, torch.zeros_like(Psi))

        # Working memory read
        q = Psi[-1].unsqueeze(0)
        attn = torch.softmax(q @ self.M_slots.T / self.D ** 0.5, dim=-1)
        M_read = (attn @ self.M_slots).squeeze(0)

        # Unified state update
        E_norm = E_gated.norm(p='fro') + 1e-8
        eta = 0.008 / E_norm
        Psi_new = Psi \
                + eta * (E_gated @ self.W_update) \
                + 0.3 * (Psi_past - Psi) \
                + 0.1 * M_read \
                - 0.001 * Psi / (Psi.norm(p='fro') + 1e-8)

        # MDL projection
        fn = Psi_new.norm(p='fro')
        if fn > 8.0:
            Psi_new = Psi_new * (8.0 / fn)

        # Surprise-gated memory write
        nmp = gria_alpha_spectral(Psi_new)
        if nmp > 0.85:
            slot = self.M_ages.argmin().item()
            self.M_slots[slot] = Psi_new[-1].detach()
            self.M_ages[slot] = 0
        self.M_ages += 1

        # Izaac store write
        self.izaac_store[izaac_vrf(Psi_new[-1])] = Psi_new.detach()

        # Output — combine RPSM top level with CyphaDIF LLR
        rpsm_logits  = self.out(Psi_new[-1].unsqueeze(0).expand(h0.shape[0], -1))

        return rpsm_logits, Psi_new
```

## Global Memory: Three Layers

### Layer 1 — Izaac Episodic Store (already in RPSM above)

```python
# Stub — replace with real Izaac VRF
import hashlib

def izaac_vrf(h: torch.Tensor, precision=3) -> str:
    q = h.detach().cpu().numpy().round(precision)
    return hashlib.sha256(q.tobytes()).hexdigest()[:16]

# Store is a plain dict for now
# Real implementation: Izaac VRF → algebraic cluster key → O(1) lookup
izaac_store: dict[str, torch.Tensor] = {}
```

Write policy: every step (top-level state snapshot).  
Read policy: on every forward pass — blended into Psi_new at weight 0.3.  
Eviction: none — write-only. Prune by age if memory exceeds limit.

### Layer 2 — Working Memory M_slots (already in RPSM above)

```python
# K slots × D dims
# Write: surprise-gated (nmp > 0.85)
# Read: soft attention from top-level state
# Eviction: LRU by M_ages
```

### Layer 3 — Gaussian Mixture World Model

```python
class GaussianMixtureWorldModel:
    """
    Slow-consolidating semantic memory.
    Identifies which domain/context type the system is currently in.
    Replaces single flat WorldPrior with K mixture components.
    """
    def __init__(self, D: int, n_components: int = 8):
        self.D = D
        self.K = n_components
        # Component parameters
        self.mu  = torch.randn(n_components, D) * 0.1
        self.var = torch.ones(n_components, D) * 0.1
        self.pi  = torch.ones(n_components) / n_components  # mixing weights
        self.counts = torch.zeros(n_components)

    def assign(self, h: torch.Tensor) -> int:
        """Soft-assign h to nearest component. Returns argmax component."""
        dists = ((h.unsqueeze(0) - self.mu).pow(2) / self.var.clamp(1e-8)).sum(-1)
        return dists.argmin().item()

    def update(self, h: torch.Tensor, alpha_signal: float):
        """
        Consolidate h into world model.
        Low α (novel) → may spawn new component.
        High α (familiar) → reinforce existing component.
        """
        k = self.assign(h)
        lr = 0.001  # slow consolidation
        self.counts[k] += 1
        self.mu[k]  += lr * (h.detach() - self.mu[k])
        self.var[k] += lr * ((h.detach() - self.mu[k]).pow(2) - self.var[k])
        self.pi = self.counts / self.counts.sum()

        # Spawn new component if α is low and no component is close
        if alpha_signal < 0.3:
            min_dist = ((h.unsqueeze(0) - self.mu).pow(2)).sum(-1).min()
            if min_dist > 5.0 and self.K < 32:  # max 32 components
                self.mu  = torch.cat([self.mu,  h.detach().unsqueeze(0)])
                self.var = torch.cat([self.var, torch.ones(1, self.D) * 0.1])
                self.counts = torch.cat([self.counts, torch.ones(1)])
                self.K += 1

    def prior(self, h: torch.Tensor) -> torch.Tensor:
        """Return component prior for current state — inject into RPSM level L-1."""
        k = self.assign(h)
        return self.mu[k]
```

Wire into RPSM: inject `world_model.prior(Psi[-1])` into `Psi[-1]` at each step with small weight (0.05). This gives the top level a slow-moving semantic anchor.

---

## Full CyphaLM Integration

```python
class CyphaLM_RPSM(nn.Module):
    """
    Full language model.
    CyphaDIF handles token-level routing (level 0).
    RPSM handles sequence-level hierarchy (levels 1..L).
    Global memory: Izaac store + working memory + GMM world model.
    """
    def __init__(self, vocab_size: int, feat_dim: int, D: int,
                 L: int, K_mem: int, n_classes: int):
        super().__init__()
        # Option A: matrix-refactored CyphaDIF
        self.cypha = CyphaDIF(n_classes=n_classes, feat_dim=feat_dim)

        # Option B: RPSM sequence layer
        self.rpsm = RPSMSequenceLayer(
            cypha_dif=self.cypha,
            D=D, L=L, K=K_mem,
            vocab_size=vocab_size
        )

        # Global memory layer 3
        self.world_model = GaussianMixtureWorldModel(D=D, n_components=8)

    def forward(self, token_seq: torch.Tensor, Psi: torch.Tensor):
        """
        token_seq: (batch, seq_len)
        Psi:       (L, D) initial state
        Returns:   logits (batch, seq_len, vocab_size)
        """
        logits_all = []

        for t in range(token_seq.shape[1]):
            logits, Psi = self.rpsm(token_seq[:, t], Psi)
            logits_all.append(logits)

            # World model consolidation (async — every 100 steps)
            if t % 100 == 0:
                alpha = gria_alpha_spectral(Psi)
                self.world_model.update(Psi[-1], alpha)
                world_prior = self.world_model.prior(Psi[-1])
                Psi[-1] = Psi[-1] + 0.05 * (world_prior - Psi[-1])

        return torch.stack(logits_all, dim=1), Psi

    def init_state(self, device='cuda'):
        return torch.randn(self.rpsm.L, self.rpsm.D, device=device) * 1.0
```

---

## Shared Utilities (used by both A and B)

```python
def gria_alpha_spectral(Psi: torch.Tensor) -> float:
    sv = torch.linalg.svdvals(Psi)
    sv_n = sv / (sv.sum() + 1e-10)
    H_max = torch.log(torch.tensor(float(len(sv))))
    H_sv = -(sv_n * (sv_n + 1e-10).log()).sum()
    return (1.0 - H_sv / (H_max + 1e-10)).item()

def gria_alpha_per_level(Psi: torch.Tensor, eps=1e-8) -> torch.Tensor:
    p = Psi.abs()
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    H_max = torch.log(torch.tensor(float(Psi.shape[1])))
    H_p = -(p * (p + eps).log()).sum(dim=-1, keepdim=True)
    return 1.0 - H_p / (H_max + eps)

def mdl_project(Psi: torch.Tensor, C=8.0) -> torch.Tensor:
    fn = Psi.norm(p='fro')
    return Psi * (C / fn) if fn > C else Psi

def normalised_update(Psi, E_gated, W_update, eta_base=0.008):
    E_norm = E_gated.norm(p='fro') + 1e-8
    return Psi + (eta_base / E_norm) * (E_gated @ W_update)

import hashlib
def izaac_vrf(h: torch.Tensor, precision=3) -> str:
    q = h.detach().cpu().numpy().round(precision)
    return hashlib.sha256(q.tobytes()).hexdigest()[:16]
```

---

## Recommended Configs

| Config | feat_dim | D | L | K | n_classes | Use case |
|---|---|---|---|---|---|---|
| Tiny | 64 | 128 | 4 | 32 | 100 | Unit tests |
| Small | 128 | 256 | 8 | 64 | 256 | BPC benchmarks |
| Medium | 256 | 512 | 16 | 128 | 512 | CyphaLM integration |
| Large | 512 | 1024 | 32 | 256 | 1024 | Full language model |

---

## Verification Checklist

### Option A — CyphaDIF Matrix Refactor

```
[ ] CyphaDIFState init — Ψ_mu row 0 matches original WorldPrior μ
[ ] batched_llr — numerically identical to per-class loop (parity fixture)
[ ] batched_update — numerically identical to per-class update (parity fixture)
[ ] TieredContextMatrix — output matches TieredContextBuffer (parity fixture)
[ ] All existing parity fixtures pass: cypha_parity, memory_train_parity,
    quantile_dif_train_parity, mke_train_step_parity, regression_m4_parity
[ ] BPC on WikiText-2 — same as pre-refactor (behaviour unchanged)
[ ] Wall time — faster than pre-refactor (matmul vs Python loop)
[ ] Native C++ batched LLR — passes nystrom_lkl_parity
```

### Option B — RPSM Sequence Layer

```
[ ] Forward pass shapes correct at Tiny config
[ ] No NaN after 1000 steps at Tiny config
[ ] gria_alpha_spectral ∈ [0.3, 0.6] after 20 warmup steps
[ ] Forgetting ratio < 0.01 after normalised_update fix
[ ] Working memory write triggers on high-NMP inputs
[ ] Izaac store read returns correct Psi_past
[ ] BPC < random baseline (log2(vocab_size)) at Tiny config
[ ] BPC < char-LSTM baseline 2.979 at Small config
[ ] BPC < hybrid_gria_lstm baseline 2.873 at Small config  ← main target
```

### Integration

```
[ ] CyphaDIF LLR feeds correctly into RPSM level 0
[ ] World model component count grows on novel input
[ ] World model component count stabilises on familiar input
[ ] Full CyphaLM_RPSM generates coherent sequences at temperature 1.0
```

---

## What Doesn't Change

- `Cypha.py` external API — same function signatures, same behaviour
- All existing benchmark scripts
- `cypha_studio` REST API
- `cypha_lm` D04/D17 benchmark fixtures
- Parity contract in `docs/port/PORT_CONTRACT.md` — extend it, don't replace it
- All attributed outputs published under Odin Loch
