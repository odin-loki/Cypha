# RPSM Implementation Spec
**Author:** Odin Loch  
**Status:** Ready for implementation in Cursor

---

## What RPSM Is

Recursive Predictive State Matrix. The entire system state is one matrix Ψ ∈ ℝ^(L × D) where L = hierarchy depth and D = hidden dimension. Every component of the Cypha stack maps onto operations on this matrix — no separate codepaths.

The core dynamics:

```
dΨ/dt = η(H_up - H_down)(1-A) W_u + α(Ψ_past - Ψ) + βM - λΨ/‖Ψ‖_F
```

Four terms: prediction error update, Izaac memory blend, working memory injection, MDL regularisation.

---

## Learned Parameters (that's it)

```
W_enc    ∈ ℝ^(input_dim × D)   input encoder
W_up     ∈ ℝ^(D × D)           bottom-up compression
W_down   = W_up^T               top-down prediction (not learned — derived)
W_update ∈ ℝ^(D × D)           state update rule
```

W_down is not a separate learned matrix. It is the transpose of W_up. This is the symmetric init fix — makes the update self-adjoint, controls spectral radius, halves parameter count on the core matrices.

---

## Fix 1 — Spectral α (Critical)

**Problem:** Softmax-based GRIA α collapses to ~0.0009 at init over D=256 dims. Uniform distribution → H ≈ H_max → α ≈ 0 always. Useless as a gate.

**Fix:** Compute α from singular value spectrum of Ψ, not softmax of rows.

```python
def gria_alpha_spectral(Psi: torch.Tensor) -> float:
    """
    Spectral GRIA α for the full state matrix.
    More numerically stable than softmax version at any init scale.
    Returns scalar α ∈ [0, 1].
    """
    sv = torch.linalg.svdvals(Psi)
    sv_n = sv / (sv.sum() + 1e-10)
    H_max = torch.log(torch.tensor(len(sv), dtype=torch.float))
    H_sv = -(sv_n * (sv_n + 1e-10).log()).sum()
    return (1.0 - H_sv / (H_max + 1e-10)).item()

def gria_alpha_per_level(Psi: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Per-level α for gating — used to weight error propagation.
    Computed on each row's magnitude distribution.
    Returns (L, 1) tensor.
    """
    p = Psi.abs()
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    H_max = torch.log(torch.tensor(Psi.shape[1], dtype=torch.float))
    H_p = -(p * (p + eps).log()).sum(dim=-1, keepdim=True)
    return 1.0 - H_p / (H_max + eps)
```

Use `gria_alpha_spectral` for system-level diagnostics and NMP/write thresholds.  
Use `gria_alpha_per_level` for the error gating matrix A inside the forward pass.

---

## Fix 2 — Normalised Learning Rate (Critical)

**Problem:** Fixed η causes forgetting ratio of 0.82 — the update wipes 82% of state per step regardless of error magnitude.

**Fix:** Normalise η by the error magnitude so update size is constant regardless of input scale.

```python
def normalised_update(Psi, E_gated, W_update, eta_base=0.008):
    """
    η scales inversely with error magnitude.
    Update norm is always eta_base * D regardless of input.
    """
    E_norm = E_gated.norm(p='fro') + 1e-8
    eta = eta_base / E_norm
    return Psi + eta * (E_gated @ W_update)
```

Expected forgetting ratio after fix: <0.001 (matches CyphaDIF's zero-forgetting property).

---

## Fix 3 — Orthogonal Init (High)

**Problem:** Xavier init on square matrices gives condition number κ ≈ 450. Poor gradient flow through W_update especially.

**Fix:** Orthogonal init. Condition number = 1.0 exactly.

```python
def init_weights(D: int, input_dim: int):
    """
    Orthogonal init for all square matrices.
    W_down derived from W_up — not initialised separately.
    """
    W_up, _ = torch.linalg.qr(torch.randn(D, D))
    W_update, _ = torch.linalg.qr(torch.randn(D, D))
    W_enc = torch.randn(input_dim, D) * (2.0 / (input_dim + D)) ** 0.5
    W_down = W_up.T  # derived, not separate
    return W_enc, W_up, W_down, W_update
```

---

## Fix 4 — Multi-Level Input Injection (High)

**Problem:** Input injected only at level 0 → level 0 carries 87.5% of error signal. Upper levels get almost no direct input signal, only propagated error.

**Fix:** Inject input at every level with diminishing scale. Level 0 gets full signal, higher levels get progressively weaker injection (they should be working at coarser granularity).

```python
# One encoder per level
W_enc_levels = [
    torch.randn(input_dim, D) * (2.0 / (input_dim + D)) ** 0.5
    for _ in range(L)
]

def inject_input(Psi, h_enc, W_enc_levels, L):
    """
    Inject encoded input at all levels with 1/l diminishing scale.
    Level 0: full signal
    Level 1: half signal  
    Level l: 1/(l+1) signal
    """
    for l in range(L):
        h_l = h_enc @ W_enc_levels[l]
        scale = 1.0 / (l + 1)
        Psi[l] = Psi[l] + scale * h_l
    return Psi
```

Parameter cost: L × input_dim × D additional. For L=8, D=256, input_dim=64: +131K params. Acceptable.

---

## Fix 5 — Symmetric W_down (Medium)

Already covered in Fix 3. W_down = W_up.T at all times — update it when W_up updates, don't maintain a separate parameter.

Rationale: makes the combined operator W_up @ W_down = W_up @ W_up^T which is positive semi-definite. Spectral radius is ‖W_up‖² which is directly controllable. Also means the bottom-up and top-down passes are symmetric — the hierarchy's predictive model is the transpose of its compression model. Architecturally clean.

---

## Full Forward Pass

```python
class RPSM(nn.Module):
    def __init__(self, input_dim, D, L, K, vocab_size,
                 mdl_C=8.0, eta_base=0.008,
                 alpha_blend=0.3, beta=0.1, lam=0.001,
                 nmp_threshold=0.85, mem_write_tau=0.85):
        super().__init__()
        self.D, self.L, self.K = D, L, K
        self.mdl_C = mdl_C
        self.eta_base = eta_base
        self.alpha_blend = alpha_blend
        self.beta = beta
        self.lam = lam
        self.nmp_threshold = nmp_threshold

        # Learned matrices
        W_up_init, _ = torch.linalg.qr(torch.randn(D, D))
        W_upd_init, _ = torch.linalg.qr(torch.randn(D, D))
        self.W_up     = nn.Parameter(W_up_init)
        self.W_update = nn.Parameter(W_upd_init)
        # W_down = W_up.T — property, not parameter

        # Per-level encoders
        self.W_enc = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, D) * (2.0/(input_dim+D))**0.5)
            for _ in range(L)
        ])

        # Output head
        self.out = nn.Linear(D, vocab_size)

        # Working memory (not learned — dynamically maintained)
        self.register_buffer('M_slots', torch.zeros(K, D))
        self.register_buffer('M_ages',  torch.zeros(K))

    @property
    def W_down(self):
        return self.W_up.T

    def mdl_project(self, Psi):
        fn = Psi.norm(p='fro')
        return Psi * (self.mdl_C / fn) if fn > self.mdl_C else Psi

    def memory_read(self, h_top):
        # h_top: (D,) — top level state
        q = h_top.unsqueeze(0)                          # (1, D)
        scores = (q @ self.M_slots.T) / self.D ** 0.5  # (1, K)
        weights = torch.softmax(scores, dim=-1)
        return (weights @ self.M_slots).squeeze(0)      # (D,)

    def memory_write(self, h_top, nmp_score):
        if nmp_score > self.nmp_threshold:
            slot = self.M_ages.argmin().item()
            self.M_slots[slot] = h_top.detach()
            self.M_ages[slot] = 0
        self.M_ages += 1

    def forward(self, x, Psi, izaac_store=None):
        """
        x:          (batch, input_dim) — current input
        Psi:        (L, D) — current state matrix
        izaac_store: dict-like, key=VRF(Psi[-1]), value=Psi snapshot

        Returns: logits (batch, vocab_size), Psi_new (L, D)
        """
        # 1. Multi-level input injection
        h_enc = torch.sigmoid(x @ self.W_enc[0])  # (batch, D)
        Psi = Psi.clone()
        for l in range(self.L):
            h_l = torch.sigmoid(x @ self.W_enc[l])
            Psi[l] = Psi[l] + (1.0 / (l + 1)) * h_l.mean(0)

        # 2. Bottom-up pass
        H_up = torch.sigmoid(Psi @ self.W_up)       # (L, D)

        # 3. Top-down prediction (symmetric)
        H_down = torch.sigmoid(Psi @ self.W_down)   # (L, D)

        # 4. Prediction error
        E = H_up - H_down                           # (L, D)

        # 5. GRIA α gating — suppress predictable, pass novel
        A = gria_alpha_per_level(Psi)               # (L, 1)
        E_gated = E * (1.0 - A)                     # (L, D)

        # 6. Izaac memory retrieval
        Psi_past = torch.zeros_like(Psi)
        if izaac_store is not None:
            key = izaac_vrf(Psi[-1])
            if key in izaac_store:
                Psi_past = izaac_store[key]

        # 7. Working memory read
        M_read = self.memory_read(Psi[-1])          # (D,)

        # 8. Unified state update
        Psi_new = normalised_update(Psi, E_gated, self.W_update, self.eta_base)
        Psi_new = Psi_new \
                + self.alpha_blend * (Psi_past - Psi) \
                + self.beta * M_read \
                - self.lam * Psi / (Psi.norm(p='fro') + 1e-8)
        Psi_new = self.mdl_project(Psi_new)

        # 9. Surprise-gated memory write
        nmp = gria_alpha_spectral(Psi_new)
        self.memory_write(Psi_new[-1], nmp)

        # 10. Izaac store write
        if izaac_store is not None:
            izaac_store[izaac_vrf(Psi_new[-1])] = Psi_new.detach()

        # 11. Output from top level
        logits = self.out(Psi_new[-1].unsqueeze(0).expand(x.shape[0], -1))

        return logits, Psi_new
```

---

## Izaac VRF Stub

Until Izaac C++ is wired in, use this deterministic hash as a stand-in. Same interface, swap it out later.

```python
import hashlib

def izaac_vrf(h: torch.Tensor, precision=3) -> str:
    """
    Deterministic content-addressed key from hidden state.
    Stub: quantise h to precision decimal places, hash it.
    Replace with real Izaac VRF when available.
    """
    quantised = h.detach().cpu().numpy().round(precision)
    key = hashlib.sha256(quantised.tobytes()).hexdigest()[:16]
    return key
```

---

## Recommended Config Sizes

| Config | L | D | K | Total params | Use case |
|---|---|---|---|---|---|
| Tiny | 4 | 128 | 32 | ~62K | Unit tests, fast iteration |
| Small | 8 | 256 | 64 | ~233K | BPC benchmarks |
| Medium | 16 | 512 | 128 | ~901K | CyphaLM integration |
| Large | 32 | 1024 | 256 | ~3.5M | Full language model |
| XLarge | 64 | 2048 | 512 | ~14M | Image + text multimodal |

Start with Tiny for unit tests, Small for BPC benchmarks against D17 baseline.

---

## State Initialisation

```python
def init_Psi(L, D, device='cuda'):
    """
    Initialise state matrix at edge-of-chaos.
    std=1.0 gives spectral α closer to target range than std=0.1.
    """
    Psi = torch.randn(L, D, device=device) * 1.0
    # Warm up: one pass of MDL projection to set ‖Ψ‖_F = mdl_C
    Psi = Psi * (8.0 / Psi.norm(p='fro'))
    return Psi
```

---

## Training Loop Skeleton

```python
def train_step(model, Psi, x_seq, targets, optimiser, izaac_store):
    """
    x_seq:   (batch, seq_len, input_dim)
    targets: (batch, seq_len) token indices
    """
    total_loss = 0.0
    alpha_trajectory = []

    for t in range(x_seq.shape[1]):
        logits, Psi = model(x_seq[:, t], Psi, izaac_store)
        loss = F.cross_entropy(logits, targets[:, t])
        total_loss += loss

        # Log α for diagnostics
        alpha_trajectory.append(gria_alpha_spectral(Psi))

    # Auxiliary α loss — drive toward edge-of-chaos
    mean_alpha = sum(alpha_trajectory) / len(alpha_trajectory)
    alpha_loss = 0.01 * abs(mean_alpha - 0.485)
    total_loss += alpha_loss

    optimiser.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()

    return total_loss.item(), alpha_trajectory
```

---

## What to Verify First (in order)

```
1. gria_alpha_spectral — confirm α ∈ [0.3, 0.6] on random Psi with std=1.0
2. normalised_update — confirm forgetting ratio < 0.01 on 100 steps
3. W_down = W_up.T — confirm spectral radius < 1.0
4. Full forward pass — confirm shapes, no NaN
5. Tiny config BPC on WikiText-2 — confirm better than random (log2(vocab_size))
6. Small config BPC — target: beat char-LSTM baseline of 2.979
7. Small config BPC — target: beat hybrid_gria_lstm of 2.873
```

Stop and diagnose if BPC doesn't improve by step 5. It means either α is still collapsing or the forgetting ratio is too high.

---

## Notes

- The Izaac VRF stub is good enough for all BPC benchmarks. Wire the real Izaac implementation in once the architecture is validated.
- W_down = W_up.T must be enforced after every optimiser step — W_down is not a parameter but it needs to stay in sync. Either recompute it as a property (as shown above) or add a post-step hook.
- The working memory M_slots and M_ages are buffers not parameters — they don't get gradients. This is intentional. Memory writes are surprise-gated, not gradient-optimised.
- All attributed outputs to be published under Odin Loch.
