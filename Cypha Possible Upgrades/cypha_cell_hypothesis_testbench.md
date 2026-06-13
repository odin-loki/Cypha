# Cypha Cell Hypothesis Testbench
**Author:** Odin Loch  
**Purpose:** Systematic sweep of 28 recurrent cell hypotheses to find a recurrent primitive that synergises with CyphaDIF and CyphaLM better than standard LSTM.  
**Baseline target:** `hybrid_gria_lstm` D17 BPC = 2.873

---

## Overview

The standard LSTM gate structure (sigmoid forget/input/output + tanh cell candidate) was discovered empirically in 1997 with no derivation from first principles. The hypothesis is that a cell update law derived from Cypha's own mathematics — GRIA, CyphaDIF, Izaac, NIG, EML — will outperform it on BPC and generalise better.

This document specifies all 28 hypotheses, the testbench scaffold, evaluation protocol, and logging requirements.

---

## Testbench Setup

### Dataset
- **Primary:** WikiText-2, character-level (matches D17)
- **Secondary (validation only):** Penn Treebank char-level
- Token budget: 300k train, full valid/test held out
- Vocabulary: raw char (≈100 tokens)

### Fixed hyperparameters across all variants
| Parameter | Value |
|---|---|
| Hidden dim | 256 |
| Layers | 1 |
| Sequence length | 200 |
| Batch size | 64 |
| Optimiser | Adam, lr=1e-3 |
| Epochs | 20 |
| Runs per variant | 5 |
| Metric | BPC (bits per character), median of 5 runs |

### Baselines (run first, lock these numbers)
| ID | Model | Expected BPC |
|---|---|---|
| B0 | 4-gram | ~3.478 |
| B1 | char-LSTM (standard) | ~2.979 |
| B2 | hybrid_gria_lstm (current best) | 2.873 |

### Hardware
RTX 3090, 24GB VRAM. Each variant should complete 20 epochs in under 60 minutes at these settings. Tier 3 variants (SR/AXIOM) are exempt — see their sections.

### Logging
Every run logs to `results/variant_NAME_run_N.json`:
```json
{
  "variant": "alpha_gate",
  "run": 1,
  "bpc_train": [],
  "bpc_valid": [],
  "alpha_trajectory": [],
  "nmp_trajectory": [],
  "final_bpc_valid": 0.0,
  "final_bpc_test": 0.0,
  "wall_time_s": 0
}
```
After all runs: `python summarise.py` produces `results/summary.csv` with median/std BPC per variant.

---

## Scaffold: Base Cell Interface

All variants implement this interface. Drop your cell in and the training loop is identical.

```python
import torch
import torch.nn as nn

class BaseCyphaCell(nn.Module):
    """
    All hypothesis cells implement this interface.
    Input:  x (batch, input_dim), h_prev (batch, hidden_dim), c_prev (batch, hidden_dim)
    Output: h (batch, hidden_dim), c (batch, hidden_dim)
    c_prev may be unused by stateless variants — just pass zeros and ignore.
    """
    def forward(self, x, h_prev, c_prev):
        raise NotImplementedError

class CyphaLM(nn.Module):
    """Wraps any BaseCyphaCell into a language model."""
    def __init__(self, vocab_size, hidden_dim, cell: BaseCyphaCell):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.cell = cell
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_seq, h0=None, c0=None):
        # x_seq: (batch, seq_len) token indices
        B, T = x_seq.shape
        H = self.embed.embedding_dim
        h = torch.zeros(B, H, device=x_seq.device) if h0 is None else h0
        c = torch.zeros(B, H, device=x_seq.device) if c0 is None else c0
        logits = []
        for t in range(T):
            x = self.embed(x_seq[:, t])
            h, c = self.cell(x, h, c)
            logits.append(self.out(h))
        return torch.stack(logits, dim=1)  # (B, T, vocab)
```

---

## Tier 1 — Highest Priority (run first)

These are directly derivable from existing Cypha maths. Highest prior probability of beating baseline.

---

### H01 — α-Gate Cell

**Hypothesis:** Replace the LSTM forget gate with `σ(α_t)` where α is the GRIA order parameter computed on the hidden state. Memory decay becomes semantically tied to compression quality of what is stored.

**Derivation:** GRIA defines α = 1 − H(f(X))/H(X). High α = ordered/compressible state = remember more. Low α = chaotic state = forget more. This is a principled decay law absent from standard LSTM.

**Implementation:**
```python
def gria_alpha(h: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """Approximate GRIA α from hidden state distribution."""
    p = torch.softmax(h, dim=-1)
    H_full = torch.log(torch.tensor(h.shape[-1], dtype=torch.float))
    H_p = -(p * (p + eps).log()).sum(-1, keepdim=True)
    return 1.0 - H_p / (H_full + eps)  # (batch, 1)

class AlphaGateCell(BaseCyphaCell):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h_prev, c_prev):
        xh = torch.cat([x, h_prev], dim=-1)
        f = gria_alpha(h_prev)          # forget = GRIA α
        i = torch.sigmoid(self.W_i(xh))
        o = torch.sigmoid(self.W_o(xh))
        c_hat = torch.tanh(self.W_c(xh))
        c = f * c_prev + i * c_hat
        h = o * torch.tanh(c)
        return h, c
```

**What to log:** α trajectory per step (mean across batch). Hypothesis: α should stabilise near 0.5 (edge-of-chaos) on good runs.

---

### H02 — EML Activation Cell

**Hypothesis:** Replace sigmoid and tanh throughout with the EML Sheffer operator `eml(x, y) = exp(x) − ln(y)`. This is native to GRIA/AXIOM's algebra and functionally complete.

**Derivation:** EML is a universal primitive in your symbolic grammar. If the cell's activation is expressed in the same algebra as the fitness function (GRIA α), gradient flow should align better with the objective.

**Implementation:**
```python
def eml(x, y, eps=1e-6):
    """EML Sheffer operator: exp(x) - ln(y). Clamp y to avoid log(0)."""
    return torch.exp(x.clamp(-10, 10)) - torch.log(y.clamp(eps))

def eml_gate(x):
    """Normalised EML gate: maps to (0,1) via sigmoid of eml(x, 1+exp(-x))."""
    return torch.sigmoid(eml(x, 1.0 + torch.exp(-x)))

class EMLCell(BaseCyphaCell):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h_prev, c_prev):
        xh = torch.cat([x, h_prev], dim=-1)
        f = eml_gate(self.W_f(xh))
        i = eml_gate(self.W_i(xh))
        o = eml_gate(self.W_o(xh))
        c_hat = eml(self.W_c(xh), torch.ones_like(self.W_c(xh)))
        c_hat = c_hat / (1.0 + c_hat.abs())  # bounded
        c = f * c_prev + i * c_hat
        h = o * c.tanh()
        return h, c
```

---

### H03 — CausalField Cell (SGEMV Recurrence)

**Hypothesis:** Replace LSTM entirely with the CausalField SGEMV update already in CyphaLM. This is Cypha's own SSM primitive — test whether it is a better recurrent base than LSTM gates.

**Derivation:** CausalField is a linear recurrence `h_t = A h_{t-1} + B x_t`. This is architecturally equivalent to S4/Mamba's selective state space. If GRIA's α already characterises its dynamics well, it may be the right primitive.

**Implementation:**
```python
class CausalFieldCell(BaseCyphaCell):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.A = nn.Parameter(torch.eye(hidden_dim) * 0.9)
        self.B = nn.Linear(input_dim, hidden_dim, bias=False)
        self.C = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h_prev, c_prev):
        h = self.A @ h_prev.unsqueeze(-1)
        h = h.squeeze(-1) + self.B(x)
        h = torch.tanh(self.C(h))
        return h, c_prev  # c unused
```

**Note:** Also test with selective A (input-dependent, Mamba-style): `A_t = sigmoid(W_a @ x_t) * self.A`.

---

### H04 — Pure CyphaDIF LM (No Neural Recurrence)

**Hypothesis:** Token prediction is entirely CyphaDIF `argmax_k [LLR_k + log p(k|context)]` with context = TieredContextBuffer over the character sequence. No neural recurrence at all.

**Derivation:** This is the purest test of whether CyphaDIF's information-geometric classifier can replace neural sequence modelling entirely at the character level. If it beats B1 (char-LSTM 2.979) it is a major result.

**Implementation:** Wire `CyphaDIF` from `Cypha.py` directly as the LM head. Feed rolling character n-gram as feature vector via `RFFEncoder`. Context prior from `TieredContextBuffer`.

**Note:** This variant does not use the `CyphaLM` scaffold above — run it standalone. Log BPC only.

---

### H05 — α-Fitness Auxiliary Loss

**Hypothesis:** Add auxiliary loss `L_α = λ * mean(|α_t - 0.5|)` penalising deviation from edge-of-chaos during training. Applied on top of standard LSTM (B1) and hybrid_gria_lstm (B2).

**Derivation:** Empirical finding: trained SwiGLU FFN layers sit at spec_alpha ∈ [0.44, 0.53] (edge-of-chaos). This loss explicitly trains the hidden state toward that regime.

```python
def alpha_loss(h_trajectory, target=0.5, lam=0.01):
    alphas = torch.stack([gria_alpha(h) for h in h_trajectory])
    return lam * (alphas - target).abs().mean()

# In training loop:
# loss = ce_loss + alpha_loss(h_trajectory)
```

**Test on:** B1 (LSTM + α-loss), B2 (hybrid + α-loss). Four conditions total.

---

## Tier 2 — Medium Complexity

---

### H06 — NIG-State Cell

**Hypothesis:** Cell state is a Normal-Inverse-Gaussian posterior (μ, λ, α_nig, β) rather than a point estimate. Update is Bayesian conjugate update. Uncertainty is tracked explicitly in memory.

**Derivation:** CyphaDIF already uses NIG via `NIGField`. This cell ports that posterior update into the recurrent state, giving calibrated uncertainty over what the cell remembers.

**Implementation:** Maintain `(mu, precision)` as cell state. Update: `mu_new = (lambda * mu + x) / (lambda + 1)`, precision via NIG conjugate. Output h = sample or mean.

---

### H07 — Differential Gate Cell

**Hypothesis:** Cell has one shared `WorldHidden` h₀ plus per-class offset `Δh_k`, mirroring CyphaDIF's `θ₀ ⊕ Δk`. Hidden state is `h₀ + Δh_predicted_class`.

**Derivation:** Direct port of CyphaDIF's structural decomposition into the recurrent domain. The cell learns what is common across all context (world prior) and what is class-specific.

**Implementation:**
```python
class DifferentialGateCell(BaseCyphaCell):
    def __init__(self, input_dim, hidden_dim, n_classes=100):
        super().__init__()
        self.world_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.delta = nn.Embedding(n_classes, hidden_dim)  # Δh per class
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, h_prev, c_prev):
        h0, c = self.world_cell(x, (h_prev, c_prev))
        k = self.classifier(h0).argmax(-1)
        delta_k = self.delta(k)
        h = h0 + delta_k
        return h, c
```

---

### H08 — TieredContext Cell

**Hypothesis:** Three parallel cell states at short (window=32), mid (EMA=0.98), long (Welford) timescales. Output is confidence-weighted blend. Mirrors `TieredContextBuffer`.

**Derivation:** TieredContextBuffer is already proven in CyphaDIF. This makes the recurrent state explicitly multi-scale with the same three tiers.

**Implementation:** Three `nn.LSTMCell` instances. Blend weights = softmax of per-tier confidence scores (entropy of output distribution).

---

### H09 — GRIA-Gated Mixture Cell

**Hypothesis:** Mixture of two cells — one optimised for ordered/compressible context (high α), one for chaotic/novel context (low α). Gate = `σ(α_t - 0.5)`.

**Derivation:** GRIA α distinguishes ordered from chaotic regimes. Different memory dynamics may be optimal in each. High-α context is predictable — use a conservative cell. Low-α context is novel — use a more plastic cell.

**Implementation:**
```python
class GRIAMixtureCell(BaseCyphaCell):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.ordered_cell = nn.LSTMCell(input_dim, hidden_dim)   # conservative
        self.chaotic_cell = nn.LSTMCell(input_dim, hidden_dim)   # plastic
        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev, c_prev):
        alpha = gria_alpha(h_prev)  # (batch, 1)
        gate = torch.sigmoid(alpha - 0.5)
        h_o, c_o = self.ordered_cell(x, (h_prev, c_prev))
        h_c, c_c = self.chaotic_cell(x, (h_prev, c_prev))
        h = gate * h_o + (1 - gate) * h_c
        c = gate * c_o + (1 - gate) * c_c
        return h, c
```

---

### H10 — NMP Regularised Cell

**Hypothesis:** Add NMP (Neural Manifold Proximity) score as a training regulariser. Pushes hidden state spectral alpha toward [0.44, 0.53] during training.

**Derivation:** Empirical finding: trained neural layers at edge-of-chaos have spec_alpha ∈ [0.44, 0.53], NMP > 0.97. Regularising toward this regime during language model training should improve generalisation.

**Implementation:** Compute `spec_alpha` from hidden state weight matrix via power-law fit to singular value spectrum. Add `L_nmp = λ * |spec_alpha - 0.485|` to loss.

---

### H11 — Reversible Cell

**Hypothesis:** Cell update is explicitly bijective — `h_t` can be exactly recovered from `h_{t+1}`. This eliminates vanishing gradients by design and constrains α (reversible = entropy-preserving = α = 0).

**Derivation:** GRIA distinguishes reversible (α=0) from irreversible (α=1) operations. A fully reversible cell is a controlled α=0 regime — pure geometric transformation, no information loss. Test whether this improves gradient flow enough to beat LSTM despite losing the forgetting mechanism.

**Implementation:** Use coupling layers (RevNet-style): split h into (h_a, h_b), update `h_a' = h_a + F(h_b)`, `h_b' = h_b + G(h_a')`. Invertible by construction.

---

### H12 — MDL Forget Cell

**Hypothesis:** Forget gate replaced by MDL norm projection `h ← h * min(1, C/‖h‖_F)`. Solomonoff-grounded forgetting — simpler states are preferred when evidence is weak.

**Derivation:** CyphaDIF's ClassDifferential uses `‖Δk‖_F ≤ C`. Same prior applied to recurrent state: when the hidden state grows complex (large norm), project it back. Complexity-constrained memory.

```python
class MDLForgetCell(BaseCyphaCell):
    def __init__(self, input_dim, hidden_dim, mdl_C=8.0):
        super().__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.mdl_C = mdl_C

    def forward(self, x, h_prev, c_prev):
        h, c = self.lstm(x, (h_prev, c_prev))
        norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        h = h * (self.mdl_C / norm).clamp(max=1.0)
        return h, c
```

---

### H13 — Priority Replay Recurrence

**Hypothesis:** Hidden state incorporates surprise-weighted replay from a priority buffer. High-surprise tokens (large LLR residual) are re-injected into h at subsequent steps.

**Derivation:** CyphaDIF Phase 4 priority replay (capacity 10k, replay_ratio=0.30) improves classification. The same mechanism applied to recurrent state: surprising tokens leave a stronger memory trace.

**Implementation:** Maintain a `PriorityBuffer` of (h, surprise_score) pairs. At each step, blend current h with top-k buffer entries weighted by surprise score.

---

### H14 — OOD-Branching Cell

**Hypothesis:** `anomaly_score` gates between two cell update paths — one for in-distribution context, one for OOD context. Separate memory dynamics for familiar vs novel input.

**Derivation:** CyphaDIF's OOD detection (threshold 3.0) is a first-class output. Novel context should update memory differently from familiar context — higher plasticity, lower retention.

**Implementation:** `anomaly_score` from CyphaDIF inference drives a hard or soft gate between `cell_familiar` and `cell_novel`.

---

## Tier 3 — Expensive / SR Level

These require either symbolic regression sweeps or AXIOM-level search. Run after Tier 1 and 2 results are in — use those results to constrain the search grammar.

---

### H15 — AXIOM-Evolved Cell

**Hypothesis:** Use AXIOM genetic programming to evolve the entire cell update equation from scratch. Grammar primitives: EML, σ, tanh, +, ×, GRIA α. Fitness signal: BPC on WikiText-2 dev set after 5 epochs.

**Protocol:**
1. Define grammar: terminals = {x, h, c, α(h), W·[·]}, functions = {eml, σ, tanh, +, ×, /}
2. Population = 50 programs, tournament selection
3. Each candidate: 5-epoch BPC eval on 50k token subset (fast proxy)
4. Run 20 generations
5. Top-5 candidates: full 20-epoch eval

**Expected runtime:** ~48 hours on 3090. Run overnight in batches.

---

### H16 — Symbolic Regression on LSTM Gates

**Hypothesis:** Treat the trained `hybrid_gria_lstm` gate activations as a dataset. Run SR (PySR or custom) to find closed-form expressions for what the gates are actually computing. Then implement that closed form directly.

**Protocol:**
1. Train `hybrid_gria_lstm` to convergence
2. Log (input, gate_value) pairs for f, i, o gates across 1M tokens
3. Run PySR with operators {exp, log, /, +, ×, α, eml} on each gate's dataset
4. Extract best closed-form expression per gate
5. Implement as `SRDerivedCell` and benchmark

**This is the most scientifically grounded approach** — you're discovering what the network learned rather than hypothesising it.

```bash
pip install pysr
# PySR config: binary_operators=["+", "*", "/"], unary_operators=["exp", "log", "eml"]
# maxsize=20, populations=30, niterations=100
```

---

### H17 — Sheffer-Only Cell

**Hypothesis:** All activations expressed solely via EML Sheffer operator. Since EML is functionally complete, any gate function is representable. This is the most extreme version of H02.

**Derivation:** A functionally complete basis can represent any boolean/continuous function. If EML is truly the right primitive for Cypha's algebra, a Sheffer-only cell should be expressive enough.

**Implementation:** Express σ(x) and tanh(x) as compositions of EML. Then replace every activation in LSTM with those compositions. Extremely sensitive to numerical stability — use aggressive clamping.

---

### H18 — Cellular Automaton State

**Hypothesis:** Cell state evolves by a learnable 1D CA rule before gating. Edge-of-chaos CA rules (Wolfram class IV) are known to sit at α ≈ 0.5 — same as the edge-of-chaos critical point in GRIA.

**Derivation:** CA class IV behaviour is computationally universal and sits at the order/chaos boundary. This may be a natural match for a GRIA-grounded cell.

**Implementation:** Hidden state reshaped to 1D grid, evolved by a learned local rule (3-neighbourhood convolution), then gated by standard LSTM gates.

---

### H19 — Izaac-Seeded Initialisation

**Hypothesis:** Deterministic initialisation of all cell weights via Izaac VRF instead of random. Tests whether deterministic structure in init affects convergence basin and final BPC.

**Derivation:** Izaac is a deterministic randomness algorithm. If weight init matters (and it does — see lottery ticket hypothesis), Izaac-seeded init may find better basins.

**Implementation:** Replace `nn.init.xavier_uniform_` with Izaac VRF output for all weight matrices. Test on B1, B2, and best Tier 1 variant.

**Note:** This is a modifier, not a standalone cell. Apply to top-3 variants from Tier 1/2.

---

### H20 — Spectral State Cell

**Hypothesis:** Cell state lives in frequency domain. Update is convolution. Tests whether harmonic structure (`σ_k ∝ 1/k`) in your theoretical backbone gives BPC gains.

**Implementation:** FFT of hidden state, multiply by learned frequency weights, IFFT. Gate in frequency domain.

---

### H21 — Free Energy Cell

**Hypothesis:** Cell update minimises variational free energy F = KL[q(h)||p(h)] − log p(x|h) explicitly. Active inference as the recurrence rule.

**Derivation:** CyphaDIF's FEP component contributes the WorldPrior/ClassDifferential structural decomposition. Extending that to minimise free energy at each recurrent step is the full active inference model.

**Implementation:** Reparameterised h ~ N(μ, σ²), ELBO loss replaces standard CE. Matches variational RNN literature but grounded in CyphaDIF's existing FEP derivation.

---

### H22 — Algebraic Fingerprint Cell

**Hypothesis:** Hidden state is a cluster membership vector in Izaac's VRF algebraic fingerprint space. Transitions are learned cluster-to-cluster probability matrices.

**Derivation:** Izaac's VRF certifies cluster membership via algebraic sequence fingerprinting. If character sequences have stable algebraic cluster structure, this cell operates directly in that space.

**Implementation:** Encode each token via Izaac fingerprint → cluster assignment. Transition matrix T[i,j] = learned probability of cluster j following cluster i. Hidden state = distribution over clusters.

---

## Execution Order

```
Phase 1 (Week 1): Baselines B0, B1, B2 — lock numbers
Phase 2 (Week 1): Tier 1: H01, H02, H03, H04, H05 — 5 variants × 5 runs
Phase 3 (Week 2): Tier 2: H06–H14 — best 5 from Tier 1 get Tier 2 modifiers applied
Phase 4 (Week 3+): Tier 3: H15 (AXIOM), H16 (SR on gates) — constrained by Phase 2/3 findings
Phase 5: Combine top-3 variants into a hybrid, benchmark against B2
```

---

## Result Table Template

Fill this in as runs complete.

| ID | Variant | Median BPC | Std | Δ vs B2 | Notes |
|---|---|---|---|---|---|
| B0 | 4-gram | | | | baseline |
| B1 | char-LSTM | | | | baseline |
| B2 | hybrid_gria_lstm | 2.873 | | 0.000 | current best |
| H01 | α-gate | | | | |
| H02 | EML activation | | | | |
| H03 | CausalField | | | | |
| H04 | Pure CyphaDIF LM | | | | |
| H05a | LSTM + α-loss | | | | |
| H05b | hybrid + α-loss | | | | |
| H06 | NIG-state | | | | |
| H07 | Differential gate | | | | |
| H08 | TieredContext cell | | | | |
| H09 | GRIA-mixture | | | | |
| H10 | NMP regularised | | | | |
| H11 | Reversible cell | | | | |
| H12 | MDL forget | | | | |
| H13 | Priority replay recurrence | | | | |
| H14 | OOD-branching | | | | |
| H15 | AXIOM-evolved | | | | |
| H16 | SR on gates | | | | |
| H17 | Sheffer-only | | | | |
| H18 | CA state | | | | |
| H19a | Izaac init (LSTM) | | | | |
| H19b | Izaac init (best variant) | | | | |
| H20 | Spectral state | | | | |
| H21 | Free Energy cell | | | | |
| H22 | Algebraic fingerprint | | | | |

---

## What to Look For

**Primary signal:** BPC < 2.873 (beats current best).

**Secondary signals:**
- α trajectory stabilising near 0.5 (edge-of-chaos) — confirms GRIA alignment
- NMP > 0.97 on trained hidden states — confirms edge-of-chaos in weight spectrum
- Catastrophic forgetting ratio (run a forgetting probe after training) — Cypha's zero-forgetting property should be preserved
- Calibrated uncertainty on OOD characters (unseen Unicode) — NIG-state and Free Energy variants should excel here

**Decision rule for Phase 5 combination:**
Take top-3 variants by BPC. If they are architecturally compatible (e.g. H01 + H05 + H12), combine into a single cell and rerun. If they conflict (e.g. H11 Reversible + H13 Priority Replay), run both combinations and pick the winner.

---

## Notes

- H16 (SR on trained gates) is scientifically the strongest approach — it discovers what the network actually learned rather than hypothesising it. Prioritise this if Tier 1 results are ambiguous.
- H04 (Pure CyphaDIF LM) is the highest-variance bet — could fail completely or be a major result. Run it in parallel with Tier 1, not sequentially.
- H15 (AXIOM-evolved) is the long-horizon bet. Start it running in the background during Phase 3 on a reduced grammar if AXIOM is available.
- All attributed outputs to be published under Odin Loch.
