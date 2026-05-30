# Cypha — Self-Organising Upgrades Plan

**Status:** Design & Implementation Roadmap  
**Scope:** CyphaDIF core · CellAI SSM · CyphaLM interface  
**Philosophy:** Every upgrade is isolated, benchmarked, and revertible. Nothing merges until it beats the baseline.

---

## Table of Contents

1. [Weak Spots Catalogue](#1-weak-spots-catalogue)
2. [Upgrade Definitions](#2-upgrade-definitions)
3. [Baseline Protocol](#3-baseline-protocol)
4. [Upgrade 1 — GNG Expert Management](#upgrade-1--gng-expert-management)
5. [Upgrade 2 — SOM over RFF Encoder](#upgrade-2--som-over-rff-encoder)
6. [Upgrade 3 — GRIA α Live Topology Controller](#upgrade-3--gria-α-live-topology-controller)
7. [Upgrade 4 — Discriminative Feedback CellAI ← CyphaDIF](#upgrade-4--discriminative-feedback-cellai--cyphadif)
8. [Upgrade 5 — Competitive Hebbian Rewiring in CellAI](#upgrade-5--competitive-hebbian-rewiring-in-cellai)
9. [Upgrade 6 — Temporal SOM for SSM Timescales](#upgrade-6--temporal-som-for-ssm-timescales)
10. [Integration Order](#10-integration-order)
11. [Revert Protocol](#11-revert-protocol)
12. [Tuning Reference](#12-tuning-reference)

---

## 1. Weak Spots Catalogue

A concise statement of what is broken and why before any fix is applied.

| ID | Component | Problem | Failure Mode |
|----|-----------|---------|--------------|
| W1 | RFF Encoder | Fixed random projections, no adaptation to distribution shift | Accuracy degrades silently on domain drift; no signal |
| W2 | Expert routing | No Winner-Takes-All competition; all experts process every input | Experts never specialise; redundant, unscalable |
| W3 | Expert count | Static buffer with manual resize; no principled growth or pruning | Prototype bloat or coarse coverage; no self-regulation |
| W4 | CellAI topology | Fixed ring, connectivity never changes after init | Topographically incoherent state space; co-active neurons never cluster |
| W5 | SSM decay rates | Fixed α_l per scale, baked at construction | Wrong temporal bandwidth when sequence statistics change |
| W6 | GRIA α usage | Computed post-hoc as a fitness oracle, not a live control signal | Edge-of-chaos target described but not enforced |
| W7 | CellAI→CyphaDIF | One-way interface; CellAI never learns what was discriminative | CellAI representation not shaped by downstream task |

---

## 2. Upgrade Definitions

| Upgrade | Fixes | Complexity | Payoff | Priority |
|---------|-------|------------|--------|----------|
| U1 — GNG Expert Management | W2, W3 | Medium | High | 1 |
| U2 — SOM Encoder Wrapper | W1 | Low | High | 2 |
| U3 — GRIA α Topology Controller | W6 | Low | Very High | 3 |
| U4 — Discriminative Feedback Path | W7 | Medium | High | 4 |
| U5 — Competitive Hebbian Rewiring | W4 | High | Medium | 5 |
| U6 — Temporal SOM for SSM Timescales | W5 | High | Medium | 6 |

**Rule:** U1–U4 must be stable before touching U5–U6. Each upgrade is built on a branch, passes the full baseline suite, then merges.

---

## 3. Baseline Protocol

Establish before writing any upgrade code. Every upgrade is measured against these numbers.

### 3.1 Metrics to Record

```
Classification:
  - Accuracy (in-distribution)
  - Accuracy under 15 adversarial injections (GH-protection benchmark)
  - OOD detection rate (chi-gate recall)
  - False positive rate on OOD (chi-gate precision)

Online learning:
  - Accuracy after N=100 / N=1000 / N=10000 sequential train steps
  - Convergence speed (steps to 90% of asymptotic accuracy)

Robustness:
  - Prototype poisoning recall (dos_recall benchmark from Section 9)
  - Accuracy after targeted 15-injection attack

Efficiency:
  - Wall time per train_step (ms)
  - Wall time per infer (ms)
  - Memory footprint (MB)
  - Expert/prototype count at end of run

Stability:
  - Accuracy variance across 5 random seeds
  - Accuracy at step 10k vs step 1k (no forgetting check)
```

### 3.2 Baseline Datasets

- **Primary:** sklearn `make_classification` 10-class, 50-feature, 10k samples (matches existing Section 9 benchmark)
- **Drift test:** Two-phase run — 5k samples distribution A, 5k samples distribution B (different class means)
- **Adversarial:** Standard 15-injection poisoning benchmark already in Cypha test suite
- **Language (CyphaLM only):** WikiText-2 character-level perplexity

### 3.3 Baseline Run Procedure

```bash
# Run before any upgrade branch is cut
python benchmark_baseline.py --dataset classification --seeds 5 --output results/baseline.json
python benchmark_baseline.py --dataset drift --seeds 5 --output results/baseline_drift.json
python benchmark_baseline.py --dataset adversarial --output results/baseline_adv.json
```

Save `results/baseline.json` to version control. This file is the contract every upgrade must beat or match.

### 3.4 Pass/Fail Thresholds

An upgrade passes if **all** of the following hold:

| Metric | Required |
|--------|----------|
| In-distribution accuracy | ≥ baseline − 0.5% |
| Adversarial dos_recall | ≥ baseline − 2% |
| OOD detection rate | ≥ baseline − 2% |
| train_step wall time | ≤ baseline × 1.25 (25% overhead budget) |
| Memory footprint | ≤ baseline × 1.5 |
| Accuracy variance (5 seeds) | ≤ baseline × 1.5 |

If any threshold fails, the upgrade is either fixed or reverted. No exceptions for "it's mostly better."

---

## Upgrade 1 — GNG Expert Management

**Fixes:** W2 (no routing competition), W3 (static expert count)

### What It Does

Replaces the static prototype set with a **Growing Neural Gas** that dynamically inserts and removes prototype nodes. GNG inserts a new node where cumulative reconstruction error is highest, and removes nodes that accumulate no traffic past a staleness threshold. Expert count becomes a function of data complexity, not a construction-time guess.

### Mathematics

Each node `i` has a weight vector `w_i ∈ ℝ^d` and an error accumulator `E_i`.

**Per-input step:**
```
BMU   = argmin_i ||x - w_i||          # best matching unit
2BMU  = second closest node

w_BMU  += ε_b · (x - w_BMU)           # pull winner toward x
w_nbr  += ε_n · (x - w_nbr)           # pull topological neighbours

E_BMU  += ||x - w_BMU||²              # accumulate error at winner
age(BMU, 2BMU) += 1                   # age the edge between them
```

**Every λ steps (insert):**
```
q = argmax_i E_i                       # highest error node
f = argmax_{j ∈ nbrs(q)} E_j          # highest error neighbour of q
insert r: w_r = (w_q + w_f) / 2
E_q *= α_gng;  E_f *= α_gng;  E_r = E_q
```

**Every step (prune):**
```
remove edges with age > age_max
remove isolated nodes (degree 0)
```

### Implementation Sketch

```python
class GNGExpertManager:
    def __init__(self, d, eps_b=0.05, eps_n=0.006,
                 lam=100, age_max=50, alpha_gng=0.5):
        self.nodes   = {}          # id -> weight vector
        self.errors  = {}          # id -> float
        self.edges   = {}          # (i,j) -> age
        # ... init with 2 random nodes

    def step(self, x):
        bmu, bmu2 = self._two_closest(x)
        self._update_weights(x, bmu, bmu2)
        self._age_edges(bmu)
        self._prune_old_edges()
        if self._step_count % self.lam == 0:
            self._insert_node()
        self._decay_errors()
        self._step_count += 1
        return bmu

    def get_prototypes(self):
        return np.stack(list(self.nodes.values()))
```

Plug `GNGExpertManager` into `DIFMemory` so each call to `train()` also runs `gng.step(encoded_x)`, and the class prototype set is replaced by `gng.get_prototypes()` at inference time.

### Testing This Upgrade

**Test 1 — Node count tracks complexity**
```
Dataset A: 3 linearly separable classes
Dataset B: 10 overlapping classes

Expected: GNG converges to ~3 nodes on A, ~10-20 nodes on B
Fail if: Node count identical on both, or grows unboundedly
```

**Test 2 — No accuracy regression on standard benchmark**
```
Run: python benchmark_baseline.py --upgrade U1 --seeds 5
Compare to: results/baseline.json
Must pass all thresholds in §3.4
```

**Test 3 — Drift recovery**
```
Phase 1 (5k steps): Measure accuracy, record node positions
Phase 2 (5k steps, shifted distribution): Measure accuracy recovery rate
Expected: GNG inserts nodes in new regions; accuracy recovers within 500 steps
Fail if: Recovery takes > 2000 steps or accuracy never recovers
```

**Test 4 — Staleness pruning works**
```
Train on 3-class problem, then remove class 2 entirely for 5k steps
Expected: Nodes covering class 2's old region get pruned
Fail if: Dead nodes persist indefinitely
```

### Tuning Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `eps_b` | 0.05 | 0.01–0.2 | Winner pull rate. Higher → faster adaptation, less stable |
| `eps_n` | 0.006 | 0.001–0.05 | Neighbour pull rate. Keep ~10× smaller than eps_b |
| `lam` | 100 | 50–500 | Insert frequency. Lower → more nodes, higher cost |
| `age_max` | 50 | 20–200 | Edge lifespan. Lower → sparser graph, more fragmentation risk |
| `alpha_gng` | 0.5 | 0.3–0.8 | Error decay on insert. Lower → more aggressive future inserts |

**Tuning procedure:** Grid search `lam ∈ {50, 100, 200}` × `age_max ∈ {30, 50, 100}` on the drift benchmark. Pick the pair with best drift recovery at lowest node count.

### Revert Condition

Revert if any of:
- train_step overhead exceeds +25%
- Accuracy variance across seeds > 2× baseline
- Node count grows past `10 × initial_class_count` without plateau

Revert action: `git revert U1-merge-commit`. The static `_D_buf` code path is untouched; flip the flag `use_gng = False`.

---

## Upgrade 2 — SOM over RFF Encoder

**Fixes:** W1 (static encoder, no drift adaptation)

### What It Does

Wraps the existing RFF encoder output in a **Self-Organising Map** trained online. The SOM organizes the encoded space topographically — similar inputs map to nearby SOM units — and self-heals when the input distribution drifts, because SOM weight vectors chase new modes via the neighbourhood update rule.

### Mathematics

SOM has `K × K` units, each with weight vector `w_i ∈ ℝ^d_rff`.

**Per-input update:**
```
BMU = argmin_i ||z - w_i||             # z = RFF(x)

Δw_i = η(t) · h(i, BMU, t) · (z - w_i)

h(i, BMU, t) = exp(-||r_i - r_BMU||² / (2σ(t)²))

η(t) = η_0 · exp(-t / T_η)            # decaying learning rate
σ(t) = σ_0 · exp(-t / T_σ)            # decaying neighbourhood
```

After the SOM step, the output fed to CyphaDIF is `w_BMU` (the winning unit's weight vector), not the raw RFF vector. This replaces a noisy projection with a smoothed prototype.

### Implementation Sketch

```python
class OnlineSOMEncoder:
    def __init__(self, d_in, k=16, eta0=0.3, sigma0=4.0, T=10000):
        self.k = k
        self.W = np.random.randn(k*k, d_in) * 0.1   # SOM weights
        self.positions = np.array([[i,j] for i in range(k) for j in range(k)])
        self.eta0, self.sigma0, self.T = eta0, sigma0, T
        self.t = 0

    def encode(self, z, train=True):
        dists = np.linalg.norm(self.W - z, axis=1)
        bmu = np.argmin(dists)
        if train:
            eta = self.eta0 * np.exp(-self.t / self.T)
            sigma = self.sigma0 * np.exp(-self.t / self.T)
            d2 = np.sum((self.positions - self.positions[bmu])**2, axis=1)
            h = np.exp(-d2 / (2 * sigma**2))
            self.W += eta * h[:, None] * (z - self.W)
            self.t += 1
        return self.W[bmu]
```

Chain: `x → RFF → OnlineSOMEncoder.encode() → CyphaDIF`

### Testing This Upgrade

**Test 1 — Topographic ordering emerges**
```
Feed 3-class data. After 5k steps, plot SOM unit activations per class.
Expected: Each class occupies a contiguous region of the SOM grid
Fail if: Class assignments are scattered with no spatial coherence
```

**Test 2 — Drift self-repair**
```
Phase 1: Train on distribution A. Record BMU assignment map.
Phase 2: Switch to distribution B (shifted means).
Measure: How many steps until SOM BMU assignments stabilise on new distribution?
Expected: < 500 steps to re-stabilise
Fail if: SOM freezes (η decayed to zero too fast) or oscillates
```

**Test 3 — No accuracy regression**
```
Run full baseline suite with SOM in chain.
Must pass all thresholds in §3.4.
```

**Test 4 — SOM overhead acceptable**
```
Profile encode() call time vs raw RFF.
Must stay within 25% overhead budget.
```

### Tuning Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `k` (grid side) | 16 | 8–32 | Larger → finer organisation, more memory and compute |
| `eta0` | 0.3 | 0.05–0.5 | Initial learning rate. Too high → instability |
| `sigma0` | 4.0 | 1.0–k/2 | Initial neighbourhood radius. Start broad, let decay |
| `T_eta` | 10000 | 5k–50k | Decay timescale. Should match expected dataset size |
| `T_sigma` | 10000 | 5k–50k | Keep equal to T_eta for synchronised decay |

**Tuning procedure:** Vary `k ∈ {8, 16, 32}` and measure topographic organisation quality (U-matrix entropy) vs inference overhead. Pick smallest k that shows clear topographic separation on the 10-class benchmark.

### Revert Condition

Revert if:
- Topographic organisation never emerges after 10k steps (SOM is noisy, not useful)
- encode() overhead exceeds 25% of total infer time
- Accuracy on frozen distribution regresses vs baseline

Revert action: Bypass the SOM, feed RFF output directly. The `use_som` flag in the encoder config controls this.

---

## Upgrade 3 — GRIA α Live Topology Controller

**Fixes:** W6 (α is post-hoc, not a control signal)

### What It Does

Computes α locally per expert at training time, then uses the value to trigger structural actions — split, merge, or leave alone — on GNG nodes and SSM layers. The edge-of-chaos attractor becomes a homeostatic mechanism, not just a description.

### Mathematics

Recall: `α = 1 − H(f(X)) / H(X)`

Compute per-expert over a rolling window of W recent activations:

```
α_i(t) = 1 − H(activations_i[t-W : t]) / H(inputs[t-W : t])
```

Control law:
```
α_i < 0.35  →  over-ordered: call GNG insert at node i (split)
α_i > 0.65  →  chaotic:      merge node i with nearest neighbour
0.35 ≤ α_i ≤ 0.65  →  healthy: standard update only
```

For SSM layers:
```
α_layer < 0.35  →  decay rates too large (over-smoothing): increase α_l by δ_ssm
α_layer > 0.65  →  decay rates too small (too reactive): decrease α_l by δ_ssm
```

### Implementation Sketch

```python
class GRIAController:
    def __init__(self, window=200, low=0.35, high=0.65, delta_ssm=0.01):
        self.window = window
        self.low, self.high = low, high
        self.delta_ssm = delta_ssm
        self._act_buf  = deque(maxlen=window)
        self._inp_buf  = deque(maxlen=window)

    def push(self, x, activations):
        self._inp_buf.append(x)
        self._act_buf.append(activations)

    def alpha(self):
        if len(self._inp_buf) < self.window:
            return 0.5                         # not enough data yet
        H_x = entropy(np.stack(self._inp_buf))
        H_f = entropy(np.stack(self._act_buf))
        return 1.0 - H_f / (H_x + 1e-9)

    def act(self, node_id, gng, ssm):
        a = self.alpha()
        if a < self.low:
            gng.force_insert(node_id)          # split
        elif a > self.high:
            gng.merge_with_nearest(node_id)    # merge
        # SSM adjustment
        if a < self.low:
            ssm.adjust_decay(+self.delta_ssm)
        elif a > self.high:
            ssm.adjust_decay(-self.delta_ssm)
```

Run `GRIAController.act()` every `C=50` train steps per expert.

### Testing This Upgrade

**Test 1 — α converges toward 0.5**
```
Train on standard benchmark.
Record per-expert α at steps 100, 500, 1000, 5000, 10000.
Expected: Mean α across experts converges to 0.45–0.55 range by step 5000.
Fail if: α diverges to 0 or 1, or oscillates above amplitude 0.2 past step 2000.
```

**Test 2 — Structural actions are triggered sensibly**
```
Force an artificially ordered dataset (single-cluster), run 1000 steps.
Expected: α < 0.35 consistently → GNG inserts new nodes → count grows.
Force high-entropy noise, run 1000 steps.
Expected: α > 0.65 consistently → merges reduce count.
Fail if: No structural actions occur when they should.
```

**Test 3 — Controller doesn't destabilise a healthy run**
```
Run full baseline. GRIAController active.
Must pass all thresholds in §3.4 — controller should be a no-op on healthy data.
```

**Test 4 — Window sensitivity**
```
Test window ∈ {50, 200, 500}.
Expected: Shorter window reacts faster but is noisier; longer is smoother but slower.
Pick window that gives α convergence < 2000 steps with variance < 0.05.
```

### Tuning Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `window` | 200 | 50–1000 | Smoothing window for α estimate |
| `low` | 0.35 | 0.2–0.45 | Split threshold. Lower → less aggressive splitting |
| `high` | 0.65 | 0.55–0.8 | Merge threshold. Higher → less aggressive merging |
| `C` (control interval) | 50 | 10–200 | Steps between control actions |
| `delta_ssm` | 0.01 | 0.001–0.05 | SSM decay rate adjustment per action |

### Revert Condition

Revert if:
- α never converges (oscillates indefinitely)
- Structural actions cascade (insert triggers chaos → merge triggers over-order → loop)
- Any baseline accuracy threshold fails

Revert action: Set `gria_control = False`. The GNG and SSM still operate; controller is bypassed.

---

## Upgrade 4 — Discriminative Feedback CellAI ← CyphaDIF

**Fixes:** W7 (one-way CellAI→CyphaDIF interface)

### What It Does

After CyphaDIF computes the LLR classification on hidden state `h`, it backprojects a **discriminative weighting vector** `d` that scores each dimension of `h` by its contribution to the LLR gap. This is fed back to CellAI as a Hebbian modulation signal — dimensions that were useful get reinforced, dimensions that contributed nothing get attenuated. No full backprop through layers.

### Mathematics

After LLR classification:
```
LLR_k = δμ_k^T · Σ⁻¹ · h         # per-class log-likelihood ratio
```

Compute discriminative weight:
```
d = Σ_k |δμ_k ⊙ Σ_diag^{-1}|     # element-wise importance of h dims
d = d / ||d||_1                    # normalise
```

Modulate CellAI's Hebbian update:
```
ΔW_CellAI +=  β · diag(d) · (Δw_Hebbian)
```

Dimensions of `h` with high `d_i` get stronger Hebbian reinforcement. Dimensions with low `d_i` are softly suppressed. `β` is a feedback gain (default 0.1 — keep it small to avoid instability).

### Implementation Sketch

```python
class DiscriminativeFeedback:
    def __init__(self, beta=0.1):
        self.beta = beta

    def compute_d(self, delta_mu, sigma_diag_inv):
        # delta_mu: [K, d], sigma_diag_inv: [d]
        importance = np.sum(np.abs(delta_mu) * sigma_diag_inv[None, :], axis=0)
        return importance / (importance.sum() + 1e-9)

    def modulate(self, dW_hebbian, d):
        # dW_hebbian: [d_out, d_in], d: [d_in]
        return dW_hebbian + self.beta * (d[None, :] * dW_hebbian)
```

Call `DiscriminativeFeedback.modulate()` on each CellAI weight update after a CyphaDIF classification step.

### Testing This Upgrade

**Test 1 — Discriminative dimensions become dominant in h**
```
After 5k steps, compute correlation between d vector and variance of h dims.
Expected: High-d dimensions have higher variance (more active representations).
Fail if: d has no correlation with h activity pattern.
```

**Test 2 — Convergence speed improves**
```
Compare steps to 90% accuracy: baseline vs U4.
Expected: U4 reaches 90% in fewer steps (feedback accelerates useful representation).
Fail if: Convergence is slower or identical.
```

**Test 3 — β sensitivity**
```
Test β ∈ {0.01, 0.1, 0.3, 0.5}.
Expected: β=0.01 negligible effect, β=0.5 instability, sweet spot 0.05–0.2.
Fail if: No β value gives improvement without instability.
```

**Test 4 — Full baseline must pass**
```
No accuracy regression, no timing regression.
```

### Tuning Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `beta` | 0.1 | 0.01–0.3 | Feedback gain. Keep small. Too large → reinforcement loop |
| Feedback interval | every step | every 1–10 steps | Less frequent = more stable, less responsive |

### Revert Condition

Revert if:
- h representation collapses (all dimensions converge to same value)
- Training instability (loss spikes, oscillating accuracy)
- Any threshold in §3.4 fails

Revert action: Set `discriminative_feedback = False`. CellAI Hebbian update reverts to unmodulated form.

---

## Upgrade 5 — Competitive Hebbian Rewiring in CellAI

**Fixes:** W4 (frozen ring topology)

### What It Does

Replaces the static ring with a dynamic adjacency structure. Edge weights between neurons grow when two neurons co-activate above a threshold, and decay when they don't. Edges below a minimum weight are pruned. New edges can form between any pair of neurons that consistently co-activate. Over time, neurons that process similar temporal patterns cluster into functional neighbourhoods.

### Mathematics

For neurons `i, j` with activations `a_i(t), a_j(t)`:

```
e_{ij}(t+1) = e_{ij}(t) + η_edge · a_i(t) · a_j(t)    # Hebbian growth
e_{ij}(t+1) *= (1 - λ_decay)                           # passive decay

if e_{ij} < θ_prune:  remove edge (i,j)
if a_i · a_j > θ_form and edge absent:  insert edge (i,j) with e=θ_form
```

The diffusion kernel in CellAI then uses the current adjacency `E` as its graph:
```
x_diffused = x + γ · (A_normalised · x)
```

where `A_normalised` is the degree-normalised adjacency matrix of `E`.

### Testing This Upgrade

This is the highest-complexity upgrade. Additional tests required:

**Test 1 — Topology converges (doesn't grow unboundedly)**
```
Run 10k steps. Record edge count every 500 steps.
Expected: Edge count plateaus within 5k steps.
Fail if: Edge count grows monotonically without bound.
```

**Test 2 — Topographic clustering emerges**
```
Feed labelled sequences. After 10k steps, compute graph modularity with respect to class labels.
Expected: Modularity > 0.2 (neurons cluster by class).
Fail if: Modularity ≤ 0 (random topology).
```

**Test 3 — Diffusion on dynamic graph is stable**
```
No NaN/Inf in x_diffused over 10k steps.
Eigenvalue check: spectral radius of A_normalised ≤ 1.
```

**Test 4 — Full baseline must pass at increased overhead**
```
Accept up to +40% overhead for this upgrade (higher budget due to complexity).
```

### Tuning Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `eta_edge` | 0.01 | 0.001–0.1 | Edge growth rate |
| `lambda_decay` | 0.001 | 0.0001–0.01 | Passive edge decay |
| `theta_prune` | 0.01 | 0.001–0.1 | Prune threshold. Higher → sparser graph |
| `theta_form` | 0.3 | 0.1–0.8 | Co-activation threshold to create new edge |
| `gamma` | 0.1 | 0.0–0.5 | Diffusion strength on current graph |

### Revert Condition

Revert if:
- Topology never converges (runaway edge growth)
- Spectral radius > 1 (diffusion instability)
- Overhead > 40%

Revert action: Restore fixed ring adjacency. `use_dynamic_topology = False`.

---

## Upgrade 6 — Temporal SOM for SSM Timescales

**Fixes:** W5 (fixed SSM decay rates)

### What It Does

Runs a small SOM over recent temporal autocorrelation statistics. Each SOM unit owns a set of decay rates. When the input temporal statistics land on a new Best Matching Unit, the SSM adopts that unit's decay rates. This creates a learned dictionary of temporal regimes; the architecture detects regime changes and switches timescales automatically.

### Mathematics

Compute a temporal feature vector at time `t`:
```
r(t) = [autocorr(x, lag=1), ..., autocorr(x, lag=L_max)]   # L_max=16
```

SOM over `r(t)` with `M` units, each unit `m` owning decay rates `Λ_m ∈ ℝ^L`.

```
BMU(t) = argmin_m ||r(t) - c_m||

Update SOM:  c_m += η_ts · h(m, BMU, t) · (r(t) - c_m)

SSM decay rates at time t: α_l = Λ_{BMU(t), l}
```

Decay rates in `Λ` are initialised uniformly and learn what temporal bandwidth each detected regime needs.

### Testing This Upgrade

**Test 1 — Regime detection**
```
Construct a sequence with two distinct temporal statistics (short-lag autocorr phase, long-lag autocorr phase).
Expected: BMU switches cleanly when regime changes. Regime boundary detection < 50 steps lag.
Fail if: BMU is stable through regime change (SOM not discriminating).
```

**Test 2 — SSM decay rates differ across regimes**
```
After training, extract Λ_BMU for each phase.
Expected: Short-range phase → higher decay rates (α closer to 1, fast forgetting).
           Long-range phase → lower decay rates (α closer to 0, long memory).
Fail if: Λ vectors are identical across BMU units.
```

**Test 3 — Long-context accuracy**
```
Construct "needle in haystack" test: correct answer depends on context 1000+ tokens ago.
Expected: U6 improves accuracy on this test vs fixed timescales.
Fail if: No improvement (temporal SOM not helping long-context recall).
```

### Tuning Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `M` (SOM units) | 8 | 4–32 | Temporal regime dictionary size |
| `L_max` | 16 | 8–64 | Autocorrelation lag window |
| `eta_ts` | 0.05 | 0.01–0.2 | Temporal SOM learning rate |

### Revert Condition

Revert if:
- BMU never changes (SOM frozen)
- Decay rates collapse to 0 or 1 (loss of timescale diversity)
- No improvement on long-context test

Revert action: `use_temporal_som = False`. SSM reverts to fixed α_l values.

---

## 10. Integration Order

Do not skip steps or combine upgrades.

```
Step 0: Cut baseline branch. Run full baseline. Save results/baseline.json.

Step 1: Branch U2 from main.
        Build SOM encoder. Run tests. Pass §3.4. Merge to main.

Step 2: Branch U1 from main.
        Build GNG expert manager. Run tests. Pass §3.4. Merge to main.

Step 3: Branch U3 from main (depends on U1 for GNG API).
        Build GRIA controller. Run tests. Pass §3.4. Merge to main.

Step 4: Branch U4 from main (depends on U1+U3 for stable expert set).
        Build discriminative feedback. Run tests. Pass §3.4. Merge to main.

Step 5: Branch U5 from main (U1–U4 must all be stable first).
        Build Hebbian rewiring. Run tests. Pass §3.4 + extended topology tests.
        Merge to main.

Step 6: Branch U6 from main (full stack should be stable first).
        Build temporal SOM. Run tests. Pass §3.4 + long-context tests.
        Merge to main.
```

After each merge, run the full baseline suite again and update `results/current.json`. This is your new comparison point for the next step.

---

## 11. Revert Protocol

### Immediate Revert (any threshold fails)

```bash
# Identify the last clean merge commit
git log --oneline main | head -20

# Hard revert to it
git revert <bad-merge-commit>

# Or if the branch hasn't merged yet, just delete it
git branch -D feature/U3-gria-controller
```

Each upgrade has a feature flag in `config.py`:

```python
USE_GNG              = True   # U1
USE_SOM_ENCODER      = True   # U2
USE_GRIA_CONTROLLER  = True   # U3
USE_DISCRIM_FEEDBACK = True   # U4
USE_DYNAMIC_TOPOLOGY = True   # U5
USE_TEMPORAL_SOM     = True   # U6
```

Set the flag to `False` to disable any upgrade without touching its code. This lets you isolate which upgrade caused a regression in a combined run.

### Regression Diagnosis Flow

```
Accuracy drops after merge?
│
├─ Disable U6 → still broken?
│   ├─ Disable U5 → still broken?
│   │   ├─ Disable U4 → still broken?
│   │   │   └─ Root is in U1/U2/U3. Bisect.
│   │   └─ U4 is the culprit. Revert and re-examine β.
│   └─ U5 is the culprit. Check spectral radius.
└─ U6 is the culprit. Check SOM freeze / decay collapse.
```

### When to Abandon an Upgrade

If an upgrade fails its tests across three distinct hyperparameter configurations, abandon it for this iteration. Log the failure mode, keep the code on a dead branch, revisit after more core stability is established.

---

## 12. Tuning Reference

Consolidated quick-reference for all tunable parameters across upgrades.

| Upgrade | Parameter | Default | Quick Adjustment |
|---------|-----------|---------|-----------------|
| U1 GNG | `eps_b` | 0.05 | ↑ if adaptation too slow; ↓ if unstable |
| U1 GNG | `lam` | 100 | ↓ for more nodes; ↑ to reduce count |
| U1 GNG | `age_max` | 50 | ↑ for denser graph; ↓ for faster pruning |
| U2 SOM | `k` | 16 | ↑ for finer organisation; watch overhead |
| U2 SOM | `T_eta` | 10000 | Match to dataset size; too small = frozen too early |
| U3 GRIA | `window` | 200 | ↑ for smoother α; ↓ for faster response |
| U3 GRIA | `C` | 50 | ↑ to reduce structural action frequency |
| U4 Feedback | `beta` | 0.1 | ↓ first if instability; never go above 0.3 |
| U5 Rewiring | `theta_form` | 0.3 | ↑ if topology grows too dense |
| U5 Rewiring | `lambda_decay` | 0.001 | ↑ for faster pruning of stale edges |
| U6 TemporalSOM | `M` | 8 | ↑ for more regime granularity |
| U6 TemporalSOM | `L_max` | 16 | ↑ for longer autocorrelation context |

### General Tuning Rules

1. **One parameter at a time.** Never change two knobs simultaneously; you can't attribute the result.
2. **Start conservative.** Halve any parameter that could cause growth or instability before moving toward the default.
3. **Re-run baseline after every tune.** A parameter that improves drift recovery may degrade adversarial protection.
4. **α is the ground truth.** If GRIA α per-expert is converging to 0.45–0.55, the system is healthy regardless of other metrics trending oddly.

---

*Document version: 1.0 — cut from architecture analysis session*
