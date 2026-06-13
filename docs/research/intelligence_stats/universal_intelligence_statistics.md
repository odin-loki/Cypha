# Universal Statistics for Intelligent Systems: From Theory to CyphaDIF Implementation

**Author:** Odin Loch
**Status:** Working Paper
**Framework:** CyphaDIF / GRIA / Izaac

---

## Abstract

This paper develops a minimal complete set of universal statistics for characterising intelligent systems — biological, artificial, or hybrid. We derive seven independent statistics from a single meta-principle: every meaningful measure of intelligence is a distance from a degenerate extreme. These statistics are shown to be orthogonal, computationally tractable, and naturally expressible as online Bayesian updates within the Normal-Inverse-Gaussian (NIG) framework of CyphaDIF. The result is a 7×3 profile matrix — seven statistics, each with epistemic uncertainty, aleatoric uncertainty, and point estimate — updated online in real time. We further demonstrate that this framework extends naturally to algorithm-intrinsic statistics for Izaac and to statistics derived from the structure of biological intelligence. The unified system constitutes a comprehensive real-time intelligence health monitor embeddable in any system where CyphaDIF operates.

---

## 1. Introduction

### 1.1 The Problem of Measuring Intelligence

What separates an intelligent system from a lookup table or a noise generator? Any useful answer must be quantitative, universal, and computable from observable data. The history of statistics suggests that the most powerful measures are discovered empirically — an operator is applied to data, its behaviour is observed, and meaning is extracted by reflection on the operator's structure. Pearson, Gosset, and Fisher all worked this way.

This paper follows the same methodology but extends it: we use structured self-reflection on the space of possible operators to derive, rather than merely discover, a minimal complete basis for intelligence measurement. The result is not a collection of ad hoc statistics but a principled framework with a single underlying meta-principle.

### 1.2 The Meta-Principle

**Every useful statistic for an intelligent system measures distance from a degenerate extreme.**

The two universal degenerate extremes are:
- **Maximum entropy** — pure randomness, no compressible structure, no generalisable representation
- **Minimum entropy** — pure repetition, fully determined, no capacity for novel response

Intelligence is the dynamical maintenance of position between these extremes. Every statistic that meaningfully characterises an intelligent system is a coordinate in this middle space, measured in some specific subspace of the system's behaviour.

This principle is not new — it is implicit in the concept of edge-of-chaos criticality, in information-theoretic treatments of learning, and in the statistical mechanics of neural systems. What is new here is the systematic derivation of a minimal orthogonal basis from this principle, and the demonstration that this basis maps cleanly onto online Bayesian inference in the NIG framework.

### 1.3 Relationship to GRIA

The Graded Reversible-Irreversible Algebra (GRIA) framework defines the universal invariant:

```
α = 1 − H(f(X)) / H(X)
```

where α ∈ [0,1] is the grade parameter measuring compression irreversibility, with α ≈ 0.5 as the critical point. This paper generalises the GRIA insight: α is not merely one statistic but the template for all statistics in this framework. Each of the seven universal statistics is an α-like quantity defined in a different space.

---

## 2. Derivation of the Seven Universal Statistics

### 2.1 Methodology

To derive a complete orthogonal basis, we enumerate the independent spaces in which an intelligent system can occupy a position between degenerate extremes. Independence here means that a system can score anywhere on one axis without determining its score on another. We identify seven such spaces.

### 2.2 The Seven Statistics

#### Statistic 1: Information Grade (α)
**Space:** Information processing
**Degenerate extremes:** Pure transmission (α = 0) / Pure destruction (α = 1)
**Definition:**
```
α = 1 − H(f(X)) / H(X)
```
**Interpretation:** Measures the irreversibility fraction of the system's information transformation. Critical systems operate at α ≈ 0.5. Directly computable from input/output entropy of any layer or system.

**Data acquisition:** Input entropy H(X) and output entropy H(f(X)) from activation histograms. Cheap on any layer.

#### Statistic 2: Representational Dimensionality (D_eff)
**Space:** Representational geometry
**Degenerate extremes:** One-hot representation (D_eff = 1) / Uniform random (D_eff = N)
**Definition:**
```
D_eff = (Σλᵢ)² / Σλᵢ²
```
where λᵢ are eigenvalues of the activation covariance matrix. This is the participation ratio.

**Interpretation:** Measures how many dimensions of the representational space are actually used. Intelligent systems use a moderate, structured subset — neither one dimension nor all dimensions.

**Data acquisition:** PCA on layer activations. O(d²n) where d is dimension and n is sample count.

#### Statistic 3: Dynamical Stability (σ_branch)
**Space:** Forward-pass dynamics
**Degenerate extremes:** Dead network (σ = 0) / Explosive network (σ → ∞)
**Definition:**
```
σ_branch = E[number of downstream units activated per upstream activation]
```
For continuous activations, use the spectral norm of the Jacobian as a proxy:
```
σ_branch ≈ ||∂f/∂x||_2
```
**Interpretation:** The branching ratio from criticality theory. σ = 1 is the critical point. σ < 1 indicates a system that dampens perturbations; σ > 1 indicates amplification toward instability.

**Data acquisition:** One forward pass with random perturbations. Jacobian-vector products via autodiff.

#### Statistic 4: Temporal Integration Depth (τ)
**Space:** Temporal processing
**Degenerate extremes:** Zero memory (τ = 0) / Infinite memory (τ → ∞)
**Definition:**
```
τ = argmax_k { I(Y_t ; X_{t-k}) > ε }
```
where I(·;·) is mutual information and ε is a noise floor.

**Interpretation:** The effective number of past timesteps that causally influence the current output. Too shallow indicates a reactive system; too deep indicates one that cannot adapt to distributional shift.

**Data acquisition:** Mutual information estimates at increasing lags. Efficient with NIG-based entropy estimators already in CyphaDIF.

#### Statistic 5: Uncertainty Decomposition Ratio (r_eu)
**Space:** Epistemic state
**Degenerate extremes:** All aleatoric (r_eu = 0, knows nothing reducible) / All epistemic (r_eu = 1, all uncertainty is knowledge deficit)
**Definition:**
```
r_eu = σ²_epistemic / (σ²_epistemic + σ²_aleatoric)
```
**Interpretation:** Measures the fraction of total uncertainty that is reducible by more data. A well-calibrated intelligent system maintains a meaningful ratio — neither dismissing all uncertainty as irreducible noise nor treating all uncertainty as ignorance.

**Data acquisition:** Native output of CyphaDIF NIG inference. This statistic is free given Cypha is already running.

#### Statistic 6: Generalisation Smoothness (L)
**Space:** Input-output topology
**Degenerate extremes:** Rigid (L = 0, no sensitivity) / Chaotic (L → ∞, hypersensitive)
**Definition:**
```
L = E[||f(x + δ) − f(x)||₂ / ||δ||₂]
```
the empirical Lipschitz constant estimated via random perturbations δ.

**Interpretation:** How smoothly the system interpolates between known inputs. Intelligent generalisation requires finite, moderate Lipschitz constants — large enough to respond meaningfully, small enough to generalise stably.

**Data acquisition:** Random perturbation pairs, two forward passes per estimate. Parallelisable.

#### Statistic 7: Calibration Fidelity (C)
**Space:** Self-knowledge
**Degenerate extremes:** Random confidence (C = 0) / Perfect calibration (C = 1)
**Definition:**
```
C = 1 − E[|P̂(correct) − P(correct)|]
```
where the expectation is over confidence bins.

**Interpretation:** The degree to which the system's expressed confidence matches its actual accuracy. An intelligent system knows what it does not know. This is the most epistemically fundamental of the seven statistics — it measures not just capability but self-awareness of capability.

**Data acquisition:** Binned confidence vs accuracy on held-out data. One forward pass.

---

## 3. Orthogonality and Completeness

### 3.1 Independence

The seven statistics are claimed to be independent in the following sense: there exist systems that score high on any subset of six while scoring poorly on the seventh. Evidence for each:

- A memorisation system: high D_eff (many represented patterns), low α (no compression), poor C (overconfident), poor generalisation L
- A random network: high σ_branch, high α, low D_eff, poor C
- A rigid lookup table: low L, low τ, high C on training distribution, zero generalisation
- An overfit transformer: low α per layer, high C on train, low C on test, low τ

No one of the seven statistics predicts any other across this diversity of failure modes.

### 3.2 Completeness

We claim no additional independent axis exists. Every other plausible statistic for intelligent systems reduces to one of the seven:

| Candidate statistic | Reduces to |
|---|---|
| Avalanche size exponent | σ_branch (Statistic 3) |
| Forgetting curve exponent | τ (Statistic 4) |
| Compression ratio across layers | α (Statistic 1) |
| Entropy of weight matrix eigenvalues | D_eff (Statistic 2) |
| Out-of-distribution detection | C (Statistic 7) |
| Adversarial robustness | L (Statistic 6) |
| Uncertainty consistency | r_eu (Statistic 5) |

### 3.3 The Profile Vector

The seven statistics form a profile vector:

```
P = (α, D_eff, σ_branch, τ, r_eu, L, C)
```

Any intelligent system can be fully characterised by its position in this seven-dimensional space. The ideal intelligent system — biological or artificial — sits near the critical manifold of each axis simultaneously. This is rare precisely because the constraints interact: a system optimised for one axis often degrades on another.

---

## 4. Online Bayesian Inference in CyphaDIF

### 4.1 The Extension

CyphaDIF currently maintains NIG distributions over prediction targets. The natural extension is to maintain NIG distributions over each component of P. This requires no new inference machinery — only seven additional NIG states per layer or system being monitored.

### 4.2 The NIG Update

For each statistic sᵢ ∈ P, at each timestep t:

**Prior:**
```
sᵢ ~ NIG(μᵢ, λᵢ, αᵢ, βᵢ)
```

**Observation:** compute sᵢ(t) from activations.

**Conjugate posterior update:**
```
μᵢ' = (λᵢμᵢ + sᵢ(t)) / (λᵢ + 1)
λᵢ' = λᵢ + 1
αᵢ' = αᵢ + 1/2
βᵢ' = βᵢ + λᵢ(sᵢ(t) − μᵢ)² / (2(λᵢ + 1))
```

**Outputs:**
```
Point estimate:    μᵢ'
Epistemic var:     σ²_e = βᵢ' / (αᵢ'(λᵢ' − 1))    [from NIG marginal]
Aleatoric var:     σ²_a = βᵢ' / (αᵢ' − 1)
```

### 4.3 The Profile Matrix

At any timestep, the full system state is represented as a 7×3 matrix:

```
        point_estimate    epistemic_var    aleatoric_var
α       μ₁               σ²_e1            σ²_a1
D_eff   μ₂               σ²_e2            σ²_a2
σ_br    μ₃               σ²_e3            σ²_a3
τ       μ₄               σ²_e4            σ²_a4
r_eu    μ₅               σ²_e5            σ²_a5
L       μ₆               σ²_e6            σ²_a6
C       μ₇               σ²_e7            σ²_a7
```

Twenty-one numbers. This is the complete real-time intelligence state of any system CyphaDIF monitors.

### 4.4 The Health Signal

The deviation of P from its learned baseline constitutes an intelligence health signal:

```
Δ(t) = ||P(t) − P̄||_Σ⁻¹
```

where P̄ is the running mean and Σ is the covariance of P under normal operation. This is a Mahalanobis distance in profile space. Spikes in Δ(t) indicate:

- **Gradual drift:** distribution shift, concept drift, or legitimate learning
- **Sharp discontinuity:** adversarial perturbation or system fault
- **Structured drift (one axis):** specific capability degradation

The NIG representation of each statistic provides uncertainty bounds on Δ(t), so the health signal is itself calibrated — an alarm is raised only when the deviation is statistically significant given current epistemic uncertainty.

---

## 5. Izaac-Intrinsic Statistics

### 5.1 Algorithm-Intrinsic Statistics

Beyond universal statistics applicable to any intelligent system, it is possible to develop statistics that are intrinsic to a specific algorithm — operators that derive their meaning from the algebraic structure of the algorithm itself rather than from general information theory. For algorithms with strong mathematical structure, these intrinsic statistics may be more sensitive and more interpretable than universal ones.

Izaac, as a deterministic randomness algorithm with VRF structure operating over GF(2ⁿ), admits a rich family of such statistics.

### 5.2 Proposed Izaac-Intrinsic Statistics

**Cycle distribution statistic (κ):**
```
κ = H(cycle_lengths) / log(max_cycle_length)
```
Measures the entropy of the cycle length distribution over the output space. A good VRF should have κ near 1 — uniformly distributed cycle lengths. Deviation indicates structural bias in the algebraic mapping.

**Mixing depth (m):**
```
m = min{k : I(Y_{t+k} ; Y_t) < ε}
```
The number of steps before outputs become statistically independent. Analogous to τ but defined in terms of Izaac's internal state transitions rather than external data.

**Monomial complexity grade (α_Izaac):**
```
α_Izaac = GRIA α applied to Izaac output sequences
```
This directly connects Izaac to GRIA. The output sequences of a well-designed VRF should sit near α ≈ 0.5 — compressible enough to be structured, irreversible enough to be secure.

**Cross-application correlation (ρ_cross):**
```
ρ_cross = max_{i≠j} |corr(Application_i output, Application_j output)|
```
Measures whether Izaac's twelve applications produce correlated outputs under any shared input conditions. Should be near zero for a well-designed system; deviations reveal algebraic dependencies.

**Adversarial uniformity deviation (δ_adv):**
```
δ_adv = max_{adversarial input set} KL(Izaac output || Uniform)
```
Measures how much the output distribution deviates from uniform under structured adversarial inputs. The natural null distribution is known analytically for a true VRF, making this testable against an exact benchmark.

### 5.3 Integration with the Profile Vector

The Izaac-intrinsic statistics extend the universal profile vector when Izaac is present in the stack:

```
P_Izaac = P ∪ (κ, m, α_Izaac, ρ_cross, δ_adv)
```

Each of these five statistics is also tracked as a NIG distribution in CyphaDIF, giving full uncertainty quantification over the cryptographic and randomness properties of Izaac in real time.

---

## 6. Brain-Derived Statistics

### 6.1 Motivation

The biological brain is the existence proof that the profile vector is achievable in a physical system. By examining what statistics the brain implicitly optimises, we gain both validation of the universal framework and additional candidate statistics that may be computationally efficient to implement.

### 6.2 Predictive Coding Statistics

The brain is widely theorised to operate via predictive coding — minimising prediction error across hierarchical levels. This suggests statistics based on prediction error structure:

**Precision-weighted surprise (S_pw):**
```
S_pw = E[precision(x) × ||x − x̂||²]
```
where x̂ is the predicted input and precision is the inverse variance of the prediction. This is the quantity minimised by predictive coding and measures how efficiently the system uses its uncertainty estimates.

**Hierarchical compression ratio (R_h):**
```
R_h = H(input layer) / H(deepest layer)
```
Measured as effective dimensionality ratio across the full hierarchy. Brains achieve compression ratios of order 10⁶ from sensory periphery to prefrontal cortex.

### 6.3 Criticality Statistics

Neural systems operate near phase transitions. Statistics that measure proximity to criticality complement the universal α:

**Avalanche exponent (β_av):**
```
P(avalanche size = s) ∝ s^{-β_av}
```
Critical systems have β_av ≈ 1.5. Estimable from activation cascade sizes across layers.

**Dynamic range (Δ):**
```
Δ = 10 log₁₀(r_max / r_min)
```
where r_max and r_min are the maximum and minimum stimulus intensities that produce discriminably different responses. Critical systems maximise dynamic range. Measurable from response curves.

### 6.4 Efficiency Statistics

**Metabolic efficiency proxy (η):**
```
η = task performance / mean activation magnitude
```
Brains minimise metabolic cost per unit of useful computation. In artificial systems, activation magnitude proxies for computational cost. High η indicates efficient use of representational capacity.

**Novelty-weighted entropy (H_n):**
```
H_n = −Σ p(x) × novelty(x) × log p(x)
```
where novelty(x) is the inverse frequency of x in recent history. This weights the entropy measure toward rare, attention-worthy patterns — what the brain attends to rather than the average information content.

---

## 7. Experimental Design

### 7.1 Experiment Architecture

The following experiment suite characterises the full profile vector for any target system using GPU-parallelised computation.

**Phase 1: Synthetic calibration**

For each statistic sᵢ, construct synthetic datasets with known ground-truth values of sᵢ. Sweep across the full range [0,1]. Verify that the measurement operator recovers the known values. This establishes the empirical sampling distribution of each operator and validates its sensitivity and specificity.

**Phase 2: Cross-statistic correlation analysis**

Run all seven statistics simultaneously on a large corpus of synthetic systems. Compute the full 7×7 correlation matrix. Off-diagonal entries should be near zero for independent statistics. Any significant correlations indicate either a genuine constraint (some combinations of statistics are physically impossible) or a methodological confound in the measurement operators.

**Phase 3: Real system characterisation**

Apply the full suite to:
- CyphaDIF itself, monitored during training
- Izaac output sequences across all twelve applications
- Each layer of a standard transformer (Qwen 3.5 8B via Ollama, accessible locally)
- Randomly initialised networks as null distributions

**Phase 4: Online integration**

Implement the 7×3 NIG profile matrix as a CyphaDIF module. Validate that the online updates converge to the batch estimates from Phase 3.

### 7.2 Compute Requirements

| Experiment | Compute | Time estimate (RTX 3090) |
|---|---|---|
| Synthetic calibration (all 7) | ~10⁶ samples × 7 operators | 2-4 hours |
| Cross-correlation analysis | 10⁴ systems × 7 statistics | 1-2 hours |
| Transformer layer profiling | 1 pass per layer, 32 layers | 30 minutes |
| Izaac intrinsic statistics | 10⁸ output samples | 1 hour |
| Online integration validation | 10⁵ timesteps | 30 minutes |

Total: approximately 6-8 GPU-hours. Runnable as a single overnight job.

### 7.3 Implementation Notes

```python
# Core profile vector computation
class IntelligenceProfiler:
    def __init__(self):
        self.nig_states = [NIGState() for _ in range(7)]
    
    def compute_profile(self, activations, targets=None):
        stats = [
            compute_alpha(activations),           # GRIA α
            compute_participation_ratio(activations),  # D_eff
            compute_branching_ratio(activations),      # σ_branch
            compute_memory_depth(activations),         # τ
            compute_eu_ratio(activations),             # r_eu
            compute_lipschitz(activations),            # L
            compute_calibration(activations, targets)  # C
        ]
        return stats
    
    def update(self, activations, targets=None):
        stats = self.compute_profile(activations, targets)
        for i, (s, nig) in enumerate(zip(stats, self.nig_states)):
            nig.update(s)
        return self.get_profile_matrix()
    
    def get_profile_matrix(self):
        # Returns 7×3 matrix: [point_estimate, epistemic_var, aleatoric_var]
        return np.array([nig.get_estimates() for nig in self.nig_states])
    
    def health_signal(self):
        P = self.get_profile_matrix()[:, 0]  # point estimates
        P_bar = np.array([nig.mean for nig in self.nig_states])
        Sigma = np.diag([nig.variance for nig in self.nig_states])
        delta = P - P_bar
        return float(delta @ np.linalg.inv(Sigma) @ delta)
```

---

## 8. Theoretical Implications

### 8.1 Intelligence as a Phase Constraint

The seven-dimensional profile space P has a natural measure: the volume of the region near the critical manifold of each axis. We conjecture that:

**Conjecture 1:** Biological intelligence occupies a region of P-space that is simultaneously near-critical on all seven axes. This co-criticality is not achievable by gradient descent alone and requires developmental processes that enforce the joint constraint.

**Conjecture 2:** The difficulty of achieving artificial general intelligence is, in part, the difficulty of enforcing the joint criticality constraint. Current training procedures optimise single objectives (task loss) and achieve near-criticality on some axes while degrading on others.

### 8.2 GRIA as the Master Statistic

The GRIA α is the deepest of the seven statistics because it is defined at the level of information transformation rather than representation, dynamics, or self-knowledge. It is arguable that the other six statistics are derived consequences of α operating in different subspaces:

- D_eff is α in representational geometry space
- σ_branch is α in dynamical flow space
- τ is α in temporal dependency space
- r_eu is α in epistemic state space
- L is α in input-output topology space
- C is α in self-model accuracy space

If this reduction is correct, then GRIA α is not merely the first statistic but the fundamental one — the universal order parameter of which all others are projections.

### 8.3 CyphaDIF as an Intelligence Measurement Theory

By embedding the seven-statistic profile vector into CyphaDIF's NIG inference framework, CyphaDIF becomes not merely an uncertainty quantification system but a theory of intelligence measurement. It provides:

- A complete characterisation of any intelligent system in 21 numbers
- Full uncertainty quantification over that characterisation
- An online update algorithm that tracks the system's intelligence state in real time
- A health signal that detects degradation, adversarial perturbation, and distributional shift
- A principled basis for comparing intelligence across systems of different types and scales

This constitutes a novel contribution at the intersection of statistical learning theory, information theory, and the science of intelligence.

---

## 9. Conclusion

We have derived a minimal complete set of seven universal statistics for intelligent systems from a single meta-principle. The statistics are orthogonal, computable from activation data, and naturally expressible as online Bayesian updates within the CyphaDIF NIG framework. The resulting 7×3 profile matrix provides a comprehensive real-time characterisation of any intelligent system in 21 numbers.

The framework extends naturally to algorithm-intrinsic statistics (demonstrated for Izaac) and to statistics derived from the structure of biological intelligence. The experimental programme is tractable on a single RTX 3090 in one overnight session.

The deepest theoretical result is that the GRIA grade parameter α appears to be the master statistic of which all six remaining statistics are projections into different subspaces. This suggests that GRIA is not merely a tool for sequence and compression analysis but a foundational framework for the science of intelligence itself.

---

## References

*This paper draws on the following frameworks developed by the author:*

- GRIA (Graded Reversible-Irreversible Algebra) — universal compression grade parameter
- CyphaDIF — online Normal-Inverse-Gaussian inference for epistemic uncertainty quantification
- Izaac — deterministic randomness algorithm with VRF and cryptographic applications

*External theoretical grounding:*

- Friston, K. — Free energy principle and predictive coding
- Beggs, J. & Plenz, D. — Neuronal avalanches and criticality
- Langton, C. — Computation at the edge of chaos
- Odrzywołek, A. (2026) — EML Sheffer operator and eml-complete activation analysis
