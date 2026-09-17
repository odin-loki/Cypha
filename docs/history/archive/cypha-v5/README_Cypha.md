Cypha.py
Technical Reference & Mathematical Introduction
File 1 of 5  ·  3,361 lines  ·  Core engine
For someone reading this for the first time

# 1. What Is This File?
Cypha.py is the entire brain of the system. Everything else — download.py, convert.py, benchmark.py, synthetic_benchmark.py — is scaffolding that feeds data into this one file. It is 3,361 lines of pure NumPy and contains:

A universal signal encoder that converts any input — SQL strings, RF radio signals, audio recordings, malware feature vectors — into a single shared vector space.
A physics-inspired neural field (the HRNA hierarchy) that processes those vectors through five stacked dynamical systems.
A prototype memory that learns by storing examples, not by adjusting weights.
A reasoning layer that detects when it is uncertain and runs a second-pass query revision to resolve ambiguous cases.

There are no learned weight matrices. There is no gradient descent. The system learns by accumulating labelled prototype vectors in memory and classifying by asking: "which stored prototype is most similar to this new input?"

| The key insight If you can encode every signal domain into the same vector space using the same mathematical operators, then a single memory-based nearest-neighbour classifier works across all of them simultaneously — with no retraining when you add a new domain. |
|---|

# 2. Architecture Overview
Every call — whether training or inference — passes through four stages in order:

| INPUT STRING (text / "iq:" hex / "pcm:" hex / "arr:" base64) │ ▼ Stage 1: Encode OmegaEncoder.encode_features() → v ∈ ℝ⁵¹² (real, L2-normalised) │ ▼ Stage 2: Project PhaseBridge.bridge() → ψ ∈ ℂ²⁵⁶ (complex, unit-sphere) │ ▼ Stage 3: Resonate (5 hierarchical levels) ResonanceField → ResonatorLevel → AssemblyLevel → ModuleLevel → GlobalLevel │ state ∈ ℂ²⁵⁶ ▼ Stage 4: Classify AnchorMemory.lookup() → k nearest prototypes by cosine similarity ThoughtProcessor → uncertainty estimate + optional query revision │ ▼ OUTPUT: (class_label, confidence) |
|---|

The rest of this document works through each stage in detail.

# 3. Stage 1 — The Omega Encoder
## 3.1 The Problem It Solves
Consider three inputs: a SQL injection string, an FM radio signal captured as int8 IQ samples, and a WAV recording of someone saying "yes". A conventional model trained on one cannot process the others. The byte histograms of all three look different in ways that do not generalise.
The Omega encoder's job is to compute features that mean the same thing regardless of the signal's origin — features like "how bursty is the rate of change?" and "where is the energy concentrated in frequency space?" These questions have well-defined answers for any 1D real signal, whether it is text, radio, or audio.
## 3.2 Definition of the Omega Operator
Let x ∈ ℝⁿ be a 1D real signal of arbitrary length n. The Omega operator is the concatenation of five feature families:

Ω(x)  =  concat[ M(x),  M(D(x)),  M(D²(x)),  R(x,K),  A(x,L) ]

Each component is defined below.
### Component 1 — M(x): Raw Moments
The moment vector extracts four statistics describing the amplitude distribution:
M(x)  =  [ μ(x),  σ(x),  κ(x),  γ(x) ]
where μ = mean, σ = standard deviation, κ = excess kurtosis (how heavy the tails are), γ = skewness (asymmetry of the distribution). These four numbers summarise the shape of the amplitude histogram. They are computed in a single-pass BLAS dot-product formulation (23× faster than four separate numpy.mean() calls).
### Component 2 — M(D(x)): Derivative Moments
D(x) is the first difference of the signal:
D(x)[i]  =  x[i+1] − x[i]   for  i = 0, …, n−2
Then M(D(x)) applies the same four-moment extraction to this derivative sequence.
The most important single feature in the entire system is κ(D(x)) — the kurtosis of the first derivative. It measures how bursty the signal changes are. A SQL injection string like ' OR 1=1 -- has sudden large ASCII value jumps (e.g., 39 → 32 → 79 → 82) that produce high kurtosis. A clean SQL query has smoother, lower-variance transitions. Empirically, κ(D(x)) alone achieves r = 0.9985 correlation with the true class boundary density across all signal domains.

| Why is κ(D(x)) so powerful? It measures burstiness independent of scale, offset, and domain. A phishing email has sudden all-caps words and suspicious punctuation. A malware PE feature vector has sudden spikes at specific feature indices. An FM signal has a different derivative autocorrelation signature than AM. All of these differences show up in κ(D(x)) without any domain-specific feature engineering. |
|---|

### Component 3 — M(D²(x)): Second Derivative Moments
The second derivative D²(x) = D(D(x)) captures acceleration — how fast the rate of change itself is changing. M(D²(x)) is the moment vector of this quantity. It is particularly useful for phase signals (equivalent to the c40 cumulant in communications theory) and for detecting abrupt inflection points in audio.
### Component 4 — R(x, K): Spectral Band Energy
Compute the FFT of x, divide the frequency axis into K = 16 equal-width bins, and measure the L1-normalised energy in each bin:
R(x, K)[k]  =  Σᵢ ∈ Bₖ |FFT(x)[i]|  /  ‖FFT(x)‖₁      k = 0,…,15
This gives a 16-dimensional spectral fingerprint. An AM radio signal concentrates energy near the carrier and its two symmetric sidebands. White noise spreads energy uniformly. A voiced speech segment has harmonic peaks at integer multiples of the fundamental. These signatures are reliable and stable across different instances of the same class.
### Component 5 — A(x, L): Autocorrelation at Log-Spaced Lags
A(x, L)[l]  =  Σᵢ x[i] · x[i+l]  /  n      for  l ∈ L = {1, 2, 4, 8, 16, 32, 64, 128}
Autocorrelation at lag l measures how similar the signal is to a time-shifted version of itself. High autocorrelation at short lags = locally smooth. High autocorrelation at a specific lag = periodic at that period. Log-spaced lags cover multiple time scales efficiently with L = 8 values.
## 3.3 Three-Scale Application
All five operators are applied three times: once over the full signal, once over the first half, and once over the second half. This gives the encoder access to temporal evolution — how the signal changes from beginning to end.
Ω₃(x)  =  concat[ Ω(x),  Ω(x[: n/2]),  Ω(x[n/2 :]) ]
The resulting feature vector has dimension 3 × (4 + 4 + 4 + 16 + 8) = 3 × 36 = 108 named statistical features, plus 256 byte-level features and 8 token-structure features for text inputs. All names are unique and deterministic.
## 3.4 Numeric-Direct Embedding
The Ω₃(x) output is a dict mapping feature names (e.g., "full_d1_kurt", "h2_band7", "byte65") to float values. This must be converted to a fixed-length vector v ∈ ℝ⁵¹². The method is:
v[  abs(hash(name)) mod d  ]  +=  value      then  v ← v / ‖v‖
No learned projection. No optimisation. The hash function is deterministic: the same feature name always maps to the same index. The output dimension d = 512 was chosen because ~143 active features occupy 512 dimensions with 28% utilisation — well below the collision pressure threshold. Empirically, cosine similarity in this space correlates reliably with feature-profile similarity: same-class inputs cluster at cosine similarity 0.80–0.95.

| Why not a learned projection? A learned embedding (e.g., a linear layer trained by gradient descent) requires seeing all input domains up front. The hash embedding works immediately on any new domain — you just start feeding new data in. The metric structure is preserved by the hash distribution, no training required. |
|---|

## 3.5 Signal Routing
The encoder checks the input prefix to select the correct decoding path:

| Prefix | Path | Decoding method |
|---|---|---|
| (plain text) | Text path | UTF-8 bytes centred to [-1,1]; Ω₃ + byte histogram + token stats |
| "iq:" | IQ/RF path | Reinterpret as int8 I/Q pairs → complex64 → 512-pt STFT power spectral density |
| "pcm:" | Audio path | Reinterpret as int16 PCM → float → mel filterbank (26 bands, 512-pt FFT) |
| "arr:" | Array path | base64 → float32 array → Ω₃ directly |

The IQ and PCM paths exist because raw-byte Omega applied to RF signals gives cosine similarity ≈ 0.99 between all RF classes — every class looks like white noise at the byte level. The spectral paths extract the actual modulation fingerprint. This was discovered empirically and the fix is documented in the source at line 422.

# 4. Stage 2 — PhaseBridge
The Omega encoder outputs a real vector v ∈ ℝ⁵¹². The resonance field (next stage) operates on complex vectors ψ ∈ ℂ²⁵⁶. PhaseBridge performs this promotion.
## 4.1 Construction
Two random matrices Wₐ, W_φ ∈ ℝ⁵¹²ˣ²⁵⁶ and a frequency vector b ∈ ℝ²⁵⁶ are initialised once at construction time with a fixed random seed and never updated:
amps  =  v Wₐ   ∈ ℝ²⁵⁶
phase  =  arctan2(‖v[256:]‖, ‖v[:256]‖)  +  0.3 · (v W_φ)   ∈ ℝ²⁵⁶
basis[k]  =  sin(b[k] · k / 256)   ∈ ℝ²⁵⁶
ψ  =  amps ⊙ exp(i · phase) ⊙ basis  /  ‖ · ‖
The amplitude component carries the magnitude information from the Omega features. The phase component encodes the geometric orientation of the input vector. The basis introduces a fixed frequency structure that helps the downstream Hamiltonian evolution discriminate between inputs with similar amplitudes but different phase structures.
The matrices are float32 (halving RAM vs float64). The output ψ is unit-normalised in the complex L2 norm. Identical inputs always produce identical ψ — the bridge is a pure function with no state.

# 5. Stage 3 — The HRNA Hierarchy
## 5.1 Why a Dynamical System?
A standard multi-layer perceptron processes input in one forward pass: multiply by weight matrix, apply activation function, repeat. Cypha instead drives a dynamical system: it injects the encoded input into a complex-valued field and lets that field evolve under a physics-inspired equation. The state that emerges after several evolution steps carries both the content of the input and its resonance with the field's structure.
The key difference: the HRNA hierarchy has internal dynamics that interact with the input across multiple timescales, not just a single feedforward transformation. This is more similar to how recurrent networks process sequences, except here the "sequence" is the repeated injection+evolution loop over the same input.
## 5.2 Level 1 — ResonanceField
### State
The field state is a complex vector ψ(t) ∈ ℂ²⁵⁶. Each element is a complex oscillator. The system is reset to a fresh random initial state before every training step and every inference call, so there is no persistent state between different inputs.
### Injection
Before each evolution step, the input encoding is mixed into the field:
ψ  ←  (1 − s) · ψ  +  s · enc(input)
ψ  ←  ψ / ‖ψ‖
with injection strength s = 0.25. This nudges the field toward the input without overwriting it — the field's own dynamics partially resist the injection, which is what produces the nonlinear interaction.
### Evolution
Each evolution step applies a Hamiltonian operator in frequency space, then a nonlinear self-interaction:
ψ_H  =  IFFT( FFT(ψ) ⊙ exp(−i Δt · H) )
ψ(t+1)  =  N[ ψ_H · exp(−i Δt · γ · (|ψ_H|² − 1) · Re(ψ_H)) ]
where H[k] = 0.5 + k·9.5/256 (linearly-spaced frequencies from 0.5 to 10), Δt = 0.3, γ = 5, and N[·] denotes L2 normalisation.
The first line (Hamiltonian step) is the discrete-time analogue of the Schrödinger equation — it rotates each frequency component by a phase proportional to its frequency, analogous to free quantum evolution. The second line (nonlinear term) pushes the field back toward the unit sphere when |ψ|² deviates from 1, while introducing the nonlinear coupling that separates inputs which are close in linear space.
During training, 3 inject+evolve loops are run (fast, approximate equilibrium). During inference, 6 loops are run (higher quality, needed for the downstream ThoughtProcessor).
### Criticality
After evolution, the field reports its criticality — a scalar measuring energy concentration:
κ  =  (Σᵢ ∈ Top₁₀ |ψ[i]| / Σ |ψ[i]|) · Var(|ψ|) · 100
Low κ = energy spread uniformly (disordered). High κ = energy concentrated in a few modes (ordered). The AdaptiveControlLoop uses κ to tune injection parameters in real time.
## 5.3 Levels 2–5
Four more levels sit above the base field. Each takes the state from the level below, applies its own dynamics, and passes an updated state upward:

| Level | Class | Key operation |
|---|---|---|
| L1 — Field | ResonanceField | FFT Hamiltonian + γ(|ψ|²−1) nonlinear self-interaction |
| L2 — Resonator | ResonatorLevel | Local coupling: ψ[i] += γ · Σⱼ ∈ 𝒩(i) ψ[j]; lateral inhibition |
| L3 — Assembly | AssemblyLevel | Resonant chain across 16 sub-fields; modulate() shifts phase per event |
| L4 — Module | ModuleLevel | Integrates 8 assemblies; produces a compressed global feature vector |
| L5 — Global | GlobalLevel | Final state readout, dim=256; feeds AnchorMemory and ThoughtProcessor |

All five levels reset between samples. The hierarchy increases the effective "receptive field" of the nonlinear dynamics — the same way stacking LSTM layers increases temporal range. The 3 ms per inference call measured in profiling is almost entirely spent in these five levels (specifically in the enhanced_resonance() calls that fire across ~18 events per forward pass).

# 6. Stage 4a — AnchorMemory
## 6.1 The Classification Approach
Cypha does not classify by passing the field state through a softmax layer. It classifies by storing labelled prototype vectors called anchors and asking: "which stored class prototype is most similar to this new input?" This is the Nearest Class Prototype approach — every anchor is a labelled point on the unit sphere, and classification is nearest-neighbour lookup.
## 6.2 Cosine Similarity and the Unit Sphere
Every anchor a ∈ ℝᵈ is stored L2-normalised. All distances are cosine similarities:
sim(u, v)  =  u · v   (since ‖u‖ = ‖v‖ = 1)
All n stored anchors are stacked as rows of a matrix V ∈ ℝⁿˣᵈ. Batch lookup for a query q is a single matrix-vector product:
sims  =  V q   ∈ ℝⁿ
O(n · d) floating-point operations, executed by BLAS. Profiling shows this is bandwidth-bound at d = 512, so lookup costs approximately 10 μs flat from n = 12 to n = 10,000 anchors — it barely scales with anchor count.
## 6.3 Storing a New Sample — Three Paths
When train_step() calls memory.store(key, v, label), one of three things happens:

#### Path 1 — Key already exists (EMA update)
If this exact input string has been stored before, update its vector with an exponential moving average:
a_new  =  N[ (1 − α) · a_old + α · v ]
where α ∈ [0.15, 0.40] is the EMA learning rate set by ThoughtProcessor based on current uncertainty. This refines the prototype toward the new observation without overwriting it. Dictionary lookup: O(1) via the _key_to_gi dict (fixed in Feb 2026 — was O(n) list.index() before).
#### Path 2 — Near-duplicate exists (dedup EMA update)
If a same-class anchor already has cosine similarity ≥ τ_dedup = 0.55 to this new vector, update that anchor instead of creating a new one:
IF  max_{a ∈ cls} sim(a, v) ≥ 0.55   THEN  a ← N[ (1−α)·a + α·v ]
This keeps the anchor set lean. τ_dedup = 0.55 is the empirically optimal threshold: accuracy is flat from 0.55 to 0.96; below 0.55, anchors accumulate faster than consolidation can remove them.
#### Path 3 — New anchor
Otherwise, add v as a new anchor. The matrix V is extended by one row (amortised O(n) vstack), class counts and index caches are updated in O(1).
## 6.4 LVQ2.1 Boundary Sharpening
After every store(), the system checks whether this input falls inside the LVQ window — i.e., whether the nearest correct-class anchor and nearest wrong-class anchor are nearly equidistant:
CONDITION:   lo / hi  >  1 − θ_w      where  lo = min(sᶜ, sʷ),  hi = max(sᶜ, sʷ),  θ_w = 0.30
If the condition fires, the two boundary anchors are nudged apart:
wᶜ ←  N[ wᶜ + η·(v − wᶜ) ]   (pull correct-class prototype closer to v)
wʷ ←  N[ wʷ − η·(v − wʷ) ]   (push wrong-class prototype away from v)
with η = 0.02. This is the LVQ2.1 rule (Kohonen 1990). It sharpens decision boundaries at exactly the regions where the model is currently confused, without touching anchors that are already well-separated.
## 6.5 Adaptive Per-Class Cap
Each class gets its own anchor count ceiling, estimated from the complexity of its current prototype distribution:
cap  =  clamp( 10 + 200·spread + 30·id_est − 100·spec_gap,  lo=10,  hi=500 )
where spread = mean pairwise cosine distance (class diffuseness), id_est = TwoNN intrinsic dimensionality (manifold complexity), spec_gap = normalised spectral gap of the similarity matrix (unimodality). A tight unimodal cluster gets cap ≈ 50. A diffuse multimodal class gets cap ≈ 200–400. This was profiled to add only 0.002 ms/step amortised.
## 6.6 Consolidation
Every 200 training steps, a consolidation pass merges redundant anchors within each class. The algorithm is greedy pivot merge: iterate through anchors; any anchor with cosine similarity ≥ 0.55 to the current pivot gets absorbed into a centroid, which replaces the group. Cost at 3,000 anchors: 0.7 ms per pass, 0.004 ms amortised per step.

# 7. Stage 4b — ThoughtProcessor
After the anchor lookup returns the top-k matches, ThoughtProcessor decides whether to accept the result or run a second-pass query revision. It is the system's uncertainty-aware reasoning layer.
## 7.1 Calibrated Uncertainty
Given the top-1 and top-2 cosine similarities s₁ and s₂, the margin is m = s₁ − s₂. The uncertainty estimate is:
u  =  exp(−m / τ)
where τ is a rolling p75 of observed margins, updated every call. τ auto-calibrates to the difficulty of the current corpus — on an easy corpus (large margins), τ grows and u stays low even at modest margins; on a hard corpus, τ shrinks and u becomes sensitive to small margin differences. At m >> τ, u → 0 (certain). At m = 0, u = 1 (maximum uncertainty).
The suggested EMA alpha returned to the memory is α = 0.15 + 0.25 · u. Samples near the decision boundary (high u) leave a stronger imprint on the prototype.
## 7.2 Rocchio Deliberation
When u > 0.4 AND each competing class has ≥ 8 anchors (density guard), the ThoughtProcessor runs Rocchio query revision. Given the two top competing classes A and B with centroids cₐ and c_b:
q_A  =  N[ q + β·cₐ − β·c_b ]     for  β ∈ {0.5, 1.0}
q_B  =  N[ q + β·c_b − β·cₐ ]     for  β ∈ {0.5, 1.0}
Four revised queries are generated (2 classes × 2 strengths). Each is looked up in the anchor memory. The class whose revised query achieves the largest margin wins:
class*  =  argmax_{X ∈ {A,B}, β} [ top1_sim(lookup(q_X)) − top2_sim(lookup(q_X)) ]
Geometrically: "if I bias my query toward class A's centroid and away from B's, does the neighbourhood become cleaner?" The Rocchio rule (Rocchio 1971) was proven optimal for this relevance feedback formulation. Here it fires only when the first-pass result was genuinely ambiguous.

| Cost of deliberation 4 additional lookups × ~10 μs each ≈ 55–70 μs overhead, flat regardless of anchor count (profiled Feb 2026). The density guard (min 8 anchors per class) prevents it firing on sparse early-training classes. In practice it fires on a minority of inputs. |
|---|

## 7.3 Confusion Memory
Every deliberation result updates a running score for the class pair (A, B):
conf(A,B)  ←  0.9 · conf(A,B) + 0.1 · confusion_signal
When conf(A,B) exceeds a threshold, the boundary is flagged as ambiguous and the deduplication threshold is dynamically lowered for that specific pair, forcing finer-grained prototype placement at the confusion region.

# 8. Training
## 8.1 train_step(input, label)
A single training step runs the following sequence:

| 1. Reset all 5 HRNA levels. 2. forward(input, training=True) → state_input (3 inject+evolve loops) 3. forward(label, training=True) → state_target (3 inject+evolve loops) 4. Mine hard negatives: 3 closest wrong-class anchors via lookup(). 5. Compute contrastive loss: L(state_input, state_target, hard_negatives). 6. lookup(encode_features(input), k=2) → top-2 candidates. 7. ThoughtProcessor.note_uncertainty(margin) → EMA alpha α. 8. memory.store(input, encode_features(input), label, ema_alpha=α). |
|---|

## 8.2 The Contrastive Loss
MetaLearning.loss() computes a contrastive loss between the input state and target state, with penalties for similarity to hard negatives:
L  =  (pos + 2 · neg) · boost
pos  =  ‖N(state_in) − N(state_tgt)‖²
neg  =  mean_{nᵢ} max(0, sim(state_in, nᵢ) + 0.1)²
boost  =  1 + mean_sim_to_recent_states   ∈ [1.0, 2.0]
The novelty boost (1 + sim_to_recent) gives more gradient to inputs similar to recently-seen examples — the hard repetitions the model needs to cement — not less. This was the inverse of the original formula.
The loss is used exclusively to drive the AdaptiveControlLoop parameters (injection strength, chunk_k, active_scales). It does not update any weight matrix. The only actual learning is in the anchor memory.
## 8.3 Hard Negative Mining
Hard negatives are the 3 closest wrong-class anchors to the current input. These are the actual confusions the model currently has — more informative than random window negatives. They are encoded via bridge-only (skipping the 6-loop HRNA pass) because only their direction in the state space matters for the contrastive margin, not their equilibrium resonance state.
## 8.4 CyphaStateful and Checkpointing
CyphaStateful wraps Cypha with two capabilities:
Byte-offset streaming: the dataset file is indexed once at startup (~8 bytes per line), then training seeks to random sample positions without loading the file into RAM. A 5 GB RF dataset trains on a 4 GB machine.
Checkpoint save/resume: after each epoch, all anchor vectors and labels are saved to a NumPy .npz archive plus a JSON metadata file. On resume, the full memory is reconstructed in a single pass with all O(1) index structures (_key_to_gi, _cls_idx, _class_counts) rebuilt correctly.

# 9. Inference
infer(text) runs the same encode + resonate pipeline as training but with 6 injection loops instead of 3, then classifies via weighted nearest-prototype voting:

| 1. Reset all 5 HRNA levels. 2. forward(text, training=False) (6 inject+evolve loops, higher quality) 3. anchor_q = encode_features(text) (same encoder path as training) 4. matches = memory.lookup(anchor_q, k=1) (k=1 is optimal for clean prototypes) 5. candidates = [(memory.get_output(k), sim) for k, sim in matches] 6. thought.note_uncertainty(margin, candidates) 7. IF uncertain AND density ok: thought.deliberate(anchor_q, candidates) → revised class 8. ELSE: top-1 nearest prototype wins. 9. conf = exp(best_sim / temperature) 10. return (class_label, confidence) |
|---|

The confidence score is exp(s / T) where T = temperature (starts at 1.5, decays 3% every 200 training steps toward 0.8). Higher similarity → higher confidence. The temperature controls how sharply confidence peaks near 1.0.

# 10. Measured Performance (February 2026)
The following table is derived from 9 profiling sessions run against the production-configured system (feature_dim=512, resonance_dim=256). All measurements on a single CPU core.

| Component | Median cost | Scales as | Notes |
|---|---|---|---|
| train_step() — text | 4.8 ms | flat | 208 steps/s; stable across all epochs |
| encode_features — text | 260 μs | O(1) | 57% in _omega_at_scale × 3 passes |
| encode_features — IQ/RF | 1,604 μs | O(1) | 5.7× text; FFT spectral extraction |
| encode_features — PCM/audio | 1,418 μs | O(1) | 5.0× text; mel filterbank |
| memory.store() EMA path | 10 μs | O(1) | _key_to_gi dict lookup |
| memory.lookup() dim=512 | 10 μs | O(n⁰·⁰¹) | BLAS sgemv, bandwidth-bound, flat to 10k |
| deliberate() fast exit | 0.5 μs | O(1) | density guard / window skip |
| deliberate() Rocchio | 65–70 μs | flat to 10k | 4 × lookup + 5 μs centroid math |
| consolidate() at 3k anchors | 0.7 ms | O(n log n) | every 200 steps; 0.004 ms/step amortised |
| _compute_class_cap() | 0.05 ms | O(n^1.14) | every 20 new anchors; 0.002 ms/step amortised |
| Field simulation per infer() | ~3 ms | O(1) | 18 events × enhanced_resonance(); dominant cost |

# 11. Full Class Inventory
All 20+ classes in Cypha.py, in order of first appearance:

| EncoderParams | Dataclass: chunk_k, damr_radius, active_scales, prev_error. Passed through AdaptiveControlLoop. |
|---|---|
| FieldStats | Dataclass: criticality, dominant_freq, mean_phase, phase_spread, energy. Returned by field.stats(). |
| Event | Dataclass: type, time, data, source, priority. Used by EventScheduler. |
| Metrics | Dataclass: step, loss, criticality, chunk_k, damr_r, n_anchors, ms, events. Returned by train_step(). |
| EventType | Enum: PATTERN, SURPRISE, RESONANCE, EXTERNAL, FEEDBACK, THOUGHT. |
| OmegaEncoder | Universal signal encoder. encode_features() dispatches to encode_text / _encode_iq / _encode_audio. |
| PhaseBridge | Projects v ∈ ℝᵈ → ψ ∈ ℂʳ via amplitude + phase + basis. Fixed random weights, never updated. |
| ResonanceField | L1: FFT Hamiltonian + γ(|ψ|²−1) nonlinear evolution. Primary field dynamics. |
| ResonatorLevel | L2: Local coupling between adjacent oscillators + lateral inhibition. |
| AssemblyLevel | L3: 16 sub-fields with resonant chain modulation. |
| ModuleLevel | L4: 8 assemblies compressed to a global feature. |
| GlobalLevel | L5: Final state readout dim=256. |
| EventScheduler | Priority queue for time-ordered event delivery to the field. |
| EventGenerator | Generates HRNA events (PATTERN, SURPRISE, etc.) from field statistics. |
| RecursiveProcessor | Iterative state refinement via IIR filter on psi. |
| FeedbackController | Applies corrective feedback based on MetaLearning loss signal. |
| ThoughtProcessor | Uncertainty estimation, Rocchio deliberation, confusion memory. |
| MetaLearning | Contrastive loss with hard negatives and novelty boost. Drives AdaptiveControlLoop. |
| ModalityDetector | Detects input modality from prefix. Tracks per-modality accuracy. |
| AnchorMemory | Prototype store: _V matrix, _key_to_gi dict, _cls_idx dict. store/lookup/consolidate. |
| AdaptiveControlLoop | Three control laws: chunk_k ← criticality, DAMR radius ← freq, scales ← coherence. |
| SparseComputer | LRU cache for repeated computations (256 slots). |
| WorkStealer | ThreadPoolExecutor wrapper for parallel sub-tasks. |
| PrecisionController | Adaptive float32/float64 switching. |
| Cypha | Main system: assembles all components. train_step(), train(), infer(), encode_features(). |
| CyphaStateful | Wraps Cypha with byte-offset streaming, epoch checkpointing, and dataset state management. |

# 12. Quick-Start Code
## 12.1 Install
| pip install numpy # Only dependency |
|---|

## 12.2 Train and Classify
| from Cypha import Cypha  c = Cypha(feature_dim=512, resonance_dim=256)  # Training data: any list of (input_string, label_string) pairs data = [ ("SELECT id FROM users WHERE active=1", "safe_sql"), ("' OR 1=1 --", "sql_inject"), ("VirtualAllocEx PAGE_EXECUTE_READWRITE", "malware"), ("CreateFile GENERIC_READ OPEN_EXISTING", "safe_api"), ]  # Train for 5 epochs (each call resets the field, stores one anchor) for epoch in range(5): for inp, label in data: c.field.reset(); c.res_level.reset() c.assembly.reset(); c.module.reset(); c.global_l.reset() c.train_step(inp, label)  # Inference result, conf = c.infer("DELETE FROM users WHERE 1=1", verbose=False) print(f"Class: {result} Confidence: {conf:.3f}") |
|---|

## 12.3 With Checkpointing (for large datasets)
| from Cypha import CyphaStateful  # feature_dim=4096 is used in the full benchmark c = CyphaStateful(feature_dim=4096, resonance_dim=256)  # Train on a file in wire format (input|||label per line) # Resumes automatically if a checkpoint exists c.train_file_stateful("sql_injection.txt", dataset_name="sql", epochs=1)  # Inference is identical to basic Cypha result, conf = c.infer("1; DROP TABLE users --", verbose=False) |
|---|

## 12.4 Multi-modal: RF signal classification
| import numpy as np from Cypha import Cypha  c = Cypha()  # RF data: int8 IQ samples encoded as "iq:" hex prefix # In production these come from convert.py -> panoradio_rf.txt raw_iq = np.random.randint(-127, 127, 2048, dtype=np.int8) iq_str = "iq:" + raw_iq.tobytes().hex()  c.train_step(iq_str, "am") # works exactly like text training result, conf = c.infer(iq_str, verbose=False) |
|---|

# 13. Symbol Glossary

| Ω(x) | Omega operator: concat[M(x), M(D(x)), M(D²(x)), R(x,K), A(x,L)] |
|---|---|
| Ω₃(x) | Three-scale Omega: concat[Ω(x), Ω(x[:n/2]), Ω(x[n/2:])] |
| M(x) | Moment vector: [mean, std, excess_kurtosis, skewness] |
| D(x) | First difference: D(x)[i] = x[i+1] − x[i] |
| κ(D(x)) | Kurtosis of first derivative. Primary universal discriminator (r=0.9985). |
| R(x,K) | Spectral band energy: K=16 L1-normalised FFT bins |
| A(x,L) | Autocorrelation at L=8 log-spaced lags |
| v ∈ ℝᵈ | Omega feature vector, d=feature_dim=512, L2-normalised |
| ψ ∈ ℂʳ | Complex field state, r=resonance_dim=256, unit-normalised |
| H[k] | Hamiltonian: H[k] = 0.5 + k·9.5/r (linearly spaced 0.5 to 10) |
| γ | Nonlinear self-interaction coefficient. γ=5 in ResonanceField. |
| Δt | Evolution time step. Δt=0.3. |
| N[·] | L2 normalisation: N[v] = v / ‖v‖ |
| κ_field | Field criticality: top-10 energy concentration × variance × 100 |
| sim(u,v) | Cosine similarity = u·v (both unit vectors) |
| τ_dedup | Dedup threshold: sim ≥ 0.55 → EMA update instead of new anchor |
| α | EMA learning rate: α = 0.15 + 0.25·u ∈ [0.15, 0.40] |
| u | Uncertainty: u = exp(−margin/τ), τ = rolling p75 of margins |
| η | LVQ2.1 learning rate: η=0.02 |
| θ_w | LVQ2.1 window threshold: θ_w=0.30 |
| β | Rocchio bias strength: β ∈ {0.5, 1.0} |
| T | Temperature for confidence: T starts at 1.5, decays to 0.8 |
| ||| | Wire format delimiter between input and label strings |

End of Cypha.py reference.  Next: download.py
