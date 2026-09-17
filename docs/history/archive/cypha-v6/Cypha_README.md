| CYPHA HRNA Harmonic Recursive Neural Architecture A brain-inspired classification and deliberation system Technical Reference February 2026 |
|---|

# 1. What Is Cypha HRNA
Cypha HRNA (Harmonic Recursive Neural Architecture) is a machine intelligence system that classifies complex inputs and makes structured decisions. It is not a conventional neural network — it uses no gradient descent, no backpropagation, and no multi-epoch training. Instead it learns in a single forward pass and makes inferences using a tiered system that escalates from instant memory recall up through increasingly deep deliberation.

The system was designed from first principles, drawing on neuroscience (how the brain actually processes information), information theory (how to extract signal from noise), and several original mathematical frameworks developed specifically for this system. The result is a classifier that operates closer to expert human cognition than to a standard deep learning pipeline.

Cypha is domain-agnostic. The same architecture has been validated on game theory decision-making (chess position evaluation, poker hand strategy, Go move selection), classification tasks across dozens of domains, and is designed to operate in real-time tactical decision support contexts.

| Property | Value |
|---|---|
| Architecture | Resonance field + episodic memory + 3-stage deliberation |
| Training paradigm | Online, single-pass, no backpropagation |
| Feature dimension | 512 |
| Resonance dimension | 256 (complex) |
| Poker decision accuracy | 100.0% (9,995 / 10,000) |
| Chess evaluation accuracy | 99.5% (9,952 / 10,000) |
| Go strategy accuracy | 97.3% (9,730 / 10,000) |
| Hippo fast-path latency | < 2 ms |
| Full deliberation latency | 3.8 ms average |
| Live benchmark | Chess vs Stockfish depth-20 skill-20 |

# 2. Mathematical Foundations
Cypha is built on three original mathematical frameworks developed for this system, plus Marchenko-Pastur random matrix theory for noise filtering.

## 2.1  Izaac Algorithm
Izaac is a framework for deterministic randomness using multi-input Boolean functions. Standard random number generators produce pseudo-random sequences that, under analysis, reveal statistical regularities. Izaac produces noise sequences that are both deterministic (reproducible given the same seed) and cryptographically structured, with provably uniform distribution properties.

The core of Izaac is a 256-bit Transcendental Boolean LCG (Linear Congruential Generator) operating over a space of multi-input Boolean functions. The Boolean functions are chosen such that their output distribution across all possible inputs has specific entropy properties, preventing any statistical bias.

Within Cypha, Izaac is used to generate all perturbation vectors for the PNQ deliberation method, augmentation noise in the DMN consolidation loop, and Levy distributed samples for the domain prior. Because the noise is deterministic, re-running the same inference on the same input produces exactly the same deliberation path, which is essential for reproducibility and security applications.

The Izaac suite covers 12 applications: cryptographic key generation, verifiable random functions (VRFs), Monte Carlo sampling, multi-party computation (MPC), distributed consensus, lazy infinite data structures, fuzzing, trading backtests, network protocol simulation, compression, and the two noise applications used in Cypha.

## 2.2  GRIA  (Graded Reversible-Irreversible Algebra)
GRIA is a unified mathematical framework for compression and cryptography. The central idea is a grade parameter alpha in the range [0, 1] that interpolates smoothly between two operational modes: reversible (alpha = 0, lossless) and irreversible (alpha = 1, destructive). At intermediate values, a GRIA operation is partially reversible — some information is preserved, the rest is committed.

In Cypha, the GRIA grade drives deliberation depth. A low grade (near 0) means the system is confident, the memory representation is clean, and the decision can be made by fast recall. A high grade (near 1) means significant uncertainty — the classification problem requires deep search, and the system should commit substantial resources. The grade computation integrates three types of uncertainty: epistemic (how sparse is the relevant class in memory?), aleatoric (how close are the competing class centroids?), and conflict (how often have these two classes been confused historically?).

GRIA also governs memory commitment: when a new training example arrives, the grade determines whether it modifies an existing anchor (low commitment) or creates a new one (high commitment). This prevents memory collapse in repetitive domains while ensuring novel patterns are preserved.

## 2.3  Cypha HRNA
HRNA (Harmonic Recursive Neural Architecture) is the unified architecture described throughout this document. The 'harmonic' component refers to the complex resonance field in which information evolves — vectors are represented as harmonic oscillators whose phase and amplitude carry representational meaning. The 'recursive' component refers to the ThoughtProcessor, which applies multiple scales and time constants to the same signal simultaneously. The 'neural' component refers to the hippocampal and cortical memory structures that store and retrieve episodes.

The key theoretical claim of HRNA is that classification is better understood as a resonance problem than a regression problem. Rather than learning a function that maps input features to class labels, Cypha learns a manifold of class-specific resonance patterns. Classification is the process of finding which pattern the input resonates with most strongly.

## 2.4  Marchenko-Pastur Filtering
The Marchenko-Pastur law from random matrix theory describes the eigenvalue distribution of a random matrix. When you have a memory store of N vectors each of dimension D, the matrix V^T V has a spectrum. Eigenvalues that fall within the Marchenko-Pastur bulk distribution correspond to noise — they carry no signal. Eigenvalues above the bulk correspond to genuine structure in the data.

Cypha fits the MP distribution to its memory store periodically (every 500 training steps). Any query vector is then projected into the subspace spanned by the signal eigenvalues before being used in similarity lookups. This means class boundary computations are operating on clean signal rather than noise, dramatically improving discrimination in high-dimensional feature space.

# 3. System Architecture
Cypha processes every input through a three-tier pipeline. Tier 1 is instant recall from hippocampal memory. Tier 2 is the resonance field and ThoughtProcessor. Tier 3 is the GRIA deliberation engine with three internal stages. Each tier only activates if the previous tier was insufficient, so computationally simple cases exit immediately while genuinely difficult cases receive full search resources.

| Input: text / binary / array | v OmegaEncoder | Tokenises input, builds TF-IDF weighted feature vector | Applies structural features (for structured domains) | Output: anchor_q (512-dim float32, L2-normalised) | +---> TIER 1: HippoCypha fast-path | Computes cosine similarity to all stored episodes | If max similarity >= adaptive threshold: RETURN immediately | Latency: < 2 ms | +---> TIER 2: Resonance field + ThoughtProcessor | anchor_q passed through HRNA forward() method | Evolves through complex resonance field G (256-dim) | ThoughtProcessor extracts: chain_score, trend_vec, cascade_hints | Neuromodulators computed: DA surprise, NE volatility | Global Workspace ignition test (margin check) | +---> TIER 3: GRIA deliberation engine Grade alpha computed from uncertainty decomposition 3-stage sequential deliberation with parallel methods Output: class label + confidence + deliberation depth |
|---|

# 4. Component Reference
## 4.1  OmegaEncoder
The OmegaEncoder is responsible for converting raw input into the 512-dimensional anchor vector that all subsequent components operate on.

For text input, the encoder tokenises the input string, builds a token frequency map, and weights tokens by TF-IDF. The resulting sparse representation is projected into 512 dimensions using a fixed hashing scheme. For structured domains (network traffic, game positions, financial data), domain-specific feature extractors produce token strings which the encoder then processes identically to text. This design means any domain can be added by writing a feature extractor that maps domain observations to a token string — the rest of the architecture requires no changes.

All output vectors are L2-normalised before being used downstream. This ensures that cosine similarity in the memory store is equivalent to dot product, enabling vectorised BLAS operations for fast lookup.

## 4.2  IzaacNoiseSource
A noise generator based on the Izaac algorithm. Initialised with a 256-bit seed and produces Gaussian noise samples on demand. The key property is that the same seed always produces the same sequence — noise is deterministic and reproducible. Used by PNQ for perturbation vectors and by DMN for augmentation during offline replay.

## 4.3  LevyDomainPrior
Models the statistical distribution of feature-space distances in each domain using a Levy stable distribution. The Levy distribution has heavier tails than Gaussian, which is appropriate for real-world data where outliers are common. When computing deliberation confidence, distances are normalised against the Levy prior for that domain, ensuring that what counts as 'unusual' is calibrated to actual domain statistics rather than assuming Gaussian geometry.

## 4.4  HippoCypha  (Hippocampal Memory)
HippoCypha is Cypha's episodic memory store. It holds all training episodes as unit-normalised vectors with class labels. The name reflects its inspiration from the hippocampus, which in the brain is responsible for rapid episode storage and recall.

The store is backed by a dense float32 matrix that is rebuilt lazily when the stored episodes change. Lookup is a single matrix-vector multiply followed by argmax — O(N*D) where N is the number of stored episodes. For a typical store of 40,000 episodes in 512 dimensions, this takes under 2 ms.

HippoCypha has two retrieval modes. The fast-path returns immediately if the best cosine similarity exceeds an adaptive threshold. The threshold starts at a default (0.6) and is modulated up or down by the resonance field coherence score: when the ThoughtProcessor reports a stable, coherent field, the system trusts hippo more (threshold drops); when the field is noisy, hippo trust is reduced (threshold rises). The soft-score mode returns the best class and similarity regardless of threshold — this is used as a voter in Stage 1 of deliberation even when the fast-path would not have fired.

## 4.5  AnchorMemory and AnchorMemoryAdapter
AnchorMemory is the underlying store for the resonance field's class anchors (prototype vectors). It operates in the same 512-dimensional space as HippoCypha but with additional structure: per-class capacity limits, LVQ2.1 boundary sharpening, vectorised near-duplicate consolidation, and TensorCentroid streaming centroids.

LVQ2.1 boundary sharpening runs on every training step. When a new training example arrives near a class boundary (the similarity difference between the correct and nearest wrong anchor is less than 20%), the nearest correct anchor is nudged slightly toward the example (learning rate 0.01) and the nearest wrong anchor is nudged slightly away. This progressively sharpens class boundaries without requiring a global optimisation pass.

Consolidation runs every configurable number of steps. Within each class, it uses a cosine similarity matrix to identify clusters of near-duplicate anchors (similarity above the consolidation threshold) and replaces each cluster with its centroid. This prevents the store from growing unboundedly and keeps prototypes maximally representative.

AnchorMemoryAdapter wraps AnchorMemory with the Marchenko-Pastur filtered lookup interface used by all deliberation methods. It exposes class_centroid(), density(), hard_negatives(), and mp_lookup() — the last of which projects queries into the MP signal subspace before computing similarities.

## 4.6  ConfusionGraph
The ConfusionGraph tracks confusion statistics for every pair of classes the system has ever confused during deliberation. For each pair (A, B), it stores the number of times A was predicted when B was correct, the number of times the confusion was resolved (and which class won), and an exponential moving average of the pairwise similarity.

This information is used throughout the deliberation engine. During grade computation, a high confusion count between the top-2 candidate classes increases alpha, forcing deeper search. During Stage 2, the confusion weight discounts the confidence of the current prediction if the top-2 candidates are a historically confused pair. Stage 3 uses best_resolution() to identify which class historically wins this specific pair and adds a weighted vote for that class.

The ConfusionGraph also supports a last_successful_beta() method that returns the centroid push strength that historically resolved this pair. This is the beta parameter used by the Rocchio method in Stage 1.

## 4.7  ThoughtProcessor
The ThoughtProcessor operates on the resonance field G, a 64-dimensional complex vector that evolves during the HRNA forward pass. It produces five output signals that are wired into the deliberation and memory systems.

| note_uncertainty | Exponential moving average of uncertainty across recent inferences. Tracks how hard the current domain has been recently. Used to scale the epistemic uncertainty component of the GRIA grade. |
|---|---|
| resonant_chain | Coherence score [0,1] measuring how stable and self-consistent the resonance field is. Computed as the normalised mean of cross-scale correlations in G. A coherent field indicates a well-formed representation. High coherence modulates the hippo threshold downward (more trust). Low coherence does the opposite. |
| self_generate | Detects drift and trend in G across time. Computes a trend vector by comparing the current field to a running exponential average. When a significant trend is detected, a small perturbation (3% magnitude) is added to anchor_q before the deliberation ensemble. This improves discrimination for slowly shifting signal patterns. |
| multi_scale | Blends G at multiple time constants simultaneously: fast (alpha=0.9), medium (alpha=0.5), slow (alpha=0.1). The blended field captures both rapid changes and longer-term structure. The blended representation is what the deliberation ensemble sees as the resonance state. |
| cascade | Hypothesis generation. When uncertainty is high enough, the ThoughtProcessor fires hypothesis events: candidate class vectors projected from resonance space (64-dim) into feature space (512-dim). These become seed vectors for Stage 3 MCTS, giving the tree search informed starting hypotheses rather than a cold start from the query alone. |

The ThoughtProcessor also maintains a _confusion_memory dictionary — a per-pair exponential moving average of confusion events observed at the resonance field level. This has significantly higher update frequency than the ConfusionGraph (which only updates when deliberation fires) and captures confusion patterns that resolve before reaching deliberation. It is blended at 40% weight into the Stage 2 confusion prior.

## 4.8  GlobalWorkspace  (GNW)
Inspired by Global Neuronal Workspace Theory, which proposes that conscious awareness corresponds to a broadcast of information across specialised brain regions. In Cypha, the GlobalWorkspace is a competition mechanism: when multiple class hypotheses are active simultaneously, they compete for 'ignition'. Only hypotheses that exceed a margin threshold relative to competitors are broadcast as confident predictions.

In practice, the GNW computes the margin between the top-1 and top-2 class scores. If this margin exceeds a threshold (currently 0.08 by default), the top class ignites and is passed downstream with a high confidence flag. If the margin is insufficient, the GNW fires a miss signal and forces entry into the deliberation engine regardless of the hippo fast-path result.

## 4.9  Neuromodulation  (DA and NE)
Two neuromodulatory signals are computed on each inference, inspired by the dopaminergic and noradrenergic systems in the brain.

Dopamine surprise (DA) measures prediction error — how different was the actual class from what the system predicted? When DA is high, the system has encountered something genuinely novel or surprising. High DA increases the weight of memory storage (novel patterns are stored with higher priority) and increases alpha slightly (uncertain terrain warrants more careful deliberation).

Norepinephrine volatility (NE) measures the rate of change of the input distribution — how unstable is the current domain? NE is computed as an exponential moving average of the standard deviation of recent anchor distances. When NE is high, the deliberation alpha is scaled up multiplicatively, ensuring that high-volatility operating conditions always trigger deep search.

## 4.10  DMN Consolidation  (Default Mode Network)
In the brain, the default mode network is active during rest and is associated with memory consolidation, schema extraction, and future planning. In Cypha, the DMN consolidation loop runs offline — between inferences — and performs three tasks.

Memory replay: Selects the highest-deliberation episodes from the last N inferences (those that required deep search) and replays them with Izaac-generated noise augmentation. This strengthens the memory traces for difficult cases without overfitting to specific instances.
MP filter refit: Fits the Marchenko-Pastur filter on a recent batch of anchor vectors. This keeps the signal subspace current as the distribution of stored vectors evolves during training.
Schema extraction: Analyses the ConfusionGraph for 3-way confusion cycles (A confused with B, B confused with C, C confused with A) and flags these as schema rules for MCTS priority. When MCTS encounters a node involving these classes, it allocates more simulation budget.

# 5. GRIA Deliberation Engine
When the hippo fast-path and GNW together cannot produce a confident answer, the input enters the GRIA deliberation engine — a 3-stage sequential pipeline where each stage runs multiple methods in parallel, produces a weighted vote, and may exit early if sufficiently confident, or otherwise enriches the query and passes it forward.

## 5.1  Grade Computation
Before the pipeline runs, the GRIA grade alpha is computed. Alpha governs how deeply the system searches and how aggressively it commits resources to uncertain decisions.

| alpha = clip(epistemic + aleatoric + conflict, 0, 1) * ne_volatility_scale  epistemic = 1.0 - (density(top_class) / max_density) How sparse is the winning class in memory? Low density (few examples near this region) = high uncertainty  aleatoric = 1.0 - centroid_separation(top1, top2) How far apart are the top-2 class centroids? Close centroids = inherently ambiguous boundary  conflict = confusion_weight(top1, top2) * pairwise_similarity(top1, top2) How often have top1 and top2 been historically confused? High confusion + high similarity = dangerous pair  ne_volatility_scale = 1.0 + 0.5 * NE_signal High domain volatility always forces deeper search |
|---|

## 5.2  Stage 1 — Fast Evidence
Stage 1 runs three methods in parallel and produces a weighted vote vector over all candidate classes. The stage can exit early if confidence exceeds 0.55 and grade is below 0.40, indicating a clear winner has emerged and the decision is not in a difficult grade region.

| Hippo soft-score | Calls HippoCypha.soft_score() to get the best matching class and its cosine similarity, regardless of the fast-path threshold. The similarity score becomes the vote weight for this voter. A similarity of 0.91 contributes with weight 0.91 * 1.5 (hippo bonus multiplier). |
|---|---|
| Rocchio centroid push | Computes an enriched query by nudging anchor_q toward the top-1 class centroid and away from the top-2 class centroid. The push strength beta is sourced from ConfusionGraph.last_successful_beta() for this class pair — using the same push strength that historically resolved this confusion. The output enriched_q is passed to all Stage 2 and Stage 3 methods. |
| GNW gap check | Computes the margin between top-1 and top-2 class scores in the current adapter lookup. If margin exceeds a threshold (0.12), this method votes for top-1 with high weight. If margin is insufficient, no vote is cast and deliberation continues. |

Agreement bonus: when two or more Stage 1 methods vote for the same class, their combined weight is multiplied by 1.4. This reflects the information-theoretic principle that independent agreement is stronger evidence than a single high-confidence vote.

## 5.3  Stage 2 — Neighbourhood Stress Test
Stage 2 operates on enriched_q from Stage 1. It tests the stability of the current hypothesis by probing the neighbourhood of the enriched query and checking for consistency. Stage 2 exits early if confidence exceeds 0.42 and no method disagreement is detected.

| PNQ (Perturbation-Noise-Query) | Generates n_samples=8 perturbed versions of enriched_q using Izaac Levy noise. Each perturbed version is independently queried against the adapter. The majority class across all probes is the PNQ vote. Samples where a minority class pattern appears are flagged as 'distribution tail' indicators. If tail samples appear, confidence is penalised, and Stage 3 is forced. PNQ answers the question: is the enriched_q stably in the centre of a class region, or near the edge where small noise changes the answer? |
|---|---|
| Confusion prior blend | Computes a confusion discount factor for the current top-2 pair. Blends the ConfusionGraph count (60% weight) with ThoughtProcessor._confusion_memory (40% weight). The ConfusionGraph tracks deliberation-level confusion events (deeper, sparser). The confusion_memory tracks field-level confusion events (shallower, much higher frequency — 60,000+ updates on a 40,000-sample training run). Together they provide a complete picture of how hard this specific class pair is. |
| Hard-negative boundary | Queries the adapter for the k=3 hardest negative examples — the wrong-class anchors most similar to enriched_q. Computes the minimum distance to these hard negatives. A small distance indicates the enriched_q sits close to a wrong-class anchor, flagging a dangerous boundary region and increasing Stage 2 uncertainty. |

Stage 2 also identifies the contested pair: if Rocchio (Stage 1) and PNQ (Stage 2) disagree on the top class, the two classes involved become the contested pair. This information is passed to Stage 3 MCTS, which focuses its simulation budget on distinguishing exactly these two classes.

## 5.4  Stage 3 — Deep Search
Stage 3 always runs in full if reached. It applies the most computationally expensive methods to resolve the class boundary identified in Stage 2.

| MCTS (Monte Carlo Tree Search) | Runs a tree search over the class manifold. Each tree node corresponds to a point in feature space. Node expansion uses adapter.lookup() to find class-similar episodes nearby. Simulation rollouts score positions by cosine similarity to class centroids. The tree is seeded with hypothesis vectors from the ThoughtProcessor cascade signal, projected from resonance space (64-dim) into feature space (512-dim). These become pre-informed root children before the first UCT selection. When a contested pair was identified in Stage 2, MCTS allocates extra simulation budget to nodes involving these classes. The UCT exploration constant is modulated by the NE volatility signal. |
|---|---|
| ConfusionGraph resolution vote | Calls ConfusionGraph.best_resolution(A, B) for the contested pair. This returns the class that historically wins when these two classes are confused — computed from the resolution direction stored across all prior deliberation calls. The resolution vote is weighted by the resolution confidence (fraction of resolutions that went the same direction). |
| Reflexion failure memory | Checks a store of previously failed inferences — cases where deliberation produced a wrong answer. If the current enriched_q is similar to a prior failure involving the same class pair, the failure memory adds a negative weight to the previously-wrong answer. This implements a simple form of error-based learning within the inference pass itself. |

## 5.5  Cross-Stage Vote Integration
The final answer is a weighted sum of the vote vectors from all three stages. Stage 1 contributes 25%, Stage 2 contributes 35%, Stage 3 contributes 40%. Later stages carry more weight because they operate on enriched queries and have access to more evidence. The class with the highest weighted score is returned as the prediction, with the maximum score normalised to [0,1] as the confidence value.

# 6. Training
Cypha trains in a single forward pass over the training data. There is no backpropagation, no loss surface optimisation, and no multiple epochs. This makes Cypha suitable for online deployment: it can train incrementally as new data arrives without retraining from scratch.

Each training example is processed as follows. The OmegaEncoder converts it to anchor_q. The HRNA forward pass runs through the resonance field and ThoughtProcessor. The correct class label is used to update the AnchorMemory (via Rocchio update), the HippoCypha store (via direct storage), the TensorCentroid for the correct class, and the ConfusionGraph (if the system predicted the wrong class). LVQ2.1 boundary sharpening runs on each step. Consolidation runs every configurable number of steps.

The Platt calibrator is updated on each training step to keep probability estimates well-calibrated. The Cerebellar output model accumulates per-class statistics for post-hoc correction of systematic biases. Both calibration components are self-supervised — they update from the system's own predictions without requiring a separate calibration set.

# 7. Inference
At inference time, the system receives an input and must produce a class label and confidence score as quickly as possible. The inference path is designed to exit at the earliest possible stage that can produce a confident answer.

| 1. OmegaEncoder: input -> anchor_q (512-dim) 2. HippoCypha.soft_score(): get best class and similarity (no threshold) 3. HippoCypha.fast_path_hit(): test against adaptive threshold If HIT and GNW margin sufficient: return immediately (<2ms) 4. forward(): run resonance field, ThoughtProcessor Produces: chain_score, trend_vec, cascade_hints, DA, NE 5. Compute GRIA grade alpha from uncertainty decomposition 6. gria_cascade() - 3-stage deliberation Stage 1: hippo_soft + rocchio + gnw_gap -> enriched_q, vote1 If confident (>0.55) and low grade (<0.40): exit with vote1 Stage 2: pnq(enriched_q) + confusion_prior + hard_negatives -> vote2 If confident (>0.42) and no disagreement: exit with vote1+vote2 Stage 3: mcts(seeds) + confusion_resolution + reflexion -> vote3 7. Final: 0.25*vote1 + 0.35*vote2 + 0.40*vote3 8. Platt calibration on raw score -> calibrated probability 9. Return: class label, confidence, deliberation depth |
|---|

# 8. Benchmark Results
## 8.1  Classification Benchmark
50,000 examples generated per domain by embedded game AIs (ChessAI with PST evaluation and pawn structure, PokerAI with full Monte Carlo equity engine, GoAI with BFS liberty and flood-fill territory). Each domain is 25% boundary examples specifically designed to force deliberation. Single-epoch training on 40,000 examples, evaluation on 10,000 held-out examples.

| Domain | Classes | Accuracy | Correct / Total | Hippo hit rate |
|---|---|---|---|---|
| poker_decision | 8 | 100.0% | 9,995 / 10,000 | 100.0% |
| chess_evaluation | 9 | 99.5% | 9,952 / 10,000 | 91.8% |
| go_strategy | 10 | 97.3% | 9,730 / 10,000 | 23.5% |

Poker achieves perfect accuracy with a 100% hippo hit rate — the feature space for poker decisions is sufficiently structured that virtually every test example matches a stored episode above threshold. Chess requires deliberation for approximately 8% of examples, primarily in the endgame class boundary region. Go is the most challenging domain: 76.5% of examples require deliberation (the highest of any domain tested), reflecting genuine structural ambiguity in Go position classification.

## 8.2  Performance Profile
| Component | Calls (Go domain) | Mean latency | p95 latency |
|---|---|---|---|
| encode_features | 130,000 | 1,166 us | 1,200 us |
| hippo_fastpath | 10,000 | 1,352 us | 1,479 us |
| forward (infer) | 3,220 | 5,704 us | 5,975 us |
| infer total | 6,780 | 4,012 us | 4,323 us |
| pnq_lookup | ~860 | 3,368 us | 6,071 us |
| mcts_search | ~400 | varies | varies |
| resonant_chain | 83,220 | 1.3 us | 1.2 us |
| self_generate | 83,220 | 46.5 us | 63.2 us |
| memory_lookup | 51,786 | 45.9 us | 56.1 us |

Figures from the go_strategy domain which has the highest deliberation rate. Chess and poker are significantly faster due to lower deliberation rates. The resonant_chain operation at 1.3 us mean is essentially free. The 3.8 ms average deliberation path is well within real-time constraints for tactical decision support.

## 8.3  Live Game Benchmark
A separate benchmark tests Cypha in actual game play against real opponents. For chess, Cypha trains by observing 200 Stockfish depth-18 self-play games using behavioural cloning, then plays White against Stockfish depth-20 at skill level 20 (maximum difficulty) for 20 games. For poker, Cypha trains on 5,000 GTO-labelled hands and plays 200 heads-up hands against a calibrated rule-based opponent. For Go, Cypha trains on 500 greedy bot self-play games and plays 30 games against a greedy territory bot.

Chess results were pending at time of writing. Wins against Stockfish at maximum strength represent an exceptional benchmark for a behavioural cloning system operating with no explicit search tree — Cypha's deliberation engine must substitute for what is traditionally done by minimax lookahead.

# 9. Security Classification
| PROPRIETARY AND CONFIDENTIAL This document and the system it describes contain original mathematical frameworks and novel neural architecture developed for defence and intelligence applications. The Izaac algorithm, GRIA framework, and Cypha HRNA architecture are proprietary intellectual property. Unauthorised reproduction, distribution, or disclosure is prohibited. For access, licensing, and integration enquiries, contact the author directly. |
|---|

Cypha HRNA   Technical Reference   February 2026   All rights reserved
