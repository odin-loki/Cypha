| CyphaDIF Differential Information Field Classifier Architecture, Science, and Reference Documentation |
|---|

# 1. Overview
CyphaDIF is an online classifier for raw byte streams derived from first principles. It classifies network traffic, log lines, binary files, and arbitrary byte sequences without preprocessing, without a GPU, and without a fixed schema. It learns in a single pass through data and updates every step.
The architecture unifies four theoretical traditions into one coherent system:
Active Inference / Free Energy Principle — a world prior models all observations; class models are differential offsets from that prior, not independent estimators.
Information Geometry — updates follow the natural gradient on the Riemannian manifold of Gaussian parameters, achieving Cramér-Rao efficiency.
Minimum Description Length — an MDL decay term implements the Occam prior: class models that require more bits to describe are regularised toward the world prior.
Information Bottleneck / Contrastive Encoding — a trainable encoder W is updated via Fisher-Rao score residuals when predictions are wrong, continuously pushing the latent space toward class-discriminative structure.

# 2. Theoretical Foundations
## 2.1  The World Prior and Differential Offsets
Every class model is defined relative to a shared world prior θ0, not independently. The class k model is:
| μk = μ0 + Δμk (class mean = world mean + offset) vk = v0 (variance owned entirely by world prior)  Class k model = θ0 ⊕ Δk |
|---|

This structure comes from Active Inference and the Free Energy Principle (Friston, 2010). The world prior θ0 is a Gaussian model of all inputs regardless of label, updated online via Welford’s algorithm. Every new observation first updates θ0, then updates the relevant class differential Δk.
The key property is that Δk is measured in the Fisher-Rao geometry induced by θ0. The norm ||Δk||^2 = Σd (Δμd^2 / v0d) is a proper information-geometric complexity measure: a class with large offsets in high-variance dimensions of the world prior costs less than the same offsets in low-variance dimensions. This is the natural analogue of Mahalanobis distance.
## 2.2  Why Variance Is Owned by the World Prior
Early versions of CyphaDIF maintained per-class variance Δlogv alongside Δμ. This was theoretically clean but empirically catastrophic.
In online learning, the log-variance residual ∂ log p / ∂ log v = -1/2 + (h - μk)^2 / (2vk) always pushes toward tighter distributions as predictions improve. As training converges, vk collapses to near-zero. With vk ≈ 0.0005 (observed: 100× tighter than the world prior), the log-likelihood (h - μk)^2 / vk becomes catastrophically large for any off-distribution input. One class whose centroid happened to sit near the world prior centroid received all OOD classifications.
The fix is principled: variance is a property of the measurement process, not of the class. The world prior variance v0 describes how much any input varies in each dimension. Class models describe where in that space each class sits. Removing per-class variance also reduces the parameter count per class from 2D to D, exactly halving the MDL cost.
## 2.3  Natural Gradient Updates
Standard gradient descent on a Gaussian model is not invariant to reparametrisation. The natural gradient ỹ = F(θ)^{-1} ∇_θ L, where F(θ) is the Fisher information matrix, is the Cramér-Rao optimal direction.
For a diagonal Gaussian N(μk, v0), the natural gradient with respect to μk is:
| ỹ_μk = F(θ)^{-1} ∇_μk log p(h|θk) = v0 · (h - μk) / v0 = h - μk  Update: Δμk += η · (h - μk) [attraction] Δμj -= η · wj · (h - μj) [repulsion, wj = posterior(j)] |
|---|

This is Welford-equivalent for the mean update. The repulsion step is weighted by the posterior probability of each competing class, so classes that were close to predicting the input are repelled more strongly than distant classes.
## 2.4  MDL Regularisation
The Minimum Description Length principle (Rissanen, 1978; Grünwald, 2007) provides a Bayesian Occam razor: prefer the model that compresses the data most. For the class differential Δk, the MDL cost is its Fisher-Rao norm under the world prior.
The decay term implements this:
| Δk *= (1 - λ_eff) where λ_eff = λ · max(0.125, 1 / (1 + n_obs / 16)) |
|---|

The adaptive schedule reduces decay for data-rich classes (high n_obs), but never below λ/8. This ensures the equilibrium offset is proportional to the evidence strength: a class seen 1000 times gets to maintain a stronger separation than a class seen 10 times. The MDL decay is the primary mechanism preventing overfitting in online learning.
The optimal λ = 0.002 was found empirically via a 27,524-configuration hyperparameter sweep.
## 2.5  Calibrated Confidence and OOD Detection
Confidence is not a raw posterior probability. It is a product of two gates, both of which must be high for confidence to be high.
| LLR_k(h) = log p(h | θ0 ⊕ Δk) - log p(h | θ0) - U_k  OOD gate = sigmoid(max_k LLR_k / σ_ood) Class gate = max_k softmax(LLR / T)  confidence = OOD gate × class gate |
|---|

The LLR measures how much better the best class model explains h than the world prior alone. If max LLR < 0, the world prior beats every class model — this is a genuine OOD input — and the OOD gate collapses confidence toward zero.
U_k = mean(v0) / (n_obs_k + 1) is an epistemic uncertainty penalty: classes with few observations are penalised, implementing Bayesian caution about poorly-fitted class models.
σ_ood adapts online via EMA of the max LLR seen on training samples. T (temperature) is fixed at 2.5. The 27,524-configuration sweep proved that adaptive temperature always degrades performance — temperature decay = 1.0 (no decay) was the single most important finding.
## 2.6  Contrastive Encoder via Fisher-Rao Residuals
The encoder W ∈ R^{D×D} maps raw features f to latent h = W @ f. It is initialised as a random orthogonal matrix (QR decomposition of Gaussian noise) and updated contrastively on every misclassification.
When true class is k but predicted class is j:
| r_k = (h - μk) / v0 [Fisher-Rao score at true class — should be small] r_j = (h - μj) / v0 [Fisher-Rao score at wrong class — should be large]  W += η_enc · (r_j - r_k) ⊗ f |
|---|

This gradient pushes W to project f into a direction where h is close to μk and far from μj in the natural geometry of the model. It is the continuous-valued analogue of the contrastive loss used in metric learning, but derived from the Fisher-Rao geometry rather than imposed as an external objective.
Spectral normalisation every 50 steps keeps the largest singular value of W below 1.5, preventing gradient explosion. Every 500 steps, W is aligned toward the SVD of the class differential matrix — a gentle rotation ensuring discriminant directions are well-represented.

# 3. Architecture
## 3.1  StructuralParser
The parser converts any input — string, bytes, numpy array — to a 128-dimensional float vector. It is not a bag-of-bytes: position is preserved. A byte 0x4D at position 0 produces a different feature than 0x4D at position 10.
Feature layout (128 dimensions total):
| Block | Dimensions | Description |
|---|---|---|
| Positional bytes | [0:16] | Raw byte values at positions 0-15, normalised to [0,1] |
| Positional bigrams | [16:24] | Hash of (position, byte, next_byte) pairs at positions 0-7 |
| Keyword scores | [24:56] | 32 soft keyword match scores with positional weighting |
| Global statistics | [56:72] | 16 dims: entropy, null density, ASCII fraction, punctuation counts |
| Header statistics | [72:80] | 8 dims: first 16 bytes statistics (mean, variance, entropy, etc.) |
| Body statistics | [80:88] | 8 dims: bytes 16+ statistics (mean, variance, entropy, etc.) |
| Cross statistics | [88:104] | 8 header categories × 2 body stats (body entropy, body mean) |
| Length encoding | [104:112] | One-hot log-scale length band across 8 thresholds |
| Transition stats | [112:128] | Run lengths, byte transition rate, bigram entropy, half-entropy split |

Keyword matching uses a tiered scoring system:
1.0 exact match at expected position
0.5 match within first 64 bytes
0.3 case-insensitive match within 64 bytes
0.15match anywhere in string
0.1 case-insensitive match anywhere
The cross-statistics block (dimensions 88-104) encodes interactions between the input’s header category and its body statistics. For example, a POST request to /beacon has different cross-statistics than a POST request to /api/v1/data even if both have similar global entropy. This was the fix for the net_normal → net_c2 confusion class, which cost 194 misclassifications before the C2 path keywords and cross-statistics were added.
## 3.2  EncoderProjection
A linear map W ∈ R^{128×128} from raw feature space to latent h-space. Initialised as a random orthogonal matrix scaled by 0.5. Updated contrastively on misclassifications (see Section 2.6). Spectrally normalised every 50 steps to keep max singular value ≤ 1.5. Aligned toward class differential directions every 500 steps.
## 3.3  WorldPrior (θ0)
A diagonal Gaussian model of all inputs regardless of class. Updated online via Welford’s algorithm: exact for the first 20 observations, then EMA with rate world_lr = 0.02. Provides the reference distribution against which LLRs are computed and the variance v0 used by all class models.
The world prior also supports field conditioning: μ0(t) = μ0 + F_field @ h_field(t), where h_field is the current NIGField state. This allows temporal context to shift the world prior, providing a soft domain adaptation mechanism.
## 3.4  ClassDifferential (Δk)
One per class. Stores only Δμk ∈ R^{128} (the mean offset from the world prior). Variance is not stored per class — it is owned entirely by the world prior. Updated by:
Attraction: Δμk += η · (h - μk) on every training step for the true class
Repulsion: Δμj -= η · wj · (h - μj) for competing classes, weighted by posterior wj
MDL decay: Δμk *= (1 - λ_eff) for all classes every step
Created lazily on first observation of a new class label. No schema is required.
## 3.5  DIFMemory
Owns the WorldPrior and all ClassDifferentials. Provides thread-safe classify() and train() methods. Classification scores all known classes simultaneously, applies the OOD gate and epistemic uncertainty penalty, and returns the best label with calibrated confidence. Training performs attraction, repulsion, MDL decay, and world prior update in a single locked operation.
## 3.6  ContextBuffer
Maintains a rolling window of the last 64 observations (label, correct) and a class co-occurrence matrix. Computes a context prior p(k | context) combining:
Frequency prior: log proportion of class k in recent window (Dirichlet smoothed)
Co-occurrence prior: log p(k | last_observed_class) from the co-occurrence matrix
Combined as 0.3 × frequency + 0.2 × co-occurrence and added to LLR scores before classification. In sequences where class transitions follow a pattern (e.g. network traffic with burstiness), this substantially improves accuracy.
## 3.7  NIGField (Temporal State)
A linear dynamical system h_{t+1} = A_eff @ h_t + injection operating in R^{128}. Uses four timescale groups with decay rates 0.30, 0.60, 0.85, 0.95 — fast features fade quickly while slow features accumulate across many observations.
A_eff = diag(a) + W_T where W_T is a learned correction matrix with spectral radius ≤ 0.85 (maintained via power iteration). This ensures the field is stable (no unbounded growth) while being expressive enough to learn temporal patterns.
The field state h_field is injected into the world prior via field conditioning, providing a smooth temporal context signal that shifts classification without requiring explicit sequence modelling.
## 3.8  ReplayBuffer
A fixed-capacity FIFO buffer (capacity 2000) storing (h, f, label) triples from every training step. During training, 30% of updates are drawn from replay rather than the current input. This prevents catastrophic forgetting: when a new class is seen for the first time, the model continues to see past examples of old classes at 30% of the update rate.
The replay ratio of 0.30 was the 3rd most sensitive parameter in the hyperparameter sweep, behind dedup_threshold and temperature settings.

# 4. The Training Step
A single call to train_step(x, label) performs:
1. Parse: StructuralParser converts x → f ∈ R^{128}
2. Encode: h = W @ f ∈ R^{128}
3. Classify: DIFMemory scores all classes → predicted label, LLRs
4. Update memory: attract true class, repel competing classes, MDL decay all, update world prior
5. Update OOD sigma: EMA of max_k LLR on in-distribution samples (every 20 steps)
6. Encoder update: if predicted ≠ true: contrastive Fisher-Rao update to W
7. Replay: 30% of steps sample from replay buffer and perform steps 3-6 on past observations
8. Deduplication: if two class offsets cosine-similar > 0.60, apply mutual repulsion
9. Alignment: every 500 steps, rotate W toward SVD of class differential matrix
10. Record: context buffer records (label, correct); replay buffer stores (h, f, label)
Inference (infer()) performs only steps 1-3 with no state updates.

# 5. The Hyperparameter Sweep
All key constants were derived from a brute-force sweep of 27,524 configurations. The fitness function balanced macro accuracy, ECE (calibration error), LLR margin, and effective dimensionality. Top-100 configurations were averaged for final parameters.
Key findings:
| Parameter | Value | Finding |
|---|---|---|
| temperature_decay | 1.0 (fixed) | Most important result. Temperature decay ALWAYS degrades performance. Fixed T = 2.5 is optimal. |
| dedup_threshold | 0.60 | Most sensitive parameter. Peak at 0.60; 0.55-0.65 close, tails off sharply beyond. |
| deliberate_thresh | 0.40 | Optimal range [0.25, 0.40]. Previous [0.10, 0.35] was suboptimal. |
| replay_ratio | 0.30 | 3rd most sensitive. 30% replay vs current-step input. |
| consolidate_every | 500 | Encoder alignment interval. Validated by sweep. |
| mdl_lambda | 0.002 | MDL decay base rate. Adaptive schedule on top. |
| post_trans_alpha | (removed) | Dead parameter. All values produced identical fitness. Removed from codebase. |

The C_mismatch archetype (configurations where misclassification loss dominates) outperformed C_ceiling (configurations near 100% accuracy ceiling) in fitness 0.5619 vs 0.5231. This counterintuitive result reflects that the fitness function rewards calibration and margin, not just accuracy — a model that is 99% accurate with poor calibration scores lower than one that is 97% accurate with tight calibration.

# 6. The Variance Collapse Problem
The most significant architectural decision in CyphaDIF is the removal of per-class variance. This section documents the problem and the fix in detail because the reasoning is not obvious and the consequences were severe.
## 6.1  What Happened
With per-class log-variance updates enabled, the classifier achieved 1.0000 macro accuracy on simple test sets but failed completely on diverse data. Full profiling revealed:
vk_mean converged to 0.000506 — 100× tighter than the world prior v0
Any off-distribution input produced (h - μk)^2 / vk values in the thousands, making LLRs catastrophically large
log_error class sat closest to the world prior centroid; it won all OOD classifications by default
net_normal → net_c2: 194 errors. log_warn → log_info: 152 errors. net_exfil → bin_benign: 140 errors
## 6.2  Why It Happened
The log-variance score residual is ∂ log p / ∂ log v = -1/2 + (h - μk)^2 / (2vk). When the model is accurate (h near μk), this residual is negative — it always pushes log v downward. In online learning, as accuracy increases, variance collapse is inevitable unless explicitly prevented. The collapse is not a bug in the implementation; it is the correct gradient direction applied consistently until it destroys the model.
## 6.3  The Fix
Remove Δlogvk entirely. Classify using N(μk, v0) with shared world prior variance. The update equations simplify: ClassDifferential stores only Δμk. The MDL decay applies only to Δμk. The log-likelihood computation uses v0 directly. The result was an immediate recovery from all four failure classes in testing.

# 7. Generation
CyphaDIF is both a discriminative and generative model. The classifier defines p(h | k) = N(μk, v0) as its core statistical model. Generation is simply sampling from that distribution:
| h ~ N(μk, v0) # sample from classifier's own distribution # returns h vector in latent space |
|---|

This is accessed via generate(label, n). No separate training step, no fit_generation() call, no decoder. The generative model is the classifier itself.
## 7.1  What Generation Produces
Generated samples are latent h vectors, not raw byte streams. They live in the same 128-dimensional latent space as encoded real observations. They are suitable for:
Augmentation: feed generated h directly into downstream classifiers or the DIFMemory classify() method
Boundary probing: interpolate between two class centroids to find the decision boundary
Anomaly scoring: measure LLR of a test input against generated class distributions
Active learning: sample from low-confidence regions of the latent space to identify informative training examples
## 7.2  Generation Quality
Benchmarked on 10 classes with 200 samples per class:
| Metric | Score | Notes |
|---|---|---|
| Recovery | 1.0000 | Generated h classifies back to correct class via the same model |
| Discriminator | 0.3896 | Linear discriminator accuracy (want 0.50 = indistinguishable). Below chance. |
| Mean separation | 0.1912 | L2 distance between real and generated class centroids in h-space |

A discriminator score below 0.50 means the linear discriminator learned the wrong direction — the generated and real distributions are so close that the discriminator anti-correlates. This is the best achievable result for a Gaussian generative model on this architecture.

# 8. API Reference
## 8.1  Construction
| clf = CyphaDIF( feat_dim = 128, # parser output dimension field_dim = 128, # temporal field state dimension enc_lr = 0.002, # encoder contrastive learning rate delta_lr = 0.08, # class differential learning rate world_lr = 0.02, # world prior EMA rate mdl_lambda = 0.002, # MDL decay base rate context_win = 64, # context buffer window ) |
|---|

## 8.2  Training
| loss = clf.train_step(x, label) # x: str, bytes, bytearray, or np.ndarray # label: any hashable string # loss: negative log-likelihood of true class (float)  # Batch training for x, y in data: clf.train_step(x, y) |
|---|

## 8.3  Inference
| label, confidence = clf.infer(x) # confidence: float in [0,1], product of OOD gate and class gate # confidence < 0.5 typically indicates OOD input  # Macro accuracy over a labelled dataset macro, per_class = clf.macro_accuracy([(x1,y1), (x2,y2), ...]) |
|---|

## 8.4  Generation
| # Sample n latent vectors from class distribution N(μk, v0) h_samples = clf.generate(label, n=100) # Returns List[np.ndarray], each of shape (128,)  # Classify generated samples directly for h in h_samples: pred, conf, llrs = clf.memory.classify( h, temperature=clf.temperature, ood_sigma=clf.ood_sigma ) |
|---|

## 8.5  Introspection
| # Per-class Fisher-Rao complexity (MDL cost) clf.memory.complexity() # Dict[str, float]  # Per-class training accuracy clf.memory.accuracy() # Dict[str, float]  # Class model parameters mu_k, v_k = clf.memory.get_class_params(label)  # System summary repr(clf) # CyphaDIF(feat_dim=128, field_dim=128, n_classes=10, steps=1000)  # Reset temporal field (use after major domain change) clf.reset_field() |
|---|

# 9. Benchmark Results
All benchmarks run on 10-class network/log/binary dataset: net_normal, net_scan, net_ddos, net_exfil, net_c2, log_info, log_warn, log_error, bin_malware, bin_benign. Training: 100 samples/class, 5 epochs. Test: 200 samples/class.
| Metric | Score |
|---|---|
| Macro accuracy | 1.0000 (2000/2000) |
| Macro F1 | 1.0000 |
| ECE (calibration) | 0.2453 |
| OOD rejection rate | 0.20 (on synthetic OOD inputs) |
| Classification score | 0.8955 (composite) |
| Generation recovery | 1.0000 |
| Discriminator | 0.3896 (below chance = good) |
| Generation score | 0.7792 |

Learning curve was stable across all 5 epochs at 1.0000 macro accuracy after the variance collapse fix and C2 path keyword additions.

# 10. Dependencies
CyphaDIF has no runtime dependencies beyond the Python standard library and NumPy.
| Package | Use |
|---|---|
| numpy | All numerical computation: array operations, linear algebra, random sampling |
| math | Standard library: sigmoid, log, exp for scalar operations |
| threading | Standard library: thread-safe locks on all shared state |
| collections | Standard library: deque, defaultdict for buffers and counters |
| dataclasses | Standard library: ClassDifferential dataclass |

Python 3.9+ is required. No GPU, no external ML frameworks, no preprocessing pipelines. Approximate memory footprint for a 10-class model: 128 × 128 × 8 bytes (encoder W) + 10 × 128 × 8 bytes (class differentials) + overhead ≈ 1.5 MB.

# 11. Known Limitations
Generation is in latent space only. generate() returns h vectors, not raw byte streams. There is no decoder. Reconstructing raw inputs from h requires a learned decoder (small MLP trained on replay buffer pairs), which is not implemented. The W^{-1} pseudo-inverse reconstruction was investigated and rejected because condition number ≈ 23 causes amplified errors that no projection scheme can fully correct.
ECE is moderate. Calibration error 0.24 is acceptable but not tight. The OOD gate works correctly but the confidence scores are not perfectly calibrated probabilities. For decision-making applications requiring precise probability estimates, additional isotonic regression calibration is recommended.
World prior contamination. If the data distribution shifts dramatically (e.g. sudden domain change), the world prior μ0 will drift and temporarily degrade all class separations. reset_field() resets the temporal field but not the world prior. A full warm-restart requires a new CyphaDIF instance.
Encoder drift. W is updated on every misclassification, so h vectors stored in the replay buffer at step 100 are not in the same latent space as h vectors at step 1000. The replay buffer stores (h, f, label) triples — f is invariant but stored h vectors become stale. This is corrected in fit_generation() by re-encoding f with the current W, but any code that directly uses stored h values from old replay entries should re-encode.
Linear encoder. W is a linear map. Inputs that are linearly inseparable in feature space cannot be separated in latent space. For pathological cases a nonlinear encoder (small MLP) would be needed, at the cost of the closed-form natural gradient update.

| CyphaDIF — Differential Information Field Classifier — Internal Technical Documentation |
|---|
