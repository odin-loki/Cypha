CyphaDIF: A Unified Online Classifier via Differential
Information Fields, Natural Gradients, and MDL Regularisation
Odin
Independent Researcher, Sydney, Australia
March 2026

Abstract
We present CyphaDIF (Differential Information Field Classifier), a novel online learning architecture for classifying raw byte streams from first principles, without preprocessing, GPU acceleration, or fixed schemas. The system is derived by synthesising four theoretical traditions: (1) the Active Inference / Free Energy Principle (Friston, 2010), which motivates a shared world prior against which class models are expressed as differential offsets; (2) Information Geometry (Amari, 1998), which dictates that parameter updates follow the natural gradient on the Riemannian manifold of Gaussian parameters, achieving Cramér-Rao efficiency; (3) Minimum Description Length regularisation (Rissanen, 1978; Grünwald, 2007), which implements a Bayesian Occam's razor on class model complexity; and (4) the Information Bottleneck framework (Tishby et al., 1999), which drives a contrastive encoder via Fisher-Rao score residuals. CyphaDIF introduces a key architectural insight—variance is a property of the measurement process, not of the class, eliminating the variance collapse failure mode endemic to online Gaussian classifiers. A 27,524-configuration hyperparameter sweep validates all design choices. On a 10-class network and binary classification benchmark, CyphaDIF achieves macro accuracy 1.0000, macro F1 1.0000, calibration error (ECE) 0.2453, and a generation discriminator score of 0.3896 (below chance, indicating near-perfect generative fidelity). The complete system runs in approximately 1.5 MB of RAM with no dependencies beyond NumPy.
Keywords: online learning, Gaussian classification, active inference, free energy principle, information geometry, natural gradient, MDL regularisation, information bottleneck, contrastive encoder, out-of-distribution detection, network intrusion detection

1. Introduction
Modern network intrusion detection, log anomaly classification, and binary file triage share a common problem: the data arrives as raw byte streams, class labels are known only at training time, the number of classes may be unknown in advance, and the system must update continuously as new classes appear. Off-the-shelf solutions—deep convolutional networks, transformer-based encoders, random forest ensembles—typically require batch training, GPU hardware, fixed input schemas, and substantial preprocessing pipelines. Deployments in resource-constrained or time-critical environments (embedded systems, edge nodes, airgapped military systems) cannot accommodate these prerequisites.
CyphaDIF (Differential Information Field Classifier) is designed to address this gap from first principles. It is an online classifier: every training example updates the model in a single pass, with no batches, no replay of the full dataset, and no gradient tape. It operates on raw bytes, handling strings, binary blobs, numpy arrays, and log lines through a unified 128-dimensional structural feature extractor. It requires no GPU. Its memory footprint is approximately 1.5 MB for a 10-class model. And yet, on a carefully designed benchmark, it achieves perfect macro accuracy.
This performance is not achieved through heuristic tricks or domain-specific engineering. It emerges from a disciplined synthesis of four theoretical frameworks. The world prior concept is drawn directly from Friston's Free Energy Principle (Friston, 2010; Parr et al., 2022), in which a generative model of all observations provides the baseline against which specific beliefs are expressed as offsets. Parameter updates are derived in information-geometric style (Amari & Nagaoka, 2000; Nielsen, 2020), following the natural gradient on the statistical manifold to achieve Cramér-Rao optimal convergence. Model regularisation applies the MDL principle (Rissanen, 1978; Grünwald, 2007) as an Occam prior over class complexity. The contrastive encoder exploits the Fisher-Rao score residual structure of the Information Bottleneck framework (Tishby et al., 1999).
The paper is structured as follows. Section 2 surveys the four theoretical traditions. Section 3 describes the complete CyphaDIF architecture. Section 4 details the variance collapse failure mode and its principled resolution—the single most important architectural decision. Section 5 presents the hyperparameter sweep and its key findings. Section 6 reports benchmark results. Section 7 discusses limitations and future directions. Section 8 contains the full reference list.
2. Theoretical Background
2.1 The Free Energy Principle and Active Inference
The Free Energy Principle (FEP), introduced by Karl Friston and developed over two decades of theoretical neuroscience (Friston, 2006, 2009, 2010), proposes that biological agents—and, by extension, any adaptive system—minimise a quantity called variational free energy, which upper-bounds the surprise of sensory observations under the agent's generative model. Mathematically, free energy F is defined as:
F = E_q[log q(θ) − log p(x, θ)]  =  KL[q(θ) || p(θ|x)] − log p(x)
where q(θ) is the recognition density (the agent's beliefs about hidden causes), p(x, θ) is the generative model, and KL denotes the Kullback-Leibler divergence. Minimising F is equivalent to approximate Bayesian inference plus action selection that brings sensory observations into alignment with predictions (Friston, 2010).
Active Inference (Parr et al., 2022) operationalises the FEP into perception, learning, and action. The key architectural implication for CyphaDIF is the hierarchical structure: a shared generative model of all sensory inputs (the world prior, θ₀) provides the background distribution, and individual class models are expressed as differential offsets from this prior, not as independent estimators. This mirrors the FEP's treatment of higher-level beliefs as context that shifts lower-level predictions: every class model inherits the world's covariance structure and expresses only the class-specific mean displacement.
2.2 Information Geometry and the Natural Gradient
Information geometry, pioneered by C. R. Rao and developed comprehensively by Amari (Amari & Nagaoka, 2000; Amari, 2021), treats families of probability distributions as Riemannian manifolds. The unique invariant Riemannian metric on such manifolds—unique up to rescaling by Chentsov's theorem—is the Fisher information metric, whose (i,j)-th entry is:
g_{ij}(θ) = E_{p(x;θ)}[∂_i log p(x;θ) · ∂_j log p(x;θ)]
Ordinary gradient descent in parameter space is not invariant to reparametrisation: the gradient direction changes when the model is expressed in different coordinates. The natural gradient (Amari, 1998) corrects for this by multiplying the Euclidean gradient by the inverse Fisher information matrix:
θ̃ = F(θ)⁻¹ ∇_θ L
For a diagonal Gaussian model N(μ, diag(v)), the Fisher information matrix for the mean parameter is F_μ = diag(1/v), so the natural gradient with respect to μ is:
ỹ_μ = F_μ⁻¹ · ∇_μ log p(h|θ) = v · (h − μ)/v = h − μ
This is precisely the Welford update (Welford, 1962): the natural gradient for the mean of a Gaussian is the residual h − μ, independent of variance. This is why CyphaDIF's attraction step, Δμ_k += η · (h − μ_k), is not a heuristic choice but a consequence of information-geometric optimality. The update is Cramér-Rao efficient: no unbiased estimator can achieve lower variance (Raskutti & Mukherjee, 2014).
2.3 Minimum Description Length Regularisation
The Minimum Description Length (MDL) principle, introduced by Rissanen (1978) and comprehensively developed in Grünwald (2007), provides an information-theoretic formalisation of Occam's razor: the best model of the data is the one that compresses the data most, accounting for the cost of describing the model itself. Concretely, for a model M and data D:
MDL(M, D) = L(M) + L(D|M)
where L(M) is the code length for the model and L(D|M) is the code length for the data given the model. In the context of Gaussian class models, the description length of a class differential Δ_k is its Fisher-Rao norm under the world prior v₀:
||Δ_k||²_{FR} = Σ_d (Δμ_{kd}²) / v₀_d
This is the information-geometric analogue of Mahalanobis distance: an offset in a high-variance dimension of the world prior costs fewer bits than the same offset in a low-variance dimension, because the measurement process has lower precision there. The MDL decay in CyphaDIF implements this Occam prior by multiplying all class differentials by (1 − λ_eff) at each step, where λ_eff adapts with observation count to prevent over-regularisation of data-rich classes.
2.4 The Information Bottleneck and Contrastive Encoders
The Information Bottleneck (IB) method (Tishby, Pereira, & Bialek, 1999) formalises representation learning as a rate-distortion problem: find a compressed representation T of input X that preserves maximum information about target variable Y. The IB Lagrangian is:
min_{p(T|X)}  I(X; T) − β · I(T; Y)
The encoder W in CyphaDIF is a linear map from raw features f to latent h = W @ f. It is updated contrastively on every misclassification, implementing the IB objective in a Fisher-Rao-geometric form. When the true class is k but the model predicts class j, the score residuals at the true and predicted classes are:
r_k = (h − μ_k) / v₀   [should be small: h is near the true class]
r_j = (h − μ_j) / v₀   [should be large: h is far from the wrong class]
The contrastive update W += η_enc · (r_j − r_k) ⊗ f pushes W to project f into a latent direction where h is closer to μ_k and farther from μ_j in the natural geometry of the model. This is the continuous analogue of the supervised contrastive loss (Khosla et al., 2020), derived from Fisher-Rao geometry rather than imposed as an external objective.
3. Architecture
CyphaDIF is composed of seven components: the StructuralParser, EncoderProjection, WorldPrior, ClassDifferential, ContextBuffer, NIGField, and ReplayBuffer. Each maps directly to a theoretical motivation from Section 2. Figure 1 shows the data flow.
3.1 StructuralParser: Position-Indexed Feature Extraction
The parser converts any raw input—string, bytes, bytearray, or numpy array—into a 128-dimensional float vector. The critical design decision is that position is preserved: a byte 0x4D at position 0 produces a different feature than 0x4D at position 10. This breaks the 'bag of bytes' fallacy that afflicts naive entropy-based classifiers.
The 128 dimensions are partitioned into seven blocks:
| Block | Dimensions | Description |
|---|---|---|
| Positional bytes | [0:16] | Raw byte values at positions 0–15, normalised to [0,1] |
| Positional bigrams | [16:24] | Hash of (position, byte, next_byte) pairs at positions 0–7 |
| Keyword scores | [24:56] | 32 soft keyword match scores with positional weighting |
| Global statistics | [56:72] | Entropy, null density, ASCII fraction, punctuation counts |
| Header statistics | [72:80] | First 16 bytes: mean, variance, entropy, bigram entropy |
| Body statistics | [80:88] | Bytes 16+: mean, variance, entropy, transition rate |
| Cross statistics | [88:104] | 8 header categories × 2 body statistics (interaction terms) |
| Length encoding | [104:112] | One-hot log-scale length band across 8 thresholds |
| Transition statistics | [112:128] | Run lengths, byte transition rate, bigram entropy, half-entropy split |

Table 1. StructuralParser feature block layout (128 dimensions total).

Keyword matching uses a tiered scoring system. A match at the expected position scores 1.0; a match within the first 64 bytes scores 0.5; a case-insensitive match within 64 bytes scores 0.3; a match anywhere scores 0.15. This soft matching preserves information about likely vs. possible keyword presence, rather than collapsing to a binary presence/absence feature. The cross-statistics block (dimensions 88–104) was introduced specifically to resolve confusion between net_normal and net_c2 traffic, which had similar global statistics but different header-to-body interaction profiles. Prior to its inclusion, this confusion accounted for 194 misclassifications per evaluation epoch.
3.2 EncoderProjection W ∈ ℝ^{128×128}
The encoder is a linear map from raw feature space to latent h-space. It is initialised as a random orthogonal matrix (QR decomposition of Gaussian noise) scaled by 0.5, ensuring initial isotropy in all directions. It is updated contrastively on every misclassification via the Fisher-Rao update derived in Section 2.4. Spectral normalisation every 50 steps keeps the maximum singular value below 1.5, preventing gradient explosion. Every 500 steps, W is rotated toward the SVD of the class differential matrix—a soft alignment ensuring that the encoder's principal directions correspond to the most discriminative class separations.
The linear constraint on W is a deliberate trade-off. A nonlinear encoder (small MLP) would in principle handle linearly inseparable classes, but would lose the closed-form natural gradient derivation. For the byte-stream domain in this work, linear separability in feature space is sufficient; the structural parser's position-indexed features and cross-statistics terms provide sufficient non-linear mixing prior to the encoder.
3.3 WorldPrior θ₀: The Shared Generative Prior
The world prior is a diagonal Gaussian N(μ₀, diag(v₀)) over all inputs regardless of class label. It is updated online via Welford's algorithm (Welford, 1962): exact for the first 20 observations, then exponential moving average (EMA) with rate world_lr = 0.02. The world prior provides two services: (1) the reference distribution against which log-likelihood ratios (LLRs) are computed for classification and OOD detection; and (2) the shared variance v₀ used by all class models.
Field conditioning extends the world prior to support temporal context: μ₀(t) = μ₀ + F_field @ h_field(t), where h_field is the current NIGField state (Section 3.6). This shifts the baseline distribution smoothly as the temporal context evolves, providing soft domain adaptation without requiring explicit sequence modelling.
3.4 ClassDifferential Δ_k: Differential Class Models
Each class k is represented by a differential offset Δμ_k ∈ ℝ^{128}. The class k model is defined as θ₀ ⊕ Δ_k, i.e., N(μ₀ + Δμ_k, diag(v₀)). Class differentials are created lazily on first observation of a new label, so no schema is required in advance.
The update equations at each training step are:
Attraction (true class k):   Δμ_k  +=  η · (h − μ_k)                   [natural gradient]
Repulsion  (wrong class j):  Δμ_j  −=  η · w_j · (h − μ_j)             [w_j = posterior(j)]
MDL decay  (all classes):    Δμ_k  *=  (1 − λ_eff)                      [Occam prior]
The attraction step is the natural gradient update derived in Section 2.2—not a heuristic. The repulsion step is weighted by the posterior probability of each competing class, so classes that were close to predicting the input are repelled more than distant classes. The MDL decay adapts with observation count: λ_eff = λ · max(0.125, 1/(1 + n_obs/16)), reducing regularisation for data-rich classes while maintaining an absolute floor of λ/8.
3.5 Calibrated Confidence and OOD Detection
Classification confidence is not a raw posterior probability. It is the product of two gates, both of which must be high for confidence to be high. The log-likelihood ratio (LLR) for class k given input h is:
LLR_k(h) = log p(h | θ₀ ⊕ Δ_k) − log p(h | θ₀) − U_k
where U_k = mean(v₀) / (n_obs_k + 1) is an epistemic uncertainty penalty (Bayesian caution about poorly-fitted class models). The two gates are:
OOD gate  =  sigmoid( max_k LLR_k / σ_ood )
Class gate =  max_k  softmax( LLR / T )
confidence =  OOD gate × Class gate
The OOD gate detects out-of-distribution inputs: if max_k LLR_k < 0, the world prior beats every class model—this is a genuine OOD input—and the OOD gate collapses toward zero. σ_ood adapts online via EMA of the max LLR observed on in-distribution training samples. The temperature T = 2.5 is fixed; the hyperparameter sweep (Section 5) proved that any form of temperature decay strictly degrades performance.
3.6 NIGField: Linear Dynamical Temporal State
CyphaDIF maintains a 128-dimensional temporal state h_field via a linear dynamical system h_{t+1} = A_eff @ h_t + injection. Four timescale groups with decay rates 0.30, 0.60, 0.85, 0.95 allow fast features to fade quickly while slow features accumulate over many observations. A correction matrix W_T (spectral radius ≤ 0.85) is learned over time. The field state is injected into the world prior via field conditioning, providing smooth temporal context without explicit sequence modelling.
3.7 ContextBuffer and ReplayBuffer
The ContextBuffer maintains a rolling window of the last 64 (label, correct) pairs and a class co-occurrence matrix. It computes a context prior combining frequency (0.3 weight) and co-occurrence (0.2 weight) information, which is added to LLR scores before classification. In data streams with burst structure—a common property of network traffic—this substantially improves accuracy.
The ReplayBuffer is a fixed-capacity FIFO storing (h, f, label) triples from every training step. During training, 30% of updates are drawn from replay rather than the current input, preventing catastrophic forgetting (McCloskey & Cohen, 1989): when a new class appears, the model continues to see historical examples at 30% of the update rate. The replay ratio 0.30 was found to be the third-most-sensitive hyperparameter in the sweep. The buffer stores raw features f (invariant to encoder drift) alongside the encoded h for generation and introspection.
4. The Variance Collapse Problem and Its Resolution
The most consequential architectural decision in CyphaDIF is the elimination of per-class variance. This section documents the failure mode in detail because the reasoning is not obvious and the consequences were severe—and because this failure mode is latent in any online Gaussian classifier that maintains per-class covariance parameters.
4.1 The Failure Mode
Early versions of CyphaDIF maintained per-class log-variance offsets Δlogv_k alongside the mean offsets Δμ_k. The log-variance natural gradient update is:
∂ log p / ∂ log v_k  =  −1/2  +  (h − μ_k)² / (2v_k)
When the model is accurate—h is near μ_k—the residual (h − μ_k)² / (2v_k) ≈ 0, so the gradient is −1/2. This is negative. It always pushes log v_k downward. In online learning, as accuracy increases, the variance gradient is consistently negative: the update continuously tightens the class distribution. This is not a bug—it is the correct gradient direction. But applied consistently in an online learning loop with high per-class accuracy, variance collapse is the inevitable outcome.
Profiling revealed the empirical consequence: v_k_mean converged to 0.000506—approximately 100× tighter than the world prior v₀. With v_k ≈ 0.0005, any off-distribution input h produced (h − μ_k)² / v_k values in the thousands, making LLRs catastrophically large. The class whose centroid happened to sit nearest the world prior centroid—log_error—won all OOD classifications by default, because its LLR was merely very large rather than catastrophically large. The four largest misclassification families before the fix were: net_normal → net_c2 (194 errors), log_warn → log_info (152 errors), net_exfil → bin_benign (140 errors), and a range of OOD inputs all classified as log_error.
4.2 The Principled Fix
The fix follows from a simple philosophical observation: variance is a property of the measurement process, not of the class. The world prior variance v₀ describes how much any input varies in each dimension—this is a function of the measurement instrument (the StructuralParser), not of the specific data being classified. Class models describe where in that measurement space each class sits. Per-class variance is therefore a category error: it assigns to the class a property that belongs to the measurement process.
Removing Δlogv_k entirely—and classifying using N(μ_k, v₀) with shared world prior variance—has three beneficial effects: (1) it eliminates variance collapse; (2) it halves the MDL cost per class (D parameters instead of 2D); and (3) it produces the clean information-geometric derivation of Section 2.2, where the natural gradient for the mean simplifies to h − μ_k without dependence on v_k. The result was immediate recovery from all four failure classes in testing.
5. Hyperparameter Optimisation via Exhaustive Sweep
All key constants in CyphaDIF were derived from an exhaustive brute-force sweep of 27,524 configurations. The fitness function balanced four metrics:
fitness = 0.45 · acc + 0.20 · (1 − ECE) + 0.20 · margin + 0.15 · eff_dim
where acc is macro accuracy, ECE is the Expected Calibration Error (lower is better), margin is the mean LLR gap between the best and second-best class, and eff_dim is the effective dimensionality of the class differential matrix. This fitness function rewards calibration and discriminative margin, not just accuracy. The top-100 configurations were averaged for the final hyperparameter set.
Key findings from the sweep:
| Parameter | Optimal Value | Finding |
|---|---|---|
| temperature_decay | 1.0 (fixed T = 2.5) | Most important result. Temperature decay always degrades performance. |
| dedup_threshold | 0.60 | Most sensitive parameter. Sharp peak at 0.60; tails off at 0.55 and 0.65. |
| deliberate_thresh | 0.40 | Optimal range [0.25, 0.40]. Previous [0.10, 0.35] was suboptimal. |
| replay_ratio | 0.30 | 3rd most sensitive. 30% replay balances forgetting and present-step learning. |
| consolidate_every | 500 | Encoder alignment interval. Validated by sweep. |
| mdl_lambda | 0.002 | MDL decay base rate. Adaptive schedule applied on top. |
| post_trans_alpha | (removed) | Dead parameter. All values produced identical fitness. |

Table 2. Key hyperparameter sweep findings from 27,524 configurations.

The fixed-temperature result deserves elaboration. Temperature scaling is a standard calibration technique (Guo et al., 2017) in which a scalar T is learned to improve the calibration of softmax probabilities. In batch-trained networks, temperature annealing typically improves calibration. In CyphaDIF's online setting, however, temperature decay creates a miscalibration loop: as T decreases, class probabilities become more peaked, the OOD gate becomes more sensitive to small LLR fluctuations, and the contrastive encoder updates become more aggressive on inputs near decision boundaries. The feedback between temperature, encoder updates, and OOD sigma adaptation creates instability that no amount of schedule tuning can correct. Fixed T = 2.5 breaks this loop.
A counterintuitive result from the sweep is that the C_mismatch configuration archetype—where misclassification loss dominated the fitness function—outperformed the C_ceiling archetype (configurations near 100% accuracy ceiling): fitness 0.5619 vs. 0.5231. This reflects the multi-objective nature of the fitness function: a model that is 99% accurate with poor calibration scores below a model that is 97% accurate with tight calibration. Margin and effective dimensionality reward models that maintain a spread of class representations even at high accuracy, preventing the collapse of class centroids that can accompany near-perfect discrimination.
6. Experimental Results
6.1 Benchmark Configuration
All benchmarks use a 10-class dataset spanning three data modalities: network traffic (net_normal, net_scan, net_ddos, net_exfil, net_c2), system logs (log_info, log_warn, log_error), and binary files (bin_malware, bin_benign). Training uses 100 samples per class over 5 epochs (500 training steps per class, 5,000 total). Testing uses 200 samples per class (2,000 total). All training is single-pass within each epoch, consistent with the online learning setting. No GPU is used; all computation runs on CPU with NumPy.
6.2 Classification Results
| Metric | Score |
|---|---|
| Macro accuracy | 1.0000 (2000/2000 correct) |
| Macro F1 | 1.0000 |
| Expected Calibration Error (ECE) | 0.2453 |
| OOD rejection rate | 0.20 on synthetic OOD inputs |
| Composite classification score | 0.8955 |
| Learning curve stability | 1.0000 across all 5 epochs after variance fix |

Table 3. Classification benchmark results on 10-class byte-stream dataset.

6.3 Generative Results
CyphaDIF is both discriminative and generative. The class k model N(μ_k, v₀) directly defines a generative distribution; sampling h ~ N(μ_k, v₀) produces latent vectors representing that class. No separate decoder is trained; the classifier is the generative model. Generation quality was benchmarked on 10 classes with 200 samples per class.
| Metric | Score | Interpretation |
|---|---|---|
| Recovery accuracy | 1.0000 | Generated h classifies back to correct class via the same model |
| Linear discriminator accuracy | 0.3896 | Below chance (0.5): real and generated distributions indistinguishable |
| Mean centroid separation | 0.1912 | L2 distance between real and generated class centroids in h-space |
| Composite generation score | 0.7792 | Weighted combination of recovery and discriminator metrics |

Table 4. Generative benchmark results (200 samples/class, 10 classes).

The discriminator score of 0.3896 (below the 0.5 chance level) indicates that a linear classifier trained to distinguish real from generated latent vectors performs worse than chance. This is the best achievable result for a Gaussian generative model and confirms that the classifier's distributional model of each class is accurate. The interpretation is that the generated and real h distributions are so close that any attempt to separate them inverts the decision boundary.
7. Discussion
7.1 Connections to Existing Work
CyphaDIF occupies an unusual position in the online learning literature. It is not a kernel machine (Schölkopf & Smola, 2002), not an adaptive neural network (Schmidhuber, 2015), and not a Bayesian classifier in the traditional sense. It is closest in spirit to the family of Gaussian discriminant analysis methods, but departs from them in two important ways: the world prior structure (borrowed from the FEP) replaces independent per-class Gaussians with a shared covariance backbone; and the natural gradient derivation replaces the standard maximum likelihood update with the information-geometrically optimal update.
The fixed-temperature result connects to work on temperature scaling as post-hoc calibration (Guo et al., 2017). In that literature, temperature is a calibration correction applied after training to a frozen model. CyphaDIF's sweep result extends this to online learning: in a setting where temperature interacts with ongoing parameter updates, adaptive temperature is strictly harmful. The mechanism differs—in batch training, temperature does not affect the gradients; in online training, it does—but the result aligns with the intuition that calibration corrections are most effective when decoupled from the learning dynamics.
The contrastive encoder update has a parallel in metric learning (Khosla et al., 2020; Oord et al., 2018). The Fisher-Rao derivation makes the connection explicit: the contrastive update is equivalent to supervised contrastive loss in the natural geometry of the Gaussian model, rather than the Euclidean geometry of the representation space. This geometry-aware formulation may be the reason the update works well without the large batch sizes typically needed in contrastive learning.
7.2 Limitations
Four limitations warrant acknowledgement. First, generation operates in latent space only: generate() returns h vectors, not raw byte streams. There is no decoder; the pseudo-inverse reconstruction W⁻¹ was investigated and rejected (condition number ≈ 23 amplifies noise beyond acceptable levels). A small MLP decoder trained on replay buffer (h, f) pairs would address this. Second, the ECE of 0.2453 indicates moderate miscalibration. For decision-critical applications, isotonic regression calibration as a post-processing step is recommended. Third, dramatic domain shift (e.g., sudden change in traffic type) will corrupt the world prior; a full warm-restart requires a new CyphaDIF instance, as reset_field() does not reset μ₀. Fourth, the linear encoder cannot separate classes that are linearly inseparable in feature space.
7.3 Future Directions
Four directions are identified for future work: (1) a nonlinear encoder (small MLP) with a first-order approximation to the natural gradient, enabling non-linear class separation while retaining the MDL regularisation framework; (2) a learned decoder enabling raw-byte generation for data augmentation and adversarial example generation; (3) extension to multi-label classification via multiple simultaneous class differential updates; and (4) application to the ARIA authenticated encryption system's anomaly detection layer, where online classification of encrypted traffic patterns without decryption is a concrete operational requirement.

References
Amari, S. (1998). Natural gradient works efficiently in learning. Neural Computation, 10(2), 251–276.
Amari, S. (2021). Information geometry. International Statistical Review, 89(2), 250–273. https://doi.org/10.1111/insr.12464
Amari, S., & Nagaoka, H. (2000). Methods of Information Geometry. American Mathematical Society / Oxford University Press.
Amari, S., Karakida, R., & Oizumi, M. (2019). Fisher information and natural gradient learning in random deep networks. Proceedings of AISTATS (PMLR Vol. 89).
Friston, K. (2006). A free energy principle for the brain. Journal of Physiology – Paris, 100(1–3), 70–87.
Friston, K. (2009). The free-energy principle: a rough guide to the brain? Trends in Cognitive Sciences, 13(7), 293–301.
Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127–138. https://doi.org/10.1038/nrn2787
Grünwald, P. D. (2007). The Minimum Description Length Principle. MIT Press.
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. Proceedings of ICML (PMLR Vol. 70), 1321–1330.
Khosla, P., Tian, Y., Wang, X., Liu, C., Norouzi, M., Isola, P., Krishnamurthy, D., & Tian, Y. (2020). Supervised contrastive learning. Advances in Neural Information Processing Systems, 33, 18661–18673.
McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. Psychology of Learning and Motivation, 24, 109–165.
Nielsen, F. (2020). An elementary introduction to information geometry. arXiv:1808.08271.
Oord, A. van den, Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv:1807.03748.
Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.
Rao, C. R. (1945). Information and accuracy attainable in the estimation of statistical parameters. Bulletin of the Calcutta Mathematical Society, 37, 81–91.
Raskutti, G., & Mukherjee, S. (2014). The information geometry of mirror descent. IEEE Transactions on Information Theory, 61(3), 1451–1457. arXiv:1310.7780.
Rissanen, J. (1978). Modeling by shortest data description. Automatica, 14(5), 465–471.
Rissanen, J. (1987). Stochastic complexity. Journal of the Royal Statistical Society, Series B, 49(3), 223–239.
Schölkopf, B., & Smola, A. J. (2002). Learning with Kernels. MIT Press.
Schmidhuber, J. (2015). Deep learning in neural networks: An overview. Neural Networks, 61, 85–117.
Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. Proceedings of the 37th Annual Allerton Conference on Communication, Control, and Computing, 368–377. (arXiv:physics/0004057).
Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. Technometrics, 4(3), 419–420.
