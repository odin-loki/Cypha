| Harmonic Analysis of the Differential Information Field Classifier Transfer Functions • Gram Spectra • PSD Analysis • Graph Fourier Transform • Harmonic Decomposition Unpublished Technical Report — 2026 |
|---|

Abstract
| We apply harmonic analysis to the CyphaDIF classifier’s encoder, parser, and class representations across ten probes. The encoder matrix W is treated as a linear filter with a measurable transfer function; the class offset vectors δ_k as discrete signals; the centroid graph as a domain for a graph Fourier transform; and the sequence of encoder outputs per class as a multivariate time series. Key findings: (1) The encoder W has a mild 1/f⁰·² spectral decay (power-law slope −0.20) across its 128 singular modes, effective rank 113.8 (near-full), and condition number 13.27. The singular spectrum is nearly flat (spectral flatness 0.965), indicating W acts as a near-white filter rather than a selective frequency amplifier. (2) Encoder rows concentrate power in the mid-to-high frequency band (CoM = 31/64, 64.8% of rows mid-dominated, HF fraction 0.50). The encoder is a high-pass filter relative to index space: it suppresses slowly-varying index patterns and amplifies rapidly oscillating ones. The input–output gain ratio G(high)/G(low) = 44.3×, confirming strong high-pass behaviour. (3) The Gram matrix of all 1,000 encoder outputs has effective dimension 1.24, with a single dominant eigenvalue capturing 89.7% of variance and a spectral gap of 0.960. Per-class Gram matrices have effective rank ≈1.00–1.06: each class’s encoder outputs lie essentially on a ray through the origin, a consequence of the near-deterministic encoding of structured inputs. (4) The class offset vectors δ_k have broadly distributed harmonic content (flatness 0.78–0.90) with peak frequencies scattered across [8, 52]. bin_malware has the highest total power (2.52), consistent with its diffuse, high-variance latent distribution. Pairwise harmonic similarity averages 0.799: class offset spectra are moderately but not identically shaped. (5) The graph Laplacian of the K=10 centroid graph has Fiedler value λ₁ = 4.85 (large algebraic connectivity, meaning the centroid graph is well-connected). The graph Fourier transform of the self-LLR signal concentrates 90.4% of power in mode 0 (the DC component), confirming that LLR values are smooth on the class graph — nearby classes have similar self-LLR values. (6) All 10 class output sequences have near-identical spectral fingerprints (cosine similarity ≥ 0.9999), meaning the encoder maps all traffic classes to the same power distribution over frequency. The LLR sequences are high-SNR (SNR up to 7.5×10⁶ for log classes) with near-flat AC spectra, confirming that within-class LLR variation is spectrally white. |
|---|

# 1. Introduction
Harmonic analysis provides a frequency-domain characterisation of the CyphaDIF pipeline that complements the spatial (Euclidean, topological) analyses of previous papers. By treating the encoder matrix W as a linear filter, the class offset vectors δ_k as discrete signals indexed by dimension, the encoder output sequences as time series, and the centroid graph as a signal domain, we obtain a multi-scale spectral picture of what the classifier has learned.
Six distinct spectral objects are analysed: the singular value spectrum of W (Section 2), the frequency content of W’s rows and columns (Section 3), the eigenspectrum of the Gram matrix (Section 4), the power spectral density of encoder output sequences (Section 5), the harmonic content of the class offset vectors δ_k (Section 6), and the graph Fourier transform of the LLR signal on the centroid graph (Section 8). Together these reveal the frequency-domain fingerprint of the learned representation.

# 2. Encoder Transfer Function: SVD Analysis
## 2.1 Singular Value Spectrum
The encoder matrix W ∈ ℝ^{128×128} maps feature vectors f to latent representations h = Wf. In the frequency-domain language, the singular values σ_i of W are the “gains” of the encoder along its principal modes. The singular value decomposition W = UΣVᵀ decomposes the encoder into input modes (rows of Vᵀ), amplitudes (Σ), and output modes (columns of U).
| Metric | Value | Interpretation |
|---|---|---|
| σ_max | 1.697 | Largest gain (dominant encoding mode) |
| σ_min | 0.128 | Smallest gain (least-amplified mode) |
| σ_mean | 0.451 | Average gain |
| Condition number κ(W) | 13.27 | Moderate: no extreme mode dominance |
| Effective rank | 113.8 | Near-full: W uses 113.8 of 128 effective modes |
| Spectral flatness | 0.965 | Near 1.0: singular spectrum is nearly flat (white) |
| Power-law slope α | −0.204 | Mild 1/f⁰·² decay across all 128 modes |
| Slope (first 32 modes) | −0.407 | Steeper 1/f⁰·⁴ decay in high-gain modes |

| Key result The encoder is a near-full-rank, near-white-noise filter with mild 1/f spectral decay. The effective rank of 113.8 out of 128 (88.9%) confirms that W is almost a bijection in terms of information capacity. The spectral flatness of 0.965 (close to the maximum of 1.0) means the singular values are nearly uniform, with no single mode dominating. The power-law slope −0.20 is much flatter than the −1/f slope of 1/f noise (slope −1) or the −2 slope of Brownian motion, placing the encoder’s spectral character between white and pink noise in the dimension-mode domain. |
|---|

Cumulative power in top-k singular modes is broadly distributed. The top-1 mode captures only 9.83% of total power, the top-8 capture 28.3%, and 64 modes are needed to reach 64.2%. This broad distribution is consistent with the high effective rank: no small subset of modes dominates the encoder. By contrast, a low-rank encoder would have ≥90% of power in the first few modes. CyphaDIF’s W is high-rank by design, preserving information from all 128 feature dimensions equally.
## 2.2 Coupling Between Singular Modes and Variance Precision
The right singular vectors v_i (rows of Vᵀ) define the input directions probed by each singular mode. We compute the precision-weighted norm of each v_i: Π_i = Σ_d |v_{i,d}|² / v_{0,d}, which measures how much the mode concentrates weight in high-precision (low-variance) dimensions.
| Rank | Mode | Singular value σ | Precision weight Π | Interpretation |
|---|---|---|---|---|
| 1 (highest Π) | 1 | 1.697 | 123.6 | Highest-gain mode also highest-precision |
| 2 | 100 | 0.412 | 99.98 |  |
| 3 | 101 | 0.412 | 96.63 |  |
| 4 | 49 | 0.414 | 96.13 |  |
| 5 | 83 | 0.413 | 95.13 |  |
| Low Π (rank 128) | 127 | 0.341 | 67.00 | Lowest-precision mode, also lowest-σ |
| Low Π (rank 127) | 11 | 0.521 | 63.53 |  |

Pearson r(σ_i, Π_i) = 0.193: weak positive coupling. The dominant singular mode (mode 1, σ = 1.697) also has the highest precision weight (Π = 123.6), suggesting the encoder’s principal amplification direction is aligned with high-precision input dimensions. However, the overall correlation is weak (r = 0.193), meaning the singular spectrum is not systematically driven by the precision structure of v₀. Most modes probe a mixture of high- and low-precision dimensions.

# 3. Frequency Content of Encoder Rows and Columns
## 3.1 Row Frequency Analysis
Each row of W defines a linear functional f ↦ wᵀ_d f, mapping the 128-dim feature vector to a single encoder dimension. Treating f as a 128-point discrete signal (indexed by position in the feature vector), the DFT of each row reveals which “input patterns” that encoder dimension responds to. A row concentrated at low frequencies responds to slowly-varying patterns (DC, trends); a high-frequency row responds to rapid oscillations in the feature index.
| Key result Encoder rows concentrate power in the mid-to-high frequency band (CoM ≈ 31/64). The mean centre-of-mass of the row frequency content is 30.98 (out of 63), with 64.8% of rows mid-dominated (CoM 8–31) and 35.2% high-dominated (CoM ≥32). Zero rows are low-frequency dominated (CoM < 8). The mean power fractions are: LF = 0.135, MF = 0.369, HF = 0.496. The encoder is predominantly a mid-to-high-frequency filter in the feature index domain — it suppresses slowly-varying patterns and emphasises rapidly oscillating index patterns. |
|---|

Both row and column frequency distributions are centred near the Nyquist midpoint. Row CoM: mean 31.0, std 2.3, range [24, 36]. Column CoM: mean 31.6, std 2.4. The narrow standard deviation (2.3–2.4) indicates that all rows have similar frequency content — the encoder does not specialise different output dimensions for different frequency bands. This uniform distribution is consistent with the near-flat singular spectrum: the encoder is an “omnidirectional” filter without strong frequency selectivity.
## 3.2 Input–Output Transfer Function
The effective transfer function of the parser+encoder pipeline is the frequency-dependent gain G(f) = |FFT(h)|² / |FFT(input_bytes)|², measuring how much each frequency component of the raw input is amplified or attenuated by the pipeline.
| Frequency band | Mean gain G(f) | Relative to DC | Interpretation |
|---|---|---|---|
| DC (f=0) | 3.8×10⁻⁷ | 1.0 | DC completely suppressed |
| Low (f=1–7) | 3.6×10⁻⁵ | 95× | Low freq also strongly attenuated |
| Mid (f=8–31) | 5.9×10⁻⁴ | 1,550× | Mid freq moderately passed |
| High (f=32+) | 1.6×10⁻³ | 4,200× | High freq most amplified |
| Peak (f=41) | 5.8×10⁻³ | 15,300× | Global peak near Nyquist |

| The pipeline is a strong high-pass filter (gain ratio 44.3× high vs low). The 44.3× ratio of high-to-low frequency gain is a direct consequence of the StructuralParser’s position-encoding scheme. The parser maps input bytes to a 128-dim feature vector via position-indexed functions (e.g., position × byte value, modular arithmetic over position). These position encodings suppress the DC component (constant offset) and emphasise positional patterns that vary rapidly along the feature index. The encoder W, trained to separate classes, then further amplifies these high-frequency positional features because they carry more class-discriminative information than the low-frequency (slowly-varying) components. Peak gain varies by class. net_scan peaks at f=41 (G=2.1×10⁻²), log_warn peaks at f=35 (G=4.3×10⁻², the largest peak gain overall), and bin_malware peaks at f=44 (G=8.3×10⁻⁴). Binary classes have the lowest peak gains — consistent with their random payload bytes providing broadband input signal power that distributes across all output frequencies without concentrating at any peak. |
|---|

# 4. Gram Matrix Spectral Analysis
## 4.1 Global Gram Matrix
The Gram matrix G = (1/D) H Hᵀ, where H ∈ ℝ^{1000×128} is the matrix of all encoder outputs, characterises the effective kernel of the encoder: G_{ij} = ⟨h_i, h_j⟩/D measures the inner-product similarity between all 1,000 encoded samples. Its eigenspectrum reveals the intrinsic dimensionality of the encoder’s output.
| λrank | Eigenvalue | % variance | Cumul. % | Interpretation |
|---|---|---|---|---|
| 1 | 103.7 | 89.65% | 89.65% | Dominant mode: shared centroid structure |
| 2 | 4.2 | 3.61% | 93.26% | Class-cluster separation |
| 3 | 2.0 | 1.73% | 95.00% |  |
| 4 | 1.6 | 1.41% | 96.41% |  |
| 5 | 1.3 | 1.13% | 97.54% |  |
| 10 | 0.15 | 0.13% | 99.37% |  |
| 50+ | <0.02 | <0.03% | 100% | Near-zero (noise floor) |

| Effective dimension = 1.24 with spectral gap 0.960. The Gram matrix is dominated by a single eigenvalue (λ₁ = 103.7) capturing 89.7% of variance, with a gap of (λ₁−λ₂)/λ₁ = 0.960. This reflects the fact that all 1,000 encoder outputs, across all 10 classes, share a common large-magnitude component — the norm of each h is O(10) — while the inter-class structure (the components that distinguish classes) is captured by the remaining eigenvalues (λ₂ through λ₅, sum ≈7%). The effective dimension of 1.24 means the 1,000-point encoder output distribution is essentially 1-dimensional at the global scale, with weak multi-class structure on top of the dominant radius component. |
|---|

## 4.2 Per-Class Gram Matrices
Computing the Gram matrix G_k for each class’s 100 encoder outputs separately reveals the within-class structure:
| Class | Eff. rank | Top eigenvalue λ₁ | λ₁/Σλ | Interpretation |
|---|---|---|---|---|
| net_normal | 1.02 | 12.12 | 0.988 | Near-rank-1: tight cluster on a ray |
| net_scan | 1.01 | 7.96 | 0.993 |  |
| net_ddos | 1.00 | 11.88 | 0.999 | Exactly rank-1 |
| net_exfil | 1.01 | 12.86 | 0.996 |  |
| net_c2 | 1.01 | 13.85 | 0.993 |  |
| log_info | 1.00 | 11.52 | 1.000 | Exactly rank-1 |
| log_warn | 1.00 | 11.35 | 1.000 | Exactly rank-1 |
| log_error | 1.00 | 10.88 | 1.000 | Exactly rank-1 |
| bin_malware | 1.06 | 10.71 | 0.971 | Slightly above rank-1 (random payload) |
| bin_benign | 1.05 | 11.55 | 0.975 | Slightly above rank-1 |

Per-class Gram matrices are rank-1 to within 1.5×10⁻². The effective rank of 1.00–1.06 for all classes means that each class’s 100 encoder outputs are nearly collinear — they all point in the same direction in ℝ^{128} up to a scalar amplitude. This is the harmonic manifestation of the tight within-class distributions found in the persistent homology analysis: the log classes have exactly rank-1 Gram matrices (λ₁/Σλ = 1.000), while binary classes have slightly higher rank (eff_rank ≈1.05–1.06) due to their random payload bytes introducing weak orthogonal variance.

# 5. Power Spectral Density of Encoder Outputs
## 5.1 Per-Class PSD
Treating the 100 encoder outputs per class as a multivariate time series H_k ∈ ℝ^{100×128} (100 samples × 128 dimensions), the power spectral density is computed along the sample dimension (axis 0), then averaged over the 128 encoder dimensions.
| Class | DC power | Spectral flatness | Dominant AC freq | Character |
|---|---|---|---|---|
| net_normal | 12.09 | 0.882 | 39 | High flatness, near-white |
| net_scan | 7.95 | 0.765 | 10 | Moderate flatness |
| net_ddos | 11.88 | 0.670 | 5 | Most structured (lowest flatness) |
| net_exfil | 12.86 | 0.895 | 48 | Near-white |
| net_c2 | 13.85 | 0.773 | 24 |  |
| log_info | 11.52 | 0.874 | 16 | High flatness |
| log_warn | 11.35 | 0.843 | 26 |  |
| log_error | 10.88 | 0.810 | 44 |  |
| bin_malware | 10.68 | 0.959 | 37 | Near-maximal flatness |
| bin_benign | 11.51 | 0.963 | 43 | Near-maximal flatness |

Mean spectral flatness = 0.843: encoder output sequences are near-white. The spectral flatness of 0.843 (maximum = 1.0 for pure white noise) indicates that the encoder output variations across samples are broadly distributed in frequency with no strongly dominant periodicity. The binary classes (bin_malware, bin_benign) have the flattest spectra (0.959–0.963), consistent with their random payload bytes producing i.i.d. encoder outputs. The network classes (net_ddos flatness 0.670, net_scan 0.765) have the most structured PSDs, reflecting the structured numerical variation in their inputs (packet rates, port numbers).
| All 45 pairs of class spectral fingerprints have cosine similarity ≥0.9999. The frequency fingerprint (mean PSD vector over 50 one-sided frequencies) is identical across all 10 classes. This is not merely high similarity — it is numerical identity to 4 decimal places. The encoder maps all traffic classes to the same power distribution over sample-index frequency, regardless of class. This confirms that the encoder’s frequency response is class-invariant: it does not selectively amplify different frequency components for different classes. The class-discriminative information is encoded in the DC component (the mean h value per class, i.e., the centroid μ_k) rather than in any frequency structure. |
|---|

# 6. Harmonic Content of Class Offset Vectors
## 6.1 FFT of δ_k Vectors
Each class offset vector δ_k ∈ ℝ^{128} (the learned deviation of class k from the world prior in latent space) is a 128-point discrete signal indexed by encoder dimension d = 0, …, 127. The DFT of δ_k reveals whether the class offset is a “smooth” pattern (low-frequency) or “oscillatory” (high-frequency) in the dimension index.
| Class | DC amplitude | Peak frequency | Peak amplitude | Flatness | Total power |
|---|---|---|---|---|---|
| net_normal | 0.97 | 15 | 1.68 | 0.786 | 0.488 |
| net_scan | 0.92 | 14 | 1.96 | 0.807 | 0.981 |
| net_ddos | 2.41 | 35 | 2.43 | 0.770 | 1.720 |
| net_exfil | 2.23 | 52 | 2.47 | 0.895 | 1.595 |
| net_c2 | 1.91 | 34 | 3.89 | 0.836 | 1.875 |
| log_info | 0.41 | 32 | 2.55 | 0.871 | 1.159 |
| log_warn | 0.50 | 44 | 2.65 | 0.820 | 1.131 |
| log_error | 1.13 | 32 | 2.09 | 0.875 | 0.867 |
| bin_malware | 0.91 | 8 | 3.74 | 0.860 | 2.523 |
| bin_benign | 0.25 | 22 | 2.41 | 0.843 | 1.209 |

net_ddos and net_c2 have the highest DC amplitude (2.41, 1.91). A large DC component in FFT(δ_k) means the class offset has a large constant bias across all 128 dimensions — the class centroid μ_k is uniformly shifted from μ₀ in all directions. This is consistent with the network traffic classes (which have consistent structural patterns such as fixed payload rates or beacon URIs) learning a globally offset representation.
bin_malware has the highest total power (2.523) and low peak frequency (f=8). The large total power of bin_malware’s offset spectrum reflects its large ||delta_k||_V = 11.53 (the largest precision-weighted offset). The peak at f=8 (relatively low frequency) suggests the malware class offset has a structured, slowly-varying pattern in dimension index space rather than the high-frequency oscillation seen in other classes. This may reflect the parser’s position-encoding scheme amplifying certain byte-position patterns in MZ headers.
Pairwise harmonic similarity averages 0.799 (range 0.706–0.910). The class offset spectra are moderately similar: the pairwise cosine similarity of |FFT(δ_i)| and |FFT(δ_j)| ranges from 0.706 (net_ddos↔bin_benign) to 0.910 (bin_malware↔bin_benign). The binary classes are harmonically the most similar pair (0.910), reflecting their common byte-array structure. net_ddos and bin_benign are the most harmonically distinct (0.706), reflecting fundamentally different input types (structured network floods vs. random ELF content).

# 7. Variance Spectrum
The per-dimension variance vector v₀ ∈ ℝ^{128} (shared across world prior and classes) is itself a 128-point signal whose spectral structure characterises the prior’s frequency-domain profile.
| Metric | Value | Interpretation |
|---|---|---|
| DC amplitude of FFT(v₀) | 1.967 | Large DC: variance is nearly uniform across dims |
| Peak AC frequency | 5 | Mild periodicity at low freq (dim 0–25) |
| AC peak amplitude | 0.168 | Small relative to DC (8.5% of DC) |
| Spectral flatness of FFT(v₀) | 0.876 | Near-flat: variance spectrum broadly distributed |
| v₀ range | [0.0048, 0.0439] | 9.1× dynamic range across dims |

The variance spectrum is predominantly DC (nearly uniform v₀). The DC amplitude of FFT(v₀) dominates (1.967) with AC components at most 0.168 (8.5% of DC). The spectral flatness of 0.876 confirms that the small AC variation in v₀ is itself broadly distributed, not concentrated at any particular frequency. The 9.1× dynamic range ([0.0048, 0.0439]) is real variation in per-dimension precision but is not organised in a periodic pattern.

# 8. Graph Fourier Transform on the Centroid Graph
## 8.1 Graph Laplacian
We define a weighted graph on K=10 class centroids with edge weights A_{ij} = exp(−||μ_i−μ_j||²/(2σ²)) (Gaussian kernel, σ² = median squared distance). The graph Laplacian L = D − A encodes the centroid graph’s connectivity structure. Its eigenvectors form a basis for signals on the class graph (the graph Fourier basis).
| Mode | Eigenvalue λ | Interpretation |
|---|---|---|
| 0 | 0.0000 | DC mode (constant on all classes) |
| 1 | 4.845 | Fiedler mode (algebraic connectivity) |
| 2 | 5.562 | Second AC mode |
| 3 | 5.771 |  |
| 4 | 5.926 |  |
| 5 | 6.291 |  |
| 6–9 | 6.58–7.17 | High-frequency graph modes |

| Fiedler value λ₁ = 4.845: high algebraic connectivity. The Fiedler value (second-smallest Laplacian eigenvalue) measures how well-connected the graph is. λ₁ = 4.845 is large, reflecting the dense weighting of the Gaussian kernel on a set of 10 centroids with relatively uniform inter-centroid distances (all pairwise distances in [1.06, 1.67]). The graph is approximately a complete weighted graph with no bottlenecks or cluster cuts — consistent with the topological analysis showing a single connected component with a narrow merge-scale range. |
|---|

The Fiedler vector (eigenvector corresponding to λ₁) shows a dominant negative loading on bin_malware (−0.903) with small positive loadings on all other classes. This indicates bin_malware is the class most “different” from the remaining centroid cluster, consistent with its being the last class to merge in the dendrogram (Section 3 of the persistent homology paper) and having the largest ||delta_k||_V.
## 8.2 Graph Fourier Transform of LLR Signal
We define a scalar signal on the centroid graph: s_k = LLR_k(μ_k) (the self-LLR of each class evaluated at its own centroid). The graph Fourier transform (GFT) ŝ = Uᵀ s decomposes this signal into the graph eigenbasis.
| Mode | Eigenvalue | GFT power |ŝ|² | % of total | Significance |
|---|---|---|---|---|
| 0 (DC) | 0.000 | 18,896 | 90.4% | Dominant: LLR is nearly constant over classes |
| 1 | 4.845 | 352 | 1.7% | bin_malware separation |
| 2 | 5.562 | 310 | 1.5% |  |
| 3 | 5.771 | 143 | 0.7% |  |
| 4 | 5.926 | 544 | 2.6% |  |
| 9 (highest freq) | 7.175 | 505 | 2.4% |  |

| 90.4% of LLR signal power in the DC graph mode. The self-LLR values s_k = LLR_k(μ_k) range from 19.0 (net_normal) to 66.5 (bin_malware), but their variation over the centroid graph is mostly smooth (DC-dominated). A DC-dominated GFT means the LLR values do not vary sharply between adjacent (strongly-connected) classes — geometrically nearby classes in the centroid graph have similar self-LLR values. This is expected: the self-LLR of class k is proportional to ||δ_k||²_V/2, the precision-weighted squared norm of the class offset. Since adjacent classes tend to have similar offset magnitudes, the LLR signal is smooth on the graph. |
|---|

Total variation TV = 12,163 confirms moderate graph smoothness. The total variation TV = ŝᵀ Λ ŝ measures the energy in the AC components weighted by their frequency. TV = 12,163 is large in absolute terms (dominated by the large LLR magnitudes) but small relative to the DC power (18,896): the ratio TV/||ŝ||² = 12,163/20,871 = 0.583. A perfectly smooth signal would have TV = 0; a maximally rough signal (anti-correlated on all edges) would have TV ≈ 7.17 × 20,871 ≈ 150,000. The LLR signal sits at 0.583/7.17 ≈ 8% of maximum roughness — smooth but not trivially so.

# 9. Spectral Analysis of LLR Sequences
For each class k, the sequence of self-LLRs LLR_k(h_n) for n=1, …, 100 (ordered training samples) is a scalar time series. Its PSD reveals whether LLR values vary randomly across samples (flat spectrum) or with systematic periodic structure.
| Class | LLR flatness | Dominant AC freq | SNR (DC/AC) | Interpretation |
|---|---|---|---|---|
| net_normal | 0.692 | 42 | 4.0×10² | Moderate structure (URL/method variation) |
| net_scan | 0.633 | 28 | 1.6×10⁴ | Structured (port range variation) |
| net_ddos | 0.570 | 5 | 4.4×10⁴ | Most structured (packet rate range) |
| net_exfil | 0.607 | 48 | 3.3×10⁴ |  |
| net_c2 | 0.621 | 24 | 5.7×10³ |  |
| log_info | 0.650 | 33 | 5.6×10⁶ | Near-constant LLR (rigid format) |
| log_warn | 0.601 | 36 | 7.5×10⁶ | Highest SNR: most constant LLR |
| log_error | 0.522 | 14 | 1.3×10⁶ | Most structured log class |
| bin_malware | 0.588 | 37 | 1.8×10³ | Random payload → variable LLR |
| bin_benign | 0.687 | 20 | 1.4×10³ |  |

Log class LLR sequences have SNR up to 7.5×10⁶: effectively constant. The DC-to-AC power ratio (SNR) for log_warn is 7.5 million, meaning the mean LLR (DC) is 7.5 million times larger than any sample-to-sample variation. The rigid [TYPE] HH:MM:SS format produces near-identical encoder outputs for all log samples, resulting in a near-constant LLR sequence with negligible AC components. This is the spectral signature of the tight within-class distributions found in both the statistical analysis and persistent homology papers.
net_ddos has the lowest spectral flatness (0.570) and highest AC structure. The packet-count range (500–500,000) in net_ddos generates the largest within-class variation in numerical features, producing an LLR sequence that is the most structured (least flat) among all classes. The dominant AC frequency at bin 5 (period ≈20 samples) may reflect a quasi-periodic pattern in how the random packet counts interact with the parser’s position encoding.

# 10. Synthesis
The encoder is a near-white, high-rank linear filter (κ=13.27, eff_rank=113.8, flatness=0.965). No small subset of encoder modes dominates; information is spread across all 128 dimensions. The mild 1/f⁰·² spectral decay is a common property of learned representations and is far from the strongly correlated spectra of structured signals.
The encoder is a high-pass filter in the feature index domain (gain ratio 44.3×). The parser+encoder pipeline strongly suppresses DC and low-frequency patterns in the feature index, amplifying mid-to-high frequency positional patterns. This is the spectral signature of position-encoded feature extraction: slowly-varying byte patterns (globally uniform bytes) are suppressed in favour of rapidly-varying positional structure.
All encoder output sequences have identical spectral fingerprints (cosine sim ≥0.9999). The encoder’s frequency response is class-invariant. Class-discriminative information is encoded in the mean latent vector (centroid) rather than in any frequency-domain feature. The classifier is “DC-coded” in the sample-sequence domain.
The Gram matrix has effective dimension 1.24 globally (rank-1 per class). The 89.7% dominance of the first Gram eigenvalue confirms that all 1,000 encoder outputs share a common large-magnitude direction (the norm component), with class structure captured by the remaining 10.3%. Per-class Gram matrices are rank-1 to within 1.5×10⁻², with log classes exactly rank-1.
The self-LLR signal on the centroid graph is 90.4% DC (smooth). LLR values vary smoothly over the centroid graph: geometrically adjacent classes have similar self-LLR values. The Fiedler value 4.845 confirms a well-connected centroid graph with no bottlenecks or cluster cuts.
Log classes have LLR SNR up to 7.5×10⁶; binary classes have SNR ≈1,400–1,800. The spectral analysis quantifies the within-class LLR consistency: log classes produce near-constant LLR sequences while binary classes produce moderately variable ones. This 3-order-of-magnitude SNR gap directly reflects the 3-order-of-magnitude difference in within-class variance (tight logs vs. diffuse binaries).

# 11. Conclusion
The harmonic analysis reveals a consistent spectral picture of CyphaDIF: the encoder is a near-white, high-pass, high-rank linear filter that maps structured traffic inputs to near-collinear class representations (rank-1 per-class Gram matrices) with class-invariant frequency fingerprints. The class-discriminative information is encoded entirely in the mean latent vector (centroid), not in any frequency-domain variation. The LLR signal is smooth on the centroid graph (90.4% DC), confirming that the classifier’s confidence structure is a slowly-varying function of class geometry. The large Fiedler value (4.845) confirms a well-connected, bottleneck-free centroid graph.

# References
[1] Oppenheim, A. V., & Schafer, R. W. (2009). Discrete-Time Signal Processing (3rd ed.). Pearson.
[2] Strang, G. (1986). Introduction to Applied Mathematics. Wellesley-Cambridge Press.
[3] Mallat, S. (2009). A Wavelet Tour of Signal Processing (3rd ed.). Academic Press.
[4] Bracewell, R. N. (2000). The Fourier Transform and Its Applications (3rd ed.). McGraw-Hill.
[5] Candès, E. J., & Wakin, M. B. (2008). An introduction to compressive sampling. IEEE Signal Processing Magazine, 25(2), 21–30.
[6] Wiener, N. (1930). Generalised harmonic analysis. Acta Mathematica, 55, 117–258.
[7] Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., & Vandergheynst, P. (2013). The emerging field of signal processing on graphs. IEEE Signal Processing Magazine, 30(3), 83–98.
[8] Sandryhaila, A., & Moura, J. M. F. (2013). Discrete signal processing on graphs. IEEE Transactions on Signal Processing, 61(7), 1644–1656.
[9] Luxburg, U. von (2007). A tutorial on spectral clustering. Statistics and Computing, 17(4), 395–416.
[10] Fiedler, M. (1973). Algebraic connectivity of graphs. Czechoslovak Mathematical Journal, 23(2), 298–305.
[11] Chung, F. R. K. (1997). Spectral Graph Theory. American Mathematical Society.
[12] Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.). Johns Hopkins University Press.
[13] Trefethen, L. N., & Bau, D. (1997). Numerical Linear Algebra. SIAM.
[14] Schoenholz, S. S., Gilmer, J., Ganguli, S., & Sohl-Dickstein, J. (2017). Deep information propagation. ICLR 2017.
[15] Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. ICLR 2014.
[16] Percival, D. B., & Walden, A. T. (1993). Spectral Analysis for Physical Applications. Cambridge University Press.
[17] Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra. IEEE Transactions on Audio and Electroacoustics, 15(2), 70–73.
[18] Yaglom, A. M. (1987). Correlation Theory of Stationary and Related Random Functions. Springer.
[19] Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13, 281–305.
[20] Mahoney, M. W. (2011). Randomized algorithms for matrices and data. Foundations and Trends in Machine Learning, 3(2), 123–224.
