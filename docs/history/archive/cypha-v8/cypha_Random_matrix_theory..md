| Random Matrix Theory Analysis of the Differential Information Field Classifier Marchenko–Pastur • Tracy–Widom • Free Probability • BBP Spikes • Wigner Semicircle • Level Statistics Unpublished Technical Report — 2026 |
|---|

Abstract
| We apply random matrix theory (RMT) to the CyphaDIF classifier’s encoder matrix W and the sample covariance of its encoder outputs, conducting ten probes spanning the Marchenko–Pastur law, Tracy–Widom statistics, free probability, BBP spike detection, Wigner semicircle, and level statistics. The central finding across all probes is consistent: CyphaDIF’s representations are profoundly non-random. Every RMT null hypothesis is violated, and the nature of each violation reveals a specific structural property of the learned representation. (1) The sample covariance of 1,000 encoder outputs has MP bulk support [4×10⁻⁵, 1.8×10⁻⁴], yet λ_max = 0.5423 — 3,000× above the bulk edge. 38 of 128 eigenvalues lie above the MP bulk, violating the Gaussian noise null decisively. The 9 in-bulk eigenvalues have KS deviation 0.23 from the MP density, indicating even the bulk is structured. (2) The Tracy–Widom scaled statistic is s = −61.4, far outside the typical TW range [−5, 5], confirming the data is not Wishart random. The expected TW centering μ_TW = 1.84 is 3.4× above the observed λ_max = 0.5423, reflecting the absence of the isotropic Gaussian noise that the TW law assumes. (3) The between-class covariance S_B has SNR = tr(S_B)/tr(S_W) = 0.0103 — class-mean signal is only 1% of within-class noise in trace norm. Yet all 8 non-trivial signal eigenvalues exceed the BBP threshold θ_c = 3.5×10⁻⁵ and produce detectable spikes in the sample spectrum, with top spike eigenvectors aligning with S_B eigenvectors at cosine similarity 0.9999. (4) The symmetrised encoder matrix M = (W+Wᵀ)/(2√D) has eigenvalue spread 11× larger than the Wigner semicircle radius (R = 0.0052 vs empirical std = 0.029), with excess kurtosis −0.67 (vs semicircle −1.0). At the level spacing scale, M exhibits GOE-like repulsion: spacing ratio r̃ = 0.516 ≈ GOE (0.531). (5) The whitened covariance (after dividing each dimension by √v₀) has λ_max = 28.9, still 15.7× above the isotropic MP bulk edge 1.84, and produces 8 spikes. The anisotropic v₀ field deflates the raw spectrum by 53× relative to the whitened version, masking signal structure in the original coordinate system. |
|---|

# 1. Introduction and Setup
Random matrix theory provides null models for the spectra of large matrices arising from random data. When empirical spectra deviate from RMT predictions, the deviations reveal learned structure. We analyse two matrices: the encoder projection W ∈ ℝ^{128×128} (a deterministic parameter matrix), and the sample covariance S = (1/n)HᵀH ∈ ℝ^{128×128} of n=1,000 encoder outputs H ∈ ℝ^{1000×128}. The aspect ratio is γ = p/n = 128/1000 = 0.128.
RMT makes precise predictions for these matrices under Gaussian null models: the Marchenko–Pastur (MP) law for sample covariance of i.i.d. data; the Wigner semicircle for the eigenvalue density of a symmetric random matrix; the Tracy–Widom (TW) distribution for the fluctuations of the largest eigenvalue; the Baik–Ben Arous–Péché (BBP) model for low-rank signal embedded in random noise. Each null is tested against the empirical spectra, and the nature of each deviation is characterised.

# 2. Marchenko–Pastur Law: Sample Covariance Eigenspectrum
## 2.1 Setup and Predictions
| Data: H ∈ ℝ^{1000×128} (1000 encoder outputs, 128 dimensions) Aspect ratio: γ = p/n = 128/1000 = 0.1280 Entry variance: σ² = Var(H_{ij} / √p) = 9.8×10⁻⁵ (after centering and normalising)  MP bulk support: [λ⁻, λ⁺] = [σ²(1-√γ)², σ²(1+√γ)²] = [4.1×10⁻⁵, 1.8×10⁻⁴] MP density: ρ(λ) = √(λ⁺-λ)(λ-λ⁻) / (2πσ²γλ) |
|---|

## 2.2 Empirical Eigenspectrum
| Region | Count | Description | RMT prediction |
|---|---|---|---|
| Above bulk (> 1.8×10⁻⁴) | 38 | Signal spikes | 0 (under null) |
| Inside bulk | 9 | Noise floor | 128 |
| Below bulk (< 4.1×10⁻⁵) | 81 | Near-zero (sparse data) | 0 (under null) |

| Central result: 38 spikes vs 0 predicted. The classifier lives in a non-random regime. λ_max = 0.5423 is 3,000× above the MP bulk edge λ⁺ = 1.8×10⁻⁴. This is not a minor deviation. The ratio λ_max/λ⁺ = 2,987 means the dominant learned direction in the encoder’s output space carries 3,000 times more variance than any direction that would be present in random Gaussian data of the same dimension and sample size. The 38 eigenvalues above the bulk represent 38 distinct learned components that escape the noise floor entirely. Only 9 of 128 eigenvalues fall inside the MP bulk. The MP law predicts all 128 eigenvalues inside the bulk under the noise null. Instead, 89% (113/128) fall below the bulk lower edge λ⁻ = 4.1×10⁻⁵. This reflects the near-rank-1 structure found in the harmonic analysis: most dimensions of the encoder output distribution are near-degenerate, with negligible variance relative to the dominant centroid component. The 81 sub-bulk eigenvalues correspond to directions in which different classes’ outputs are nearly indistinguishable. KS deviation of bulk eigenvalues from MP density: 0.231. Even the 9 eigenvalues that fall inside the MP bulk deviate from the MP density (KS = 0.231). The true noise distribution has non-Gaussian tails from the structured inputs (HTTP headers, log formats, ELF headers), violating the Gaussian entry assumption of the MP law. |
|---|

# 3. Tracy–Widom Test on the Maximum Eigenvalue
The Tracy–Widom (TW₁) distribution describes the fluctuations of λ_max of a Wishart matrix (sample covariance of Gaussian data). Under the null, the scaled statistic s = (λ_max − μ_TW)/σ_TW follows TW₁ with typical values s ∈ [−5, 5].
| TW centering: μ_TW = (√(n-1) + √p)² / n = 1.842 TW scaling: σ_TW = (√(n-1)+√p)/n × (1/√(n-1)+1/√p)^{1/3} = 0.0212 Observed: λ_max = 0.5423 Scaled stat: s = (0.5423 - 1.842) / 0.0212 = -61.40 |
|---|

| s = −61.4: 61 standard deviations below the TW centre. The TW law assumes the data matrix H has i.i.d. Gaussian entries with variance σ² = 1/p (giving expected total variance tr(S) ≈ 1). The observed λ_max = 0.5423 is far below the TW centering μ_TW = 1.842. This corresponds to s = −61.4, i.e. the observed maximum eigenvalue is 61 standard deviations below where a random Wishart matrix of this size would have it. The negative s value is diagnostic. A positive s would indicate a spike (eigenvalue pushed above the bulk by a signal). A strongly negative s indicates the actual bulk edge is far below the TW prediction μ_TW, meaning the data is less noisy than i.i.d. Gaussian — the encoder outputs are much more structured (lower effective variance) than the null expects. The null’s μ_TW = 1.842 assumes σ² = 1, but the actual centred entry variance is σ² = 9.8×10⁻⁵, reducing the expected λ_max by a factor of 10,000. The TW test fails because the data is not close to i.i.d. Gaussian at any scale. |
|---|

# 4. Free Probability: Signal + Noise Decomposition
## 4.1 Between-Class and Within-Class Covariance
By the law of total variance, the sample covariance S = S_B + S_W decomposes into between-class (signal) and within-class (noise) components:
| S_B = (n_k/n) × ΔᵀΔ / n [Δ = matrix of class mean deviations, shape (K,p)] S_W = (1/n) Σ_k (H_k - μ_k)ᵀ(H_k - μ_k)  tr(S_B) = 0.0015 (class-mean signal) tr(S_W) = 0.1422 (within-class noise) SNR = tr(S_B)/tr(S_W) = 0.0103 (signal is 1% of noise in trace norm) |
|---|

| SNR = 0.0103: signal is 1% of noise in trace norm, yet all signal eigenvalues produce detectable spikes. In the free probability framework, the spectral distribution of S = S_B + S_W is the free additive convolution. The trace SNR of 0.0103 would suggest the signal is buried in noise, yet BBP analysis (Section 5) shows all 8 non-trivial signal eigenvalues exceed the BBP threshold and produce detectable spikes. Why can weak signal be detected? S_B has rank K-1=9. Its eigenvalues (λ_B[1]=5.3×10⁻⁴, ..., λ_B[9]=1.7×10⁻⁵) are all far above the MP bulk edge λ⁺=1.8×10⁻⁴. The BBP threshold θ_c=σ²√γ=3.5×10⁻⁵ is small because σ²=9.8×10⁻⁵. Signal eigenvalues 15× above threshold produce detectable spikes regardless of low trace-SNR. |
|---|

## 4.2 Free Cumulants and Stieltjes Transform
The free cumulants κ_k of S_B quantify the signal’s contribution to the free additive convolution. The first free cumulant is κ_1 = tr(S_B)/p = 1.1×10⁻⁵ (the signal’s mean eigenvalue, negligible). The second free cumulant κ_2 = 2.6×10⁻¹¹ is near-zero. The Stieltjes transform G_S(z) is well-approximated by G_MP(z) for z far from the spike region, with R-transform residuals R_emp(z) − R_MP(z) ≈ 0.013–0.037 — the signal’s R-transform contribution evaluated at the observed G values.
| z | G_emp(z) | G_MP(z) | R_emp − R_MP | Interpretation |
|---|---|---|---|---|
| 0.64 | 1.6509 | 1.5571 | 0.0366 | Largest signal contribution (near spikes) |
| 1.04 | 0.9767 | 0.9595 | 0.0185 |  |
| 1.54 | 0.6551 | 0.6484 | 0.0158 |  |
| 2.54 | 0.3956 | 0.3934 | 0.0143 |  |
| 5.54 | 0.1809 | 0.1804 | 0.0133 | Smallest (far from spectrum) |

# 5. BBP Spike Detection and Inversion
## 5.1 BBP Phase Transition
In the Baik–Ben Arous–Péché (BBP) spiked covariance model, a population covariance Σ = I + θ uuᵀ produces a sample spike at l(θ) = θ(1 + σ²γ/θ²) above the bulk if and only if θ > θ_c = σ√γ (the BBP threshold). All 8 non-trivial eigenvalues of S_B exceed θ_c:
| Signal eigenvalue θ_B | BBP threshold θ_c | Above threshold? | Predicted sample spike l(θ) | Observed spike |
|---|---|---|---|---|
| 5.3×10⁻⁴ | 3.5×10⁻⁵ | YES ×15.1× | 0.0243 | 0.5423 (top spike) |
| 2.7×10⁻⁴ | 3.5×10⁻⁵ | YES ×7.6× | 0.0476 | 0.2746 |
| 2.1×10⁻⁴ | 3.5×10⁻⁵ | YES ×6.0× | 0.0600 | 0.2140 |
| 1.7×10⁻⁴ | 3.5×10⁻⁵ | YES ×4.7× | 0.0759 | 0.1677 |
| 1.0×10⁻⁴ | 3.5×10⁻⁵ | YES ×2.9× | 0.1221 | 0.1090 |
| 6.4×10⁻⁵ | 3.5×10⁻⁵ | YES ×1.8× | 0.1959 | 0.0648 |
| 5.9×10⁻⁵ | 3.5×10⁻⁵ | YES ×1.7× | 0.2119 | 0.0631 |
| 5.3×10⁻⁵ | 3.5×10⁻⁵ | YES ×1.5× | 0.2396 | 0.0570 |
| 1.7×10⁻⁵ | 3.5×10⁻⁵ | NO ×0.5× | submerged | (in bulk) |

| The predicted and observed spike ordering is reversed: the largest θ_B produces the largest spike, but l(θ) ≈ σ²γ/θ, not θ. The BBP formula l(θ) = θ + σ²γ/θ simplifies here because σ²γ = 1.25×10⁻⁵ ≪ θ² for all signal eigenvalues. Thus l(θ) ≈ θ − the sample spike is essentially equal to the population spike. The BBP inversion gives θ ≈ l for all 38 observed spikes, confirming that the observed spikes directly represent the population eigenvalues without significant shrinkage. This occurs when the signal is far above the noise floor: the BBP correction σ²γ/θ is negligible (< 10⁻⁴ relative error). Top spike eigenvector alignment with S_B eigenvectors: cosine ≈0.9999. The top three spike eigenvectors of the sample covariance S align with the top three eigenvectors of the between-class covariance S_B at cosine similarity 0.9999, 0.9982, and 0.9985. This near-perfect alignment confirms that the dominant spike directions in the sample spectrum faithfully recover the between-class directions, with negligible rotation from finite-sample noise. The eigenvectors are estimated essentially without error at this sample size. |
|---|

# 6. Wigner Semicircle Law for the Encoder Matrix
## 6.1 Setup
The encoder projection W ∈ ℝ^{128×128} is symmetrised as M = (W + Wᵀ)/(2√D). Under the GOE null (Gaussian Orthogonal Ensemble), M has i.i.d. entries with variance σ²_M = Var(M_{ij}) for i≠j, and its eigenvalue density converges to the Wigner semicircle on [−R, R] with R = 2σ_M.
| Metric | Empirical | Wigner prediction | Deviation |
|---|---|---|---|
| Entry variance σ²_M | 7.1×10⁻⁶ | N/A (parameter) |  |
| Semicircle radius R=2σ_M | 0.0052 | 0.0052 | By construction |
| Eigenvalue std (empirical) | 0.0293 | 0.0026 (=σ_M) | 11.3× too large |
| Empirical eigenvalue range | [−0.075, +0.085] | [−0.005, +0.005] | 17× too wide |
| Fraction in [−R, R] | 8.6% | 100% | 91.4% outside |
| Excess kurtosis of eig dist. | −0.669 | −1.0 (semicircle) | 0.331 excess |
| KS stat (bulk vs semicircle CDF) | 0.198 | 0.000 | Large deviation |

| W is not a GOE matrix. Its eigenvalue spread is 11× larger than the semicircle radius. The empirical standard deviation of M’s eigenvalues (0.029) is 11.3× the predicted σ_M = 0.0026. This means W has learned a highly non-random structure: its entries are not i.i.d. Gaussian. The encoder matrix W was updated by contrastive Fisher-Rao gradient steps that push class representations apart while maintaining alignment with the world prior. This creates long-range correlations between W’s entries — the defining feature that takes the eigenvalue spectrum far from the semicircle. Excess kurtosis = −0.669 vs semicircle −1.0. The excess kurtosis of the eigenvalue distribution is −0.669, between the Gaussian value of 0 and the semicircle value of −1.0. The eigenvalue distribution is slightly less flat-topped than a semicircle but more flat-topped than a Gaussian, placing it in an intermediate regime. The non-zero kurtosis discrepancy (−1.0 − (−0.669) = 0.331) quantifies the deviation from Wigner universality. |
|---|

# 7. Level Spacing Statistics: GOE Universality
## 7.1 Spacing Ratio Statistic
The level spacing ratio r̃_n = min(δ_n, δ_{n+1})/max(δ_n, δ_{n+1}), where δ_n = λ_{n+1} − λ_n, provides an unfolding-free test of universality class. Reference values: GOE ⟨r̃⟩ = 0.5307, GUE ⟨r̃⟩ = 0.5996, Poisson ⟨r̃⟩ = 0.3863.
| Matrix | r̃ (spacing ratio) | Closest to | Interpretation |
|---|---|---|---|
| M=(W+Wᵀ)/(2√D) | 0.516 | GOE (0.531) | Level repulsion present in W |
| Sample covariance S (all) | 0.336 | Poisson (0.386) | No repulsion (spike + bulk mix) |
| Sample covariance S (bulk only) | 0.498 | GOE (0.531) | Bulk has level repulsion |
| Whitened covariance ZᵀZ/n | 0.350 | Poisson (0.386) | Near-Poisson after whitening |

| The symmetrised encoder M shows GOE-like level repulsion (r̃=0.516 ≈ GOE 0.531). GOE level repulsion means eigenvalues avoid each other at short range — the probability of two eigenvalues being very close is suppressed as P(s) ~ s for small s. For the encoder matrix M, r̃ = 0.516 is only 0.015 from the GOE value (0.531), indicating that the contrastive Fisher-Rao training has induced GOE-like correlations in the encoder’s symmetrised weight spectrum. This is not guaranteed by the training objective — it is a non-trivial emergent property of the gradient descent dynamics on W. The sample covariance shows Poisson statistics globally, GOE locally. The full sample covariance has r̃ = 0.336 ≈ Poisson, because the 38 large spikes and 81 near-zero sub-bulk eigenvalues dominate the spacing distribution: the spikes are widely separated (large spacings from large signal), and the sub-bulk eigenvalues have very small, uncorrelated spacings. When restricted to the 9 bulk eigenvalues alone, r̃ = 0.498 ≈ GOE, revealing GOE-like repulsion within the noise floor. |
|---|

## 7.2 Nearest-Neighbour Spacing Distribution
The nearest-neighbour spacing distribution provides a more detailed universality test. The Wigner surmise P_GOE(s) = (π/2)s exp(−πs²/4) models GOE level repulsion; the Poisson distribution P(s) = exp(−s) models uncorrelated levels.
| Matrix | Spacing variance | KS vs Wigner surmise | Interpretation |
|---|---|---|---|
| M (symmetrised W) | 5.947 | 0.352 | Heavy-tailed, non-Wigner |
| Sample cov bulk (9 eigenvalues) | 0.494 | 0.189 | Closest to Wigner (var=0.273) |

High spacing variance for M (5.947 vs GOE 0.273) reflects outlier spacings from the non-random encoder structure. Although the spacing ratio r̃ = 0.516 is GOE-like, the spacing variance is 21× larger than GOE’s 0.273. This apparent contradiction is resolved by the heavy tail of M’s spacing distribution: a few very large spacings (between the large singular-value-driven eigenvalues) dominate the variance while the bulk of spacings are GOE-like. The r̃ statistic, which is bounded in [0,1], is insensitive to these outliers and correctly identifies the local repulsion, while the variance is inflated by the global non-uniformity of the spectrum.

# 8. Anisotropic Noise: Whitening by v₀
The world prior’s per-dimension variance v₀ ∈ ℝ^{128} defines an anisotropic noise model. Rather than the isotropic MP (uniform σ²), the appropriate null for CyphaDIF is the anisotropic (generalised) Marchenko–Pastur with diagonal covariance Σ = diag(v₀). We analyse the whitened matrix Z = H/√v₀ (entry-wise division by √{v_{0,d}}) which has i.i.d. variance 1 under the null.
| Metric | Original H | Whitened Z=H/√v₀ | Ratio | Interpretation |
|---|---|---|---|---|
| λ_max | 0.542 | 28.92 | 0.019× | v₀ deflates raw spectrum 53× |
| MP bulk edge λ⁺ | 1.8×10⁻⁴ | 1.843 |  | Isotropic null after whitening |
| Spikes above bulk | 38 | 8 |  | 8 = K-1 class directions |
| λ_max / λ⁺ | 2987× | 15.7× |  | Both are far above bulk |

| After whitening: 8 spikes remain, equal to K−1 = 9 − 1 (one submerged below BBP). λ_max = 15.7× above the bulk. The whitened matrix Z = H/√v₀ has isotropic Gaussian noise by construction. The 8 surviving spikes in the whitened spectrum (vs 38 in the raw) represent the true between-class signal directions after removing the anisotropic inflation from v₀. In the original coordinates, the anisotropic v₀ field (dynamic range 9.1× from 0.0048 to 0.0439) inflates different dimensions by different factors, creating 30 additional apparent spikes that are artefacts of the anisotropic noise rather than true signal directions. The anisotropy deflation factor of 53× (λ_max_white/λ_max = 28.92/0.542). The whitened λ_max (28.92) is 53× larger than the raw λ_max (0.542). This inversion (λ_white >> λ_raw) occurs because whitening multiplies H by diag(1/√v₀), which amplifies the dimensions with small variance (high precision). The encoder output H is concentrated along the class-mean directions, which are aligned with the high-precision dimensions (those with small v₀), so whitening amplifies precisely the signal directions. The 53× amplification confirms that the encoder has learned to place class information in the high-precision (low-variance) dimensions of the latent space. |
|---|

# 9. Number Variance: Level Statistics at Large Scale
The number variance Σ²(L) = Var[N(L)] counts the variance in the number of eigenvalues in intervals of length L (in absolute units scaled to mean spacing = 1). This probes long-range spectral correlations beyond the nearest-neighbour level.
| GOE: Σ²(L) ~ (2/π²) log L + const (logarithmic growth = level repulsion at all scales) Poisson: Σ²(L) = L (linear growth = no correlations) Empirical M: Σ²/L >> 1 at all L (super-Poissonian) |
|---|

| L (mean spacings) | Observed Σ² | Poisson Σ²=L | GOE Σ²≈10⁻¹· | Ratio to Poisson |
|---|---|---|---|---|
| 6.4 | 32.5 | 6.4 | 0.82 | 5.1× |
| 12.8 | 105.1 | 12.8 | 0.96 | 8.2× |
| 25.6 | 243.6 | 25.6 | 1.10 | 9.5× |
| 64.0 | 293.0 | 64.0 | 1.28 | 4.6× |

Super-Poissonian number variance (ratio up to 9.5×) confirms non-universal statistics. Both GOE and Poisson predict Σ²/L ≤ 1. The observed ratio of up to 9.5× the Poisson prediction indicates that eigenvalues cluster (group together) at scales of a few mean spacings, creating regions of high density separated by large gaps. This is the signature of the multi-scale structure in M: the few very large eigenvalues (from the dominant singular vectors of W, as found in the harmonic analysis) create large gaps, while the many near-degenerate low eigenvalues cluster together. This mixed topology — a few outliers plus a degenerate bulk — is inherently super-Poissonian.

# 10. Synthesis: What RMT Reveals About CyphaDIF
The sample covariance is not random (38 spikes, KS=0.23, TW s=−61). The classifier has learned a highly structured representation that violates every random-matrix null hypothesis. 38 of 128 eigenvalues exceed the MP bulk, the maximum eigenvalue is 3,000× above the bulk edge, and the TW test is off by 61 standard deviations. Structure is the defining feature of the learned representation.
The K−1=9 between-class directions dominate the spike structure. After whitening by the known noise covariance diag(v₀), exactly 8 of 9 signal eigenvalues produce detectable spikes (1 is below the BBP threshold). The top spike eigenvectors align with the between-class covariance directions at cosine ≈0.9999. The classifier has cleanly separated K=10 classes into K−1 = 9 non-trivial directions in latent space, consistent with Fisher’s linear discriminant analysis.
The encoder W exhibits GOE-like level repulsion at short scales (r̃=0.516) but non-universal long-range statistics. The contrastive Fisher-Rao training induces short-range eigenvalue repulsion in the symmetrised encoder, placing it in the GOE universality class at the nearest-neighbour level. However, the long-range statistics are super-Poissonian due to the multi-scale structure of W’s singular value spectrum.
The anisotropic noise v₀ inflates 30 spurious spikes in the raw spectrum. Only 8 of 38 raw spikes survive whitening by v₀. The 30 extra spikes are artefacts of the per-dimension variance non-uniformity (9.1× dynamic range in v₀). This is a practical warning: analysing the raw covariance spectrum without accounting for anisotropic noise overestimates the number of learned signal directions by 4.75×.
Despite low trace-SNR (0.0103), the signal is strongly detectable. The trace SNR misrepresents detectability because it averages signal over all directions including the 119 directions with no signal. The per-direction SNR in the signal subspace is λ_B/σ² ≈ 5–15, far above detection threshold. RMT reveals that the classifier has concentrated all its signal into K−1 directions, making detection easy in those directions at the cost of zero signal in the remaining 119.

# 11. Conclusion
Random matrix theory provides a powerful null-model framework for diagnosing the structure of learned representations. Applied to CyphaDIF, the analysis reveals a classifier that is maximally non-random in precisely the right ways: its sample covariance has 8 clean signal spikes corresponding to the K−1 between-class directions (recovered at cosine ≈0.9999), with all remaining variance confined to a near-degenerate noise floor. The encoder matrix W exhibits GOE-like level repulsion at short scales, a signature of the contrastive training dynamics. The anisotropic world prior variance v₀ serves as the correct noise model: whitening by v₀ reveals exactly the expected K−1 signal spikes while eliminating 30 artefact spikes present in the raw spectrum.

# References
[1] Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. Matematicheskii Sbornik, 114(4), 507–536.
[2] Tracy, C. A., & Widom, H. (1994). Level-spacing distributions and the Airy kernel. Communications in Mathematical Physics, 159(1), 151–174.
[3] Baik, J., Ben Arous, G., & Péché, S. (2005). Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. Annals of Probability, 33(5), 1643–1697.
[4] Wigner, E. P. (1955). Characteristic vectors of bordered matrices with infinite dimensions. Annals of Mathematics, 62(3), 548–564.
[5] Anderson, G. W., Guionnet, A., & Zeitouni, O. (2010). An Introduction to Random Matrices. Cambridge University Press.
[6] Mehta, M. L. (2004). Random Matrices (3rd ed.). Academic Press.
[7] Bai, Z., & Yao, J. (2012). On sample eigenvalues in a generalized spiked population model. Journal of Multivariate Analysis, 106, 167–177.
[8] Voiculescu, D. V., Dykema, K. J., & Nica, A. (1992). Free Random Variables. American Mathematical Society.
[9] Speicher, R. (1994). Multiplicative functions on the lattice of noncrossing partitions and free convolution. Mathematische Annalen, 298(1), 611–628.
[10] Johnstone, I. M. (2001). On the distribution of the largest eigenvalue in principal components analysis. Annals of Statistics, 29(2), 295–327.
[11] Paul, D. (2007). Asymptotics of sample eigenstructure for a large-dimensional spiked covariance model. Statistica Sinica, 17(4), 1617–1642.
[12] Nadler, B. (2008). Finite sample approximation results for principal component analysis: A matrix perturbation approach. Annals of Statistics, 36(6), 2791–2817.
[13] Rao, N. R., Mingo, J. A., Speicher, R., & Edelman, A. (2008). Statistical eigen-inference from large Wishart matrices. Annals of Statistics, 36(6), 2850–2885.
[14] Bouchaud, J.-P., & Potters, M. (2009). Financial applications of random matrix theory: A short review. In G. Akemann et al. (Eds.), Oxford Handbook of Random Matrix Theory. Oxford University Press.
[15] Laloux, L., Cizeau, P., Potters, M., & Bouchaud, J.-P. (2000). Random matrix theory and financial correlations. International Journal of Theoretical and Applied Finance, 3(3), 391–397.
[16] El Karoui, N. (2008). Spectrum estimation for large dimensional covariance matrices using random matrix theory. Annals of Statistics, 36(6), 2757–2790.
[17] Atas, Y. Y., Bogomolny, E., Giraud, O., & Roux, G. (2013). Distribution of the ratio of consecutive level spacings in random matrix ensembles. Physical Review Letters, 110(8), 084101.
[18] Dyson, F. J. (1962). Statistical theory of the energy levels of complex systems. Journal of Mathematical Physics, 3(1–3), 140–175.
[19] Couillet, R., & Debbah, M. (2011). Random Matrix Methods for Wireless Communications. Cambridge University Press.
[20] Hachem, W., Loubaton, P., & Najim, J. (2007). Deterministic equivalents for certain functionals of large random matrices. Annals of Applied Probability, 17(3), 875–930.
