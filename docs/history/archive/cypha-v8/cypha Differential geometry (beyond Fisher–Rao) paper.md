| Differential Geometry of the Differential Information Field Classifier Riemannian Geometry • Christoffel Symbols • Holonomy • Jacobi Fields • Cartan Frames • Sectional Curvature Unpublished Technical Report — 2026 |
|---|

Abstract
| We analyse the differential geometry of the CyphaDIF classifier across three geometric domains: the statistical parameter manifold — the space of Gaussian distributions N(μ, v₀) parameterised by the class means μ_k; the encoder weight manifold — the Lie group GL(128) in which the encoder matrix W lives; and the feature space equipped with the encoder-induced Riemannian metric. Across ten probes, three geometric regimes are found. Flat regime (statistical manifold): The Gaussian family with fixed variance is a flat Riemannian manifold with zero Riemann curvature tensor, zero Christoffel symbols, trivial holonomy group {Id}, and path-independent parallel transport. Geodesics are straight lines γ(t) = μ₀ + tδ_k. The Fisher-Rao metric inflates distances by 8.06× relative to Euclidean; class geodesic distances range from 8.71 to 17.85. Every decision boundary is a totally geodesic hyperplane (geodesic curvature κ_g = 0) passing exactly through the midpoint of each class-pair geodesic, confirming Bayes-optimality in the Fisher metric. Curved regime (encoder manifold GL(128)): The encoder matrix W lives in GL(128), a curved Lie group. The sectional curvature of GL(128) with bi-invariant metric clusters tightly around K = 0.500 (mean 0.5004, std 0.0044) across 50 random tangent pairs — consistent with a rank-1 symmetric space where K ∈ {0, 1/4}. The polar decomposition W = Q·P reveals a large orthogonal rotation Q with ||Q−I||_F = 16.46 (of max 22.63) and mean rotation angle 93.35°. Ricci and scalar curvatures are approximately 63.6 and 8,135. Induced geometry (feature space): The encoder-induced metric g_enc = Wᵀ diag(1/v₀) W on feature space has eigenvalues spanning [1.07, 218.3], condition number 203.96, effective rank 51.38, and log-determinant 337.48. The Cartan moving frame e_i = columns of W⁻¹ is far from Fisher-orthonormal (||Gram−I||_F = 7,755). Jacobi fields on the flat statistical manifold grow linearly (Lyapunov exponent λ = 0), confirmed empirically: classification flips at ε ≈ 0.9 along the geodesic toward the nearest class. |
|---|

# 1. Geometric Setup
Three distinct geometric domains arise in CyphaDIF:
Statistical manifold Mₛ: {N(μ, v₀) : μ ∈ ℝ^d} with the Fisher-Rao (FR) metric G_ij = δ_ij/v_{0,i}. Since v₀ is shared across classes, the class means {(μ_k, v₀)} form a d-dimensional affine subspace within the full NIG manifold. Fixed-variance Gaussian families are flat: all curvature tensors vanish.
Weight manifold M_W: GL(128) = {W ∈ ℝ^{128×128} : det W ≠ 0}, a Lie group. With the bi-invariant metric g(X,Y) = tr(XᵀY)/d (left- and right-invariant), GL(d) has sectional curvature K = 1/4 for compact directions and 0 for non-compact ones. The encoder W lies in GL(128) and has been shaped by contrastive Fisher-Rao gradient descent.
Induced feature manifold M_f: The feature space ℝ^d equipped with the pulled-back metric g_enc = Wᵀ G_0 W, where G_0 = diag(1/v₀). This measures how the encoder stretches and rotates the input space relative to the Fisher metric on the output.
| Notation: d = 128 (dimension of encoder output and feature space) K = 10 (number of classes) v₀ ∈ ℝ^d_+ (shared world-prior variance, mean=0.0154) G₀ = diag(1/v₀) (Fisher metric matrix, precision matrix) μ₀ ∈ ℝ^d (world-prior mean) δ_k = μ_k - μ₀ (class offset, ||D||_G = Fisher distance from prior) W ∈ GL(128) (encoder projection matrix) |
|---|

# 2. Fisher-Rao Metric and Geodesic Distances
## 2.1 Riemannian Metric on the Statistical Manifold
For the Gaussian family N(μ, v₀) with fixed diagonal covariance diag(v₀), the Fisher information matrix is:
| G_ij(μ) = E[∂_i log p(x|μ) ∂_j log p(x|μ)] = δ_ij / v_{0,i} (diagonal, position-independent)  This is a flat metric: Christoffel symbols Γ^k_ij = 0 for all i,j,k Riemann tensor: R^l_kij = 0 (flat) Geodesic equation: dμ/dt = const (straight lines) |
|---|

The flatness of the Fisher metric on fixed-variance Gaussians is a standard result in information geometry. The Riemannian distance between two class means is:
| d_G(μ_i, μ_j) = ||μ_i - μ_j||_G = √( Σ_d (μ_{i,d} - μ_{j,d})^2 / v_{0,d} )  = ||δ_i - δ_j||_G (since μ_i = μ_0 + δ_i, terms cancel) |
|---|

## 2.2 Geodesic Distance Matrix
| Class pair | d_G (Fisher-Rao) | d_Eucl (Euclidean) | FR/Eucl ratio | Classification difficulty |
|---|---|---|---|---|
| bin_malware ↔ bin_benign | 8.71 | 1.054 | 8.26× | Closest pair (FR) |
| log_warn ↔ log_error | 9.39 | 1.165 | 8.06× |  |
| log_info ↔ log_warn | 9.89 | 1.227 | 8.06× |  |
| net_normal ↔ log_warn | 9.97 | 1.237 | 8.06× |  |
| … (mean) | 13.71 | 1.700 | 8.06× |  |
| net_c2 ↔ bin_malware | 17.85 | 2.215 | 8.06× | Farthest pair (FR) |
| net_exfil ↔ bin_malware | 17.48 | 2.174 | 8.04× |  |
| net_c2 ↔ net_ddos | 16.23 | 2.019 | 8.04× |  |

| FR/Euclidean ratio = 8.06× (nearly constant across all pairs). The near-constant ratio d_G/d_Eucl ≈ 8.06 across all 45 pairs is a striking geometric fact. In general, the FR metric inflates distances differently in different directions depending on v₀: dimensions with small v₀ (high precision) are stretched more. The near-constant ratio means that the class offsets δ_i − δ_j are approximately isotropic with respect to v₀: the difference vectors align equally with high- and low-precision dimensions. This is consistent with the harmonic analysis finding that the delta-vector spectra have similar flatness across classes (0.77–0.90). Fisher distances from the world prior μ₀: range [6.17, 11.53]. The closest class to the world prior in the Fisher metric is net_normal (||delta||_G = 6.17), consistent with HTTP traffic being the most ‘generic’ traffic type. The farthest is bin_malware (||delta||_G = 11.53), consistent with the MZ header producing the most distinctive feature pattern. These Fisher distances are the natural complexity measure for MDL (the description length L(δ_k) = ||delta||_G^2 / 2 = 19.0–66.5 nats). |
|---|

# 3. Christoffel Symbols, Connection, and Flat Geometry
The Christoffel symbols Γ^k_ij of a Riemannian manifold measure the failure of the coordinate basis to be parallel. For the Fisher metric G_ij = δ_ij/v_{0,i} (diagonal, position-independent):
| Γ^k_ij = (1/2) G^{kl} (∂_i G_{lj} + ∂_j G_{li} - ∂_l G_{ij})  Since G is constant (∂_i G_{lj} = 0 for all i,l,j): Γ^k_ij = 0 for all i, j, k  Geodesic equation: d²μ^k/dt² + Γ^k_ij (dμ^i/dt)(dμ^j/dt) = 0 Reduces to: d²μ^k/dt² = 0 ⇒ μ(t) = μ_0 + t·v (straight lines) |
|---|

The Levi-Civita connection on the statistical manifold is the standard Euclidean connection. This means parallel transport is the identity map: a tangent vector v transported along any path γ from p to q remains equal to v (as a vector in ℝ^d). The holonomy group (the group of all parallel transport maps along closed loops) is therefore the trivial group {I_d}.
Covariant derivatives reduce to ordinary derivatives. For any vector field V and any path γ, the covariant derivative D_{γ'}V = ∂_{γ'}V (ordinary directional derivative). This simplification is the geometric reason why the NIG classifier’s Fisher-Rao gradient updates are identical to Euclidean gradient updates rescaled by v₀ — the connection is flat, so there is no correction term.

# 4. Holonomy and Parallel Transport
## 4.1 Statistical Manifold Holonomy
The holonomy group Hol(∇, p) at a point p is the group of all linear maps on T_pM obtained by parallel transport around all closed loops through p. For a flat manifold:
| Hol(∇, μ_0) = {Id} (trivial holonomy)  Angle defect for a geodesic triangle (μ_i, μ_j, μ_k): α + β + γ = π (Euclidean angle sum, Gauss-Bonnet with K=0)  A tangent vector v transported around any closed loop returns to itself. |
|---|

## 4.2 Encoder Polar Decomposition and Rotation
The encoder W ∈ GL(128) admits the unique polar decomposition W = Q·P where Q is orthogonal (det Q = ±1) and P is symmetric positive definite. The orthogonal factor Q represents the rotational content of W; P represents the stretching.
| Metric | Value | Interpretation |
|---|---|---|
| ||Q − I||_F | 16.46 | 72.8% of maximum possible (2√128 = 22.63) |
| tr(Q)/d | −0.058 | Near-zero: Q rotates by ∼90° on average |
| Mean rotation angle | 93.4° | Near-maximal: Q is a large rotation |
| det(Q) | −1.000 | W has negative orientation (improper rotation) |
| ||P − I||_F | Varies | Stretching component: non-uniform scaling |
| Condition number κ(g_enc) | 203.96 | Strong anisotropy in encoder-induced metric |

| Q has mean rotation 93.4° and det(Q) = −1: the encoder applies a near-maximal improper rotation. The orthogonal factor Q of the polar decomposition has ||Q−I||_F = 16.46 out of a maximum of 22.63 (72.8%). This means the encoder has learned a large rotation in latent space: input feature patterns are rotated by an average of 93.4° before being fed to the class score functions. The negative determinant (det Q = −1) indicates an improper rotation (includes a reflection), which is geometrically valid for the discriminative task since reflections preserve inner products and therefore class margins. The symmetric factor P (stretching) has condition number 203.96 in the Fisher metric. The induced metric g_enc = Wᵀ G₀ W has eigenvalues ranging from 1.07 to 218.33, with effective rank 51.38 out of 128. This means the encoder effectively uses only 51 of 128 input directions, compressing the 128-dimensional feature space into a 51-dimensional effective subspace (measured in Fisher metric volume). The log-determinant log det(g_enc) = 337.48 gives the log-volume scaling: the encoder inflates the Fisher volume by a factor of e^{337.48/2} ≈ 10^{73}. |
|---|

# 5. Geodesic Curvature of Decision Boundaries
Decision boundary B_{ij} between classes i and j is the hyperplane:
| B_{ij} = { h ∈ ℝ^d : LLR_i(h) = LLR_j(h) } = { h : ⟨w_i - w_j, h⟩ = b_j - b_i } = { h : ⟨δ_i/v₀ - δ_j/v₀, h⟩ = const } |
|---|

As a subset of the flat Riemannian manifold (ℝ^d, G₀), a hyperplane is a totally geodesic submanifold. Its geodesic curvature κ_g measures how much the boundary curves relative to geodesics tangent to it:
| κ_g(B_{ij}) = 0 for all pairs (i,j)  All 45/45 boundaries have geodesic curvature exactly zero. All boundaries pass exactly through the midpoint of the class-pair geodesic. (midpoint μ_mid = (μ_i + μ_j)/2, confirmed: LLR_i(μ_mid) = LLR_j(μ_mid)) |
|---|

| Boundary alignment theorem: all 45 decision boundaries are perpendicular to the corresponding class geodesic (angle 0.000°) and pass through the geodesic midpoint. This is an analytic result, not a numerical coincidence. The LLR difference LLR_i(h) − LLR_j(h) = ⟨w_i−w_j, h⟩ + (b_i−b_j). The boundary normal in dual space is w_i−w_j = (δ_i−δ_j)/v₀. Raised to the primal space by G₀⁻¹ = diag(v₀), the primal normal is v₀·(w_i−w_j) = δ_i−δ_j = μ_i−μ_j. This is exactly the class geodesic direction. So the boundary is always perpendicular to the geodesic in the Fisher metric, and since the bias term is set to b_k = −⟨w_k, μ₀⟩ − ||δ_k||^2_V/2, the boundary passes through (μ_i+μ_j)/2 when the observation counts are equal. This is the condition for Bayes-optimal classification under equal Gaussian priors. The boundary is totally geodesic: it is a flat (K−2)-dimensional submanifold. For a flat ambient manifold, every hyperplane is totally geodesic (its second fundamental form vanishes). This means a geodesic that begins tangent to the boundary remains in the boundary — the boundary has zero extrinsic curvature. Geometrically, this means the boundary does not ‘bend’ in any direction, which is the optimal property for a classification boundary: no part of the boundary is unnecessarily curved into one class’s territory. |
|---|

# 6. Sectional Curvature of the Encoder Manifold
## 6.1 GL(d) with Bi-invariant Metric
The Lie group GL(d) equipped with the bi-invariant metric g(X,Y) = tr(XᵀY)/d (defined on the Lie algebra gl(d) and extended by left-translation) has sectional curvature:
| K(X,Y) = (1/4) ||[X,Y]||^2 / ( ||X||^2 ||Y||^2 - ⟨X,Y⟩^2 )  where [X,Y] = XY - YX is the Lie bracket (matrix commutator) and ||X||^2 = tr(X^T X)/d  For GL(d): K(X,Y) ≥ 0 (non-negative sectional curvature) For SO(d) (compact subgroup): K = 1/4 (constant curvature) For upper-triangular matrices: K = 0 (flat solvable subgroup) |
|---|

## 6.2 Measured Curvature at W
| Tangent pair | K(X_i, X_j) | Singular values | Interpretation |
|---|---|---|---|
| Random pairs (mean ± std) | 0.500 ± 0.004 | N/A | Clusters near 1/2 |
| (σ_1, σ_2) | 0.264 | 1.697, 1.143 | Large SVs: lower curvature |
| (σ_1, σ_3) | 0.016 | 1.697, 1.026 | Near-equal SVs: near-zero K |
| (σ_1, σ_5) | 1.486 | 1.697, 0.836 | Most disparate: highest K |
| (σ_3, σ_5) | 1.395 | 1.026, 0.836 | Near-equal small SVs: high K |
| (σ_4, σ_5) | 0.770 | 0.883, 0.836 |  |

| Mean sectional curvature K = 0.500 ≈ 1/2 across 50 random pairs (std 0.004). Consistent with a symmetric space. The tight clustering of K around 0.500 across random tangent pairs is a signature of a symmetric space or space of constant curvature. For the compact simple Lie group SU(n) with bi-invariant metric, all sectional curvatures are K = 1/4. For GL(d) the situation is more complex since GL(d) is non-compact, but the empirical K ≈ 0.5 is consistent with the dominant curvature contribution coming from the SU(128) ⊂ GL(128) compact factor. Curvature along singular directions ranges from 0.016 to 1.486. The curvature K(σ_1, σ_3) ≈ 0.016 (nearly flat) occurs between the dominant singular direction (σ_1 = 1.697) and a nearly equal singular direction (σ_3 = 1.026). The highest curvature K = 1.486 occurs between σ_1 and σ_5 (0.836), the most disparate singular value pair. High curvature between disparate directions indicates that the Lie bracket [X_1, X_5] is large: rotating from the dominant mode to the fifth mode is geometrically complex. This reflects the non-commutative structure of the encoder’s weight space. |
|---|

# 7. Ricci Curvature, Scalar Curvature, and Volume
The Ricci tensor Ric and scalar curvature R_scal are derived from the Riemann tensor by contraction. For our two geometric domains:
| Manifold | Riemann tensor | Ricci tensor | Scalar curvature | Note |
|---|---|---|---|---|
| Statistical M_s = {N(μ,v₀)} | R = 0 | Ric = 0 | R_scal = 0 | Flat: all curvature zero |
| Encoder M_W = GL(128) | K ≈ 0.500 | Ric ≈ 63.6·g | R_scal ≈ 8,135 | Positively curved |
| Feature M_f = (ℝ^d, g_enc) | R = 0 | Ric = 0 | R_scal = 0 | Flat: linear encoder |

The statistical manifold is flat in every geometric sense. Riemann = Ricci = scalar curvature = 0. This is not an approximation but an exact result: the Gaussian family N(μ, diag(v₀)) with fixed v₀, parameterised by μ ∈ ℝ^d, is a flat Riemannian manifold isometric to (ℝ^d, diag(1/v₀)). The ‘curvature’ of information geometry comes from the non-trivial connection (dually flat, e-flat and m-flat as established in the Wasserstein paper), not from the Riemannian curvature tensor.
The encoder manifold GL(128) has Ricci curvature ≈ 63.6 and scalar curvature ≈ 8,135. These are derived from the mean sectional curvature K ≈ 0.5 via the approximations Ric(X,X) ≈ (d−1)·K_avg·||X||^2 = 63.6·||X||^2 and R_scal ≈ d(d−1)·K_avg = 8,135. Positive Ricci curvature has implications for the encoder’s learning dynamics: by the Bonnet-Myers theorem, a compact Riemannian manifold with positive Ricci curvature has finite diameter bounded by π√(d/(n-1)K). For our values: diameter ≤ π√(128/(127×0.5)) ≈ 4.48, consistent with the observed encoder singular value range [0.128, 1.697].

# 8. Shape Operator of the Class Centroid Submanifold
## 8.1 The Delta Subspace
The K=10 class offset vectors {δ_1, ..., δ_K} span a subspace of ℝ^d. Their SVD reveals the intrinsic dimensionality of the class structure:
| SVD of Δ (K×D = 10×128 matrix of class offsets): Singular values: [2.107, 1.581, 1.437, 1.273, 1.010, 0.800, 0.768, 0.719, 0.389, 0.058] Effective rank = 7.59 (of maximum K-1 = 9) Numerical rank = 10 (all singular values > 0.1% of σ_max) |
|---|

The delta subspace has effective rank 7.59 in ℝ^{128}. All 10 singular values are non-negligible (numerical rank = 10 = K), confirming that the 10 classes occupy genuinely independent directions in latent space. The effective rank of 7.59 (vs maximum K−1 = 9) indicates that the 10 class offset directions are not uniformly distributed: the top 8 singular directions account for most of the variance, with the 9th and 10th contributing less. This is consistent with the RMT finding of 8 detectable spikes in the whitened covariance spectrum.
## 8.2 Per-Class Sample Covariance Geometry
| Class | tr(Σ_k) | ||Σ_k||_F | λ_max(Σ_k) | Isotropy | Shape |
|---|---|---|---|---|---|
| net_normal | 0.2358 | 0.1238 | 0.0908 | 0.000 | Anisotropic ray (URL diversity) |
| net_scan | 0.0862 | 0.0568 | 0.0457 | 0.000 | Anisotropic ray |
| net_ddos | 0.0218 | 0.0200 | 0.0200 | 0.000 | Near-rank-1 |
| net_exfil | 0.0645 | 0.0308 | 0.0244 | 0.000 | Anisotropic |
| net_c2 | 0.1351 | 0.0975 | 0.0816 | 0.000 | Anisotropic |
| log_info | 0.0005 | 0.0003 | 0.0002 | 0.005 | Most isotropic (rigid format) |
| log_warn | 0.0003 | 0.0002 | 0.0002 | 0.008 | Most isotropic |
| log_error | 0.0025 | 0.0019 | 0.0019 | 0.001 | Near-rank-1 |
| bin_malware | 0.4624 | 0.1553 | 0.1003 | 0.000 | Highest variance |
| bin_benign | 0.4270 | 0.1363 | 0.0785 | 0.000 | High variance |

Log classes have 500× smaller covariance trace than binary classes. tr(Σ_k) ranges from 0.0003 (log_warn) to 0.4624 (bin_malware), a 1,541× range. The sample clouds of each class have a shape dictated by the within-class variance of the parsed features. Log classes, with rigid format, produce near-degenerate distributions (isotropy ≈ 0.005–0.008). Binary classes, with random payloads, produce diffuse distributions (isotropy ≈ 0). The low isotropy (0.000–0.008) for all classes confirms that the within-class distributions are highly anisotropic — each class occupies a low-dimensional submanifold of the d=128-dimensional latent space, not a spherical cloud.

# 9. Exponential Map, Logarithmic Map, and Geodesics
## 9.1 The Statistical Manifold
On the flat statistical manifold, the exponential and logarithmic maps are linear:
| exp_μ(v) = μ + v (flat: exp is just translation) log_μ(ν) = ν - μ (flat: log is subtraction)  Geodesic γ_{μ,ν}(t) = μ + t(ν - μ) = (1-t)μ + tν (straight line, t ∈ [0,1])  Class centroid geodesic: γ_k(t) = μ_0 + t·δ_k Geodesic speed ||dγ/dt||_G = ||d_k||_G (Fisher distance from prior): Range: [6.17 (net_normal), 11.53 (bin_malware)] |
|---|

## 9.2 The Full NIG Manifold
If class-specific variances v_k were allowed (the full NIG model), the manifold of (mean, variance) parameters would have non-trivial curvature. The variance component lives in ℝ^d_+ with the log-metric d(v, w) = ||log(v/w)||, giving:
| Sectional curvature of (ℝ^d_+, g_var) = -1/2 (hyperbolic half-space model) Geodesic on ℝ^d_+: v(t) = v_0^{1-t} ⊙ v_1^t (geometric interpolation)  In CyphaDIF: all classes share v₀ ⇒ all classes at the same point in variance space The class manifold is the fibre {μ ∈ ℝ^d} × {v₀} ⊂ NIG manifold Curvature contribution from variance: 0 (all classes at same variance point) |
|---|

# 10. Jacobi Fields and Geodesic Stability
## 10.1 Theoretical Analysis
A Jacobi field J along a geodesic γ measures the deviation between nearby geodesics. It satisfies the Jacobi equation:
| D²J/dt² + R(J, γ')γ' = 0 (Jacobi equation)  For flat manifolds (R = 0): D²J/dt² = 0 ⇒ J(t) = J_0 + t·J_0' (linear growth)  Lyapunov exponent λ = lim_{t→∞} (1/t) log||J(t)|| = 0 (neutral stability) Geodesics are neither focusing (K<0 case) nor defocusing (K>0 case). |
|---|

## 10.2 Empirical Geodesic Stability
We test geodesic stability by perturbing a point along the geodesic from the net_ddos centroid toward the nearest class (net_normal), and measuring how the LLR gap changes:
| Distance ε along geodesic | LLR (net_ddos) | LLR gap | Correctly classified? |
|---|---|---|---|
| 0.0 (centroid) | +57.54 | 82.24 | Yes (margin = 82.2) |
| 0.1 | +50.07 | 72.06 | Yes |
| 0.5 | +20.20 | 31.34 | Yes |
| 1.0 | −17.15 | −19.55 | No (flip at ε ≈ 0.9) |
| 2.0 | −91.84 | −121.3 | No |
| 5.0 | −315.9 | −426.7 | No |

| LLR decays linearly with distance ε (slope −75.5/unit), consistent with flat geometry. Classification flips at ε ≈ 0.9. The linear decay of LLR with ε is exactly what flat geometry predicts. Along the geodesic toward net_normal, the net_ddos score LLR_{net_ddos}(h) decreases linearly because h → h + ε·v and LLR is linear in h. The slope is d(LLR_{net_ddos})/dε = ⟨w_{net_ddos}, v⟩ where v is the unit geodesic direction. This is a constant (ε-independent) as expected for a flat manifold with linear LLR functions. On a curved manifold with K > 0, geodesics would defocus and the LLR would decrease faster than linear; for K < 0, slower. The flat K = 0 case gives exactly linear decay. Classification boundary crossed at ε ≈ 0.9 (midpoint of geodesic at ε = 1.0). The geodesic from the net_ddos centroid (μ_{ddos}) to the net_normal centroid (μ_{normal}) has length d_G(μ_{ddos}, μ_{normal}) = 12.37 in Fisher units. The boundary is at the midpoint t = 0.5, corresponding to absolute distance ε ≈ 0.9 in the latent space units used. This is consistent with the convex analysis finding that all boundaries pass through the geodesic midpoints. |
|---|

# 11. Cartan Moving Frame and Differential Forms
## 11.1 Moving Frame
The encoder W defines a global frame field on the feature space: e_i = columns of W⁻¹ (the pullback of the standard coordinate frame). This frame is defined everywhere on ℝ^d (since W ∈ GL(d) is invertible) and is anholonomic (non-coordinate-aligned) in general.
| Moving frame: {e_1, ..., e_d} = columns of W^{-1} Co-frame: {θ^1, ..., θ^d} = rows of W (dual frame: θ^i(e_j) = δ^i_j)  Gram matrix in Fisher metric: G(e_i, e_j) = (W^{-T} G_0 W^{-1})_{ij} = (g_enc^{-1})_{ij} ||Gram - I||_F = 7,755.2 (W is NOT Fisher-orthonormal) (For a Fisher-orthonormal frame: W would satisfy W^T G_0 W = I) |
|---|

## 11.2 Maurer-Cartan Form and Curvature 2-Form
The Maurer-Cartan form is the gl(d)-valued 1-form ω = W⁻¹ dW defined on GL(d). Evaluated on a tangent vector V at W, it gives ω_W(V) = W⁻¹V ∈ gl(d). The curvature 2-form is Ω = dω + ω ∧ ω.
| Maurer-Cartan structure equation: dω + ω ∧ ω = 0 ⇒ Ω = 0 (flat connection on GL(d) via MC form)  ||MC form ω(V_i)||_F for top singular directions: mean=1.246, min=0.589, max=1.890  Torsion T^i = de^i + ω^i_j ∧ e^j = 0 (torsion-free) |
|---|

The Maurer-Cartan form gives a flat connection on the frame bundle. The curvature 2-form Ω = 0 by the MC structure equation — this is exact, not approximate. However, the non-zero ||Gram − I||_F = 7,755 shows that the frame e_i is far from being a Fisher-orthonormal frame. The frame is adapted to the encoder W, not to the Fisher geometry. Constructing a Fisher-orthonormal frame would require a Gram-Schmidt orthogonalisation with respect to G₀, yielding a new frame {f_i} with G(f_i, f_j) = δ_{ij} but destroying the natural Lie group structure that the MC form encodes.

# 12. Synthesis
The statistical manifold is flat (K = 0) in every differential-geometric sense. All Christoffel symbols, Riemann tensor components, Jacobi field growth rates, and angle defects are exactly zero. Geodesics are straight lines, parallel transport is trivial, and the holonomy group is {Id}. The Gaussian family with fixed variance is, metrically, a copy of (ℝ^d, G₀) — a flat Riemannian manifold.
Decision boundaries are totally geodesic and Bayes-optimal in Fisher metric. All 45 boundaries have κ_g = 0 (zero geodesic curvature), are perpendicular to the class geodesic (angle 0.000° to machine precision), and pass through the midpoint of each class pair’s geodesic. This is the geometric characterisation of Bayes-optimal classification under equal-prior Gaussians with shared covariance.
The encoder manifold GL(128) has positive sectional curvature K ≈ 0.500. The tight clustering of K around 1/2 (std 0.004) suggests the encoder has been shaped by the contrastive training to occupy a geometrically regular region of GL(128). The polar decomposition reveals a large improper rotation Q (||Q−I||_F = 16.46, angle 93.4°) and a strongly anisotropic stretching P (condition number 203.96, effective rank 51.38).
Jacobi fields grow linearly on the flat manifold (λ = 0). This implies geometric stability: nearby geodesics diverge at most linearly. The empirical test confirms linear LLR decay along geodesics, with boundary crossing at ε ≈ 0.9 (midpoint of the geodesic to the nearest class). No exponential mixing or focusing occurs.
The Cartan frame is non-orthonormal (||Gram−I||_F = 7,755) but torsion-free. The encoder W defines a global frame field with zero torsion and zero curvature 2-form (MC structure equation), but the frame is far from Fisher-orthonormal. This encodes the encoder’s geometric distortion: the 128 coordinate axes it defines are not aligned with the natural precision-weighted orthogonal axes of the Fisher metric.

# References
[1] do Carmo, M. P. (1992). Riemannian Geometry. Birkhäuser.
[2] Lee, J. M. (2018). Introduction to Riemannian Manifolds (2nd ed.). Springer.
[3] Amari, S., & Nagaoka, H. (2000). Methods of Information Geometry. AMS.
[4] Amari, S. (2016). Information Geometry and Its Applications. Springer.
[5] Murray, M. K., & Rice, J. W. (1993). Differential Geometry and Statistics. Chapman & Hall.
[6] Milnor, J. (1976). Curvatures of left invariant metrics on Lie groups. Advances in Mathematics, 21(3), 293–329.
[7] Cartan, É. (1926). La géométrie des espaces de Riemann. Mémorial des sciences mathématiques, Fasc. 9.
[8] Bishop, R. L., & Crittenden, R. J. (1964). Geometry of Manifolds. Academic Press.
[9] Kobayashi, S., & Nomizu, K. (1963). Foundations of Differential Geometry, Vol. I. Wiley.
[10] Cheeger, J., & Ebin, D. G. (1975). Comparison Theorems in Riemannian Geometry. North-Holland.
[11] Bhatia, R. (2007). Positive Definite Matrices. Princeton University Press.
[12] Moakher, M. (2005). A differential geometric approach to the geometric mean of symmetric positive-definite matrices. SIAM Journal on Matrix Analysis and Applications, 26(3), 735–747.
[13] Pennec, X., Fillard, P., & Ayache, N. (2006). A Riemannian framework for tensor computing. International Journal of Computer Vision, 66(1), 41–66.
[14] Skovgaard, L. T. (1984). A Riemannian geometry of the multivariate normal model. Scandinavian Journal of Statistics, 11(4), 211–223.
[15] Eriksen, E. (1987). On the measures of geodesic curvature on statistical manifolds. Journal of Statistical Planning and Inference, 15, 281–292.
[16] Murray, M. K. (1993). The geometry of Gaussian distributions. Geometry and Statistics, 22, 165–183.
[17] Petersen, P. (2016). Riemannian Geometry (3rd ed.). Springer.
[18] Gallot, S., Hulin, D., & Lafontaine, J. (2004). Riemannian Geometry (3rd ed.). Springer.
[19] Jost, J. (2011). Riemannian Geometry and Geometric Analysis (6th ed.). Springer.
[20] Postnikov, M. M. (2001). Geometry VI: Riemannian Geometry. Springer.
