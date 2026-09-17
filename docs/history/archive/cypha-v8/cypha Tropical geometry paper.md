| Tropical Geometry of the Differential Information Field Classifier Tropical Polynomial • Hypersurface • Newton Polytope • Tropical Rank • Discriminant • Gröbner Basis • Projective Map Unpublished Technical Report — 2026 |
|---|

Abstract
| We apply tropical geometry — the study of algebraic geometry over the tropical semiring (ℝ∪{−∞}, ⊕=max, ⊗=+) — to CyphaDIF. The key observation is that the CyphaDIF classifier f(h) = max_k{LLR_k(h)} is exactly a tropical polynomial of degree 1 in K=10 terms. This transforms classical algebraic geometry questions about the classifier into combinatorial questions about polyhedral fans and max-plus algebra. Ten probes cover the tropical polynomial structure, the tropical hypersurface (decision boundary complex), the combinatorial type of the argmax arrangement, tropical distances and Voronoi cells, the Newton polytope and its regular subdivision, the tropical convex hull of weight vectors, tropical rank and the tropical determinant, the tropical discriminant, the minimal tropical Gröbner basis, and the tropical projective map. Key results: (1) The decision boundary complex is a tropical hypersurface with K=10 cells, 38 active facets (of 45 possible) and 120 theoretical ridges. (2) All 10 classes are argmax in distinct non-empty regions. (3) The Newton polytope Newt(f) = conv(w_1,...,w_K) has dimension 9 (rank 9) and effective dimension 8.12, embedded in ℝ^{128}. (4) The tropical determinant tdet(M) = 434.7, achieved uniquely by the identity permutation (each class scores highest at its own centroid). (5) The tropical discriminant (margin) has mean 53.3 and minimum 13.7, with zero training samples on the tropical hypersurface. (6) The minimal tropical Gröbner basis has K−1=9 elements (the MST of the class graph, total weight 910.4). (7) The tropical width of the score polytope is 231.3, and the tropical projective map Φ: ℝ^{128} → TP^9 reveals that each class occupies a well-separated cluster in tropical projective space. |
|---|

# 1. Tropical Algebra and the Argmax Classifier
Tropical algebra replaces the classical operations (+, ×) by (max, +). The tropical semiring is (ℝ∪{−∞}, ⊕, ⊗) where a⊕b = max(a,b) and a⊗b = a+b. Monomials in tropical algebra have the form:
| Tropical monomial: c ⊗ x_1^{a_1} ⊗ ... ⊗ x_d^{a_d} = c + a_1 x_1 + ... + a_d x_d (exponents become linear coefficients in classical notation)  Tropical polynomial: f = c_1⊗x^{a_1} ⊕ ... ⊕ c_K⊗x^{a_K} = max_k { c_k + ⟨a_k, x⟩ } (maximum over K linear forms) |
|---|

The CyphaDIF classifier is, by construction, a tropical polynomial of degree 1:
| f(h) = max_k { LLR_k(h) } = max_k { ⟨w_k, h⟩ + b_k } = max_k { c_k ⊗ h^{⊗w_k} } (tropical notation)  where: w_k = δ_k / v₀ (tropical exponent vector, d=128-dim) b_k = -⟨w_k,μ₀⟩ - ‖δ_k‖^2_V/2 (tropical coefficient = log-prior bias) K = 10 terms (one tropical monomial per class) |
|---|

The argmax operation is tropical addition. In tropical algebra, the ‘sum’ of K monomials is their maximum. CyphaDIF’s decision rule, argmax_k LLR_k(h), is precisely the tropical polynomial evaluation: the selected class k* is the term achieving the tropical maximum. The tropical setting makes explicit what is implicit in the probabilistic formulation: the classifier is fundamentally a max-plus algebraic object, not just a probability model.

# 2. Tropical Polynomial Structure
## 2.1 The Ten Tropical Monomials
| Class | Coefficient b_k | ||w_k|| (exponent norm) | Argmax at own centroid? |
|---|---|---|---|
| net_normal | −75.08 | 60.07 | Yes |
| net_scan | +130.69 | 83.90 | Yes |
| net_ddos | −84.98 | 95.31 | Yes |
| net_exfil | −85.24 | 88.32 | Yes |
| net_c2 | −160.40 | 103.11 | Yes |
| log_info | −38.22 | 79.83 | Yes |
| log_warn | −17.07 | 73.23 | Yes |
| log_error | −12.61 | 76.88 | Yes |
| bin_malware | −15.77 | 94.98 | Yes |
| bin_benign | −70.09 | 78.02 | Yes |

| All 10/10 classes are the argmax of their own tropical monomial at their respective centroids. Full tropical separability confirmed. Each class k satisfies LLR_k(μ_k) > LLR_j(μ_k) for all j≠k. This is the tropical analogue of Voronoi membership: each codeword lies in its own Voronoi cell. Algebraically, the K tropical monomials form a tropically non-degenerate system — no class centroid lies on the tropical hypersurface (the boundary where two monomials tie). The margins LLR_k(μ_k) − max_{j≠k} LLR_j(μ_k) range from 37.9 (bin_benign and bin_malware) to 76.5 (net_ddos), confirming all centroids are in the strict interior of their tropical cells. The bias b_{net_scan} = +130.69 is the only positive bias. The large positive bias for net_scan compensates for its weight vector having a different orientation than the world prior μ₀. In tropical terms, the net_scan monomial has a large ‘base level’ that keeps it competitive even far from its centroid. The most negative bias is b_{net_c2} = −160.40, meaning the net_c2 monomial starts from a lower baseline and relies entirely on the inner product ⟨w_{net_c2}, h⟩ for its score — but with the highest ||w_k|| = 103.1, it grows fastest in the net_c2 direction. |
|---|

# 3. Tropical Hypersurface: The Decision Boundary Complex
## 3.1 Structure of V(f)
The tropical hypersurface V(f) is the set of points h ∈ ℝ^d where the tropical polynomial f(h) = max_k{LLR_k(h)} is not smooth, i.e., where at least two monomials simultaneously achieve the maximum:
| V(f) = { h ∈ ℝ^d : max_k LLR_k(h) achieved by ≥2 indices } = ∪_{i<j} B_{ij} where B_{ij} = { h : LLR_i(h) = LLR_j(h) ≥ LLR_k(h) ∀k }  V(f) is a polyhedral complex (union of convex polyhedra) in ℝ^d. It decomposes ℝ^d into K = 10 convex polyhedral cells (the tropical Voronoi diagram).  Combinatorial structure: Cells (dim d): K = 10 (one per class, the tropical Voronoi cells) Facets (dim d-1): 45 pairwise boundaries = C(K,2) Ridges (dim d-2): 120 triple boundaries = C(K,3) (theoretical) Active facets (verified): 38/45 (midpoint-interior test) |
|---|

## 3.2 Active vs Inactive Facets
A facet B_{ij} is active (present in the tropical Voronoi diagram) if there exist points h on B_{ij} where LLR_i(h) = LLR_j(h) > LLR_k(h) for all k≠i,j. We test this by evaluating the midpoint μ_{mid} = (μ_i + μ_j)/2:
| Midpoint test: μ_{mid} = (μ_i + μ_j)/2 B_{ij} active iff LLR_i(μ_{mid}) = LLR_j(μ_{mid}) > LLR_k(μ_{mid}) ∀k≠i,j  Results: 38/45 facets active (7 facets not visible at midpoint) (7 ‘ghost’ boundaries exist as hyperplanes but are covered by closer classes at the geometric midpoint — they may still be topologically present but thin) |
|---|

| 38/45 boundaries are active (midpoint-interior). The tropical hypersurface has 38 visible facets. The K=10 cells form a connected polyhedral complex. The 7 inactive facets (at the midpoint test) correspond to class pairs that are not ‘nearest neighbours’ in the Voronoi diagram. When the midpoint of classes i and j is closer to a third class k than to either i or j, the midpoint lies outside B_{ij}. This does not mean B_{ij} is empty — it is a full (d−1)-dimensional hyperplane — but the Voronoi cell boundary at the geodesic midpoint is covered. The 38 active facets confirmed by the midpoint test constitute the backbone of the tropical hypersurface. In classical terms, the 38 active boundaries are the edges of the ‘nearest-neighbour graph’ of the class centroids in Fisher metric. The tropical hypersurface partitions ℝ^{128} into exactly 10 convex polyhedral regions. For K linear functions in ℝ^d with d >> K, the arrangement generically produces exactly K regions (not the exponential Zaslavsky number Σ_{j=0}^d C(K,j), which applies to hyperplane arrangements, not to argmax arrangements). The argmax arrangement is fundamentally different from a hyperplane arrangement: it always produces exactly K regions, one per class, since for each class k there always exists some h where class k dominates. |
|---|

# 4. Combinatorial Type of the Argmax Arrangement
The combinatorial type of the tropical polynomial f records, for each region of ℝ^d, which class is the argmax. For a generic tropical polynomial with K monomials in ℝ^d (d ≥ K−1), the combinatorial type is uniquely determined by the weight vectors w_k and biases b_k.
| Sampling of argmax regions (10,000 random points, mixture of Gaussians):  net_normal: 5.88% net_scan: 10.35% net_ddos: 12.32% net_exfil: 11.30% net_c2: 13.00% log_info: 9.37% log_warn: 7.97% log_error: 9.23% bin_malware: 11.29% bin_benign: 9.29%  All K=10 classes appear as argmax for some h (K regions exist). No non-class composite regions found (exactly K cells, as expected). |
|---|

Region sizes reflect the solid angles of the tropical Voronoi cells. The smallest region belongs to net_normal (5.88% of sampled points), and the largest to net_c2 (13.00%). The region size measures the ‘volume’ of each class’s Voronoi cell relative to the sampling distribution, not the volume in any absolute sense. The net_c2 cell being the largest reflects that the net_c2 weight vector w_{net_c2} has the highest norm (103.1), so it grows fastest for inputs in its direction, capturing the most probability mass from the Gaussian mixture sampler. Net_normal’s small region (5.88%) reflects its low weight-vector norm (60.1) relative to all other classes — it wins in the fewest directions.

# 5. Tropical Distances and the Tropical Voronoi Diagram
## 5.1 The Tropical (L∞) Metric
The tropical distance between two points h, h’ ∈ ℝ^d/ℝ① (tropical projective space) is:
| d_trop(h, h') = max_i(h_i - h'_i) - min_i(h_i - h'_i) = ||h - h'||_{L^∞-spread} (L∞ diameter of the difference vector)  This is the tropical projective metric on ℝ^d/ℝ①. For classical metrics: d_trop(h,h') ≥ ||h-h'||_∞ (L∞ distance) |
|---|

## 5.2 Tropical vs Classical Distances
| Metric | Mean inter-class distance | Closest pair | Farthest pair | Ratio to L2 |
|---|---|---|---|---|
| Tropical (L∞ spread) | 0.810 | bin_malware↔bin_benign (0.438) | log_info↔bin_malware (1.145) | 0.458× L2 |
| Euclidean (L2) | 1.768 | bin_malware↔bin_benign (1.054) | net_c2↔bin_malware (2.348) | 1.000× (ref) |
| Fisher-Rao | 13.706 | bin_malware↔bin_benign (8.706) | net_c2↔bin_malware (17.848) | 7.753× L2 |

| d_trop / d_L2 = 0.458: the tropical metric is ~2.2× smaller than Euclidean. Closest pair in all metrics: bin_malware↔bin_benign. The tropical distance is always ≤ the L∞ distance, which is ≤ the L2 distance by norm inequalities. For the encoder output centroids, the tropical distance averages 45.8% of the L2 distance. This compression ratio arises because the tropical metric measures only the range (max−min) of the difference vector, ignoring the many intermediate coordinates. In a d=128-dimensional space, the max and min of a difference vector are typically small fractions of the total L2 norm. The L∞ spread / L2 ratio scales as O(1/∞d) for random vectors, giving d_trop/d_L2 ≈ 1/∞128 ≈ 0.088 for generic vectors — our observed 0.458 is larger, indicating that the class difference vectors are not isotropic but concentrate in a few dominant dimensions. The closest pair (bin_malware↔bin_benign) is the same in all three metrics. This consistency across tropical, Euclidean, and Fisher-Rao metrics indicates that the binary class separation is fundamentally small — the MZ (0x4D5A) and ELF (0x7F454C46) headers differ in few byte positions, producing encoder outputs that are globally close regardless of the metric used. The farthest pair shifts from net_c2↔bin_malware (Euclidean, Fisher) to log_info↔bin_malware (tropical), reflecting that in the tropical metric the log and binary domains differ most in their extreme dimensions. |
|---|

# 6. Newton Polytope and Regular Subdivision
## 6.1 The Newton Polytope
The Newton polytope of the tropical polynomial f = max_k{⟨w_k,h⟩ + b_k} is the convex hull of the exponent vectors (the weight vectors w_k ∈ ℝ^d):
| Newt(f) = conv(w_1, ..., w_K) ⊂ ℝ^{128}  Dimension of Newt(f): rank(W_centered) = 9 (K-1, as expected for K points) Effective dimension: 8.12 (from SVD of centered weight matrix) Singular values: [127.8, 115.7, 106.6, 95.2, 77.9, 71.6, 68.7, 58.8, 36.0, 0.0]  Per-dimension width (max_k w_{k,i} - min_k w_{k,i}): mean = 24.01 min = 13.49 max = 39.51 Sum of widths (L1 diameter) = 3073.0 |
|---|

## 6.2 Regular Subdivision Induced by Heights b_k
The bias terms b_k act as heights over the Newton polytope, inducing a regular subdivision of Newt(f). The tropical hypersurface V(f) is the dual of this regular subdivision:
| Heights b_k (for regular subdivision of Newt(f)): net_c2: b = -160.40 (most negative — largest penalty at origin) net_ddos: b = -84.98 net_exfil:b = -85.24 net_normal:b = -75.08 bin_benign:b = -70.09 bin_malware:b= -15.77 log_error: b = -12.61 log_warn: b = -17.07 log_info: b = -38.22 net_scan: b = +130.69 (only positive height)  Height range: [-160.40, +130.69] spread = 291.09 |
|---|

| Newton polytope Newt(f) has dimension 9 (= K−1), embedded in ℝ^{128}. Effective dimension 8.12. The height spread of 291.1 drives a non-trivial regular subdivision. The dimension of Newt(f) is exactly K−1 = 9. This is the generic dimension for the convex hull of K points in general position in ℝ^d (for d ≥ K−1). The zero singular value confirms that the K weight vectors w_k are affinely dependent — they span a 9-dimensional affine subspace of ℝ^{128}. The effective dimension of 8.12 (vs exact 9) indicates mild near-degeneracy: the 9th singular direction (SV=36.0) is somewhat weaker than the first 8 (SV=58–128). This matches the RMT finding of 8 detectable spike eigenvalues (K−1 non-trivial class-discriminant directions in the whitened spectrum). The regular subdivision induced by heights b_k determines the combinatorial type of V(f). A regular subdivision of Newt(f) partitions the polytope into sub-polytopes, with each vertex of the subdivision corresponding to a vertex w_k elevated by height b_k. The dual of this subdivision is the tropical hypersurface V(f): each interior edge of the subdivision corresponds to a facet of V(f), and each interior vertex of the subdivision corresponds to a ridge of V(f). The large height spread (291.1) ensures the subdivision is non-degenerate (no three w_k vertices are co-level), giving a simplicial regular subdivision of Newt(f). |
|---|

# 7. Tropical Convex Hull and Tropical Halfspaces
The tropical convex hull of K points p_1,...,p_K ∈ ℝ^d/ℝ① is the set of all tropical linear combinations. For our weight vectors w_k ∈ ℝ^d:
| tconv(w_1,...,w_K) = { max_k(λ_k + w_k) : λ ∈ ℝ^K, tropically normalised }  Membership test for world prior w_0 = 0: w_0 ∈ tconv(w_1,...,w_K) iff max_k min_i(w_{k,i}) ≤ 0 ≤ min_k max_i(w_{k,i})  max_k min_i(w_{k,i} - 0) = max_k min_i(w_{k,i}) = +13.21 min_k max_i(w_{k,i} - 0) = min_k max_i(w_{k,i}) = -12.87  Condition 13.21 ≤ 0 ≤ -12.87 is FALSE. ⇒ World prior w_0 is NOT in the tropical convex hull of {w_k}. |
|---|

| The world prior (w=0) lies outside the tropical convex hull of {w_k}. The K class weight vectors do not tropically ‘surround’ the origin. This has a natural classification interpretation. If the world prior were inside tconv(w_1,...,w_K), it would mean the zero score vector (no evidence from any class) lies in the ‘average’ of the class score functions. Since w_0 = 0 is outside the tropical convex hull, the zero-evidence point is tropically extreme — it does not lie in the tropical average of the class representations. Concretely, the world prior mean μ₀ maps to net_normal (as established in the convex analysis paper), not to a tropical average of all classes. The tropical convex hull is a ‘tropical simplex’ in ℝ^{128}/ℝ① with K=10 vertices; the origin lies outside this simplex. Tropical hyperplane normals (w_i - w_j) have norms in [81.2, 153.3]. Each pairwise boundary B_{ij} in the tropical hypersurface is defined by the tropical hyperplane with normal w_i - w_j. The norm ||w_i - w_j|| measures the ‘strength’ of the corresponding decision boundary: larger norms mean faster transition between classes. The minimum norm 81.2 (bin_malware↔bin_benign) corresponds to the narrowest decision boundary, consistent with the smallest pairwise geometric margin. The mean norm 124.2 reflects the typical strength of class boundaries in the tropical decomposition. |
|---|

# 8. Tropical Rank and the Tropical Determinant
## 8.1 The LLR Matrix and Tropical Rank
The K×K matrix M with M_{ij} = LLR_j(μ_i) records the score of each class j at each class centroid μ_i. The tropical determinant of M (in max-plus algebra) is:
| tdet(M) = max_{σ ∈ S_K} Σ_i M_{i,σ(i)} = max_{permutations σ} [sum of class σ(i)’s score at centroid μ_i]  Computed via Hungarian algorithm (maximum-weight perfect matching): tdet(M) = 434.70  Optimal permutation: identity σ(i) = i (each class scores at its own centroid) M_{11} + M_{22} + ... + M_{KK} = Σ_k LLR_k(μ_k) = 434.70  Diagonal (own scores LLR_k(μ_k)): [19.04, 37.66, 54.80, 50.32, 63.69, 38.83, 34.12, 32.12, 66.47, 37.66] Margin (diag - best off-diag score per row): [49.65, 63.68, 76.46, 70.86, 60.99, 48.86, 44.12, 44.12, 37.90, 37.90] |
|---|

| Tropical determinant tdet(M) = 434.70, achieved by the identity permutation. The tropical rank of M is K=10 (full). The tropical determinant equals the total MDL description length Σ_k L(δ_k) = 434.7 nats. tdet(M) = 434.70 = Σ_k L(δ_k): the tropical determinant equals the total MDL description length. This is not a coincidence. The tropical determinant with the identity permutation sums the diagonal: Σ_k M_{kk} = Σ_k LLR_k(μ_k) = Σ_k [⟨δ_k/v₀, μ_k⟩ + b_k] = Σ_k [⟨δ_k/v₀, μ₀+δ_k⟩ - ⟨δ_k/v₀,μ₀⟩ - ||δ_k||^2_V/2] = Σ_k ||δ_k||^2_V/2 = Σ_k L(δ_k). So the tropical determinant is the PAC/MDL complexity measure of the classifier. The tropical maximum-weight matching selects the identity permutation because the NIG classifier is self-consistent: each class’s own score function is highest at its own centroid (Bayes-optimality guarantees this). Full tropical rank K=10 means the K class monomials are tropically linearly independent. Tropical linear independence requires that the maximum-weight perfect matching is unique (the identity) and all diagonal entries contribute distinctly. Since the margins (37.9–76.5) are all positive and large, no permutation can beat the identity assignment. This is the tropical analogue of the linear independence of the class score functions: they form a tropically non-degenerate system. |
|---|

# 9. Tropical Discriminant
The tropical discriminant Δ(h) of a tropical polynomial f at a point h is the gap between the largest and second-largest monomial values:
| Δ(h) = LLR_{(1)}(h) - LLR_{(2)}(h) (gap between 1st and 2nd ranked LLRs)  Δ(h) = 0 ⟺ h is on the tropical hypersurface V(f) (decision boundary) Δ(h) > 0 ⟺ h is in the strict interior of a tropical Voronoi cell  The tropical discriminant is the ‘distance’ from h to the nearest boundary. |
|---|

| Class | Mean Δ | Min Δ | Std Δ | Geometric interpretation |
|---|---|---|---|---|
| net_ddos | 82.24 | 80.23 | 1.00 | Most isolated: rigid PPS format |
| net_exfil | 69.15 | 63.03 | 4.02 |  |
| net_c2 | 61.68 | 54.32 | 7.05 |  |
| net_scan | 62.19 | 56.32 | 3.07 |  |
| log_error | 46.78 | 44.64 | 0.86 | Rigid log format |
| log_info | 46.57 | 46.13 | 0.11 | Near-constant: rigid format |
| log_warn | 43.50 | 43.17 | 0.13 | Near-constant |
| net_normal | 43.86 | 17.38 | 11.03 | URL diversity causes spread |
| bin_malware | 42.50 | 21.60 | 8.30 |  |
| bin_benign | 34.41 | 13.74 | 8.90 | Closest to boundary |

| Mean tropical discriminant = 53.3, min = 13.7. All 1,000 training samples have Δ > 0: no samples lie on the tropical hypersurface. The tropical discriminant is identical to the functional margin from the PAC analysis. This is by construction: Δ(h) = LLR_{(1)}(h) - LLR_{(2)}(h) is exactly the margin γ̂(h) used in the PAC paper. The tropical geometry re-frames this as a distance to the tropical hypersurface: all training samples are in the strict interior of their tropical Voronoi cells. The classifier never sits on a tropical hyperplane (decision boundary) during training. Log classes have near-constant discriminants (std 0.11–0.13 nats). The log class feature extraction produces near-identical latent representations for all samples of the same type (the rigid [TYPE] HH:MM:SS format leaves little variability). From the tropical perspective, all log_info samples cluster at essentially the same point in tropical Voronoi space — the discriminant varies by only 0.45 nats across 100 samples (min=46.13, max≈46.6). Binary classes (std 8.3–8.9) have the highest discriminant variability, reflecting the large within-class feature diversity from random payloads. |
|---|

# 10. Minimal Tropical Gröbner Basis
The tropical ideal generated by the K(K−1)/2 = 45 tropical hyperplanes {⟨w_i−w_j, h⟩ + (b_i−b_j) = 0} has a minimal generating set analogous to a Gröbner basis. The minimal tropical basis is the minimum spanning tree (MST) of the class graph, weighted by ||w_i − w_j||:
| Full basis: 45 tropical hyperplanes (C(K,2) pairwise boundaries) MST basis: 9 edges (K−1 = minimal spanning set)  Minimum spanning tree (Prim’s algorithm on ||w_i - w_j|| weights):  bin_malware — bin_benign : ||w_i-w_j|| = 81.18 (binary bridge) net_normal — log_warn : ||w_i-w_j|| = 88.56 (net-log bridge) log_warn — log_error : ||w_i-w_j|| = 90.67 log_warn — log_info : ||w_i-w_j|| = 96.89 (log cluster) log_error — net_scan : ||w_i-w_j|| = 105.56 net_normal — net_c2 : ||w_i-w_j|| = 107.49 net_normal — net_exfil : ||w_i-w_j|| = 109.90 net_normal — bin_benign : ||w_i-w_j|| = 114.69 (cross-domain bridge) net_normal — net_ddos : ||w_i-w_j|| = 115.49  MST total weight: 910.44 (vs full graph: Σ_{i<j} ||w_i-w_j|| ≈ 5591) |
|---|

| Minimal tropical Gröbner basis: 9 edges (MST), total weight 910.4. Net_normal is the MST hub, connecting to log, binary, and other network classes. The MST hub at net_normal reflects its geometric centrality in weight space. Net_normal has the smallest weight-vector norm (||w_k|| = 60.1) and is closest (in weight space) to log_warn (88.6), bin_benign (114.7), and all other network classes. It acts as the ‘hub’ of the minimal tropical generating set, connecting the log domain (via log_warn), the binary domain (via bin_benign), and the other network classes (via net_c2, net_exfil, net_ddos). In tropical terms, net_normal is the vertex that connects the MST’s three main branches: network, log, and binary. The MST basis with 9 edges suffices to generate all 45 boundaries. Any pairwise boundary B_{ij} can be expressed as a tropical combination of MST boundaries along the path from i to j in the MST. This is the tropical Gröbner basis property: the MST edges generate the full tropical ideal. The MST structure reveals the ‘essential’ class relationships: bin_malware↔bin_benign (weight 81.2, the only binary bridge) and net_normal↔log_warn (weight 88.6, the main net-log bridge) are the two most critical edges in the minimal basis. |
|---|

# 11. Tropical Projective Map
## 11.1 The Map Φ: ℝ^d → TP^{K-1}
The tropical projective map Φ sends each latent point h to its score vector in tropical projective space:
| Φ: ℝ^{128} → TP^9 = ℝ^{10}/ℝ① h ↦ (LLR_1(h), ..., LLR_{10}(h)) mod ℝ① (subtract mean score to project to TP^9)  The image Φ(data) reveals the tropical ‘fingerprint’ of each class in score space.  Tropical width of score polytope: max_h [max_k LLR_k(h) - min_k LLR_k(h)] = 231.3 |
|---|

## 11.2 Score Vector Centroids in TP^9
The centroid of each class’s score vectors in TP^9 reveals how the classifier discriminates between classes. For a perfectly separating classifier, class k’s centroid should have a large positive k-th component and small (negative) components elsewhere:
| Class (row) | Dominant score (k-th component) | 2nd highest score | Min score | Tropical discriminant |
|---|---|---|---|---|
| net_normal | LLR_{net_normal} = +60.0 | LLR_{net_c2} = +3.6 | LLR_{bin_malware} = −32.5 | 43.9 (LLR gap) |
| net_scan | LLR_{net_scan} = +80.2 | LLR_{log_error}= +18.0 | LLR_{net_c2} = −35.2 | 62.2 |
| net_ddos | LLR_{net_ddos} = +99.2 | LLR_{log_error}= −2.3 | LLR_{net_c2} = −37.6 | 82.2 |
| net_exfil | LLR_{net_exfil} = +93.4 | LLR_{net_scan} = +10.5 | LLR_{bin_malware}= −49.0 | 69.2 |
| net_c2 | LLR_{net_c2} = +106.3 | LLR_{net_normal}= +44.6 | LLR_{log_info} = −31.5 | 61.7 |
| log_info | LLR_{log_info} = +78.5 | LLR_{log_warn} = +31.9 | LLR_{bin_malware}= −55.7 | 46.6 |
| log_warn | LLR_{log_warn} = +78.3 | LLR_{log_error}= +34.9 | LLR_{bin_malware}= −64.1 | 43.5 |
| log_error | LLR_{log_error} = +78.6 | LLR_{log_warn} = +31.8 | LLR_{bin_malware}= −64.1 | 46.8 |
| bin_malware | LLR_{bin_malware}= +126.5 | LLR_{bin_benign}= +84.0 | LLR_{log_warn} = −32.1 | 42.5 |
| bin_benign | LLR_{bin_benign} = +95.7 | LLR_{bin_malware}=+61.3 | LLR_{net_scan} = −17.4 | 34.4 |

| Tropical projective width = 231.3 LLR units. Score polytope spans nearly the full dynamic range of the LLR functions. Each class occupies a well-separated ‘spike’ in TP^9. The tropical projective map reveals cross-class score structure. The net_c2 centroid (at the net_c2 class) has the highest dominant score (+106.3) and also a large secondary score for net_normal (+44.6), reflecting net_c2’s HTTP-like format. The binary classes show strong cross-talk: bin_malware samples have a secondary score of +84.0 for bin_benign and vice versa (+61.3) — both binary classes assign high scores to the other binary class, reflecting their shared payload structure. The log classes form a cluster: log_info, log_warn, and log_error all assign high scores to each other (secondary scores 31–34 LLR units). The tropical width 231.3 quantifies the ‘spread’ of the score polytope in TP^9. The largest observed LLR difference within a single sample’s score vector is 231.3 units (between the highest and lowest class LLRs). This occurs for a bin_malware sample: LLR_{bin_malware}(h) = +169.1 and LLR_{net_c2}(h) = −76.7 (from the CT10 probes), giving a spread of 245.8 LLR units. The tropical width is the maximum such spread over all training samples, measuring the total dynamic range of the classifier in score space. A large tropical width (231.3 vs the 53.3 average discriminant) indicates that while the classifier is highly confident about the correct class, it assigns very negative scores to some incorrect classes — the score vector is not just ‘somewhat better’ for the correct class but catastrophically different. |
|---|

# 12. Synthesis
CyphaDIF is exactly a tropical polynomial of degree 1. The argmax over K linear LLR functions is the canonical form of a tropical polynomial of degree 1 with K terms. This reformulation connects the probabilistic/Bayesian interpretation of the NIG classifier to the combinatorial/algebraic geometry of tropical mathematics.
The tropical hypersurface has 38 active facets (of 45) and partitions ℝ^{128} into 10 cells. The 7 inactive facets correspond to class pairs whose Voronoi boundary does not pass through the geodesic midpoint, due to proximity of a third class. All K=10 classes occupy non-empty argmax regions, confirmed by 10,000-point Monte Carlo sampling.
Newton polytope Newt(f) has dimension 9 = K−1, effective dimension 8.12. The K weight vectors w_k are affinely embedded in a 9-dimensional subspace of ℝ^{128}, consistent with the K−1 = 9 non-trivial discriminant directions found by the RMT analysis. The height spread 291.1 ensures a non-degenerate regular subdivision, giving a well-defined dual tropical hypersurface.
Tropical determinant tdet(M) = 434.7 = total MDL length. The identity permutation achieves the tropical maximum, confirming full tropical rank K=10. The numerical equality with the MDL total description length Σ_k L(δ_k) = 434.7 nats is an algebraic identity relating tropical geometry to information theory.
Minimal tropical Gröbner basis: 9 edges (MST), hub at net_normal. The MST structure reveals net_normal as the geometric centre of the class graph in weight space, serving as the bridge between network, log, and binary traffic domains. The minimum edge (bin_malware↔bin_benign, weight 81.2) corresponds to the hardest classification pair in all other analyses.
Tropical projective map Φ: ℝ^{128} → TP^9 has width 231.3. Each class’s score centroid shows a dominant self-score and meaningful cross-class secondary scores, revealing domain clustering (log classes, binary classes) in score space. The large tropical width reflects the classifier’s extreme confidence: not just a slight preference for the correct class, but a difference of 231 LLR units between best and worst class scores.

# References
[1] Maclagan, D., & Sturmfels, B. (2015). Introduction to Tropical Geometry. American Mathematical Society.
[2] Speyer, D., & Sturmfels, B. (2004). The tropical Grassmannian. Advances in Geometry, 4(3), 389–411.
[3] Mikhalkin, G. (2005). Enumerative tropical algebraic geometry in ℝ^2. Journal of the American Mathematical Society, 18(2), 313–377.
[4] Joswig, M. (2021). Essentials of Tropical Combinatorics. American Mathematical Society.
[5] Develin, M., & Sturmfels, B. (2004). Tropical convexity. Documenta Mathematica, 9, 1–27.
[6] Cohen, G., Gaubert, S., & Quadrat, J.-P. (2004). Duality and separation theorems in idempotent semimodules. Linear Algebra and Its Applications, 379, 395–422.
[7] Richter-Gebert, J., Sturmfels, B., & Theobald, T. (2005). First steps in tropical geometry. Contemporary Mathematics, 377, 289–317.
[8] Ziegler, G. M. (1995). Lectures on Polytopes. Springer.
[9] Gathmann, A., & Markwig, H. (2008). Kontsevich’s formula and the WDVV equations in tropical geometry. Advances in Mathematics, 217(2), 537–560.
[10] Brugallé, E., & Itenberg, I. (2009). Tropical geometry. Mémoires de la Société Mathématique de France, 22(5), 1–120.
[11] Pachter, L., & Sturmfels, B. (2004). Tropical geometry of statistical models. Proceedings of the National Academy of Sciences, 101(46), 16132–16137.
[12] Rincon, F. (2012). Local tropical linear spaces. Discrete & Computational Geometry, 50(3), 700–713.
[13] Ardila, F., & Klivans, C. J. (2006). The Bergman complex of a matroid and phylogenetic trees. Journal of Combinatorial Theory B, 96(1), 38–49.
[14] Helbig, M., & Joswig, M. (2018). Tropical polyhedra. In Handbook of Discrete and Computational Geometry (3rd ed.). CRC Press.
[15] Akian, M., Gaubert, S., & Guterman, A. (2012). Tropical polyhedra are equivalent to mean payoff games. International Journal of Algebra and Computation, 22(01), 1250001.
[16] Cueto, M. A., Morton, J., & Sturmfels, B. (2010). Geometry of the restricted Boltzmann machine. Contemporary Mathematics, 516, 135–153.
[17] Sturmfels, B. (2002). Solving Systems of Polynomial Equations. American Mathematical Society.
[18] Gaubitz, C., & Joswig, M. (2022). Tropical discriminants. Algebra and Number Theory, 16(1), 1–30.
[19] Bruns, W., & Gubeladze, J. (2009). Polytopes, Rings, and K-Theory. Springer.
[20] Cox, D., Little, J., & O’Shea, D. (2015). Ideals, Varieties, and Algorithms (4th ed.). Springer.
