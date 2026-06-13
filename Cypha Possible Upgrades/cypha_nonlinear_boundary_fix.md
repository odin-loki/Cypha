# CyphaDIF Nonlinear Boundary Fix
**Author:** Odin Loch  
**Problem:** CyphaDIF LLR discriminant is linear in latent space. XOR-style tasks hard-ceiling at ~48.2% vs kernel SVM 83.5% — a 32.3pp gap. This also limits CyphaLM's ability to model nonlinear token dependencies.

---

## Root Cause

CyphaDIF classifies via:

```
y* = argmax_k [ log p(h | θ₀ ⊕ Δk) + log p(k | context) ]
```

Where `log p(h | θ₀ ⊕ Δk)` is a log-likelihood under a diagonal Gaussian. This is a **quadratic discriminant in h-space** — which reduces to linear when class covariances are tied (the default). The `EncoderProjection W_enc` maps raw features to latent h via a linear projection. So the full pipeline is:

```
x → W_enc (linear) → h → LLR (linear in h) → y*
```

**No nonlinear transformation anywhere in the discriminant path.** XOR is not linearly separable in any basis reachable by a linear encoder. This is structural, not a tuning problem.

---

## Fix Taxonomy

Five independent approaches, ordered by implementation cost and theoretical fit with Cypha's maths.

---

## Fix 1 — Nyström Kernel LLR (Priority: Critical)

**What:** Replace the linear LLR with a kernel LLR using Nyström approximation. Already flagged in `docs/FUTURE.md` as highest-priority architectural upgrade.

**How it works:**

Instead of computing `log p(h | N(μ_k, diag v_k))` in the original latent space, map h to a higher-dimensional reproducing kernel Hilbert space (RKHS) via Nyström features, then compute LLR there.

Nyström approximation:
```
φ(h) = K(h, Z) K(Z,Z)^{-1/2}    where Z = m landmark points
```

The LLR becomes:
```
log p(φ(h) | N(μ_k^φ, diag v_k^φ))
```

Which is now a kernel discriminant — can separate XOR, concentric rings, any kernel-separable boundary.

**Implementation:**

```python
class NystromEncoder(nn.Module):
    """
    Nyström kernel feature map.
    Approximates RBF kernel k(x,y) = exp(-||x-y||² / 2σ²)
    with m landmark points.
    """
    def __init__(self, input_dim, m=256, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        # Landmarks: learnable or fixed from training data subset
        self.landmarks = nn.Parameter(torch.randn(m, input_dim))
        self.K_inv_sqrt = None  # computed after landmark init

    def fit_landmarks(self, X_sample: torch.Tensor):
        """Call once on a representative data sample to compute K(Z,Z)^{-1/2}."""
        with torch.no_grad():
            K_zz = self._rbf(self.landmarks, self.landmarks)
            # Eigendecomposition for stable sqrt inverse
            L, V = torch.linalg.eigh(K_zz + 1e-6 * torch.eye(K_zz.shape[0]))
            L_inv_sqrt = torch.diag(1.0 / L.clamp(min=1e-8).sqrt())
            self.K_inv_sqrt = V @ L_inv_sqrt @ V.T

    def _rbf(self, A, B):
        dist_sq = torch.cdist(A, B).pow(2)
        return torch.exp(-dist_sq / (2 * self.sigma ** 2))

    def forward(self, h):
        K_xz = self._rbf(h, self.landmarks)          # (batch, m)
        if self.K_inv_sqrt is None:
            return K_xz                               # fallback: no normalisation
        return K_xz @ self.K_inv_sqrt                 # (batch, m)


class CyphaDIF_Nystrom(CyphaDIF):
    """
    CyphaDIF with Nyström kernel LLR.
    Drop-in replacement: swap EncoderProjection output through NystromEncoder
    before feeding to DIFMemory.
    """
    def __init__(self, *args, nystrom_m=256, nystrom_sigma=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.nystrom = NystromEncoder(
            input_dim=self.feat_dim,
            m=nystrom_m,
            sigma=nystrom_sigma
        )
        # Resize DIFMemory to Nyström output dim
        self._reinit_memory(nystrom_m)
```

**Hyperparameters to sweep:**
| Parameter | Range | Notes |
|---|---|---|
| m (landmarks) | 64, 128, 256, 512 | More = better approx, more memory |
| σ (RBF bandwidth) | 0.1, 0.5, 1.0, 2.0 | Median heuristic: σ = median(‖x_i − x_j‖) |
| Kernel type | RBF, Matérn-3/2, polynomial | RBF first; Matérn if RBF overfits |

**Expected outcome:** XOR ceiling lifts from 48.2% to >80%. Gap vs kernel SVM (83.5%) should close to <5pp.

**Parity requirement:** Python NystromEncoder must have a byte-identical C++ equivalent in `native/`. Add to parity matrix as `nystrom_lkl_parity`.

---

## Fix 2 — RFF Kernel Approximation (Faster Alternative to Nyström)

**What:** Random Fourier Features (RFF) approximate the same RBF kernel via random projections. Already partially present as `RFFEncoder` in Cypha — extend it to the LLR path.

**How it works:**

Bochner's theorem: a shift-invariant kernel k(x,y) = k(x-y) can be approximated by:
```
φ(x) = √(2/D) [cos(ω₁ᵀx + b₁), ..., cos(ω_Dᵀx + b_D)]
where ω_i ~ p(ω), b_i ~ Uniform[0, 2π]
```

For RBF: `p(ω) = N(0, σ⁻² I)`.

**Key difference from Nyström:** RFF uses random projections (no landmark fitting). Faster to initialise, slightly less accurate per feature than Nyström. Good for online/streaming settings.

**Implementation:**

```python
class RFFKernelLLR(nn.Module):
    """
    Replaces diagonal Gaussian LLR with RFF-approximated kernel LLR.
    Uses existing RFFEncoder infrastructure.
    """
    def __init__(self, input_dim, D=512, sigma=1.0):
        super().__init__()
        self.D = D
        self.sigma = sigma
        # Fixed random weights (not learned — that's the RFF guarantee)
        omega = torch.randn(input_dim, D) / sigma
        bias = torch.rand(D) * 2 * torch.pi
        self.register_buffer('omega', omega)
        self.register_buffer('bias', bias)

    def forward(self, h):
        """Map h to RFF feature space."""
        proj = h @ self.omega + self.bias          # (batch, D)
        return (2.0 / self.D) ** 0.5 * torch.cos(proj)

    def kernel_llr(self, h, mu_k, v_k):
        """LLR in RFF space under diagonal Gaussian."""
        phi_h = self.forward(h)
        phi_mu = self.forward(mu_k.unsqueeze(0))
        diff = phi_h - phi_mu
        return -0.5 * (diff.pow(2) / v_k.clamp(min=1e-8)).sum(-1)
```

**Comparison vs Nyström:**
| | Nyström | RFF |
|---|---|---|
| Accuracy per feature | Higher | Lower |
| Init cost | O(m³) eigendecomp | O(D) sampling |
| Online update | Requires re-fit | No update needed |
| Recommended for | Batch/offline | Online/streaming |

**Use RFF for CyphaLM (streaming), Nyström for CyphaDIF classification (batch).**

---

## Fix 3 — Learned Nonlinear Encoder (Neural Featuriser)

**What:** Replace the linear `EncoderProjection W_enc` with a shallow MLP (2 layers, GELU or EML activation). The nonlinearity lives in the encoder, not the discriminant.

**How it works:**

```
x → MLP_enc(x) → h (nonlinear features) → linear LLR → y*
```

A 2-layer MLP with EML activations is sufficient to separate XOR. The LLR stays linear in h, but h is now a nonlinear function of x.

**Implementation:**

```python
class NonlinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='eml'):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        h = self.fc1(x)
        if self.activation == 'eml':
            h = eml_gate(h)
        elif self.activation == 'gelu':
            h = F.gelu(h)
        else:
            h = torch.tanh(h)
        return self.fc2(h)
```

**Trade-offs:**
- Loses the Cramér-Rao efficiency guarantee (natural gradient property of linear encoder)
- Fisher-Rao contrastive update needs to be rewritten for MLP encoder
- Simpler than Nyström, faster to implement
- May hurt calibration — NIG gate and OOD detection depend on the Gaussian structure of h

**Recommendation:** Use as a diagnostic first. If nonlinear encoder alone closes the XOR gap, it tells you the bottleneck is purely in the encoder. If not, Nyström is needed.

---

## Fix 4 — GRIA-α Kernel (Custom Kernel Derived from Your Maths)

**What:** Define a custom kernel derived from GRIA's order parameter α. Instead of RBF k(x,y) = exp(-‖x-y‖²), use:

```
k_α(x, y) = exp(−|α(x) − α(y)|² / 2τ²) · exp(−‖x−y‖² / 2σ²)
```

This kernel measures similarity in both feature space AND in compression-complexity space. Two points that are close in feature space but have very different α (one ordered, one chaotic) are considered dissimilar.

**Derivation:** GRIA α is an invariant of the data generating process, not just the features. Using it as a kernel dimension means the discriminant operates on information-geometric structure, not just Euclidean distance.

**Implementation:**

```python
def gria_alpha_kernel(X, Y, sigma=1.0, tau=0.5):
    """
    Custom GRIA-α kernel.
    X, Y: (n, d) and (m, d) feature matrices
    """
    # Standard RBF component
    rbf = torch.exp(-torch.cdist(X, Y).pow(2) / (2 * sigma**2))
    # GRIA α component
    alpha_X = gria_alpha(X)   # (n, 1)
    alpha_Y = gria_alpha(Y)   # (m, 1)
    alpha_dist = (alpha_X - alpha_Y.T).pow(2)
    alpha_rbf = torch.exp(-alpha_dist / (2 * tau**2))
    return rbf * alpha_rbf
```

**Use with Nyström:** Feed `gria_alpha_kernel` as the kernel function to `NystromEncoder`. This is the most theoretically coherent fix — the kernel is derived from the same maths as the rest of Cypha.

---

## Fix 5 — Spectral Mixture Kernel

**What:** Replace diagonal Gaussian LLR with a spectral mixture kernel (Wilson & Adams 2013). Learns a mixture of RBF kernels at multiple length scales, each with learned frequency and bandwidth.

**Connection to Cypha:** Your theoretical backbone's `σ_k ∝ 1/k` harmonic structure is directly expressible as a spectral mixture kernel component. This makes the kernel derivable from first principles.

**Spectral mixture kernel:**
```
k_SM(τ) = Σ_q w_q · exp(−2π²τ²v_q) · cos(2πτμ_q)
```

Where `(w_q, μ_q, v_q)` are learned mixture weights, frequencies, and variances.

**Implementation:** Use GPyTorch's `SpectralMixtureKernel` as a drop-in. Set initial frequencies from GRIA spectral analysis of training data.

---

## Test Protocol

### Diagnostic Battery (run in order)

**D1 — Confirm the ceiling:**
```python
# XOR dataset, 1000 points, 2D
X = torch.randn(1000, 2)
y = (X[:, 0] * X[:, 1] > 0).long()
# Run CyphaDIF default → confirm ~48% accuracy
# Run kernel SVM → confirm ~83% accuracy
# Gap = structural baseline
```

**D2 — Encoder nonlinearity test:**
```python
# Same XOR dataset
# Fix 3: NonlinearEncoder (2-layer MLP, EML activation)
# Expected: should reach >80% if encoder is the bottleneck
```

**D3 — Nyström LLR test:**
```python
# Same XOR dataset
# Fix 1: NystromEncoder, m=256, σ=median heuristic
# Expected: >80% if LLR linearity is the bottleneck
```

**D4 — Combined (Fix 1 + Fix 3):**
```python
# Nonlinear encoder feeding Nyström LLR
# Expected: ceiling should match or exceed kernel SVM
```

**D5 — Real benchmark:**
```
Once XOR is solved, run full diagnostic suite from docs/reports/DIAGNOSTIC_REPORT.md
Target: all tasks ≥ SVM ceiling (current gap on XOR = 32.3pp)
```

### Datasets for full validation
| Dataset | Current CyphaDIF | SVM ceiling | Target |
|---|---|---|---|
| S1 linearly-separable | 0.783 | 0.898 | >0.90 |
| S3 XOR | 0.482 | 0.825 | >0.80 |
| R1 Iris | 0.900 | 0.968 | >0.95 |
| R3 Digits (10-class) | 0.922 | 0.982 | >0.97 |

---

## Parity Requirements

Any fix merged to `native/` must pass parity tests. Add to `docs/port/PORT_CONTRACT.md`:

| Test | What it verifies |
|---|---|
| `nystrom_lkl_parity` | Python ↔ C++ Nyström LLR agreement |
| `rff_kernel_parity` | Python ↔ C++ RFF kernel LLR agreement |
| `nonlinear_enc_parity` | Python ↔ C++ MLP encoder agreement |
| `alpha_kernel_parity` | Python ↔ C++ GRIA-α kernel agreement |

---

## Recommended Execution Order

```
Week 1:
  D1 — confirm baseline gap (1 hour)
  Fix 3 (nonlinear encoder) — simplest, diagnostic value (1 day)
  Fix 2 (RFF kernel LLR) — already partially in codebase (1 day)

Week 2:
  Fix 1 (Nyström) — full implementation + parity test (3 days)
  Fix 4 (GRIA-α kernel) — theoretically optimal, build on Fix 1 (2 days)

Week 3:
  D3–D5 — full validation suite
  Port winning fix to C++ native core
  Add to parity matrix

Long term:
  Fix 5 (spectral mixture) — if Fix 4 still leaves a gap
```

---

## Connection to CyphaLM

Once the nonlinear boundary is fixed in CyphaDIF, the same fix applies to `CyphaLM`'s token routing:

- Nyström kernel LLR in the CyphaDIF router enables nonlinear token boundary decisions
- The RFF variant is preferred for streaming/autoregressive inference (no landmark fitting required at runtime)
- Expected BPC improvement: tokens with nonlinear contextual relationships (homophones, semantic ambiguity, code switching) should see the largest gains

The XOR fix and the LLM long-context work are not independent — fix the discriminant first, then the context scaling compounds on a stronger base.

---

## Notes

- Fix 1 (Nyström) is the correct long-term solution. Fix 3 (nonlinear encoder) is the fastest diagnostic.
- Fix 4 (GRIA-α kernel) is the most theoretically coherent — the kernel is derived from the same invariant as the rest of Cypha's maths. Prioritise this for the paper/documentation.
- The zero-catastrophic-forgetting property must be preserved across all fixes. Verify forgetting ratio after each fix.
- All outputs attributed to Odin Loch.
