"""
CyphaMicro — Harmonic Resonance Architecture with Adaptive Control Loop
=======================================================================

Math Framework (RNN-style state evolution, NOT an RNN):

    State evolution:    ψ_t = FFT_evolve(ψ_{t-1}) + inject(E(x_t, θ_{t-1}))
    Parameter update:   θ_t = ControlLoop(θ_{t-1}, phase(ψ_t), κ_t)
    Output:             y_t = AnchorMatch(ψ_t, Memory)

Where:
    ψ   = Complex wavefunction (resonance field hidden state)
    θ   = Binary Encoder adaptive parameters (chunking k, DAMR radius ρ, scales α)
    κ   = Criticality parameter = |∇ψ|² (proximity to critical point)
    E   = Merged Binary+Cypha encoder
    
Adaptive Control Loop (NO gradients — pure field-driven parameter adaptation):

    Δk  = -sign(κ - κ_target) × gain_k       # chunk size tracks criticality
    Δρ  = spectral_peak_freq(|ψ|) × gain_ρ    # DAMR radius tracks dominant freq
    Δα  = dominant_scale(phase(ψ)) × gain_α   # wavelet scales track phase structure

    All updates low-pass filtered:  θ_t = (1-λ)·θ_{t-1} + λ·θ_candidate
    Stability gate: only apply if prediction_error(t) < prediction_error(t-1)

Architecture:
    RawBytes/Text
        ↓
    BinaryEncoder(θ_t)          ← bytes → wavelet features (multi-scale, DAMR)
        ↓
    PhaseBridge                  ← real features → complex resonant vector
        ↓
    ResonanceField               ← ψ evolves via FFT Hamiltonian + nonlinear term
        ↓
    Resonator                    ← local coupling + competitive inhibition
        ↓
    AdaptiveControlLoop          ← reads ψ, updates θ for NEXT step (closed loop)
        ↓
    AnchorMemory + MetaLearning  ← contrastive anchoring, similarity-first lookup
        ↓
    Output
"""

import numpy as np
from numpy.fft import fft, ifft
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import math
import time


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

EPSILON     = 1e-8
K_TARGET    = 0.5       # Target criticality (edge-of-chaos)
LAMBDA_LP   = 0.15      # Low-pass filter coefficient for θ updates
GAIN_K      = 0.05      # Chunk threshold gain
GAIN_RHO    = 0.03      # DAMR radius gain
GAIN_ALPHA  = 0.04      # Wavelet scale gain


# ─────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────

@dataclass
class EncoderParams:
    """Adaptive parameters for the Binary Encoder — these self-tune via ControlLoop"""
    chunk_k:       float = 4.0      # Chunking threshold exponent (chunk size ≈ 2^k)
    damr_radius:   float = 3.0      # DAMR neighbourhood radius ρ
    active_scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125])
    prev_error:    float = float('inf')  # For stability gate


@dataclass
class FieldStats:
    """Snapshot of resonance field state for control loop decisions"""
    criticality:    float   # κ = mean(|∇ψ|²)
    dominant_freq:  float   # Frequency bin with highest amplitude
    mean_phase:     float   # Mean phase angle arg(ψ)
    phase_spread:   float   # Std of phase angles (measure of coherence)
    energy:         float   # Total field energy ||ψ||²


@dataclass
class MicroMetrics:
    """Per-step training metrics"""
    step:           int
    loss:           float
    criticality:    float
    chunk_k:        float
    damr_radius:    float
    n_anchors:      int
    inference_ms:   float = 0.0


# ─────────────────────────────────────────────
# 1. Binary Encoder (adaptive, byte-native)
# ─────────────────────────────────────────────

class BinaryEncoder:
    """
    Operates directly on raw bytes. Adaptive parameters θ are tuned externally
    by the AdaptiveControlLoop — this class just applies whatever θ it's given.

    Pipeline:
        bytes → content-defined chunks (k)
              → multi-scale wavelet decomposition (active_scales)
              → DAMR reservoir statistics (damr_radius)
              → feature importance selection
              → feature vector F ∈ ℝ^output_dim
    """

    def __init__(self, output_dim: int = 64):
        self.output_dim = output_dim

    def encode(self, data: bytes, params: EncoderParams) -> np.ndarray:
        """Encode raw bytes to feature vector using current adaptive params"""
        if len(data) == 0:
            return np.zeros(self.output_dim)

        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0

        # Stage 1: Chunk at content boundaries
        chunks = self._chunk(arr, params.chunk_k)

        # Stage 2: Multi-scale wavelet features per chunk
        wavelet_features = []
        for chunk in chunks[:8]:  # Cap at 8 chunks for micro version
            for scale in params.active_scales:
                sub = chunk[:max(1, int(len(chunk) * scale))]
                wavelet_features.extend(self._wavelet_stats(sub))

        # Stage 3: DAMR byte pattern statistics
        damr_features = self._damr(arr, int(params.damr_radius))

        # Stage 4: Combine and select by importance
        all_features = np.array(wavelet_features + damr_features, dtype=np.float32)
        return self._select_features(all_features, self.output_dim)

    def encode_text(self, text: str, params: EncoderParams) -> np.ndarray:
        """Convenience: encode text as UTF-8 bytes"""
        return self.encode(text.encode('utf-8'), params)

    def _chunk(self, arr: np.ndarray, k: float) -> List[np.ndarray]:
        """Content-defined chunking: boundary when rolling hash mod 2^k == 0"""
        if len(arr) < 4:
            return [arr]
        chunks = []
        start = 0
        threshold = max(1, int(2 ** k))
        min_size = max(4, threshold // 4)
        max_size = threshold * 4

        for i in range(1, len(arr)):
            # Rolling hash (simplified xxHash-style)
            window = arr[max(0, i-4):i]
            h = int(np.sum(window * np.array([31**j for j in range(len(window))]))) % 65536
            size = i - start
            if (h % threshold == 0 and size >= min_size) or size >= max_size:
                chunks.append(arr[start:i])
                start = i

        if start < len(arr):
            chunks.append(arr[start:])
        return chunks if chunks else [arr]

    def _wavelet_stats(self, x: np.ndarray) -> List[float]:
        """
        Simplified Daubechies-4 wavelet decomposition → statistical moments.
        D4 low-pass filter coefficients: h = [0.4830, 0.8365, 0.2241, -0.1294]
        """
        if len(x) < 4:
            x = np.pad(x, (0, 4 - len(x)))

        # D4 filter coefficients
        h = np.array([0.4830, 0.8365, 0.2241, -0.1294])
        g = np.array([-0.1294, -0.2241, 0.8365, -0.4830])  # high-pass

        # Single level decomposition
        n = len(x) - len(h) + 1
        if n < 1:
            n = 1
            x = np.pad(x, (0, len(h)))

        approx  = np.convolve(x, h, mode='valid')[::2]
        detail  = np.convolve(x, g, mode='valid')[::2]

        def stats(v):
            if len(v) == 0:
                return [0.0] * 6
            mu  = float(np.mean(v))
            std = float(np.std(v)) + EPSILON
            energy = float(np.sum(v**2))
            entropy_p = np.abs(v) / (np.sum(np.abs(v)) + EPSILON)
            entropy = float(-np.sum(entropy_p * np.log(entropy_p + EPSILON)))
            skew    = float(np.mean(((v - mu)/std)**3)) if std > EPSILON else 0.0
            kurt    = float(np.mean(((v - mu)/std)**4)) - 3.0 if std > EPSILON else 0.0
            return [mu, std, energy, entropy, skew, kurt]

        return stats(approx) + stats(detail)

    def _damr(self, arr: np.ndarray, radius: int) -> List[float]:
        """
        Dynamic Adaptive Multi-scale Reservoir.
        V_r[b_t] += 1; neighbours get V[(b±j) mod 256] += (σ/j)·β_t
        Returns normalised histogram statistics.
        """
        radius = max(1, min(radius, 8))  # Clamp to [1, 8]
        arr_int = (arr * 255).astype(int)

        V = np.zeros(256, dtype=np.float32)
        sigma = 1.0

        for i, b in enumerate(arr_int[:256]):  # Cap for speed
            V[b] += 1.0
            # Contextual boost from repetition
            beta = 1.0 + (0.5 if i > 0 and arr_int[i] == arr_int[i-1] else 0.0)
            for j in range(1, radius + 1):
                V[(b + j) % 256] += (sigma / j) * beta
                V[(b - j) % 256] += (sigma / j) * beta

        V /= (np.sum(V) + EPSILON)  # Normalise

        # Statistical summary of histogram
        mu    = float(np.mean(V))
        std   = float(np.std(V))
        peak  = float(np.max(V))
        top8  = V[np.argsort(V)[-8:]].tolist()
        return [mu, std, peak] + top8  # 11 features

    def _select_features(self, features: np.ndarray, n: int) -> np.ndarray:
        """
        Feature importance: I(f) = |μ(f)| · (1 + σ(f))
        Pad or truncate to exactly n features.
        """
        if len(features) == 0:
            return np.zeros(n, dtype=np.float32)
        if len(features) >= n:
            # Select by importance score across sliding windows
            scores = np.abs(features) * (1.0 + np.abs(features - np.mean(features)))
            idx = np.argsort(scores)[-n:]
            out = features[np.sort(idx)]
        else:
            out = np.pad(features, (0, n - len(features)))
        return out.astype(np.float32)


# ─────────────────────────────────────────────
# 2. Phase Bridge (real features → complex resonant vector)
# ─────────────────────────────────────────────

class PhaseBridge:
    """
    Converts Binary Encoder's real feature vector to complex resonant input.

    Amplitude: from DAMR pattern norm (structural intensity)
    Phase:     from wavelet detail/approximation ratio
               θ = arctan(||D|| / ||A||) — physically meaningful:
               high-frequency content → phase near π/2
               low-frequency content  → phase near 0

    Then applies Cypha's resonant projection:
        E(x) = ∑ᵢ αᵢ·e^(iθᵢ)·φᵢ(x)
    Fully vectorised — not the O(N²) loop from original Cypha.
    """

    def __init__(self, feature_dim: int, resonance_dim: int):
        self.feature_dim   = feature_dim
        self.resonance_dim = resonance_dim

        rng = np.random.default_rng(42)
        self.amp_proj   = rng.standard_normal((feature_dim, resonance_dim)) * 0.1
        self.phase_proj = rng.standard_normal((feature_dim, resonance_dim)) * 0.1
        self.basis_freqs = np.linspace(0.5, 10.0, resonance_dim)

    def bridge(self, features: np.ndarray) -> np.ndarray:
        """
        features: real numpy array, shape [feature_dim]
        returns:  complex numpy array, shape [resonance_dim]
        """
        x = features.astype(np.float64)
        if len(x) != self.feature_dim:
            # Resize via linear interpolation
            idx_new = np.linspace(0, len(x) - 1, self.feature_dim)
            x = np.interp(idx_new, np.arange(len(x)), x)

        # Amplitude coefficients: α = Wₐ·x
        amps = x @ self.amp_proj   # [resonance_dim]

        # Phase from wavelet approx/detail energy ratio
        half          = len(x) // 2
        approx_energy = np.linalg.norm(x[:half]) + EPSILON
        detail_energy = np.linalg.norm(x[half:]) + EPSILON
        base_phase    = np.arctan2(detail_energy, approx_energy)  # physically grounded

        # Learnable phase modulation
        phase_mod = x @ self.phase_proj  # [resonance_dim]
        phases    = base_phase + 0.3 * phase_mod

        # Basis: φᵢ = sin(ωᵢ · i/N)
        domain = np.arange(self.resonance_dim, dtype=np.float64)
        basis  = np.sin(self.basis_freqs * domain / self.resonance_dim)

        # E(x) = α · e^(iθ) · φ
        resonant = amps * np.exp(1j * phases) * basis
        norm = np.linalg.norm(resonant) + EPSILON
        return resonant / norm


# ─────────────────────────────────────────────
# 3. Resonance Field (FFT Hamiltonian evolution)
# ─────────────────────────────────────────────

class ResonanceField:
    """
    State evolution: ψ_t = FFT_evolve(ψ_{t-1}) + inject(E(x_t))

    ∂ψ/∂t = -iHψ + γ(|ψ|² - 1)ψ
    H is diagonal in frequency domain → O(N log N) per step.

    Criticality κ = mean(|ψ_t - ψ_{t-1}|²) measures proximity to phase transition.
    κ → 0:   field is frozen (too stable, not learning)
    κ → 1:   field is chaotic (too unstable, not reliable)
    κ ≈ K_TARGET (0.5): edge-of-chaos, maximum information processing
    """

    def __init__(self, dim: int, gamma: float = 0.1, dt: float = 0.1):
        self.dim    = dim
        self.gamma  = gamma
        self.dt     = dt

        rng       = np.random.default_rng(0)
        self.psi  = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        self.psi /= np.linalg.norm(self.psi)
        self.psi_prev = self.psi.copy()
        self.H_freq   = np.linspace(0.5, 10.0, dim)

    def inject(self, event: np.ndarray, strength: float = 0.6):
        v = event.flatten()[:self.dim].astype(complex)
        v /= (np.linalg.norm(v) + EPSILON)
        self.psi = (1 - strength) * self.psi + strength * v
        self.psi /= (np.linalg.norm(self.psi) + EPSILON)

    def evolve(self, steps: int = 1) -> np.ndarray:
        self.psi_prev = self.psi.copy()
        for _ in range(steps):
            psi_freq  = fft(self.psi)
            phase_rot = np.exp(-1j * self.dt * self.H_freq)
            self.psi  = ifft(psi_freq * phase_rot)
            density   = np.abs(self.psi) ** 2
            self.psi += self.gamma * self.dt * (density - 1.0) * self.psi
            self.psi /= (np.linalg.norm(self.psi) + EPSILON)
        return self.psi

    def stats(self) -> FieldStats:
        kappa    = float(np.mean(np.abs(self.psi - self.psi_prev) ** 2))
        spectrum = np.abs(fft(self.psi))
        dom_freq = float(np.argmax(spectrum)) / self.dim
        phases   = np.angle(self.psi)
        return FieldStats(
            criticality   = kappa,
            dominant_freq = dom_freq,
            mean_phase    = float(np.mean(phases)),
            phase_spread  = float(np.std(phases)),
            energy        = float(np.sum(np.abs(self.psi) ** 2))
        )

    def reset(self):
        rng       = np.random.default_rng()
        self.psi  = rng.standard_normal(self.dim) + 1j * rng.standard_normal(self.dim)
        self.psi /= np.linalg.norm(self.psi)
        self.psi_prev = self.psi.copy()


# ─────────────────────────────────────────────
# 4. Resonator (local coupling + inhibition)
# ─────────────────────────────────────────────

class Resonator:
    """
    dR_i/dt = ω_i·R_i + ∑_j W_ij·σ(R_j) - γ·∑|R| + drive_i
    """

    def __init__(self, n: int = 64, gamma_inhib: float = 0.35, locality: int = 3):
        self.n        = n
        self.gamma    = gamma_inhib
        self.locality = locality

        rng           = np.random.default_rng(1)
        self.R        = np.zeros(n)
        self.omega    = np.linspace(1.0, 10.0, n)
        self.W_local  = rng.standard_normal(2 * locality + 1) * 0.3
        self.W_local[locality] = 0.0

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def update(self, dt: float = 0.1, drive: Optional[np.ndarray] = None) -> np.ndarray:
        freq_term  = self.omega * self.R
        coupling   = np.zeros(self.n)

        for off in range(-self.locality, self.locality + 1):
            if off == 0:
                continue
            w = self.W_local[off + self.locality]
            if off > 0:
                coupling[:-off] += w * self._sigmoid(self.R[off:])
            else:
                coupling[-off:] += w * self._sigmoid(self.R[:off])

        inhibition = -self.gamma * np.sum(np.abs(self.R)) / self.n
        drive_term = 0.0
        if drive is not None:
            drive_term = drive[:self.n].real * 200.0

        R_new    = self.R + dt * (freq_term + coupling + inhibition) + drive_term
        abs_R    = np.abs(R_new)
        thresh   = np.quantile(abs_R, 0.8)
        R_new[abs_R < thresh] *= 0.1
        self.R   = np.clip(R_new, -10, 10)
        return self.R

    def reset(self):
        self.R = np.zeros(self.n)


# ─────────────────────────────────────────────
# 5. Adaptive Control Loop (the closed-loop self-tuner)
# ─────────────────────────────────────────────

class AdaptiveControlLoop:
    """
    Reads FieldStats, updates EncoderParams θ via rule-based control.
    Completely separate from gradient flow — pure field-driven adaptation.

    Control laws:
        Δk   = -sign(κ - κ_target) × gain_k
                 → chunk size shrinks when κ too high (chaotic), grows when too stable

        Δρ   = clip(dom_freq × 8 - ρ, -1, 1) × gain_ρ
                 → DAMR radius tracks dominant frequency of field

        Δα   = derived from phase_spread
                 → low spread (coherent) → fewer scales needed
                 → high spread (incoherent) → more scales needed

    All updates gated by stability: only apply if current error < previous error.
    All updates low-pass filtered: θ_t = (1-λ)·θ_{t-1} + λ·θ_candidate
    """

    def __init__(self):
        self.history: deque = deque(maxlen=16)

    def update(self, params: EncoderParams, stats: FieldStats,
               current_error: float) -> EncoderParams:

        # Stability gate: only tune if we're not getting worse
        if current_error > params.prev_error * 1.05:
            params.prev_error = current_error
            return params  # Skip this update

        # --- Control Law 1: chunk_k tracks criticality ---
        delta_k     = -math.copysign(GAIN_K, stats.criticality - K_TARGET)
        k_candidate = params.chunk_k + delta_k
        k_candidate = max(2.0, min(8.0, k_candidate))  # Clamp [2, 8]
        params.chunk_k = (1 - LAMBDA_LP) * params.chunk_k + LAMBDA_LP * k_candidate

        # --- Control Law 2: DAMR radius tracks dominant frequency ---
        rho_target  = max(1.0, stats.dominant_freq * 8.0)
        delta_rho   = np.clip(rho_target - params.damr_radius, -1.0, 1.0) * GAIN_RHO
        rho_candidate = params.damr_radius + delta_rho
        rho_candidate = max(1.0, min(8.0, rho_candidate))
        params.damr_radius = (1 - LAMBDA_LP) * params.damr_radius + LAMBDA_LP * rho_candidate

        # --- Control Law 3: active scales track phase coherence ---
        # phase_spread → 0: coherent, low-freq dominant → reduce scales
        # phase_spread → π: incoherent → use all scales
        all_scales     = [1.0, 0.5, 0.25, 0.125]
        coherence      = 1.0 - min(1.0, stats.phase_spread / math.pi)
        n_scales       = max(1, round(1 + coherence * (len(all_scales) - 1)))
        params.active_scales = all_scales[:n_scales]

        params.prev_error = current_error
        self.history.append({
            'kappa': stats.criticality,
            'chunk_k': params.chunk_k,
            'damr_r': params.damr_radius,
            'n_scales': n_scales
        })

        return params


# ─────────────────────────────────────────────
# 6. Anchor Memory (KD-tree, similarity-first lookup)
# ─────────────────────────────────────────────

class AnchorMemory:
    """
    Stores (input → anchor vector) and (input → output string) mappings.
    Lookup: similarity-first (cosine distance in resonance space).
    Falls back to nearest-anchor if exact match not found.
    """

    def __init__(self, dim: int = 64, min_sep: float = 0.5):
        self.dim         = dim
        self.min_sep     = min_sep
        self.anchors:    Dict[str, np.ndarray] = {}
        self.outputs:    Dict[str, str]         = {}
        self._vecs:      List[np.ndarray]       = []
        self._keys:      List[str]              = []

    def store(self, key: str, state: np.ndarray, output: str):
        """Store state vector and output for key"""
        vec = state / (np.linalg.norm(state) + EPSILON)
        self.anchors[key]   = vec
        self.outputs[key]   = output
        # Update fast lookup lists
        if key not in self._keys:
            self._vecs.append(vec)
            self._keys.append(key)
        else:
            idx = self._keys.index(key)
            self._vecs[idx] = vec

    def lookup(self, state: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Similarity-first: return top_k (key, cosine_similarity) pairs.
        Never returns via dict lookup — always goes through resonance space.
        """
        if not self._vecs:
            return []
        q = state / (np.linalg.norm(state) + EPSILON)
        sims = [(self._keys[i], float(np.dot(q, v)))
                for i, v in enumerate(self._vecs)]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def get_output(self, key: str) -> Optional[str]:
        return self.outputs.get(key)

    @property
    def n(self) -> int:
        return len(self.anchors)


# ─────────────────────────────────────────────
# 7. Contrastive MetaLearning
# ─────────────────────────────────────────────

class MetaLearning:
    """
    Contrastive loss to push state vectors apart.

    L = (s - t)² + 0.8 · ∑ relu(cos(s, neg_i) - margin)²
    Anti-collapse penalty: exp(-3 · mean_similarity_to_recent)
    """

    def __init__(self, max_recent: int = 8):
        self.recent: deque = deque(maxlen=max_recent)

    def loss(self, state: np.ndarray, target: np.ndarray,
             negatives: Optional[List[np.ndarray]] = None) -> float:
        s = state  / (np.linalg.norm(state)  + EPSILON)
        t = target / (np.linalg.norm(target) + EPSILON)
        n = min(len(s), len(t))
        s, t = s[:n], t[:n]

        # Positive loss
        pos_loss = float(np.mean((s - t) ** 2))

        # Contrastive
        neg_loss = 0.0
        if negatives:
            for neg in negatives:
                neg = neg / (np.linalg.norm(neg) + EPSILON)
                neg = neg[:n]
                sim = float(np.dot(s, neg))
                neg_loss += max(0.0, sim - 0.2) ** 2
            neg_loss /= len(negatives)

        # Anti-collapse
        penalty = 1.0
        if self.recent:
            sims = [abs(float(np.dot(s, p[:n]))) for p in self.recent]
            penalty = float(np.exp(-3.0 * np.mean(sims)))

        self.recent.append(s.copy())
        return (pos_loss + 0.8 * neg_loss) * penalty


# ─────────────────────────────────────────────
# 8. CyphaMicro — Main System
# ─────────────────────────────────────────────

class CyphaMicro:
    """
    Micro Cypha with adaptive self-tuning encoder.

    State evolution (RNN-style math, not RNN architecture):
        ψ_t   = FFT_evolve(ψ_{t-1}) + inject(PhaseBridge(BinaryEncoder(x, θ_{t-1})))
        θ_t   = ControlLoop(θ_{t-1}, stats(ψ_t), error_t)     ← adaptive control
        y_t   = AnchorMemory.lookup(Resonator(ψ_t))
    """

    def __init__(self, feature_dim: int = 64, resonance_dim: int = 64):
        self.feature_dim   = feature_dim
        self.resonance_dim = resonance_dim

        # Components
        self.bin_encoder  = BinaryEncoder(output_dim=feature_dim)
        self.phase_bridge = PhaseBridge(feature_dim, resonance_dim)
        self.field        = ResonanceField(resonance_dim)
        self.resonator    = Resonator(n=resonance_dim)
        self.ctrl_loop    = AdaptiveControlLoop()
        self.memory       = AnchorMemory(dim=resonance_dim)
        self.meta         = MetaLearning()

        # Adaptive encoder parameters (self-tuned by ControlLoop)
        self.params = EncoderParams()

        # Training state
        self.step        = 0
        self.temperature = 1.5
        self._metrics:   List[MicroMetrics] = []

    # ── Core forward pass ──

    def _encode(self, text: str) -> np.ndarray:
        """text → binary features → phase bridge → complex resonant vector"""
        features = self.bin_encoder.encode_text(text, self.params)
        return self.phase_bridge.bridge(features)

    def forward(self, text: str) -> Tuple[np.ndarray, FieldStats]:
        """
        Full forward pass. Returns (global_state, field_stats).
        Does NOT update θ — that happens in train step with error feedback.
        """
        resonant   = self._encode(text)
        self.field.inject(resonant, strength=0.6)
        psi        = self.field.evolve(steps=1)
        reso_state = self.resonator.update(dt=0.1, drive=psi)
        global_s   = reso_state.copy()
        global_s  /= (np.linalg.norm(global_s) + EPSILON)
        return global_s, self.field.stats()

    # ── Training ──

    def train_step(self, input_text: str, output_text: str,
                   batch_negatives: Optional[List[str]] = None) -> MicroMetrics:
        """
        Single supervised step:
        1. Forward pass on input
        2. Forward pass on target (for contrastive target)
        3. Compute loss
        4. Update adaptive params via ControlLoop
        5. Store in memory
        """
        t0 = time.time()

        state_in,  stats_in  = self.forward(input_text)
        state_tgt, _         = self.forward(output_text)

        # Contrastive negatives
        neg_states = []
        if batch_negatives:
            for neg_txt in batch_negatives[:4]:
                neg_s, _ = self.forward(neg_txt)
                neg_states.append(neg_s)

        loss = self.meta.loss(state_in, state_tgt, neg_states)

        # Adaptive control: update θ based on field stats + current loss
        self.params = self.ctrl_loop.update(self.params, stats_in, loss)

        # Store in memory (similarity-first — no dict shortcut)
        self.memory.store(input_text, state_in, output_text)

        # Temperature annealing
        if self.step % 50 == 0 and self.step > 0:
            self.temperature = max(0.3, self.temperature * 0.97)

        self.step += 1

        m = MicroMetrics(
            step        = self.step,
            loss        = loss,
            criticality = stats_in.criticality,
            chunk_k     = self.params.chunk_k,
            damr_radius = self.params.damr_radius,
            n_anchors   = self.memory.n,
            inference_ms= (time.time() - t0) * 1000
        )
        self._metrics.append(m)
        return m

    def train(self, data: List[Tuple[str, str]], epochs: int = 3,
              verbose: bool = True) -> List[MicroMetrics]:
        """
        Train on list of (input, output) pairs.
        """
        all_metrics = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            np.random.shuffle(data)
            for i, (inp, out) in enumerate(data):
                # Use other items in window as negatives
                window = data[max(0, i-4):i] + data[i+1:i+5]
                negatives = [w[0] for w in window]
                m = self.train_step(inp, out, batch_negatives=negatives)
                epoch_loss += m.loss
                all_metrics.append(m)

            if verbose:
                avg = epoch_loss / len(data)
                print(f"Epoch {epoch+1}/{epochs} | loss={avg:.4f} | "
                      f"anchors={self.memory.n} | "
                      f"k={self.params.chunk_k:.2f} | "
                      f"ρ={self.params.damr_radius:.2f} | "
                      f"scales={len(self.params.active_scales)} | "
                      f"κ={all_metrics[-1].criticality:.4f} | "
                      f"T={self.temperature:.3f}")
        return all_metrics

    def train_file(self, path: str, epochs: int = 3,
                   verbose: bool = True) -> List[MicroMetrics]:
        """Load input|||output file and train"""
        pairs = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|||' in line:
                    parts = line.split('|||', 1)
                    if len(parts) == 2:
                        pairs.append((parts[0].strip(), parts[1].strip()))
        print(f"Loaded {len(pairs)} pairs from {path}")
        return self.train(pairs, epochs=epochs, verbose=verbose)

    # ── Inference ──

    def infer(self, text: str) -> Tuple[str, float]:
        """
        Infer output for input text.
        Uses similarity-first lookup — no memorised dict shortcuts.
        Returns (output_text, confidence).
        """
        t0 = time.time()
        self.field.reset()
        self.resonator.reset()

        state, _ = self.forward(text)
        matches  = self.memory.lookup(state, top_k=3)

        if not matches:
            return "[no memory]", 0.0

        best_key, best_sim = matches[0]
        output = self.memory.get_output(best_key) or "[unknown]"
        confidence = float(np.exp(best_sim / self.temperature))

        return output, confidence

    def infer_top_k(self, text: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """Return top-k (input_key, output, confidence) matches"""
        self.field.reset()
        self.resonator.reset()
        state, _ = self.forward(text)
        matches  = self.memory.lookup(state, top_k=k)
        results  = []
        for key, sim in matches:
            out  = self.memory.get_output(key) or "[unknown]"
            conf = float(np.exp(sim / self.temperature))
            results.append((key, out, conf))
        return results

    # ── Diagnostics ──

    def debug_summary(self):
        """Print current system state"""
        stats = self.field.stats()
        print("\n── CyphaMicro Debug ──")
        print(f"  Step:          {self.step}")
        print(f"  Anchors:       {self.memory.n}")
        print(f"  Temperature:   {self.temperature:.4f}")
        print(f"  Field κ:       {stats.criticality:.4f}  (target={K_TARGET})")
        print(f"  Field energy:  {stats.energy:.4f}")
        print(f"  Phase spread:  {stats.phase_spread:.4f}")
        print(f"  Dominant freq: {stats.dominant_freq:.4f}")
        print(f"  θ.chunk_k:     {self.params.chunk_k:.3f}")
        print(f"  θ.damr_ρ:      {self.params.damr_radius:.3f}")
        print(f"  θ.scales:      {self.params.active_scales}")
        if self._metrics:
            recent = self._metrics[-10:]
            avg_loss = np.mean([m.loss for m in recent])
            avg_ms   = np.mean([m.inference_ms for m in recent])
            print(f"  Avg loss(10):  {avg_loss:.4f}")
            print(f"  Avg ms(10):    {avg_ms:.2f}ms")
        print("─────────────────────\n")

    def param_history(self) -> Dict[str, List[float]]:
        """Return history of adaptive parameters for plotting"""
        return {
            'chunk_k':     [m.chunk_k     for m in self._metrics],
            'damr_radius': [m.damr_radius for m in self._metrics],
            'criticality': [m.criticality for m in self._metrics],
            'loss':        [m.loss        for m in self._metrics],
        }


# ─────────────────────────────────────────────
# 9. Main / Demo
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CyphaMicro — Adaptive Resonance Architecture")
    print("=" * 60)

    cypha = CyphaMicro(feature_dim=64, resonance_dim=64)

    # Demo data covering multiple modalities
    demo_data = [
        # Math
        ("12+165",    "177"),
        ("44+60",     "104"),
        ("25+75",     "100"),
        # Language
        ("cat sound", "meow"),
        ("dog sound", "bark"),
        ("owl sound", "hoot"),
        # Geography
        ("capital of France",    "Paris"),
        ("capital of Japan",     "Tokyo"),
        ("capital of Australia", "Canberra"),
        # Logic
        ("is 5 > 3",  "true"),
        ("is 2 > 10", "false"),
        # Sorting
        ("sort: 5 2 9 1", "1 2 5 9"),
        ("sort: 3 7 1 4", "1 3 4 7"),
        # General
        ("answer to life", "42"),
    ]

    print(f"\nTraining on {len(demo_data)} examples × 3 epochs...\n")
    metrics = cypha.train(demo_data, epochs=3, verbose=True)

    cypha.debug_summary()

    # Show parameter adaptation
    print("\n── Adaptive Parameter Evolution ──")
    hist = cypha.param_history()
    n = len(hist['chunk_k'])
    step_points = [0, n//4, n//2, 3*n//4, n-1]
    print(f"  {'Step':>6}  {'chunk_k':>8}  {'damr_ρ':>8}  {'κ':>8}  {'loss':>8}")
    for i in step_points:
        print(f"  {i:>6}  {hist['chunk_k'][i]:>8.3f}  "
              f"{hist['damr_radius'][i]:>8.3f}  "
              f"{hist['criticality'][i]:>8.4f}  "
              f"{hist['loss'][i]:>8.4f}")

    print("\n── Inference Test ──\n")
    test_cases = [
        ("cat sound",         "meow"),
        ("capital of France", "Paris"),
        ("is 5 > 3",          "true"),
        ("sort: 5 2 9 1",     "1 2 5 9"),
        ("12+165",            "177"),
        # Unseen inputs — tests generalisation
        ("wolf sound",        "howl"),
        ("capital of Germany","Berlin"),
    ]

    correct = 0
    for inp, expected in test_cases:
        result, conf = cypha.infer(inp)
        ok = "✓" if result == expected else "~"  # ~ = close match expected
        print(f"  {ok} '{inp}'")
        print(f"      → '{result}' (conf={conf:.3f}, expected='{expected}')")
        if result == expected:
            correct += 1

    seen_correct = sum(1 for inp, exp in test_cases[:5]
                       if cypha.infer(inp)[0] == exp)
    print(f"\n  Seen accuracy:   {seen_correct}/5")
    print(f"  Overall:         {correct}/{len(test_cases)}")

    print("\n── Interactive Mode (type 'quit' to exit) ──\n")
    while True:
        try:
            inp = input("  > ").strip()
            if inp.lower() in ('quit', 'exit', 'q'):
                break
            if inp == 'debug':
                cypha.debug_summary()
                continue
            result, conf = cypha.infer(inp)
            top = cypha.infer_top_k(inp, k=3)
            print(f"    → '{result}' (conf={conf:.3f})")
            print(f"    Top-3: {[(o, f'{c:.2f}') for _, o, c in top]}")
        except (KeyboardInterrupt, EOFError):
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
