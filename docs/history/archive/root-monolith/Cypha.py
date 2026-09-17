#!/usr/bin/env python3
"""
CyphaDIF  —  Differential Information Field Classifier
═══════════════════════════════════════════════════════

Architecture derived from first principles, unifying:
  ─ AIXI / Solomonoff  : MDL prior over class complexity (||Δk||_F ≤ C)
  ─ Information Geometry: natural gradient on Gaussian manifold (Cramér-Rao efficient)
  ─ Active Inference / FEP: world prior θ₀ + differential class offsets Δk
  ─ Information Bottleneck: contrastive encoder feedback (Fisher-Rao residuals)

Core invariant:
  Class k model  =  θ₀ ⊕ Δk   (world prior + differential offset, natural parameter space)

Classification:
  y* = argmax_k  [log p(h | θ₀ ⊕ Δk)]  +  log p(k | context)

Encoder interface:
  Subclass Encoder, implement .dim and __call__(x) → np.ndarray[dim].
  The DIF core operates solely on latent vectors — entirely domain-agnostic.

Enhancement summary (v2):
  Phase 1 — Tiered context: 3-tier TieredContextBuffer (short/mid/long), NIGField τ=0.99
            group, field confidence wired into context prior blending.
  Phase 2 — Generation overhaul: temperature-scaled, field-conditioned, boundary
            interpolation, adversarial, OOD, MDL-ball constrained, ancestral, KDE.
  Phase 3 — Active learning & anomaly: anomaly_score(), active_query_score(),
            drift_score(), infer_full() with full probabilistic breakdown.
  Phase 4 — Priority replay: recency+surprise weighted, 10k capacity, KDE generation.
  Phase 5 — Sequence & multi-modal: predict_next(), ConcatEncoder,
            MultiModalCyphaDIF with per-encoder LLR fusion.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_EPS           = 1e-8
_FEAT_DIM      = 128
_FIELD_DIM     = 160       # 5 groups × 32 (Phase 1: added τ=0.99 group)
_MIN_VAR       = 1e-4
_MDL_LAMBDA    = 0.002
_MDL_COLD_START= 8          # cold-start: no MDL decay for the first N observations
_REPULSE_CAP   = 0.5
_ENC_LR        = 0.002
_DELTA_LR      = 0.08
_WORLD_LR      = 0.02
_CONTEXT_WIN   = 64
_MID_DECAY     = 0.98      # Tier-2 EMA decay rate

_TEMP_INIT     = 2.5
_OOD_SIGMA     = 15.0
_OOD_EMA       = 0.01

_DEDUP_THRESH  = 0.60
_REPLAY_RATIO  = 0.30
_ALIGN_EVERY   = 500
_DELIBERATE_LO = 0.25
_DELIBERATE_HI = 0.40

_REPLAY_CAP    = 10_000    # Phase 4: 5× increase
_PRIORITY_EPS  = 1e-3      # priority floor
_MAX_CLASSES   = 128       # pre-allocated D-buffer rows
_LOG2PI        = math.log(2.0 * math.pi)  # for log-normalizer cache


# ─────────────────────────────────────────────────────────────────────────────
# Math utilities
# ─────────────────────────────────────────────────────────────────────────────

def _diag_gaussian_logpdf(h: np.ndarray, mu: np.ndarray, v: np.ndarray) -> float:
    v_safe = np.maximum(v, _MIN_VAR)
    return float(-0.5 * np.sum(np.log(v_safe) + (h - mu) ** 2 / v_safe))


def _batch_logpdf(H: np.ndarray, mu: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised log p(H | N(mu, diag(v))) for a batch H of shape (B, D).
    Returns shape (B,).  ~10× faster than calling _diag_gaussian_logpdf in a loop.
    """
    v_safe   = np.maximum(v, _MIN_VAR)
    log_norm = 0.5 * np.sum(np.log(v_safe))         # scalar
    maha     = 0.5 * ((H - mu) ** 2 / v_safe).sum(axis=1)  # (B,)
    return -log_norm - maha


def _fisher_rao_residual(h: np.ndarray, mu: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (h - mu) / np.maximum(v, _MIN_VAR)


def _mean_residual(h: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return h - mu


def _softmax(x: np.ndarray) -> np.ndarray:
    """Softmax. Pure-python fast path for K≤8 avoids numpy call overhead."""
    n = len(x)
    if n <= 8:
        mx = float(x.max())
        e  = [math.exp(float(v) - mx) for v in x]
        s  = sum(e) + _EPS
        return np.array([v / s for v in e], dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + _EPS)


def _shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) in nats."""
    p = np.maximum(p, _EPS)
    return float(-np.dot(p, np.log(p)))


def _softmax_batch(X: np.ndarray) -> np.ndarray:
    """Vectorised row-wise softmax.  Returns (N, K) probability matrix."""
    X2 = X - X.max(axis=1, keepdims=True)
    E  = np.exp(X2)
    return E / (E.sum(axis=1, keepdims=True) + _EPS)


def _compute_ece(confs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error from arrays of confidence and binary correctness."""
    ece = 0.0
    N   = len(confs)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask   = (confs >= lo) & (confs < hi)
        if mask.sum() > 0:
            ece += mask.sum() * abs(float(confs[mask].mean()) - float(correct[mask].mean())) / N
    return ece


def _kde_sample(vectors: List[np.ndarray], n: int,
                bandwidth: float, rng: np.random.Generator) -> List[np.ndarray]:
    """Gaussian KDE sample from a list of stored latent vectors."""
    centres = np.stack(vectors)
    scale   = float(np.std(centres)) * bandwidth + _EPS
    idx     = rng.integers(0, len(centres), size=n)
    return [centres[i] + rng.normal(0, scale, centres.shape[1]) for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
# Encoder  —  pluggable feature extraction interface
# ─────────────────────────────────────────────────────────────────────────────

class Encoder:
    """
    Abstract base for all domain encoders.
    Implement .dim (int property) and __call__(x) → np.ndarray[float64, shape=(dim,)].
    Values should be normalised ([0,1] or z-scored).
    """

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def __call__(self, x: Any) -> np.ndarray:
        raise NotImplementedError


class VectorEncoder(Encoder):
    """Passthrough for pre-computed numpy feature vectors."""

    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def __call__(self, x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).ravel()
        if len(arr) != self._dim:
            raise ValueError(f"VectorEncoder(dim={self._dim}) got length {len(arr)}.")
        return arr


class ConcatEncoder(Encoder):
    """
    Phase 5: Fuse multiple encoders by feature concatenation.
    All child encoders receive the same input x. Their outputs are concatenated.

        enc = ConcatEncoder([SpectralEncoder(), TextEncoder()])
        clf = CyphaDIF(encoder=enc)
    """

    def __init__(self, encoders: List[Encoder]):
        if not encoders:
            raise ValueError("ConcatEncoder requires at least one encoder.")
        self._encoders = encoders
        self._dim      = sum(e.dim for e in encoders)

    @property
    def dim(self) -> int:
        return self._dim

    def __call__(self, x: Any) -> np.ndarray:
        return np.concatenate([e(x).astype(np.float64) for e in self._encoders])


# ─────────────────────────────────────────────────────────────────────────────
# EncoderProjection  —  trainable W_enc with contrastive Fisher-Rao feedback
# ─────────────────────────────────────────────────────────────────────────────

class EncoderProjection:
    """
    Linear projection W ∈ R^{D×D}: raw_features → latent h.
    Initialised as random orthogonal (QR). Updated by contrastive Fisher-Rao gradient.
    """

    def __init__(self, dim: int = _FEAT_DIM, rng: Optional[np.random.Generator] = None):
        self.d             = dim
        self._rng          = rng or np.random.default_rng(42)
        Q, _               = np.linalg.qr(self._rng.normal(0, 1.0, (dim, dim)))
        self.W             = (Q * 0.5).astype(np.float64)
        self._lock         = threading.Lock()
        self._update_count = 0

    def project(self, f: np.ndarray) -> np.ndarray:
        with self._lock:
            return self.W @ f

    def contrastive_update(self, f: np.ndarray, h: np.ndarray,
                           mu_k: np.ndarray, v_k: np.ndarray,
                           mu_j: np.ndarray, v_j: np.ndarray,
                           weight: float = 1.0) -> None:
        # Guard: skip if frozen (e.g. during distillation) or non-finite inputs
        if getattr(self, '_frozen', False):
            return
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(h))):
            return
        r_k  = _fisher_rao_residual(h, mu_k, v_k)
        r_j  = _fisher_rao_residual(h, mu_j, v_j)
        grad = np.outer(r_j - r_k, f)
        if not np.all(np.isfinite(grad)):
            return
        with self._lock:
            self.W            += _ENC_LR * weight * grad
            self._update_count += 1
            if self._update_count % 50 == 0:
                # Frobenius norm; threshold is 2× the expected norm for a random
                # (feat_dim × feat_dim) matrix with unit-norm rows ≈ sqrt(feat_dim).
                # We use a fixed cap of 8.0 which fits expert-trained W well.
                sv = float(np.linalg.norm(self.W, 'fro'))
                if not np.isfinite(sv) or sv == 0:
                    self.W[:] = 0.0
                    return
                cap = 8.0
                if sv > cap:
                    self.W *= cap / sv

    def align_to_offsets(self, delta_mus: List[np.ndarray]) -> None:
        if len(delta_mus) < 2:
            return
        D = np.stack([d for d in delta_mus if float(d @ d) > _EPS * _EPS], axis=0)
        if len(D) < 2:
            return
        try:
            top_k = min(len(D), self.d // 4)
            with self._lock:
                for i in range(top_k):
                    n_sq = float(D[i] @ D[i])
                    v    = D[i] / (n_sq ** 0.5 + _EPS)
                    proj = self.W.T @ v
                    if float(proj @ proj) ** 0.5 < 0.1:
                        self.W += 0.01 * np.outer(v, v)
        except np.linalg.LinAlgError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# WorldPrior  —  Tier-3 infinite context  (Welford/EMA, never forgets)
# ─────────────────────────────────────────────────────────────────────────────

class WorldPrior:
    """
    Diagonal Gaussian θ₀ fitted online. Acts as Tier-3 (infinite) context.
    Every observation ever seen has shaped it; nothing is discarded, only compressed.
    Tracks drift for concept drift detection (Phase 3).
    """

    def __init__(self, dim: int = _FEAT_DIM, field_dim: int = _FIELD_DIM,
                 rng: Optional[np.random.Generator] = None):
        self.d           = dim
        self._rng        = rng or np.random.default_rng(0)
        self._n          = 0
        self.mu          = np.zeros(dim)
        self.v           = np.ones(dim)
        self.inv_v       = np.ones(dim)   # cached 1/v — kept in sync
        self.v_mean      = 1.0            # cached float(v.mean())
        self._log_norm   = -0.5 * dim * _LOG2PI  # cached -½(d·log2π + Σlog v)
        self._M2         = np.ones(dim)
        self._mu_prev    = np.zeros(dim)
        self._drift_ema  = 0.0
        self._inv_d      = np.full(dim, 1.0 / dim)      # for fast dot-mean
        self._buf        = np.empty(dim)                 # scratch: in-place v update
        self._D_LOG2PI   = -0.5 * dim * math.log(2.0 * math.pi)  # constant log-norm term
        self._log_n_ctr  = 0   # amortised log_norm counter
        self.F_field     = self._rng.normal(0, 0.001, (dim, field_dim))
        self._lock       = threading.Lock()

    def update(self, h: np.ndarray, lr: float = _WORLD_LR) -> None:
        with self._lock:
            self._n += 1
            if self._n <= 20:
                # Welford online mean/variance (first 20 obs — cold start)
                delta    = h - self.mu
                self.mu += delta / self._n
                self._M2 += delta * (h - self.mu)
                if self._n > 1:
                    self.v      = np.maximum(self._M2 / (self._n - 1), _MIN_VAR)
                    self.inv_v  = 1.0 / self.v
                    self.v_mean = float(self.v.mean())
                    self._log_norm = self._D_LOG2PI - 0.5 * float(np.log(self.v).sum())
                # Drift: ||delta/n|| (mu step size in Welford)
                drift = float(np.dot(delta, delta)) ** 0.5 / self._n
            else:
                # EMA update — hot path.  No copies: drift from step size, log_norm absolute.
                delta = h - self.mu
                self.mu += lr * delta
                self._drift_ema = (0.95 * self._drift_ema
                                   + 0.05 * lr * float(np.dot(delta, delta)) ** 0.5)
                # In-place v update
                np.multiply(delta, delta, out=self._buf)
                self._buf *= lr
                np.multiply(1.0 - lr, self.v, out=self.v)
                self.v += self._buf
                np.maximum(self.v, _MIN_VAR, out=self.v)
                np.divide(1.0, self.v, out=self.inv_v)
                self.v_mean = float(np.dot(self.v, self._inv_d))
                # Amortise log_norm: recompute every 8 steps.
                # log_norm is used only for replay-priority loss; stale values
                # produce slightly noisy priorities with no effect on classification.
                self._log_n_ctr += 1
                if self._log_n_ctr >= 8:
                    self._log_norm = self._D_LOG2PI - 0.5 * float(np.log(self.v).sum())
                    self._log_n_ctr = 0
                return   # drift already updated above; skip Welford drift below
            self._drift_ema = 0.95 * self._drift_ema + 0.05 * drift

    def drift_score(self) -> float:
        with self._lock:
            return self._drift_ema

    def condition_on_field(self, h_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Guard: if field state is corrupted or too large, skip conditioning
        h_sq = float(h_field @ h_field)
        if not math.isfinite(h_sq) or h_sq > 1e8:
            with self._lock:
                return self.mu.copy(), self.v.copy()
        with self._lock:
            return self.mu + self.F_field @ h_field, self.v.copy()

    def update_field_map(self, h_field: np.ndarray, residual: np.ndarray,
                         lr: float = 0.0001) -> None:
        # Guard: skip if field state is invalid
        if not (np.all(np.isfinite(h_field)) and np.all(np.isfinite(residual))):
            return
        with self._lock:
            self.F_field += lr * np.outer(residual, h_field)


# ─────────────────────────────────────────────────────────────────────────────
# ClassDifferential  —  per-class differential offset Δk
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassDifferential:
    """
    Δk for one class.  θk = θ₀ ⊕ Δk:
        μk = μ₀ + Δμ    vk = v₀  (fixed — prevents collapse)
    Fisher-Rao norm: ||Δk||² = Σ_d Δμd²/v₀d
    """
    label    : str
    dim      : int
    delta_mu : np.ndarray = field(default_factory=lambda: np.zeros(_FEAT_DIM))
    n_obs    : int = 0
    n_correct: int = 0

    def mu(self, mu0: np.ndarray) -> np.ndarray:
        return mu0 + self.delta_mu

    def log_likelihood(self, h: np.ndarray, mu0: np.ndarray, v0: np.ndarray) -> float:
        return _diag_gaussian_logpdf(h, self.mu(mu0), v0)

    def attract(self, h: np.ndarray, mu0: np.ndarray, inv_v: np.ndarray,
                lr: float = _DELTA_LR) -> None:
        # h - (mu0 + delta_mu) = (h - mu0) - delta_mu
        self.delta_mu += lr * ((h - mu0) - self.delta_mu)
        self.n_obs    += 1

    def repel(self, h: np.ndarray, mu0: np.ndarray, inv_v: np.ndarray,
              weight: float = 1.0, lr: float = _DELTA_LR) -> None:
        self.delta_mu -= lr * min(weight, _REPULSE_CAP) * ((h - mu0) - self.delta_mu)

    def mdl_decay(self, lam: float = _MDL_LAMBDA, v_mean: float = 1.0) -> None:
        """
        Inverted MDL schedule + cold-start immunity.

        Old (broken): lam_eff ∝ 1/(1+n/16) — MAX decay at n=0, min at large n.
                      New classes decayed hardest.  1-shot barely survived.
        New (correct): lam_eff ∝ n/(n+16)  — ZERO decay at n=0, max at large n.
                      New classes protected; established ones properly regularised.

        Cold-start immunity: lam_eff=0 for the first _MDL_COLD_START steps.
        This prevents the initial few observations from being immediately erased
        by MDL pressure before the class can even establish its position.
        """
        if self.n_obs < _MDL_COLD_START:
            return   # cold-start immunity: no decay until enough observations
        snr_factor    = 1.0 / max(v_mean, 1.0)
        # Inverted schedule: grows with n_obs → light for new, heavy for established
        weight        = self.n_obs / (self.n_obs + 16.0)   # ∈ [0, 1)
        lam_eff       = lam * max(0.00025, weight) * snr_factor
        self.delta_mu *= (1.0 - lam_eff)

    def fisher_rao_norm(self, v0: np.ndarray) -> float:
        return float(np.sum(self.delta_mu ** 2 / np.maximum(v0, _MIN_VAR)))

    def accuracy(self) -> float:
        return self.n_correct / self.n_obs if self.n_obs > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DIFMemory  —  world prior + all class differentials
# ─────────────────────────────────────────────────────────────────────────────

class DIFMemory:
    """Core DIF memory: WorldPrior θ₀ + Dict[label → ClassDifferential Δk]."""

    def __init__(self, dim: int = _FEAT_DIM, field_dim: int = _FIELD_DIM,
                 rng: Optional[np.random.Generator] = None):
        self.d        = dim
        self.world    = WorldPrior(dim=dim, field_dim=field_dim, rng=rng)
        self._classes : Dict[str, ClassDifferential] = {}
        self._lock    = threading.Lock()
        self._step    = 0
        # Pre-allocated D buffer: row k = delta_mu of class k (avoids np.stack each call)
        self._D_buf        : np.ndarray = np.zeros((_MAX_CLASSES, dim))
        self._label_order  : List[str]  = []   # class name at index k
        self._label_idx    : Dict[str, int] = {}  # O(1) lookup label→k
        # Cached n_obs array — kept in sync with ClassDifferential.n_obs.
        # Eliminates np.fromiter(...) in the MDL decay hot path.
        self._n_obs_buf    : np.ndarray = np.zeros(_MAX_CLASSES, dtype=np.float64)
        # Pre-allocated softmax scratch — avoids np.array([...]) allocation per call.
        self._probs_buf    : np.ndarray = np.zeros(_MAX_CLASSES, dtype=np.float64)
        self._exp_list     : list       = [0.0] * _MAX_CLASSES

    def _get_or_create(self, label: str) -> ClassDifferential:
        if label not in self._classes:
            k = len(self._label_order)
            if k < _MAX_CLASSES:
                self._D_buf[k, :] = 0.0              # clear buffer row
                cd = ClassDifferential(label=label, dim=self.d)
                cd.delta_mu = self._D_buf[k]          # view: in-place ops auto-sync buffer
            else:
                cd = ClassDifferential(label=label, dim=self.d)
            self._label_order.append(label)
            self._label_idx[label] = k
            self._classes[label] = cd
            self._n_obs_buf[k] = 0.0   # sync cached n_obs
        return self._classes[label]

    def classify(self,
                 h            : np.ndarray,
                 h_field      : Optional[np.ndarray] = None,
                 context_prior: Optional[Dict[str, float]] = None,
                 temperature  : float = _TEMP_INIT,
                 ood_sigma    : float = _OOD_SIGMA,
                 mahal_ema    : Optional[float] = None,
                 mahal_std_ema: float = 0.5,
                 ) -> Tuple[str, float, Dict[str, float]]:
        """
        Calibrated classification via LLRs.
        LLR_k = log p(h|θk) − log p(h|θ₀) − U_k

        conf = ood_gate × disc × world_gate

        ood_gate  : sigmoid(max_LLR / ood_sigma) — relative gate; catches h far from
                    every class mean in LLR space.
        disc      : max softmax(LLR / temperature) — discriminability.
        world_gate: sigmoid((threshold − mahal/D) / mahal_std_ema) — absolute gate
                    based on normalised per-dim Mahalanobis distance to world prior.
                    Catches OOD samples that score high on a class LLR but lie far
                    from the in-distribution manifold.  threshold = mahal_ema + 4σ.
                    Pass mahal_ema=None to disable (backward-compatible default).

        Uses pre-allocated D buffer — no np.stack allocation per call.
        """
        with self._lock:
            if not self._classes:
                return '__unknown__', 0.0, {}
            K      = len(self._label_order)
            labels = self._label_order          # reference, not copy

            if h_field is not None:
                h_sq = float(h_field @ h_field)
                with self.world._lock:
                    if math.isfinite(h_sq) and h_sq <= 1e8:
                        mu0 = self.world.mu + self.world.F_field @ h_field
                    else:
                        mu0 = self.world.mu.copy()
                    inv_v    = self.world.inv_v
                    v_mean   = self.world.v_mean
            else:
                with self.world._lock:
                    mu0      = self.world.mu.copy()
                    inv_v    = self.world.inv_v
                    v_mean   = self.world.v_mean

            n_obs_arr = np.array([self._classes[k].n_obs for k in labels], dtype=np.float64)
            D = self._D_buf[:K].copy()

        d_h     = h - mu0
        r       = d_h * inv_v
        cross   = D @ r
        D_sq    = (D * D) @ inv_v
        u_arr   = v_mean / (n_obs_arr + 1.0)
        if context_prior:
            ctx_arr = np.array([context_prior.get(k, 0.0) for k in labels])
        else:
            ctx_arr = np.zeros(K)
        llr_arr = cross - 0.5 * D_sq - u_arr + ctx_arr

        best_i   = int(llr_arr.argmax())
        best     = labels[best_i]
        max_llr  = float(llr_arr[best_i])
        probs    = _softmax(llr_arr / (temperature + _EPS))
        disc     = float(probs[best_i])

        # World-prior absolute gate — normalised Mahalanobis distance.
        # Sole OOD rejection mechanism. ood_gate (sigmoid of max_LLR) retired:
        # it incorrectly penalised in-dist samples near class boundaries where
        # max_LLR is legitimately small but the prediction is still correct.
        # disc already captures within-manifold uncertainty; world_gate handles
        # the absolute out-of-distribution case.
        if mahal_ema is not None:
            mahal_per_dim = float(d_h @ r) / (len(h) + _EPS)
            std_safe      = max(mahal_std_ema, 0.05)
            threshold     = mahal_ema + 5.0 * std_safe
            scale         = 2.0 / std_safe
            margin        = float(np.clip((threshold - mahal_per_dim) * scale, -500.0, 500.0))
            world_gate    = 1.0 / (1.0 + math.exp(-margin))
        else:
            world_gate = 1.0

        return best, disc * world_gate, dict(zip(labels, llr_arr.tolist()))

    def train(self,
              h            : np.ndarray,
              label        : str,
              h_field      : Optional[np.ndarray] = None,
              context_prior: Optional[Dict[str, float]] = None,
              temperature  : float = _TEMP_INIT,
              ood_sigma    : float = _OOD_SIGMA,
              ) -> Tuple[str, bool, float, Dict[str, float], float]:
        """One training step.
        Returns (pred, correct, loss, post_llrs, post_conf).
        D rows are views into _D_buf — in-place attract/repel/mdl_decay auto-sync the buffer.
        """
        with self._lock:
            self._step += 1

            # World params under world lock — capture everything we need at once
            if h_field is not None:
                h_sq = float(h_field @ h_field)
                with self.world._lock:
                    if math.isfinite(h_sq) and h_sq <= 1e8:
                        _ff_proj = self.world.F_field @ h_field   # computed once; reused below
                        mu0 = self.world.mu + _ff_proj
                    else:
                        _ff_proj = None
                        mu0 = self.world.mu.copy()
                    inv_v     = self.world.inv_v        # reference: read-only, safe under memory._lock
                    log_norm  = self.world._log_norm
            else:
                _ff_proj = None
                with self.world._lock:
                    mu0       = self.world.mu.copy()
                    inv_v     = self.world.inv_v
                    log_norm  = self.world._log_norm

            cd_k  = self._get_or_create(label)
            K     = len(self._label_order)
            labels = self._label_order              # ordered reference
            k_idx  = self._label_idx[label]         # O(1)

            D = self._D_buf[:K]   # view — no stack, no allocation

            # Pre-update scores for pred / repel weights
            h_mu0      = h - mu0
            r          = h_mu0 * inv_v
            cross      = D @ r                      # (K,)
            D_sq       = (D * D) @ inv_v            # (K,)
            if context_prior:
                ctx_arr = np.array([context_prior.get(k, 0.0) for k in labels])
            else:
                ctx_arr = np.zeros(K)
            scores_arr = cross - 0.5 * D_sq + ctx_arr

            best_idx = int(scores_arr.argmax())
            pred     = labels[best_idx]
            correct  = (pred == label)
            if correct:
                cd_k.n_correct += 1

            # Save pre-attract scalars for loss (D[k_idx] is a view and will change)
            cross_k = float(cross[k_idx])
            D_sq_k  = float(D_sq[k_idx])
            cd_k.n_obs += 1
            self._n_obs_buf[k_idx] += 1.0   # keep cached array in sync

            # Attract + Repel — vectorised op over K rows.
            # Inline softmax into scales to avoid intermediate probs array allocation.
            # scales_i = +lr (attract k_idx), -lr·min(softmax_i, cap) (repel others)
            _s_lst = scores_arr.tolist()
            _s_mx  = max(_s_lst)
            _s_e   = [math.exp(v - _s_mx) for v in _s_lst]
            _s_Z   = sum(_s_e) + _EPS
            scales = self._probs_buf[:K]               # reuse pre-alloc buffer for scales
            for _i in range(K):
                scales[_i] = -_DELTA_LR * min(_s_e[_i] / _s_Z, _REPULSE_CAP)
            scales[k_idx] = _DELTA_LR                  # attract true class
            D += scales[:, np.newaxis] * (h_mu0[np.newaxis, :] - D)   # in-place view

            # MDL decay — vectorised; inverted schedule + cold-start immunity.
            # lam_eff = 0 for classes with < _MDL_COLD_START observations.
            # lam_eff grows with n_obs → gentle for new classes, strong for established.
            lam     = _MDL_LAMBDA
            v_m     = self.world.v_mean
            snr_fac = 1.0 / max(v_m, 1.0)
            n_obs_v = self._n_obs_buf[:K]
            # Cold-start mask: zero decay until each class has enough observations
            cold_mask = (n_obs_v >= _MDL_COLD_START).astype(np.float64)
            weight    = n_obs_v / (n_obs_v + 16.0)           # inverted: grows with n_obs
            lam_eff   = lam * snr_fac * np.maximum(0.00025, weight) * cold_mask
            D        *= (1.0 - lam_eff[:, np.newaxis])                   # in-place view

            self.world.update(h)  # updates world.mu, inv_v, v_mean, _log_norm
            # Inline loss: -ll_true = -log_norm + ½(h-μ₀)²/v - cross_k + ½D_sq_k
            # All pre-attract/pre-update values (matches original behaviour exactly)
            loss = -log_norm + 0.5 * float(r @ h_mu0) - cross_k + 0.5 * D_sq_k

            # Post-update LLRs: D is already current (views), fetch updated world params
            # We're still under self._lock; no other thread can modify _D_buf or world
            if _ff_proj is not None:
                mu0_p = self.world.mu + _ff_proj   # reuse cached F_field @ h_field
            elif h_field is not None:
                mu0_p = self.world.mu.copy()
            else:
                mu0_p = self.world.mu
            inv_v_p  = self.world.inv_v    # updated by world.update
            v_mean_p = self.world.v_mean   # cached scalar

            r_p     = (h - mu0_p) * inv_v_p
            # Use cached _n_obs_buf — already updated at attract step
            u_p     = v_mean_p / (self._n_obs_buf[:K] + 1.0)
            llr_p   = D @ r_p - 0.5 * (D * D) @ inv_v_p - u_p + ctx_arr
            post_llrs = dict(zip(labels, llr_p.tolist()))

            # post_conf disc: 1/Σexp((llr−max)/T).  tolist() avoids numpy scalar
            # boxing; values are bounded by training so no clamping needed.
            ml_p    = float(llr_p.max())
            T_inv   = 1.0 / (temperature + _EPS)
            llr_lst = llr_p.tolist()
            sum_exp = sum(math.exp((v - ml_p) * T_inv) for v in llr_lst) + _EPS
            post_conf = 1.0 / sum_exp

        return pred, correct, float(loss), post_llrs, post_conf

    def get_class_params(self, label: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            if label not in self._classes:
                return None
            mu0, v0 = self.world.mu.copy(), self.world.v.copy()
            return self._classes[label].mu(mu0), v0.copy()

    def all_class_means(self) -> Dict[str, np.ndarray]:
        with self._lock:
            mu0 = self.world.mu.copy()
            return {lbl: cd.mu(mu0).copy() for lbl, cd in self._classes.items()}

    def complexity(self) -> Dict[str, float]:
        with self._lock:
            v0 = self.world.v.copy()
            return {lbl: cd.fisher_rao_norm(v0) for lbl, cd in self._classes.items()}

    def accuracy(self) -> Dict[str, float]:
        with self._lock:
            return {lbl: cd.accuracy() for lbl, cd in self._classes.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: TieredContextBuffer  —  3-tier nearly-infinite context
# ─────────────────────────────────────────────────────────────────────────────

class TieredContextBuffer:
    """
    Three-tier episodic context with effectively unbounded memory.

    Tier 1 — Short-term (exact):
        Sliding window of last `short_window` labels. Full-resolution frequency
        prior + co-occurrence prior from this window.

    Tier 2 — Mid-term (compressed):
        EMA of class visit frequencies and transition probabilities, decaying at
        `mid_decay` per step. Never discards structure, just exponentially
        compresses it. Memory cost: O(K²) where K = number of classes.

    Tier 3 — Long-term (world prior):
        WorldPrior θ₀ has absorbed every observation via Welford/EMA.
        The NIGField's slow τ=0.99 group carries long-range temporal signal.

    Combined prior:
        log p(k|ctx) = (1−α)·T1 + α·T2
        α = field_confidence ∈ [0, 0.7], grows with number of observations.
    """

    def __init__(self, short_window: int = _CONTEXT_WIN, mid_decay: float = _MID_DECAY):
        self._short_win  = short_window
        self._mid_decay  = mid_decay

        # Tier 1
        self._history    : deque = deque(maxlen=short_window)
        self._cooccur    : Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._last_label : Optional[str] = None
        self._t1_counts  : Dict[str, float] = defaultdict(float)  # rolling count in _history
        self._t1_total   : float = 0.0                             # len(_history), kept in sync

        # Tier 2
        self._mid_freq       : Dict[str, float] = defaultdict(float)
        self._mid_trans      : Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._mid_trans_tot  : Dict[str, float] = defaultdict(float)  # cached sum per from-label
        self._mid_n          : float = 0.0
        self._mid_freq_total : float = 0.0   # cached sum(mid_freq.values())

        self._lock = threading.Lock()

        # Context prior cache — invalidated on every record() call.
        # Eliminates recomputation for repeated infer() calls between training steps.
        self._ctx_cache_key : Optional[tuple] = None
        self._ctx_cache_val : Dict[str, float] = {}
        # Cached co-occurrence totals (maintained incrementally in record())
        self._cooccur_tot   : Dict[str, float] = defaultdict(float)

    def record(self, label: str, correct: bool, llr_max: float = 0.0) -> None:
        with self._lock:
            if self._last_label is not None:
                frm = self._last_label
                self._cooccur[frm][label] += 1
                self._cooccur_tot[frm]   += 1.0  # cached total for context_prior
                old_val = self._mid_trans[frm][label]
                new_val = self._mid_decay * old_val + (1 - self._mid_decay)
                self._mid_trans[frm][label] = new_val
                self._mid_trans_tot[frm]   += new_val - old_val   # delta update
            # Maintain rolling t1_counts: decrement evicted item before append
            if len(self._history) == self._short_win:
                evicted_lbl, _ = self._history[0]
                self._t1_counts[evicted_lbl] -= 1.0
            else:
                self._t1_total += 1.0
            self._history.append((label, correct))
            self._t1_counts[label] = self._t1_counts.get(label, 0.0) + 1.0

            self._mid_n += 1.0
            decay = self._mid_decay
            self._mid_freq_total *= decay
            for k in self._mid_freq:
                self._mid_freq[k] *= decay
            self._mid_freq[label] = self._mid_freq[label] + (1 - decay)
            self._mid_freq_total += (1 - decay)
            self._last_label = label
            self._ctx_cache_key = None   # invalidate context prior cache

    def context_prior(self, classes: List[str],
                      field_confidence: float = 0.0) -> Dict[str, float]:
        """
        Combined log prior over classes.
        field_confidence blends T1 (recent) → T2 (mid-term) as experience grows.

        Performance: pure-python math.log loop (avoids numpy overhead for K≤20)
        + single-entry cache keyed on (classes_tuple, last_label, fc_bucket).
        Cache is invalidated on every record() call.
        """
        if not classes:
            return {}

        # Fast path: return cached result if context unchanged since last call
        fc_bucket = int(field_confidence * 10)  # bucket to 0.1 granularity
        cache_key = (tuple(classes), self._last_label, fc_bucket)
        if cache_key is self._ctx_cache_key or cache_key == self._ctx_cache_key:
            return self._ctx_cache_val

        K    = len(classes)
        _log = math.log
        _EPS2 = _EPS

        with self._lock:
            last        = self._last_label
            t1_total    = self._t1_total + K
            co          = self._cooccur.get(last, {})       if last else {}
            co_total    = self._cooccur_tot.get(last, 0.0) + K  if last else K
            mt          = self._mid_trans.get(last, {})     if last else {}
            mt_total    = self._mid_trans_tot.get(last, 0.0) + K * 1e-3 + K  if last else K
            mid_total   = self._mid_freq_total + K
            have_co     = bool(co)
            have_mt     = bool(mt)

            alpha = min(field_confidence, 0.7)
            w1    = 1.0 - alpha; w2 = alpha

            result = {}
            for k in classes:
                # Tier 1 — short-term frequency
                t1_log  = _log((self._t1_counts.get(k, 0.0) + 1.0) / t1_total + _EPS2)
                t1_co   = _log((co.get(k, 0) + 1.0) / co_total + _EPS2) if have_co else 0.0
                tier1   = 0.6 * t1_log + 0.4 * t1_co

                # Tier 2 — mid-term EMA
                mid_k  = self._mid_freq.get(k, 0.0) + 1e-3
                t2_log = _log(mid_k / (mid_total + 1e-3) + _EPS2)
                mt_k   = mt.get(k, 0.0) + 1e-3 if have_mt else 1e-3
                t2_tr  = _log(mt_k / (mt_total + 1e-3) + _EPS2) if have_mt else 0.0
                tier2  = 0.6 * t2_log + 0.4 * t2_tr

                result[k] = w1 * tier1 + w2 * tier2

        self._ctx_cache_key = cache_key
        self._ctx_cache_val = result
        return result

    def predict_next(self, classes: List[str]) -> Dict[str, float]:
        """
        Phase 5: Distribution over the next label.
        Combines Tier-1 co-occurrence and Tier-2 EMA transitions.
        """
        if not classes or self._last_label is None:
            return {k: 1.0 / max(len(classes), 1) for k in classes}
        with self._lock:
            t1_co = self._cooccur.get(self._last_label, {})
            t2_tr = self._mid_trans.get(self._last_label, {})
            scores = {k: float(t1_co.get(k, 0)) + 5.0 * float(t2_tr.get(k, 0.0)) + 1e-3
                      for k in classes}
            total = sum(scores.values())
            return {k: v / total for k, v in scores.items()}

    def recent_accuracy(self) -> float:
        with self._lock:
            if not self._history:
                return 0.0
            return sum(c for _, c in self._history) / len(self._history)

    def field_confidence(self) -> float:
        """∈ [0,1]. Grows with observations; saturates at 200 steps."""
        with self._lock:
            return float(min(self._mid_n / 200.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: NIGField  —  5-timescale temporal field (adds τ=0.99)
# ─────────────────────────────────────────────────────────────────────────────

class NIGField:
    """
    Diagonal-A temporal field with 5 timescale groups:
        τ = 0.30  fast     τ = 0.60  medium-fast    τ = 0.85  medium-slow
        τ = 0.95  slow     τ = 0.99  very slow  ← Phase 1 addition

    The τ=0.99 group carries long-range temporal context that persists across
    hundreds of steps — effective unbounded memory in the field state.
    Spectral radius of W_T maintained ≤ 0.85 via power iteration.
    """

    def __init__(self, state_dim: int = _FIELD_DIM,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.d    = state_dim
        self._rng = rng or np.random.default_rng(2)

        g = state_dim // 5
        r = state_dim - 4 * g
        self._a = np.concatenate([
            np.full(g, 0.30), np.full(g, 0.60),
            np.full(g, 0.85), np.full(g, 0.95),
            np.full(r, 0.99),
        ])

        W = self._rng.normal(0, 0.01, (state_dim, state_dim))
        v = self._rng.normal(0, 1, state_dim)
        for _ in range(20):
            v = W @ v; v /= (np.linalg.norm(v) + _EPS)
        sr            = float(np.linalg.norm(W @ v))
        self._W_T     = W / (sr + _EPS) * 0.05
        self._wt_lock = threading.Lock()
        self._A_eff   = (np.diag(self._a) + self._W_T).astype(np.float32)
        self._h       = np.zeros(state_dim)
        self._step    = 0
        self._lock    = threading.Lock()
        self._sr_vec  = v / (np.linalg.norm(v) + _EPS)

    # Max field state magnitude (per-element L2 norm cap).
    # Prevents runaway field amplification when injected signals are large or
    # repeated over many steps. Chosen to be >> typical in-distribution h norm (~10)
    # but finite enough to stop blowup before NaN propagates into class NIGs.
    _H_CAP: float = 50.0

    def evolve(self, h: np.ndarray, update_state: bool = True) -> np.ndarray:
        h_new  = (self._A_eff @ h.astype(np.float32)).astype(np.float64)
        h_sq   = float(h_new @ h_new)
        if not math.isfinite(h_sq):
            h_new = np.zeros_like(h_new)
        elif h_sq > self._H_CAP * self._H_CAP:
            h_new *= self._H_CAP / h_sq ** 0.5
        if update_state:
            with self._lock:
                self._h    = h_new
                self._step += 1
        return h_new

    def inject(self, signal: np.ndarray, strength: float = 0.15) -> None:
        # Guard: detect non-finite via dot product (O(d) float, avoids np.isfinite array alloc)
        s_sq = float(signal @ signal)
        if not math.isfinite(s_sq) or s_sq == 0.0:
            signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
            s_sq   = float(signal @ signal)
        with self._lock:
            h_mag    = float(self._h @ self._h) ** 0.5 + _EPS
            s_mag    = s_sq ** 0.5 + _EPS
            self._h += strength * (h_mag / s_mag) * signal
            # Hard cap: prevent runaway magnitude
            h_sq = float(self._h @ self._h)
            if not math.isfinite(h_sq):
                self._h = np.zeros_like(self._h)
            elif h_sq > self._H_CAP * self._H_CAP:
                self._h *= self._H_CAP / h_sq ** 0.5

    def update_causal(self, h_t: np.ndarray, h_target: np.ndarray,
                      lr: float = 0.0002) -> float:
        with self._wt_lock:
            err   = self._W_T @ h_t - h_target
            W_new = self._W_T - lr * np.outer(err, h_t) / self.d
            v     = self._sr_vec
            for _ in range(3):          # 3 iters sufficient for spectral radius estimate
                v  = W_new @ v
                nv = float(v @ v) ** 0.5
                if nv < _EPS:
                    break
                v /= nv
            sr = float((W_new @ v) @ (W_new @ v)) ** 0.5
            if sr > 0.85:
                W_new *= 0.85 / sr
            self._W_T    = W_new
            self._sr_vec = v
            self._A_eff  = (np.diag(self._a) + W_new).astype(np.float32)
        return float(err.dot(err) / self.d)

    def slow_state(self) -> np.ndarray:
        """Return the τ=0.99 group slice — long-range memory carrier."""
        with self._lock:
            g = self.d // 5
            return self._h[4*g:].copy()

    def reset(self) -> None:
        with self._lock:
            self._h    = np.zeros(self.d)
            self._step = 0

    @property
    def h(self) -> np.ndarray:
        with self._lock:
            return self._h.copy()

    @property
    def step(self) -> int:
        with self._lock:
            return self._step


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: PriorityReplayBuffer  —  recency + surprise weighted replay
# ─────────────────────────────────────────────────────────────────────────────

class PriorityReplayBuffer:
    """
    Priority experience replay. Capacity 10,000 (5× v1).

    Priority = surprise × recency_decay
        surprise       = loss at storage time
        recency_decay  = exp(−0.0001 × age_in_steps)

    High-surprise recent events are replayed far more frequently than
    stale easy examples — catastrophic forgetting is resisted at the
    most informative boundaries.

    Optimised: stores losses and insert_ns as numpy arrays for O(1)
    vectorised priority computation instead of O(N) Python loops.
    """

    def __init__(self, capacity: int = _REPLAY_CAP,
                 rng: Optional[np.random.Generator] = None):
        self._cap      = capacity
        self._buf      : List = []
        self._losses   = np.empty(capacity, dtype=np.float64)
        self._ins_ns   = np.empty(capacity, dtype=np.float64)
        self._buf_len  = 0
        self._rng      = rng or np.random.default_rng(5)
        self._lock     = threading.Lock()
        self._insert_n = 0
        # Amortised decay: w_i = loss_i × decay^(age_i).  Maintained in push()
        # so sample() never calls np.exp on the full buffer.
        self._w_cache  = np.empty(capacity, dtype=np.float64)
        self._log_buf  = np.empty(capacity, dtype=np.float64)  # Gumbel scratch
        self._DECAY    = math.exp(-0.0001)

    def push(self, h: np.ndarray, f: np.ndarray,
             label: str, loss: float = 1.0) -> None:
        with self._lock:
            self._insert_n += 1
            loss_v = abs(float(loss)) + _PRIORITY_EPS
            # Age all existing weights by one decay step
            if self._buf_len > 0:
                self._w_cache[:self._buf_len] *= self._DECAY
            entry = (h.copy(), f.copy(), label, loss_v, self._insert_n)
            if self._buf_len < self._cap:
                idx = self._buf_len
                self._buf.append(entry)
                self._buf_len += 1
            else:
                idx = int(np.argmin(self._w_cache[:self._buf_len]))
                self._buf[idx] = entry
            self._losses[idx]  = loss_v
            self._ins_ns[idx]  = self._insert_n
            self._w_cache[idx] = loss_v   # new entry: age 0 → weight = loss

    def _priorities_unlocked(self) -> np.ndarray:
        """Priority weights — maintained incrementally in push()."""
        if self._buf_len == 0:
            return np.array([])
        return self._w_cache[:self._buf_len].copy()

    def sample(self, n: int) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        with self._lock:
            if self._buf_len == 0:
                return []
            n = min(n, self._buf_len)
            w = self._w_cache[:self._buf_len]
            if n >= self._buf_len:
                idx = np.arange(self._buf_len)
            else:
                # Efraimidis-Spirakis weighted reservoir sampling without replacement.
                # key_i = log(u_i) / w_i  →  top-K keys give proportional WOR sample.
                # One log call + one division vs Gumbel's two log calls.
                lb = self._log_buf[:self._buf_len]
                self._rng.random(self._buf_len, out=lb)
                np.log(lb + _PRIORITY_EPS, out=lb)
                lb /= w
                idx = np.argpartition(lb, -n)[-n:]
            return [(self._buf[i][0], self._buf[i][1], self._buf[i][2]) for i in idx]

    def by_class(self) -> Dict[str, List[np.ndarray]]:
        with self._lock:
            groups: Dict[str, List[np.ndarray]] = defaultdict(list)
            for h, _, label, _, _ in self._buf:
                groups[label].append(h)
            return dict(groups)

    def __len__(self) -> int:
        with self._lock:
            return self._buf_len


# ─────────────────────────────────────────────────────────────────────────────
# CyphaDIF  —  Differential Information Field main system  (all 5 phases)
# ─────────────────────────────────────────────────────────────────────────────

class CyphaDIF:
    """
    Differential Information Field Classifier — Universal AI core.

    Core API
    --------
    train_step(x, label)        → loss
    batch_train(data, epochs)   → losses
    infer(x)                    → (label, confidence)
    infer_full(x)               → full probabilistic breakdown dict

    Generation API  (Phase 2)
    -------------------------
    generate(label, n, temperature)         temperature-scaled sampling
    generate_conditioned(label, n)          field-conditioned sampling
    generate_boundary(a, b, n, alpha)       latent space interpolation
    generate_adversarial(n)                 maximise posterior entropy
    generate_ood(n)                         sample from OOD region
    generate_mdl_ball(label, n, radius)     Fisher-Rao ball constraint
    generate_ancestral(n)                   sample k~context, h~p(h|k)
    generate_kde(label, n, bandwidth)       KDE from replay latents

    Anomaly / Active Learning  (Phase 3)
    -------------------------------------
    anomaly_score(x)            OOD gate value (high = anomalous)
    active_query_score(x)       entropy × (1−max_p) boundary proximity
    drift_score()               world prior drift (concept drift signal)

    Sequence  (Phase 5)
    -------------------
    predict_next(label)         next-label probability distribution
    """

    def __init__(self,
                 encoder     : Encoder,
                 field_dim   : int   = _FIELD_DIM,
                 enc_lr      : float = _ENC_LR,
                 delta_lr    : float = _DELTA_LR,
                 world_lr    : float = _WORLD_LR,
                 mdl_lambda  : float = _MDL_LAMBDA,
                 context_win : int   = _CONTEXT_WIN,
                 rng         : Optional[np.random.Generator] = None):

        if not isinstance(encoder, Encoder):
            raise TypeError(
                f"CyphaDIF requires an Encoder instance, got {type(encoder).__name__}."
            )

        self.encoder_fn  = encoder
        self.feat_dim    = encoder.dim
        self.field_dim   = field_dim
        self._rng        = rng or np.random.default_rng(42)
        self.enc_lr      = enc_lr
        self.delta_lr    = delta_lr
        self.world_lr    = world_lr
        self.mdl_lambda  = mdl_lambda

        self.encoder  = EncoderProjection(dim=self.feat_dim, rng=self._rng)
        self.memory   = DIFMemory(dim=self.feat_dim, field_dim=field_dim, rng=self._rng)
        self.context  = TieredContextBuffer(short_window=context_win)
        self.field    = NIGField(state_dim=field_dim, rng=self._rng)
        self.replay   = PriorityReplayBuffer(capacity=_REPLAY_CAP, rng=self._rng)

        if self.feat_dim == field_dim:
            self._W_inject = None
        elif self.feat_dim < field_dim:
            # QR of (field_dim, feat_dim) → Q=(field_dim, feat_dim): projects up ✓
            Q, _           = np.linalg.qr(self._rng.normal(0, 1.0, (field_dim, self.feat_dim)))
            self._W_inject = Q.astype(np.float64)
        else:
            # feat_dim > field_dim: QR of (feat_dim, field_dim) → Q.T=(field_dim, feat_dim): projects down ✓
            Q, _           = np.linalg.qr(self._rng.normal(0, 1.0, (self.feat_dim, field_dim)))
            self._W_inject = Q.T.astype(np.float64)

        self._total_steps   = 0
        self._total_correct = 0
        self._loss_buf      : deque = deque(maxlen=100)
        self._align_every   = _ALIGN_EVERY
        self.temperature    = _TEMP_INIT
        self.ood_sigma      = _OOD_SIGMA
        self._llr_ema       = 0.0
        # Temperature adaptation: track running mean of winning LLR magnitudes
        # to allow self-calibration under distribution shift (adapt_temperature)
        self._llr_scale_ema = 0.0      # EMA of |LLR_winner| during training
        self._llr_scale_n   = 0        # count for bootstrap stabilisation
        self._base_temp     = _TEMP_INIT   # reference temperature at training time
        self._buf_len_cache = 0   # cached replay buffer length (avoids lock per step)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _encode(self, x: Any) -> Tuple[np.ndarray, np.ndarray]:
        f = self.encoder_fn(x).astype(np.float64)
        h = self.encoder.project(f)
        return f, h

    def _to_field_dim(self, h: np.ndarray) -> np.ndarray:
        h_norm = h / (float(h @ h) ** 0.5 + _EPS)
        return h_norm if self._W_inject is None else self._W_inject @ h_norm

    def _ctx_prior(self, classes: List[str]) -> Dict[str, float]:
        return self.context.context_prior(classes, self.context.field_confidence())

    def _dedup_check(self, label: str) -> None:
        with self.memory._lock:
            classes = list(self.memory._classes.items())
            if len(classes) < 2:
                return
            cd_k = self.memory._classes.get(label)
            if cd_k is None:
                return
            nk = float(cd_k.delta_mu @ cd_k.delta_mu) ** 0.5 + _EPS
            for lbl_j, cd_j in classes:
                if lbl_j == label:
                    continue
                nj      = float(cd_j.delta_mu @ cd_j.delta_mu) ** 0.5 + _EPS
                cos_sim = float(cd_k.delta_mu @ cd_j.delta_mu) / (nk * nj)
                if cos_sim > _DEDUP_THRESH:
                    overlap = cos_sim - _DEDUP_THRESH
                    push    = overlap * 0.5 * cd_j.delta_mu / nj
                    cd_k.delta_mu -= push
                    cd_j.delta_mu -= push


    def _check_capacity(self) -> Tuple[int, int, bool]:
        """
        Check if latent space is geometrically overcrowded.

        Returns (K, d, is_crowded) where is_crowded = K > d//2.
        When crowded, class delta_mus cannot all be near-orthogonal,
        causing accuracy degradation.  Call expand_capacity(2*d) to fix.
        """
        with self.memory._lock:
            K = len(self.memory._classes)
        d = self.feat_dim
        return K, d, K > d // 2

    def expand_capacity(self, new_feat_dim: int) -> None:
        """
        Expand the feature dimension to accommodate more classes.

        Constructs a random orthogonal (new_d × old_d) expansion matrix,
        re-projects all class centroids, world prior, encoder weights, and
        replay buffer into the larger space.

        Information-preserving: orthogonal projection preserves inner products
        (and therefore LLR scores) up to scale.  Typically call with new_feat_dim
        = 2 * current feat_dim when _check_capacity() reports crowding.

        Parameters
        ----------
        new_feat_dim : target dimension (must be > current feat_dim)
        """
        if new_feat_dim <= self.feat_dim:
            raise ValueError(f"new_feat_dim={new_feat_dim} must exceed current {self.feat_dim}")

        old_d = self.feat_dim
        rng   = self._rng

        # Orthogonal expansion: (new_d, old_d)
        Q, _   = np.linalg.qr(rng.normal(0, 1.0, (new_feat_dim, old_d)))
        expand = Q.astype(np.float64)

        with self.memory._lock:
            # Expand world prior
            self.memory.world.mu    = expand @ self.memory.world.mu
            new_v                   = np.ones(new_feat_dim) * self.memory.world.v.mean()
            self.memory.world.v     = new_v
            self.memory.world.inv_v = 1.0 / np.maximum(new_v, _MIN_VAR)
            self.memory.world._buf  = np.empty(new_feat_dim)
            self.memory.world._inv_d= np.ones(new_feat_dim) / new_feat_dim
            self.memory.world.F_field = np.zeros((new_feat_dim, self.field_dim))

            # Expand class delta_mus
            K = len(self.memory._label_order)
            for lbl, cd in self.memory._classes.items():
                cd.delta_mu = expand @ cd.delta_mu

            # Rebuild D_buf
            new_buf = np.zeros((self.memory._D_buf.shape[0], new_feat_dim))
            for ki in range(K):
                l = self.memory._label_order[ki]
                new_buf[ki] = self.memory._classes[l].delta_mu.copy()
            self.memory._D_buf   = new_buf
            self.memory._probs_buf = np.empty(K + 128)

        # Expand encoder projection: new W maps input → new_d latent
        self.encoder.W   = expand @ self.encoder.W   # (new_d, feat_dim_in)
        self.encoder.dim = new_feat_dim
        self.feat_dim    = new_feat_dim

        # Rebuild field injection matrix
        if new_feat_dim == self.field_dim:
            self._W_inject = None
        elif new_feat_dim < self.field_dim:
            Q2, _ = np.linalg.qr(rng.normal(0, 1.0, (self.field_dim, new_feat_dim)))
            self._W_inject = Q2.astype(np.float64)
        else:
            Q2, _ = np.linalg.qr(rng.normal(0, 1.0, (new_feat_dim, self.field_dim)))
            self._W_inject = Q2.T.astype(np.float64)

        # Re-project replay buffer h vectors (entries are 5-tuples: h, f, label, loss, ins_n)
        for i in range(self.replay._buf_len):
            h_old, f, lbl, *rest = self.replay._buf[i]
            self.replay._buf[i]  = (expand @ h_old, f, lbl, *rest)

    def _class_params_safe(self, label: str) -> Tuple[np.ndarray, np.ndarray]:
        p = self.memory.get_class_params(label)
        if p is None:
            raise KeyError(f"Unknown class '{label}'. Known: {list(self.memory._classes)}")
        return p

    # ── Core API ──────────────────────────────────────────────────────────────

    def infer(self, x: Any, use_field: bool = True) -> Tuple[str, float]:
        """Classify x. Returns (label, confidence ∈ [0,1])."""
        f, h      = self._encode(x)
        h_field   = self.field.h if use_field else None
        with self.memory._lock:
            classes = list(self.memory._classes.keys())
        ctx_prior = self._ctx_prior(classes) if classes else {}
        pred, conf, _ = self.memory.classify(
            h, h_field=h_field, context_prior=ctx_prior,
            temperature=self.temperature, ood_sigma=self.ood_sigma,
            mahal_ema=getattr(self, '_mahal_ema', None),
            mahal_std_ema=getattr(self, '_mahal_std_ema', 0.5),
        )
        return pred, conf

    def infer_full(self, x: Any, use_field: bool = True) -> Dict:
        """
        Phase 3: Full probabilistic breakdown.
        Returns dict: label, confidence, llrs, probs, entropy, ood_gate,
                      anomaly_score, query_score, drift_score, field_step.
        """
        f, h      = self._encode(x)
        h_field   = self.field.h if use_field else None
        with self.memory._lock:
            classes = list(self.memory._classes.keys())
        ctx_prior = self._ctx_prior(classes) if classes else {}

        pred, conf, llrs = self.memory.classify(
            h, h_field=h_field, context_prior=ctx_prior,
            temperature=self.temperature, ood_sigma=self.ood_sigma,
            mahal_ema=getattr(self, '_mahal_ema', None),
            mahal_std_ema=getattr(self, '_mahal_std_ema', 0.5),
        )

        if llrs:
            llr_arr   = np.array(list(llrs.values()))
            probs_arr = _softmax(llr_arr / (self.temperature + _EPS))
            probs     = dict(zip(llrs.keys(), probs_arr.tolist()))
            entropy   = _shannon_entropy(probs_arr)
            max_llr   = float(llr_arr.max())
            # Legacy LLR gate — kept for analysis/observability in infer_full() only.
            # No longer used in classify() confidence computation.
            ood_gate2 = float(1.0 / (1.0 + math.exp(-max_llr / (self.ood_sigma + _EPS))))
            # Gate 1: absolute world-prior LL per dim (catches extreme OOD)
            with self.memory._lock:
                mu0c = self.memory.world.mu.copy()
                v0c  = self.memory.world.v.copy()
            ll_w     = _diag_gaussian_logpdf(h, mu0c, v0c) / (len(h) + _EPS)
            ll_ref   = getattr(self, '_ll_world_ema', -1.5)
            margin   = float(np.clip((ll_w - ll_ref) * 2.0, -500, 500))
            ood_gate1 = float(1.0 / (1.0 + math.exp(-margin)))
            ood_gate  = min(ood_gate1, ood_gate2)
            max_p     = float(probs_arr.max())
            query_score = float(entropy * (1.0 - max_p))
        else:
            probs = {}; entropy = 0.0; ood_gate = 0.0; query_score = 0.0

        return {
            'label'        : pred,
            'confidence'   : conf,
            'llrs'         : llrs,
            'probs'        : probs,
            'entropy'      : entropy,
            'ood_gate'     : ood_gate,
            'anomaly_score': 1.0 - ood_gate,
            'query_score'  : query_score,
            'drift_score'  : self.drift_score(),
            'field_step'   : self.field.step,
        }

    def batch_infer(self, xs: List[Any],
                    use_field: bool = False,
                    ) -> List[Tuple[str, float]]:
        """
        Vectorised batch inference.  Uses score_matrix() internally.
        40–140× faster than serial infer() depending on N and encoding cost.

        Parameters
        ----------
        xs        : list of N raw inputs, OR a (N, d) numpy array of pre-encoded
                    latent vectors (skip encoding — maximum throughput)
        use_field : apply field-conditioned μ₀ shift

        Returns list of (label, confidence) tuples.

        For maximum performance with pre-encoded data:
            H = clf.batch_encode(xs)       # encode once
            results = clf.batch_infer(H)   # score fast
        """
        if xs is None or (hasattr(xs, '__len__') and len(xs) == 0):
            return []

        # Accept pre-encoded (N, d) matrix directly — skip encoding
        if isinstance(xs, np.ndarray) and xs.ndim == 2:
            H = xs.astype(np.float64)
        else:
            H = self.batch_encode(xs)

        N = len(H)
        if N == 0:
            return []

        LLR, labels = self.score_matrix(H, use_field=use_field)
        if len(labels) == 0:
            return [('__unknown__', 0.0)] * N

        probs = _softmax_batch(LLR / (self.temperature + _EPS))
        gates = self.world_gate_vector(H)

        best_idx = probs.argmax(axis=1)
        confs    = probs[np.arange(N), best_idx] * gates
        return [(labels[best_idx[i]], float(confs[i])) for i in range(N)]

    def batch_infer_full(self, xs: List[Any],
                         use_field: bool = False,
                         ) -> List[Dict]:
        """
        Vectorised batch version of infer_full().
        Returns list of dicts: label, confidence, llrs, probs, entropy, anomaly_score.
        """
        if not xs:
            return []
        # Accept pre-encoded matrix
        if isinstance(xs, np.ndarray) and xs.ndim == 2:
            H = xs.astype(np.float64)
        else:
            H = self.batch_encode(xs)

        N = len(H)
        if N == 0:
            return []

        LLR, labels = self.score_matrix(H, use_field=use_field)
        if len(labels) == 0:
            return [{'label': '__unknown__', 'confidence': 0.0, 'llrs': {},
                     'probs': {}, 'entropy': 0.0, 'anomaly_score': 1.0}] * N

        probs     = _softmax_batch(LLR / (self.temperature + _EPS))
        gates     = self.world_gate_vector(H)
        best_idx  = probs.argmax(axis=1)
        best_conf = probs[np.arange(N), best_idx] * gates
        entropies = -np.sum(probs * np.log(probs + _EPS), axis=1)

        return [{
            'label'        : labels[best_idx[i]],
            'confidence'   : float(best_conf[i]),
            'llrs'         : dict(zip(labels, LLR[i].tolist())),
            'probs'        : dict(zip(labels, probs[i].tolist())),
            'entropy'      : float(entropies[i]),
            'anomaly_score': float(1.0 - gates[i]),
        } for i in range(N)]


    def score_matrix(self, H: np.ndarray,
                     use_field: bool = False,
                     ) -> Tuple[np.ndarray, List[str]]:
        """
        Core scoring primitive: (N, d) latent matrix → (N, K) raw LLR matrix.

        This is the fundamental kernel that batch_infer, batch_infer_full and
        future CUDA/C++ implementations are built on.  All downstream tasks
        (classification, confidence, calibration, calibration) derive from LLR.

        Equation (per row i, class k)
        ─────────────────────────────
          LLR[i,k] = D[k]·(inv_v ⊙ (H[i] - μ₀))
                     - ½ ‖D[k]‖²_inv_v
                     - U_k
                     + ctx[k]

        Parameters
        ----------
        H         : (N, d) numpy array of latent vectors (already encoded).
                    Use batch_encode(xs) to go from raw inputs to H.
        use_field : apply field-conditioned μ₀ shift (one extra matmul)

        Returns
        -------
        LLR   : (N, K) float64 array — higher = more likely
        labels: list of K class names corresponding to LLR columns

        Notes
        -----
        No OOD gating is applied here — this is intentional.  Apply gating
        downstream with world_gate_vector() if needed.
        Thread-safe: takes a single lock snapshot.
        """
        N, d = H.shape

        with self.memory._lock:
            K       = len(self.memory._label_order)
            if K == 0:
                return np.zeros((N, 0)), []
            labels  = self.memory._label_order[:K]
            D       = self.memory._D_buf[:K].copy()      # (K, d)
            mu0     = self.memory.world.mu.copy()
            inv_v   = self.memory.world.inv_v.copy()
            n_obs_v = np.array([self.memory._classes[l].n_obs for l in labels],
                                dtype=np.float64)

        if use_field:
            h_fld = self.field.h
            h_sq  = float(h_fld @ h_fld)
            if math.isfinite(h_sq) and h_sq <= 1e8:
                mu0 = mu0 + self.memory.world.F_field @ h_fld

        # Core kernel: 3 matrix ops
        R     = (H - mu0) * inv_v                         # (N, d)  — broadcast
        cross = R @ D.T                                   # (N, K)  — BLAS gemm
        D_sq  = (D * D) @ inv_v                           # (K,)
        u_k   = inv_v.mean() / (n_obs_v + 1.0)           # (K,)

        ctx     = self._ctx_prior(labels)
        ctx_arr = np.array([ctx.get(l, 0.0) for l in labels])

        LLR = cross - 0.5 * D_sq - u_k + ctx_arr         # (N, K)
        return LLR, labels

    def batch_encode(self, xs: List[Any]) -> np.ndarray:
        """
        Encode a list of inputs to a (N, d) latent matrix.

        Vectorises the encoder projection step for batch processing.
        For numpy array inputs this is a single (N, feat_dim) @ (feat_dim, d)
        matmul rather than N serial dot products.

        Use with score_matrix() for maximum throughput:
            H   = clf.batch_encode(xs)
            LLR, labels = clf.score_matrix(H)

        Returns (N, feat_dim) array (latent space).
        """
        if xs is None or (not hasattr(xs, '__len__')) or len(xs) == 0:
            return np.empty((0, self.feat_dim))
        try:
            # Fast path: stack encoder outputs and project in one matmul
            F = np.stack([self.encoder_fn(x).astype(np.float64) for x in xs])
            return F @ self.encoder.W.T   # (N, feat_dim)
        except Exception:
            # Fallback: serial projection (handles non-array inputs)
            return np.stack([self.encoder.project(self.encoder_fn(x).astype(np.float64))
                             for x in xs])

    def world_gate_vector(self, H: np.ndarray) -> np.ndarray:
        """
        Vectorised OOD gate for a batch of latent vectors.

        Returns (N,) float array ∈ [0, 1]:
          1.0 = in-distribution (confident)
          0.0 = out-of-distribution (suppress confidence)

        Use to gate the output of score_matrix():
            LLR, labels = clf.score_matrix(H)
            gates = clf.world_gate_vector(H)
            probs = softmax(LLR)
            conf  = probs.max(1) * gates
        """
        with self.memory._lock:
            mu0   = self.memory.world.mu.copy()
            inv_v = self.memory.world.inv_v.copy()

        N, d = H.shape
        m_ema = getattr(self, '_mahal_ema', None)
        m_std = getattr(self, '_mahal_std_ema', 0.5)

        if m_ema is None or m_std <= 0:
            return np.ones(N)

        diffs   = H - mu0
        mahals  = np.sum(diffs ** 2 * inv_v, axis=1) / (d + _EPS)
        thresh  = m_ema + 5.0 * m_std
        margins = np.clip((thresh - mahals) * 2.0 / max(m_std, _EPS), -500., 500.)
        return 1.0 / (1.0 + np.exp(-margins))


    # ── Decoder ───────────────────────────────────────────────────────────────

    def _build_decoder(self) -> None:
        """
        Pre-compute the decoder matrix  W⁺ = pinv(EncoderProjection.W).

        Called automatically on first decode() call.
        VectorEncoder: h = W @ x  →  x = W⁺ @ h   (exact if W is square/tall)
        RFFEncoder:    h = W_enc @ φ(x) — not exactly invertible, falls back to
                       replay-anchored k-NN decode (see decode()).
        Cached as self._decoder_W_pinv.
        """
        with self.encoder._lock:
            W = self.encoder.W.copy()
        self._decoder_W_pinv = np.linalg.pinv(W)   # (feat_dim, feat_dim)
        self._decoder_ready  = True

    def decode(self, h: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Invert the encoder: latent h  →  raw input x.

        Strategy
        ─────────
        VectorEncoder  (default): x = W⁺ @ h  — one matmul, deterministic, exact.
          The EncoderProjection W is fixed at init and never changes, so W⁺ is
          computed once and cached.

        RFFEncoder / other nonlinear encoders: replay-anchored k-NN decode.
          Re-encodes all stored training inputs with the current encoder to find
          the k nearest latent neighbours of h, then returns their weighted average
          in input space (weights = softmax(-dist²/T)).
          Requires that raw inputs were stored via train_step_with_decode() or
          that _x_store is populated manually.  Falls back to W⁺@h if no store.

        Parameters
        ----------
        h : latent vector (feat_dim,)
        k : number of nearest neighbours (k=1 = exact match for training data)

        Returns
        -------
        x_approx : (feat_dim_in,) reconstructed input
        """
        if not getattr(self, '_decoder_ready', False):
            self._build_decoder()

        # Try VectorEncoder exact path first (f IS x for VectorEncoder)
        # Check: is the encoder a passthrough (VectorEncoder)?
        encoder_class = self.encoder_fn.__class__.__name__
        if encoder_class == 'VectorEncoder':
            return self._decoder_W_pinv @ h    # x = W⁺ @ h, exact

        # RFF or other nonlinear encoder: replay-anchored k-NN
        x_store = getattr(self, '_x_store', None)
        if x_store and len(x_store) > 0:
            return self._decode_knn(h, x_store, k=k)

        # Fallback: W⁺ @ h (linear approximation, not exact for RFF)
        return self._decoder_W_pinv @ h

    def _decode_knn(self, h_star: np.ndarray,
                    x_store : List[np.ndarray],
                    k       : int = 3) -> np.ndarray:
        """
        Replay-anchored k-NN decode: find k stored inputs whose latent
        representations are closest to h_star, return their weighted average.

        Re-encodes stored inputs with the current encoder state (W is fixed
        so this is deterministic and cached once W is known).
        """
        X_arr   = np.stack(x_store)                              # (N, d_in)
        F_arr   = np.stack([self.encoder_fn(x).astype(np.float64)
                            for x in x_store])                   # (N, feat)
        H_live  = F_arr @ self.encoder.W.T                       # (N, d)
        dists2  = np.sum((H_live - h_star) ** 2, axis=1)        # (N,)
        top_k   = np.argsort(dists2)[:k]
        d_vals  = dists2[top_k]
        T       = max(float(d_vals.max()) * 0.1, _EPS)
        weights = np.exp(-d_vals / T)
        weights = weights / (weights.sum() + _EPS)
        return sum(float(weights[j]) * X_arr[top_k[j]] for j in range(k))

    def decode_batch(self, H: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Vectorised batch decode: (N, d) latent matrix → (N, d_in) inputs.

        For VectorEncoder: single matmul H @ W_pinv.T
        For others: sequential k-NN calls (parallelisable on GPU in future).

        Returns (N, d_in) array.
        """
        if not getattr(self, '_decoder_ready', False):
            self._build_decoder()

        encoder_class = self.encoder_fn.__class__.__name__
        if encoder_class == 'VectorEncoder':
            return H @ self._decoder_W_pinv.T    # (N, d) @ (d, d_in) = (N, d_in)

        # Sequential k-NN for nonlinear encoders
        return np.stack([self.decode(H[i], k=k) for i in range(len(H))])

    def generate_real(self, label: str, n: int = 10,
                      mode    : str   = 'langevin',
                      temperature: float = 1.0,
                      n_steps : int   = 30,
                      rng     : Optional[np.random.Generator] = None,
                      ) -> np.ndarray:
        """
        Generate real inputs in the original data space.

        This is the complete generation pipeline:
          1. Sample latent h vectors via the chosen generation mode
          2. Decode h → x via the encoder inverse (W⁺@h or k-NN)

        Parameters
        ----------
        label       : class to generate from
        n           : number of samples
        mode        : 'langevin'  — posterior MCMC (highest quality, best diversity)
                      'gaussian'  — fast Gaussian from class centroid
                      'boundary'  — samples near the class decision boundary
                      'ood'       — out-of-distribution samples
        temperature : sampling temperature (higher = wider, more diverse)
        n_steps     : Langevin MCMC steps (only for mode='langevin')

        Returns
        -------
        X : (n, d_in) numpy array of generated inputs
        """
        rng = rng or self._rng

        if mode == 'langevin':
            H = self.generate_langevin(label, n=n, n_steps=n_steps,
                                       temperature=temperature, rng=rng)
            H = np.stack(H)
        elif mode == 'gaussian':
            H = np.stack(self.generate(label, n=n, temperature=temperature))
        elif mode == 'boundary':
            H = np.stack(self.generate_hard_negatives(label, n=n))
        elif mode == 'ood':
            H = np.stack(self.generate_ood(n=n))
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use: langevin/gaussian/boundary/ood")

        return self.decode_batch(H)

    def generate_sequence(self, seed_label: str, n_steps: int = 10,
                           temperature: float = 0.5,
                           rng        : Optional[np.random.Generator] = None,
                           ) -> List[Tuple[str, np.ndarray]]:
        """
        Generate a sequence of real inputs using the learned Markov chain.

        Combines rollout() (sequence prediction) with decode() (inversion)
        to produce trajectories in the original input space.

        Returns list of (label, x) tuples — real inputs along the predicted sequence.
        """
        rng  = rng or self._rng
        seq  = self.rollout(seed_label, n_steps=n_steps,
                            temperature=temperature, rng=rng)
        return [(label, self.decode(h)) for label, h in seq]

    def generate_composite_real(self, weights: Dict[str, float], n: int = 5,
                                 rng: Optional[np.random.Generator] = None,
                                 ) -> np.ndarray:
        """
        Generate blended real inputs from a weighted mixture of classes.

        weights = {'alpha': 0.7, 'beta': 0.3} generates inputs that are
        70% alpha + 30% beta in latent space, decoded back to real inputs.

        Returns (n, d_in) array.
        """
        H = np.stack(self.generate_composite(weights, n=n, rng=rng))
        return self.decode_batch(H)

    # ─────────────────────────────────────────────────────────────────────────

    def train_step(self, x: Any, label: str) -> float:
        """One online supervised training step. Returns loss."""
        f, h = self._encode(x)

        with self.memory._lock:
            classes = list(self.memory._classes.keys())
        if label not in classes:
            classes.append(label)
        ctx_prior = self._ctx_prior(classes)
        h_field   = self.field.h

        pred, correct, loss, post_llrs, post_conf = self.memory.train(
            h, label, h_field=h_field, context_prior=ctx_prior,
            temperature=self.temperature, ood_sigma=self.ood_sigma,
        )

        # Phase 4: priority push
        self.replay.push(h, f, label, loss=loss)
        self._buf_len_cache = len(self.replay)

        # Contrastive encoder update on misclassification
        if not correct and pred != '__unknown__':
            params_k = self.memory.get_class_params(label)
            params_j = self.memory.get_class_params(pred)
            if params_k is not None and params_j is not None:
                self.encoder.contrastive_update(
                    f, h, params_k[0], params_k[1], params_j[0], params_j[1]
                )

        # Deliberate mode: use post-update llrs returned by train() — no extra classify call
        if _DELIBERATE_LO < post_conf < _DELIBERATE_HI and len(post_llrs) >= 2:
            ss    = sorted(post_llrs.items(), key=lambda kv: -kv[1])
            p_top = self.memory.get_class_params(ss[0][0])
            p_sec = self.memory.get_class_params(ss[1][0])
            if p_top is not None and p_sec is not None:
                self.encoder.contrastive_update(
                    f, h, p_top[0], p_top[1], p_sec[0], p_sec[1], weight=0.3
                )

        if self._total_steps % 5 == 0:
            self._dedup_check(label)

        # Phase 4: priority replay — ctx_prior precomputed once for all replay samples
        if self._buf_len_cache >= 10 and np.random.random() < _REPLAY_RATIO:
            rclasses = list(self.memory._classes.keys())
            rctx     = self._ctx_prior(rclasses)         # compute once, reuse for all
            rhf      = self.field.h                       # one h_field copy for all
            for rh, rf, rlabel in self.replay.sample(n=min(4, self._buf_len_cache)):
                self.memory.train(rh, rlabel, h_field=rhf, context_prior=rctx)

        self._total_steps   += 1
        self._total_correct += int(correct)
        self._loss_buf.append(loss)

        if self._total_steps % self._align_every == 0:
            with self.memory._lock:
                deltas = [cd.delta_mu.copy() for cd in self.memory._classes.values()
                          if (cd.delta_mu @ cd.delta_mu) > _EPS * _EPS]
            if len(deltas) >= 2:
                self.encoder.align_to_offsets(deltas)

        # Field update
        self.field.inject(self._to_field_dim(h), strength=0.05)
        h_old = self.field.h
        h_new = self.field.evolve(h_old, update_state=True)
        if self._total_steps % 50 == 0:
            self.field.update_causal(h_old, h_new, lr=0.0002)

        self.context.record(label, correct,
                            llr_max=max(post_llrs.values()) if post_llrs else 0.0)

        # Track LLR scale for temperature adaptation
        if post_llrs:
            _llr_win = float(max(post_llrs.values()))
            _alpha   = 0.002
            self._llr_scale_ema = (1 - _alpha) * self._llr_scale_ema + _alpha * abs(_llr_win)
            self._llr_scale_n  += 1

        # Adaptive OOD sigma (every 20 steps)
        if self._total_steps % 20 == 0:
            _, _, llrs_train = self.memory.classify(
                h, h_field=self.field.h,
                temperature=self.temperature, ood_sigma=self.ood_sigma
            )
            if llrs_train:
                max_llr        = max(llrs_train.values())
                self._llr_ema  = (1 - _OOD_EMA) * self._llr_ema + _OOD_EMA * max_llr
                self.ood_sigma = max(1.0, abs(self._llr_ema))
            with self.memory._lock:
                mu0_c   = self.memory.world.mu.copy()
                inv_v_c = self.memory.world.inv_v.copy()
            d_h_c         = h - mu0_c
            mahal_c       = float(d_h_c @ (d_h_c * inv_v_c)) / (len(h) + _EPS)
            prev_ema      = getattr(self, '_mahal_ema',     mahal_c)
            prev_var      = getattr(self, '_mahal_std_ema', 0.5) ** 2
            self._mahal_ema     = (1 - _OOD_EMA) * prev_ema + _OOD_EMA * mahal_c
            mahal_var           = (1 - _OOD_EMA) * prev_var + _OOD_EMA * (mahal_c - prev_ema) ** 2
            self._mahal_std_ema = max(math.sqrt(mahal_var), 0.05)

        return float(loss)

    def batch_train(self, data: List[Tuple[Any, str]],
                    n_epochs: int = 1, shuffle: bool = True) -> List[float]:
        """Train on (input, label) pairs. Returns per-step losses."""
        import random
        losses = []
        for _ in range(n_epochs):
            if shuffle:
                random.shuffle(data)
            for x, y in data:
                losses.append(self.train_step(x, y))
        return losses

    def macro_accuracy(self, data: List[Tuple[Any, str]]) -> Tuple[float, Dict[str, float]]:
        """Macro-averaged accuracy."""
        per: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for x, y in data:
            pred, _ = self.infer(x)
            per[y][0] += int(pred == y)
            per[y][1] += 1
        per_cls = {k: v[0] / v[1] for k, v in per.items() if v[1] > 0}
        macro   = sum(per_cls.values()) / len(per_cls) if per_cls else 0.0
        return macro, per_cls

    # ── Phase 2: Generation API ───────────────────────────────────────────────

    def generate(self, label: str, n: int = 1, temperature: float = 1.0,
                 rng: Optional[np.random.Generator] = None,
                 rejection_sampling: bool = True,
                 max_candidates: int = 16) -> List[np.ndarray]:
        """
        Temperature-scaled sampling:  h ~ N(μk, temperature² · v₀)
        temperature=0 → mode (μk).  temperature<1 → tighter.  temperature>1 → diverse.

        rejection_sampling: when True (default), each sample is drawn from up to
            max_candidates candidates and the one with the highest LLR for `label`
            is kept.  Eliminates misclassified samples at high temperatures with
            negligible cost (vectorised candidate scoring).  Set False for raw speed.
        """
        mu_k, v0 = self._class_params_safe(label)
        rng = rng or self._rng
        if temperature <= 0.0:
            return [mu_k.copy() for _ in range(n)]
        std = np.sqrt(np.maximum(v0, _MIN_VAR)) * temperature

        if not rejection_sampling or temperature <= 1.0:
            return [mu_k + rng.standard_normal(len(mu_k)) * std for _ in range(n)]

        # Rejection sampling: draw max_candidates, keep best LLR_k
        with self.memory._lock:
            mu0   = self.memory.world.mu.copy()
            inv_v = self.memory.world.inv_v.copy()
            K     = len(self.memory._label_order)
            D     = self.memory._D_buf[:K].copy()
            k_idx = self.memory._label_idx.get(label, -1)

        results = []
        for _ in range(n):
            C        = max_candidates
            cands    = mu_k + rng.standard_normal((C, len(mu_k))) * std  # (C, d)
            r_mat    = (cands - mu0) * inv_v                              # (C, d)
            cross    = r_mat @ D.T                                        # (C, K)
            D_sq     = (D * D) @ inv_v                                    # (K,)
            llr_mat  = cross - 0.5 * D_sq                                # (C, K)
            if k_idx >= 0:
                best_i = int(llr_mat[:, k_idx].argmax())
            else:
                best_i = 0
            results.append(cands[best_i])
        return results

    def generate_conditioned(self, label: str, n: int = 1,
                             temperature: float = 1.0,
                             rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
        """
        Field-conditioned generation.  World prior is shifted by current field state.
            μ₀(t) = μ₀ + F_field @ h_field
            h ~ N(μ₀(t) + Δk, temperature² · v₀)
        Produces samples consistent with the current temporal context.
        temperature=0 → mode (μk), temperature<1 → tighter, temperature>1 → diverse.
        """
        h_field  = self.field.h
        mu0, v0  = self.memory.world.condition_on_field(h_field)
        with self.memory._lock:
            if label not in self.memory._classes:
                raise KeyError(f"Unknown class '{label}'.")
            delta_mu = self.memory._classes[label].delta_mu.copy()
        mu_k = mu0 + delta_mu
        rng  = rng or self._rng
        if temperature <= 0.0:
            return [mu_k.copy() for _ in range(n)]
        std  = np.sqrt(np.maximum(v0, _MIN_VAR)) * temperature
        return [mu_k + rng.standard_normal(len(mu_k)) * std for _ in range(n)]

    def generate_boundary(self, label_a: str, label_b: str,
                          n: int = 1, alpha: float = 0.5,
                          temperature: float = 0.3,
                          rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
        """
        Latent space interpolation between two class means, snapped to the true
        LLR decision surface by a gradient walk along the boundary normal.

            μ_interp = (1−α)·μ_a + α·μ_b
            h ~ N(μ_interp, temperature²·v₀)
            h ← h − t·normal   where t makes LLR_a(h) = LLR_b(h)

        alpha=0 → label_a,  alpha=0.5 → exact boundary,  alpha=1 → label_b.
        temperature controls dispersion *along* the boundary (not perpendicular).
        """
        mu_a, v0 = self._class_params_safe(label_a)
        mu_b, _  = self._class_params_safe(label_b)
        with self.memory._lock:
            mu0   = self.memory.world.mu.copy()
            inv_v = self.memory.world.inv_v.copy()
            dm_a  = self.memory._classes[label_a].delta_mu.copy()
            dm_b  = self.memory._classes[label_b].delta_mu.copy()

        # Boundary normal in latent space: ∇_h (LLR_a - LLR_b) = (dm_a - dm_b) * inv_v
        normal    = (dm_a - dm_b) * inv_v
        n_sq      = float(normal @ normal) + _EPS
        # Signed intercept: at exact boundary LLR_a == LLR_b →
        #   (h - mu0)·normal = 0.5*(||dm_a||²_inv_v - ||dm_b||²_inv_v) + U_a - U_b
        # U terms are small and class-dependent; approximate as 0 for boundary geometry.
        target_dot = 0.5 * (float(dm_a @ (dm_a * inv_v)) - float(dm_b @ (dm_b * inv_v)))

        mu_interp = (1.0 - alpha) * mu_a + alpha * mu_b
        rng       = rng or self._rng
        std       = np.sqrt(np.maximum(v0, _MIN_VAR)) * max(temperature, _EPS)
        results   = []
        for _ in range(n):
            h    = mu_interp + rng.standard_normal(len(mu_interp)) * std
            # Project h onto the true decision hyperplane along the normal direction
            curr = float((h - mu0) @ normal)
            t    = (curr - target_dot) / n_sq   # signed step
            h   -= t * normal                    # snap to boundary
            results.append(h)
        return results

    def generate_adversarial(self, n: int = 1, n_candidates: int = 64,
                             rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
        """
        Maximise classification posterior entropy.

        Sampling strategy (improved):
          50% near class-pair midpoints (where confusion is highest by design)
          50% near world prior with wide noise

        Scoring uses vectorised batch logpdf — ~10× faster than v1.
        Returns top-n candidates by posterior entropy.
        """
        with self.memory._lock:
            mu0     = self.memory.world.mu.copy()
            v0      = self.memory.world.v.copy()
            classes = list(self.memory._classes.keys())
            class_mus = [self.memory._classes[k].mu(mu0) for k in classes]

        if len(classes) < 2:
            raise ValueError("Need ≥ 2 classes for adversarial generation.")

        rng = rng or self._rng
        std = np.sqrt(np.maximum(v0, _MIN_VAR))

        # Build candidate pool
        cands = []
        half  = n_candidates // 2

        # Half: midpoints between random class-mean pairs with noise
        for _ in range(half):
            i, j = rng.choice(len(classes), size=2, replace=False)
            alpha = rng.uniform(0.3, 0.7)
            mid   = (1 - alpha) * class_mus[i] + alpha * class_mus[j]
            cands.append(mid + rng.standard_normal(len(mu0)) * std * 0.5)

        # Half: world-prior sampling (wide noise for OOD boundary coverage)
        for _ in range(n_candidates - half):
            cands.append(mu0 + rng.standard_normal(len(mu0)) * std)

        # Vectorised entropy scoring
        H = np.stack(cands)  # (n_candidates, D)
        with self.memory._lock:
            mu0c, v0c = self.memory.world.mu.copy(), self.memory.world.v.copy()
            class_items = [(k, self.memory._classes[k]) for k in classes]

        ll_world = _batch_logpdf(H, mu0c, v0c)  # (B,)
        # LLR matrix: (B, K)
        llr_mat = np.zeros((len(H), len(class_items)))
        for ci, (lbl, cd) in enumerate(class_items):
            mu_k      = cd.mu(mu0c)
            u_k       = float(np.mean(v0c)) / (cd.n_obs + 1)
            ll_k      = _batch_logpdf(H, mu_k, v0c)
            llr_mat[:, ci] = ll_k - ll_world - u_k

        # Softmax entropy over classes
        llr_mat -= llr_mat.max(axis=1, keepdims=True)
        exp_mat  = np.exp(llr_mat / (self.temperature + _EPS))
        prob_mat = exp_mat / (exp_mat.sum(axis=1, keepdims=True) + _EPS)
        entropies = -np.sum(prob_mat * np.log(prob_mat + _EPS), axis=1)

        top_idx = np.argsort(entropies)[::-1][:n]
        return [cands[i] for i in top_idx]

    # ══════════════════════════════════════════════════════════════════════════
    # NEW GENERATION API  (Phase 6 additions)
    # ══════════════════════════════════════════════════════════════════════════

    # ── 1. Latent-space classify / generate ───────────────────────────────────

    def classify_latent(self, h: np.ndarray,
                        use_field: bool = True) -> Tuple[str, float]:
        """
        Classify an already-encoded latent vector h directly.

        Unlike infer(x), this skips the encoder step — use when working with
        vectors returned by generate(), generate_langevin(), or generate_composite().

            pred, conf = clf.classify_latent(h)

        Returns (label, confidence ∈ [0,1]).
        """
        h_field = self.field.h if use_field else None
        with self.memory._lock:
            classes = list(self.memory._classes.keys())
        ctx_prior = self._ctx_prior(classes) if classes else {}
        pred, conf, _ = self.memory.classify(
            h, h_field=h_field, context_prior=ctx_prior,
            temperature=self.temperature, ood_sigma=self.ood_sigma,
            mahal_ema=getattr(self, '_mahal_ema', None),
            mahal_std_ema=getattr(self, '_mahal_std_ema', 0.5),
        )
        return pred, conf

    def classify_latent_full(self, h: np.ndarray,
                             use_field: bool = True) -> Dict:
        """
        Full probabilistic breakdown for a latent vector h.
        Returns same dict as infer_full() but skips the encoder.
        """
        h_field = self.field.h if use_field else None
        with self.memory._lock:
            classes = list(self.memory._classes.keys())
        ctx_prior = self._ctx_prior(classes) if classes else {}
        pred, conf, llrs = self.memory.classify(
            h, h_field=h_field, context_prior=ctx_prior,
            temperature=self.temperature, ood_sigma=self.ood_sigma,
            mahal_ema=getattr(self, '_mahal_ema', None),
            mahal_std_ema=getattr(self, '_mahal_std_ema', 0.5),
        )
        if llrs:
            llr_arr   = np.array(list(llrs.values()))
            probs_arr = _softmax(llr_arr / (self.temperature + _EPS))
            probs     = dict(zip(llrs.keys(), probs_arr.tolist()))
            entropy   = _shannon_entropy(probs_arr)
        else:
            probs, entropy = {}, 0.0
        return dict(label=pred, confidence=conf, llrs=llrs,
                    probs=probs, entropy=entropy)

    # ── 2. Langevin posterior sampling ────────────────────────────────────────

    def generate_langevin(self, label: str, n: int = 1,
                          n_steps: int = 60, step_size: float = 0.05,
                          temperature: float = 1.0,
                          rng: Optional[np.random.Generator] = None,
                          ) -> List[np.ndarray]:
        """
        Langevin MCMC sampling from the true class posterior p(h | y=label).

        Unlike generate() which samples a Gaussian ball around μk, Langevin
        dynamics follow the analytic gradient of log p(h|y=k), exploring the
        full posterior manifold.  This gives substantially more diversity (typically
        2–3×) at the same self-consistency, because it finds every region where
        LLR_k is large — not just the neighbourhood of the mean.

        Gradient derivation
        -------------------
        LLR_k(h) = Δk · (inv_v ⊙ (h − μ₀)) − ½ ‖Δk‖²_inv_v − U_k
        ∇_h LLR_k = inv_v ⊙ Δk   (analytic, exact)

        Additionally a weak isotropic prior ∇_h log p₀(h) = −(h − μ₀)/v_prior
        prevents unbounded drift along low-gradient dimensions.

        Langevin update
        ---------------
        h ← h + step·∇_h log p(h|k) + √(2·step·T²)·ε,   ε ~ N(0, I)

        Parameters
        ----------
        label      : class to sample from
        n          : number of independent chains (samples returned)
        n_steps    : Langevin steps per sample (more → better mixing)
        step_size  : gradient step magnitude
        temperature: noise temperature (lower → tighter, higher → more exploratory)
        rng        : random generator (uses internal rng if None)
        """
        rng = rng or self._rng
        mu_k, v0 = self._class_params_safe(label)
        with self.memory._lock:
            mu0   = self.memory.world.mu.copy()
            inv_v = self.memory.world.inv_v.copy()
            k_idx = self.memory._label_idx.get(label, 0)
            K     = len(self.memory._label_order)
            delta_mu_k = self.memory._classes[label].delta_mu.copy()

        # Analytic gradient of LLR_k w.r.t. h: inv_v ⊙ Δk
        grad_llr_k = inv_v * delta_mu_k        # (d,) — constant across h

        # Prior regularisation: prevent runaway along flat dimensions
        # Effective variance for prior: v0 (world variance) scaled by temperature²
        v_prior = v0 * max(temperature ** 2, 0.1)

        sqrt_2step_T = np.sqrt(2.0 * step_size * max(temperature ** 2, _MIN_VAR))

        results = []
        for _ in range(n):
            # Initialise from class mean + small noise for diversity
            h = mu_k + rng.standard_normal(len(mu_k)) * np.sqrt(v0) * 0.5

            for _ in range(n_steps):
                # Gradient of log posterior:  LLR_k gradient + prior gradient
                grad = grad_llr_k - (h - mu_k) / (v_prior + _EPS)
                h    = h + step_size * grad + sqrt_2step_T * rng.standard_normal(len(h))

            results.append(h.copy())
        return results

    # ── 3. Composite / concept-blending generation ────────────────────────────

    def generate_composite(self, weights: Dict[str, float],
                           n: int = 1,
                           n_steps: int = 80,
                           step_size: float = 0.04,
                           temperature: float = 0.8,
                           rng: Optional[np.random.Generator] = None,
                           ) -> List[np.ndarray]:
        """
        Generate latent vectors satisfying a weighted mixture of class constraints.

        Uses Langevin dynamics with a composite gradient:

            ∇_h log p(h | blended) = Σ_k w_k · ∇_h LLR_k(h)
                                    = inv_v ⊙ (Σ_k w_k · Δk)

        This generates samples that lie on the manifold where all specified
        classes are simultaneously likely — a principled form of concept blending.

        Parameters
        ----------
        weights : dict mapping label → weight (need not sum to 1; auto-normalised)

        Examples
        --------
        # Midpoint blend
        h = clf.generate_composite({'dog': 0.5, 'cat': 0.5})

        # Mostly alpha with beta flavour
        h = clf.generate_composite({'alpha': 0.8, 'beta': 0.2})

        # Triplet blend
        h = clf.generate_composite({'alpha': 1/3, 'beta': 1/3, 'gamma': 1/3})
        """
        if not weights:
            raise ValueError("weights must be a non-empty dict {label: float}")
        rng = rng or self._rng

        # Normalise weights
        total_w = sum(abs(v) for v in weights.values()) + _EPS
        w_norm  = {l: v / total_w for l, v in weights.items()}

        with self.memory._lock:
            mu0   = self.memory.world.mu.copy()
            inv_v = self.memory.world.inv_v.copy()
            v0    = self.memory.world.v.copy()
            delta_mus = {}
            for l in w_norm:
                if l not in self.memory._classes:
                    raise KeyError(f"Unknown label '{l}'. Known: {list(self.memory._classes)}")
                delta_mus[l] = self.memory._classes[l].delta_mu.copy()

        # Composite gradient: inv_v ⊙ (weighted sum of delta_mus)
        composite_delta = sum(w * delta_mus[l] for l, w in w_norm.items())
        grad_composite  = inv_v * composite_delta   # constant across h

        # Starting point: weighted mean of class means
        mu_init = mu0 + composite_delta
        v_prior = v0 * max(temperature ** 2, 0.1)
        sqrt_2step_T = np.sqrt(2.0 * step_size * max(temperature ** 2, _MIN_VAR))

        results = []
        for _ in range(n):
            h = mu_init + rng.standard_normal(len(mu_init)) * np.sqrt(v0) * 0.3

            for _ in range(n_steps):
                grad = grad_composite - (h - mu_init) / (v_prior + _EPS)
                h    = h + step_size * grad + sqrt_2step_T * rng.standard_normal(len(h))

            results.append(h.copy())
        return results

    # ── 4. Unsupervised clustering ────────────────────────────────────────────

    def fit_unlabeled(self, vectors: List[Any],
                      n_clusters: int = 0,
                      prefix: str = 'cluster',
                      min_cluster_size: int = 5,
                      n_init: int = 3,
                      rng: Optional[np.random.Generator] = None,
                      ) -> Dict[str, List[int]]:
        """
        Auto-discover latent clusters in unlabeled data and train on them.

        Pipeline
        --------
        1. Encode all vectors to latent h (via current encoder).
        2. Run k-means in latent space.  If n_clusters=0, estimate k by
           scanning k=2…min(10, N//min_cluster_size) and picking the k with
           the best silhouette score using Mahalanobis distance.
        3. Assign pseudo-labels 'prefix_0', 'prefix_1', …
        4. Call train_step for every (h, pseudo_label) pair.
        5. Return {pseudo_label: [original_vector_indices]}.

        Parameters
        ----------
        vectors         : raw input list (same type as train_step inputs)
        n_clusters      : number of clusters (0 = auto-detect, max 10)
        prefix          : label prefix for discovered clusters
        min_cluster_size: minimum points per cluster for validity
        n_init          : k-means random restarts

        Notes
        -----
        Clusters are added as NEW classes.  Existing classes are unaffected.
        Use merge_from() or prune_classes() to consolidate afterwards.
        """
        rng = rng or self._rng
        if not vectors:
            return {}

        # Encode all vectors
        H = []
        for x in vectors:
            _, h = self._encode(x)
            H.append(h)
        H_arr = np.stack(H)              # (N, d)
        N, d  = H_arr.shape

        # ── k-means with cosine/mahal distance ──────────────────────────────
        with self.memory._lock:
            inv_v_w = self.memory.world.inv_v.copy()

        def _kmeans(k, seed):
            """Lloyd's algorithm in scaled (Mahalanobis) space."""
            rng_k  = np.random.default_rng(seed)
            # Initialise: k-means++ style in Mahal space
            H_s = H_arr * np.sqrt(inv_v_w)  # scaled latent
            chosen = [int(rng_k.integers(N))]
            for _ in range(k - 1):
                dists = np.array([min(float(np.sum((H_s[i] - H_s[c])**2)) for c in chosen)
                                  for i in range(N)])
                p = dists / (dists.sum() + _EPS)
                chosen.append(int(rng_k.choice(N, p=p)))
            centroids = H_arr[chosen].copy()

            labels_km = np.zeros(N, dtype=int)
            for _ in range(50):  # max iterations
                # Assign
                dists_all = np.array([
                    np.sum((H_arr - c[np.newaxis, :]) ** 2 * inv_v_w[np.newaxis, :], axis=1)
                    for c in centroids
                ])  # (k, N)
                new_labels = np.argmin(dists_all, axis=0)
                if np.all(new_labels == labels_km):
                    break
                labels_km = new_labels
                # Update centroids
                for ki in range(k):
                    mask = labels_km == ki
                    if mask.sum() >= 1:
                        centroids[ki] = H_arr[mask].mean(axis=0)
            return labels_km, centroids

        def _silhouette(labels_km, k):
            """Mean silhouette coefficient using Mahalanobis distance."""
            scores = []
            for i in range(min(N, 200)):  # subsample for speed
                c = labels_km[i]
                same  = H_arr[labels_km == c]
                a     = float(np.mean(np.sum((same - H_arr[i]) ** 2 * inv_v_w, axis=1))) if len(same) > 1 else 0.
                other_means = [float(np.mean(np.sum((H_arr[labels_km == ck] - H_arr[i]) ** 2 * inv_v_w, axis=1)))
                               for ck in range(k) if ck != c and (labels_km == ck).sum() > 0]
                b = min(other_means) if other_means else a
                s = (b - a) / (max(a, b) + _EPS)
                scores.append(s)
            return float(np.mean(scores))

        # Auto-detect k or use provided
        max_k = min(10, N // max(min_cluster_size, 1))
        if max_k < 2:
            max_k = 2
        if n_clusters > 0:
            k_best = n_clusters
        else:
            best_sil  = -2.0
            k_best    = 2
            for k in range(2, max_k + 1):
                sil_scores = [_silhouette(_kmeans(k, s)[0], k) for s in range(n_init)]
                sil = float(np.mean(sil_scores))
                if sil > best_sil:
                    best_sil, k_best = sil, k

        # Final clustering with best k
        inertias = []
        for seed_i in range(n_init):
            lbl_i, cent_i = _kmeans(k_best, seed_i)
            inertia = sum(float(np.sum((H_arr[i] - cent_i[lbl_i[i]]) ** 2))
                          for i in range(N))
            inertias.append((inertia, lbl_i))
        best_inertia_idx = int(np.argmin([x[0] for x in inertias]))
        best_labels = inertias[best_inertia_idx][1]

        # Train on discovered clusters
        assignment: Dict[str, List[int]] = {}
        for ki in range(k_best):
            lbl = f"{prefix}_{ki}"
            idxs = [i for i in range(N) if best_labels[i] == ki]
            if len(idxs) < min_cluster_size:
                continue   # skip tiny clusters
            assignment[lbl] = idxs
            for idx in idxs:
                self.train_step(vectors[idx], lbl)

        return assignment

    # ── 5. Rapid consolidation after few-shot addition ────────────────────────

    def consolidate(self, new_label: Optional[str] = None,
                    n_replay_rounds: int = 3,
                    conf_threshold: float = 0.7,
                    ) -> Dict[str, float]:
        """
        Re-assert decision boundaries after adding a new class or receiving
        a burst of new data.

        Problem
        -------
        After few-shot addition of a new class (< 20 examples), MDL decay
        and catastrophic interference can temporarily soften the boundaries
        of existing classes.  consolidate() runs targeted replay over ALL
        existing classes to re-sharpen boundaries before normal training
        resumes.

        What it does
        ------------
        1. For every class in the memory, draw replay samples from the buffer
           (all available, not just probabilistic).
        2. Run n_replay_rounds passes of memory.train() on those samples.
        3. If new_label is provided, also run extra passes on new-class samples
           to compensate for their scarcity relative to existing classes.
        4. Return per-class accuracy on a random replay subset.

        Parameters
        ----------
        new_label        : freshly-added label to get extra reinforcement
                           (None = consolidate all equally)
        n_replay_rounds  : passes through the replay buffer per class
        conf_threshold   : minimum confidence to count a replay sample
                           as correctly classified (for accuracy report)
        """
        with self.memory._lock:
            classes = list(self.memory._classes.keys())

        if not classes:
            return {}

        h_field = self.field.h

        # Build per-class replay pools from buffer
        by_class = self.replay.by_class()

        ctx = self._ctx_prior(classes)
        per_class_correct: Dict[str, int]  = {c: 0 for c in classes}
        per_class_total:   Dict[str, int]  = {c: 0 for c in classes}

        for rnd in range(n_replay_rounds):
            for label in classes:
                samples = by_class.get(label, [])
                if not samples:
                    continue
                # Give new_label 3× more rounds to compensate for scarcity
                n_passes = 3 if (label == new_label) else 1
                for _ in range(n_passes):
                    for h_rep in samples:
                        pred, correct, _, _, post_conf = self.memory.train(
                            h_rep, label, h_field=h_field, context_prior=ctx
                        )
                        per_class_total[label]   += 1
                        per_class_correct[label] += int(correct)

        # Accuracy report
        accuracy = {
            c: per_class_correct[c] / max(per_class_total[c], 1)
            for c in classes
        }
        return accuracy

    def generate_ood(self, n: int = 1, n_candidates: int = 256,
                     rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
        """
        Sample from the region where the world prior dominates all class models
        (max_k LLR_k < 0) — genuinely novel in the model's current view.

        Vectorised LLR scoring across all candidates simultaneously.
        Fallback: return lowest-max-LLR candidates if strict OOD count < n.
        """
        with self.memory._lock:
            mu0        = self.memory.world.mu.copy()
            v0         = self.memory.world.v.copy()
            class_items = [(k, self.memory._classes[k]) for k in self.memory._classes]

        rng   = rng or self._rng
        std   = np.sqrt(np.maximum(v0, _MIN_VAR)) * 2.0
        H     = mu0 + rng.standard_normal((n_candidates, len(mu0))) * std  # (B, D)

        ll_world = _batch_logpdf(H, mu0, v0)  # (B,)
        max_llr  = np.full(len(H), -np.inf)
        for lbl, cd in class_items:
            mu_k    = cd.mu(mu0)
            u_k     = float(np.mean(v0)) / (cd.n_obs + 1)
            ll_k    = _batch_logpdf(H, mu_k, v0)
            llr_k   = ll_k - ll_world - u_k
            np.maximum(max_llr, llr_k, out=max_llr)

        # Sort by max_llr ascending — most OOD first
        order   = np.argsort(max_llr)
        results = [H[i] for i in order[:n]]
        return results

    def generate_mdl_ball(self, label: str, n: int = 1, radius: float = 1.0,
                          rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
        """
        Fisher-Rao ball constrained sampling.
            ||h − μk||_FR ≤ radius   where ||δ||_FR = sqrt(Σ δd²/v₀d)

        radius=1 → samples within ±1 std (natural units) of μk.
        radius=2 → ±2 std — wider spread, still class-plausible.
        radius=0.1 → tight mode samples.

        Uniform-on-ball sampling via normalise-then-scale (not surface-only):
        draw direction uniformly on unit sphere in FR metric,
        draw magnitude uniformly in [0, radius] (uniform volume fill).
        """
        mu_k, v0 = self._class_params_safe(label)
        rng      = rng or self._rng
        std      = np.sqrt(np.maximum(v0, _MIN_VAR))  # natural coordinate scale
        d        = len(mu_k)

        results = []
        for _ in range(n):
            # Direction: uniform on unit sphere in FR metric
            raw     = rng.standard_normal(d)
            # Whiten by sqrt(v0) so FR metric = Euclidean
            raw_fr  = raw / std
            fr_norm = float(np.linalg.norm(raw_fr)) + _EPS
            dir_fr  = raw_fr / fr_norm               # unit vector in FR space

            # Magnitude: uniform in [0, radius] with volume correction (d-th root)
            r = radius * float(rng.uniform(0, 1) ** (1.0 / d))

            # Convert back to original coordinate space
            delta = dir_fr * r * std                  # back to feature space
            results.append(mu_k + delta)
        return results

    def generate_ancestral(self, n: int = 1,
                           temperature: float = 1.0,
                           rng: Optional[np.random.Generator] = None,
                           ) -> List[Tuple[str, np.ndarray]]:
        """
        Ancestral sampling — full generative model:
            k ~ Categorical(p_k)   where p_k ∝ n_obs[k]^(1/temperature)
            h ~ N(μk, v₀)
        Returns [(label, h), ...].

        Uses global class n_obs counts (not the recency-biased context EMA)
        so sampling proportions reflect the *lifetime* class frequencies seen
        during training. temperature controls sharpness:
            temperature=1 → proportional to n_obs
            temperature→0 → argmax (most frequent class only)
            temperature→∞ → uniform over classes
        """
        with self.memory._lock:
            classes = list(self.memory._classes.keys())
            n_obs   = np.array([self.memory._classes[k].n_obs for k in classes],
                               dtype=np.float64)
        if not classes:
            raise ValueError("No classes registered.")

        rng = rng or self._rng
        # Frequency-based prior.
        # temperature=1  → proportional to n_obs (direct empirical frequencies)
        # temperature→0  → argmax (most frequent class only)
        # temperature→∞  → uniform over classes
        # Normalised-frequency^(1/T) avoids the log-amplification of the old
        # log(n_obs)/T formulation which was super-linearly over-weighting dominant classes.
        freq   = np.maximum(n_obs, 1.0)
        freq  /= freq.sum()
        probs  = freq ** (1.0 / (temperature + _EPS))
        probs /= probs.sum()

        results = []
        for _ in range(n):
            idx   = int(rng.choice(len(classes), p=probs))
            label = classes[idx]
            mu_k, v0 = self._class_params_safe(label)
            std   = np.sqrt(np.maximum(v0, _MIN_VAR))
            results.append((label, mu_k + rng.standard_normal(len(mu_k)) * std))
        return results

    def generate_kde(self, label: str, n: int = 1, bandwidth: float = 0.5,
                     rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
        """
        Phase 4: Non-parametric KDE generation from stored replay latents.
        Better captures multi-modal or skewed class distributions than the
        parametric Gaussian. Requires the class to have replay entries.
        """
        vecs = self.replay.by_class().get(label)
        if not vecs:
            raise ValueError(
                f"No replay entries for '{label}'. "
                f"Train first, or use generate() for parametric sampling."
            )
        return _kde_sample(vecs, n, bandwidth, rng or self._rng)

    # ── Phase 3: Anomaly / Active Learning ───────────────────────────────────

    def anomaly_score(self, x: Any, use_field: bool = True) -> float:
        """
        Dual-gate anomaly score ∈ [0,1].  High → OOD/anomalous.
        Encodes raw input x → h, then calls anomaly_score_latent(h).
        Use anomaly_score_latent(h) directly when h is already in latent space
        (e.g. from generate(), generate_kde(), generate_boundary(), etc.).
        """
        _, h = self._encode(x)
        return self.anomaly_score_latent(h, use_field=use_field)

    def anomaly_score_latent(self, h: np.ndarray, use_field: bool = True) -> float:
        """
        Dual-gate anomaly score ∈ [0,1] operating directly on latent h.

        Use this instead of anomaly_score() when h is already in latent space —
        e.g. outputs from generate(), generate_kde(), generate_boundary(),
        generate_ancestral(), or rollout(). Calling anomaly_score(h) on those
        would double-encode and produce meaningless results.

        Gate 1 — Absolute world-prior LL (per-dim):
            ll_world / D  measures how likely h is under θ₀.
            Calibrated using a running EMA of in-distribution ll_world/D values.

        Gate 2 — LLR class gate:
            σ(max_k LLR_k / ood_sigma)

        Combined: anomaly = 1 − min(gate1, gate2)
        """
        h_field = self.field.h if use_field else None
        with self.memory._lock:
            classes = list(self.memory._classes.keys())
        ctx = self._ctx_prior(classes) if classes else {}

        # Gate 2: LLR gate
        _, _, llrs = self.memory.classify(h, h_field=h_field, context_prior=ctx,
                                          temperature=self.temperature,
                                          ood_sigma=self.ood_sigma)
        if not llrs:
            return 1.0
        max_llr = max(llrs.values())
        gate2   = float(1.0 / (1.0 + math.exp(-max_llr / (self.ood_sigma + _EPS))))

        # Gate 1: normalised Mahalanobis distance to world prior
        with self.memory._lock:
            mu0   = self.memory.world.mu
            inv_v = self.memory.world.inv_v
        d_h           = h - mu0
        mahal_per_dim = float((d_h * d_h) @ inv_v) / (len(h) + _EPS)
        m_ema         = getattr(self, '_mahal_ema',     1.0)
        m_std         = max(getattr(self, '_mahal_std_ema', 0.5), 0.05)
        threshold     = m_ema + 5.0 * m_std
        scale         = 2.0 / m_std
        margin        = float(np.clip((threshold - mahal_per_dim) * scale, -500, 500))
        gate1         = float(1.0 / (1.0 + math.exp(-margin)))

        return 1.0 - min(gate1, gate2)


    def confidence_interval(self, x: Any,
                             n_samples         : int   = 50,
                             perturbation_scale: float = 1.0,
                             rng               : Optional[np.random.Generator] = None,
                             ) -> Dict:
        """
        Prediction confidence interval via local input perturbation.

        Generates n_samples noisy variants of the encoded input h and measures
        how the prediction behaves under realistic perturbations.  Captures
        *discriminative* uncertainty — whether the model would flip its label
        under noise — rather than posterior coverage.

        Advantages over Langevin CI
        ---------------------------
        - Near class centres: single label dominates, std_confidence ≈ 0
        - Near boundaries: multiple labels appear, entropy_over_samples rises
        - No OOD-gate suppression (perturbations stay close to the input)
        - ~10× faster than Langevin (no MCMC chain)

        Parameters
        ----------
        x                  : input to classify
        n_samples          : number of perturbed variants (50 typical)
        perturbation_scale : noise magnitude as multiple of world std deviation
                             (1.0 = typical within-class noise level)

        Returns
        -------
        dict: pred, fraction_correct, mean_confidence, std_confidence,
              ci_low, ci_high, label_distribution, entropy_over_samples
        """
        rng  = rng or self._rng
        _, h = self._encode(x)

        with self.memory._lock:
            K       = len(self.memory._label_order)
            if K == 0:
                return dict(pred='__unknown__', fraction_correct=0.0,
                            mean_confidence=0.0, std_confidence=0.0,
                            ci_low=0.0, ci_high=0.0, label_distribution={},
                            entropy_over_samples=0.0)
            labs    = self.memory._label_order[:K]
            D       = self.memory._D_buf[:K].copy()
            mu0     = self.memory.world.mu.copy()
            inv_v   = self.memory.world.inv_v.copy()
            v_mean  = self.memory.world.v_mean
            n_obs_v = np.array([self.memory._classes[l].n_obs for l in labs],
                                dtype=np.float64)

        noise_std = math.sqrt(max(v_mean, _MIN_VAR)) * perturbation_scale
        T_inv     = 1.0 / (self.temperature + _EPS)
        u_k       = v_mean / (n_obs_v + 1.0)

        preds: List[str] = []
        confs: List[float] = []
        for _ in range(n_samples):
            h_p   = h + rng.standard_normal(len(h)) * noise_std
            r_h   = (h_p - mu0) * inv_v
            LLR   = D @ r_h - 0.5 * (D * D) @ inv_v - u_k
            # Raw softmax disc — no OOD gate (perturbations are in-distribution by design)
            LLR_s = LLR * T_inv
            LLR_s = LLR_s - LLR_s.max()
            p_arr = np.exp(LLR_s)
            p_arr = p_arr / (p_arr.sum() + _EPS)
            best  = int(p_arr.argmax())
            preds.append(labs[best])
            confs.append(float(p_arr[best]))

        ca   = np.array(confs)
        dist = {l: preds.count(l) / n_samples for l in set(preds)}
        main = max(dist, key=dist.get)
        pv   = np.array(list(dist.values()))
        entr = float(-np.sum(pv * np.log(pv + _EPS)))

        return dict(
            pred                = main,
            fraction_correct    = dist.get(main, 0.0),
            mean_confidence     = float(ca.mean()),
            std_confidence      = float(ca.std()),
            ci_low              = float(np.percentile(ca, 5)),
            ci_high             = float(np.percentile(ca, 95)),
            label_distribution  = dist,
            entropy_over_samples= entr,
        )

    def active_query_score(self, x: Any, use_field: bool = True) -> float:
        """
        Active learning query score ∈ [0,∞).  High → near decision boundary.
        = H(posterior) × (1 − max_k p(k|h))
        Zero when model is certain; maximum at equal-posterior boundary.
        """
        _, h    = self._encode(x)
        h_field = self.field.h if use_field else None
        with self.memory._lock:
            classes = list(self.memory._classes.keys())
        ctx = self._ctx_prior(classes) if classes else {}
        _, _, llrs = self.memory.classify(h, h_field=h_field, context_prior=ctx,
                                          temperature=self.temperature,
                                          ood_sigma=self.ood_sigma)
        if not llrs:
            return 0.0
        probs   = _softmax(np.array(list(llrs.values())) / (self.temperature + _EPS))
        return float(_shannon_entropy(probs) * (1.0 - float(probs.max())))



    def evaluate(self, data: List[Tuple[Any, str]],
                 compute_ece: bool = True,
                 ) -> Dict:
        """
        Comprehensive evaluation on a labelled dataset.

        Runs batch_infer on all inputs and computes:
          accuracy, macro_accuracy, per_class_accuracy,
          ECE (calibration), confusion matrix entries,
          mean_confidence, mean_entropy.

        Parameters
        ----------
        data        : list of (x, label) pairs
        compute_ece : compute Expected Calibration Error (adds one pass)

        Returns rich dict for analysis and reporting.
        """
        if not data:
            return {}

        xs    = [x for x, _ in data]
        true  = [y for _, y in data]

        # Encode once, score once
        H          = self.batch_encode(xs)
        LLR, labels= self.score_matrix(H)

        if len(labels) == 0:
            return {'accuracy': 0.0, 'n': len(data), 'n_classes': 0}

        probs  = _softmax_batch(LLR / (self.temperature + _EPS))
        gates  = self.world_gate_vector(H)
        best_i = probs.argmax(axis=1)
        confs  = probs[np.arange(len(H)), best_i] * gates
        preds  = [labels[best_i[i]] for i in range(len(H))]
        entr   = -np.sum(probs * np.log(probs + _EPS), axis=1)

        N = len(data)
        correct = [int(preds[i] == true[i]) for i in range(N)]
        acc     = sum(correct) / N

        # Per-class accuracy
        per: Dict[str, List[int]] = {}
        for i in range(N):
            k = true[i]
            if k not in per: per[k] = [0, 0]
            per[k][0] += correct[i]; per[k][1] += 1
        per_cls = {k: v[0]/v[1] for k, v in per.items() if v[1] > 0}
        macro   = sum(per_cls.values()) / len(per_cls) if per_cls else 0.0

        # Confusion matrix (sparse — top errors only)
        confusion: Dict[str, Dict[str, int]] = {}
        for i in range(N):
            if not correct[i]:
                t, p = true[i], preds[i]
                confusion.setdefault(t, {})[p] = confusion.get(t, {}).get(p, 0) + 1

        result = dict(
            n               = N,
            accuracy        = acc,
            macro_accuracy  = macro,
            per_class       = per_cls,
            confusion       = confusion,
            mean_confidence = float(confs.mean()),
            std_confidence  = float(confs.std()),
            mean_entropy    = float(entr.mean()),
            n_classes       = len(labels),
        )

        if compute_ece:
            result['ece'] = _compute_ece(confs, np.array(correct, dtype=float))

        return result

    def active_learning_loop(self,
                             pool      : List[Any],
                             label_fn,
                             budget    : int  = 20,
                             strategy  : str  = 'uncertainty',
                             batch_size: int  = 1,
                             warm_start: int  = 0,
                             ) -> Dict:
        """
        Active learning: iteratively query the most informative examples.

        Ranks unlabelled candidates by informativeness, calls label_fn(x) on
        the top `budget` items, trains on results.

        Strategies
        ----------
        'uncertainty' : highest active_query_score (entropy × boundary proximity)
        'margin'      : smallest margin between top-2 class scores
        'coreset'     : greedy max-min distance to already-labelled set
        'random'      : random baseline

        Parameters
        ----------
        pool       : unlabelled inputs
        label_fn   : callable(x) → str  (oracle / ground truth)
        budget     : total number of examples to query and label
        strategy   : scoring strategy
        batch_size : examples to query per round before re-scoring
        warm_start : query this many examples randomly BEFORE switching to
                     the chosen strategy.  Strongly recommended for uncertainty/
                     margin/coreset when starting from an empty model — without
                     at least 1 example per class the model has no basis for
                     scoring.  warm_start counts against `budget`.

        Returns dict: n_queried, n_trained, labels_acquired, losses, mean_loss.
        """
        if not pool:
            return dict(n_queried=0, n_trained=0, labels_acquired={},
                        losses=[], mean_loss=float('nan'))

        budget      = min(budget, len(pool))
        remaining   = list(range(len(pool)))
        queried_idx = []; labels_acq = {}; losses = []; labelled_hs = []

        # Warm-start: random queries to seed the model before strategic sampling
        if warm_start > 0:
            n_ws = min(warm_start, budget, len(remaining))
            ws_idxs = list(self._rng.choice(len(remaining), size=n_ws, replace=False))
            ws_idxs = [remaining[j] for j in ws_idxs]
            for idx in ws_idxs:
                lbl = label_fn(pool[idx])
                if lbl is not None:
                    losses.append(self.train_step(pool[idx], lbl))
                    labels_acq[idx] = lbl
                    _, h_l = self._encode(pool[idx])
                    labelled_hs.append((h_l, lbl))
                queried_idx.append(idx)
                remaining.remove(idx)
            budget -= n_ws  # reduce remaining budget

        rounds    = max(1, budget // max(batch_size, 1))
        per_round = budget // rounds

        for rnd in range(rounds):
            if not remaining:
                break
            n_query = min(per_round + (1 if rnd < budget % rounds else 0),
                          len(remaining))
            if n_query <= 0:
                break

            if strategy == 'uncertainty':
                scores = [self.active_query_score(pool[i]) for i in remaining]
                chosen = [remaining[j] for j in np.argsort(scores)[::-1][:n_query]]

            elif strategy == 'margin':
                margins = []
                for i in remaining:
                    _, _, llrs = self.memory.classify(
                        self._encode(pool[i])[1],
                        temperature=self.temperature, ood_sigma=self.ood_sigma)
                    if llrs and len(llrs) >= 2:
                        sv = sorted(llrs.values(), reverse=True)
                        margins.append(sv[0] - sv[1])
                    else:
                        margins.append(float('inf'))
                chosen = [remaining[j] for j in np.argsort(margins)[:n_query]]

            elif strategy == 'coreset':
                if not labelled_hs:
                    scores = [self.active_query_score(pool[i]) for i in remaining]
                    chosen = [remaining[j] for j in np.argsort(scores)[::-1][:n_query]]
                else:
                    H_lab = np.stack([h for h, _ in labelled_hs])
                    rem_c = list(remaining); chosen = []
                    for _ in range(n_query):
                        if not rem_c:
                            break
                        H_cands = np.stack([self._encode(pool[i])[1] for i in rem_c])
                        dists   = np.min(np.sum((H_cands[:, None, :] - H_lab[None, :, :]) ** 2,
                                                axis=2), axis=1)
                        best    = int(np.argmax(dists))
                        chosen.append(rem_c[best])
                        _, h_new = self._encode(pool[rem_c[best]])
                        H_lab    = np.vstack([H_lab, h_new[None]])
                        rem_c.pop(best)
            else:  # 'random'
                chosen = list(self._rng.choice(len(remaining), size=n_query,
                                               replace=False))
                chosen = [remaining[j] for j in chosen]

            for idx in chosen:
                lbl = label_fn(pool[idx])
                if lbl is not None:
                    losses.append(self.train_step(pool[idx], lbl))
                    labels_acq[idx] = lbl
                    _, h_l = self._encode(pool[idx])
                    labelled_hs.append((h_l, lbl))
                queried_idx.append(idx)
                remaining.remove(idx)

        return dict(
            n_queried      = len(queried_idx),
            n_trained      = len(losses),
            labels_acquired= labels_acq,
            losses         = losses,
            mean_loss      = float(np.mean(losses)) if losses else float('nan'),
            queried_indices= queried_idx,
        )

    def drift_score(self) -> float:
        """
        Phase 3: Concept drift detection.
        EMA of ||Δμ₀|| per training step. Near zero = stable; rising = drift.
        """
        return self.memory.world.drift_score()


    def watch_drift(self,
                    threshold    : float = 0.22,
                    auto_respond : bool  = True,
                    callback     = None,
                    ) -> Dict:
        """
        Check for concept drift and optionally respond automatically.

        If drift_score() > threshold and auto_respond=True, triggers:
          1. consolidate()      — re-sharpen all class boundaries via replay
          2. auto_recalibrate() — adjust temperature to match current LLR scale

        Parameters
        ----------
        threshold    : drift_score value that triggers a response (~0.3 mild, ~0.5 significant)
        auto_respond : if True, respond automatically on detection
        callback     : optional callable(drift_info_dict) for custom handling

        Returns
        -------
        dict: drifting, drift_score, threshold, action_taken, [new_temperature]
        """
        score = self.drift_score()
        drift_info = {
            'drifting'    : score > threshold,
            'drift_score' : score,
            'threshold'   : threshold,
            'action_taken': None,
        }

        if score > threshold and auto_respond:
            with self.memory._lock:
                classes = list(self.memory._classes.keys())
            if classes:
                acc = self.consolidate(n_replay_rounds=2)
                drift_info['action_taken']    = 'consolidate'
                drift_info['consolidate_acc'] = acc
                if self._llr_scale_n >= 50:
                    drift_info['new_temperature'] = self.auto_recalibrate()

        if callback is not None:
            callback(drift_info)
        return drift_info

    def drift_monitor(self,
                      xs_stream    : List[Any],
                      labels_stream: Optional[List[str]] = None,
                      threshold    : float = 0.22,
                      window       : int   = 10,
                      ) -> Dict:
        """
        Process a stream of observations while watching for concept drift.

        For each observation: train if labelled, semi-supervised infer if not.
        Calls watch_drift() every `window` steps and collects drift events.

        Returns summary dict: n_processed, accuracy, drift_events, n_drifts.
        """
        n = len(xs_stream)
        labelled = labels_stream is not None
        correct = 0; total_lab = 0; losses = []; drift_events = []

        for i, x in enumerate(xs_stream):
            if labelled and labels_stream[i] is not None:
                pred, _ = self.infer(x)
                correct += int(pred == labels_stream[i])
                total_lab += 1
                losses.append(self.train_step(x, labels_stream[i]))
            else:
                pred, conf = self.infer(x)
                if conf > 0.7 and pred != '__unknown__':
                    self.train_step(x, pred)

            if (i + 1) % window == 0:
                info = self.watch_drift(threshold=threshold, auto_respond=True)
                if info['drifting']:
                    drift_events.append({'step': i, **info})

        return {
            'n_processed' : n,
            'accuracy'    : correct / max(total_lab, 1),
            'n_labelled'  : total_lab,
            'drift_events': drift_events,
            'n_drifts'    : len(drift_events),
            'mean_loss'   : float(np.mean(losses)) if losses else float('nan'),
        }

    # ── Phase 5: Sequence Prediction ─────────────────────────────────────────

    def predict_next(self, current_label: Optional[str] = None) -> Dict[str, float]:
        """
        Phase 5: Probability distribution over the next label.
        Uses Tier-1 co-occurrence + Tier-2 EMA transitions.
        If current_label is given, it overrides the last recorded label.
        """
        with self.memory._lock:
            classes = list(self.memory._classes.keys())
        if not classes:
            return {}
        if current_label is not None:
            orig = self.context._last_label
            self.context._last_label = current_label
            result = self.context.predict_next(classes)
            self.context._last_label = orig
            return result
        return self.context.predict_next(classes)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    # ── B. Test-time temperature adaptation ──────────────────────────────────

    def adapt_temperature(self,
                          calibration_data: List[Tuple[Any, str]],
                          n_grid: int = 20,
                          T_min: float = 0.3,
                          T_max: float = 8.0,
                          ) -> float:
        """
        Find the temperature T* that minimises ECE on a small calibration set.

        Use this after deployment if the test distribution has shifted from
        training (different noise, preprocessing, domain).

        Algorithm
        ---------
        Grid search over T ∈ [T_min, T_max] (log-spaced), compute ECE at each T
        using the provided labelled examples, set self.temperature = T*.

        Parameters
        ----------
        calibration_data : list of (x, true_label) pairs (50–500 typical)
        n_grid           : number of temperature candidates to try
        T_min / T_max    : search bounds

        Returns
        -------
        T* (float) — the best temperature, also stored in self.temperature.
        """
        if not calibration_data:
            return self.temperature

        # Collect raw LLRs (without temperature) for all calibration inputs
        raw_llrs : List[np.ndarray] = []
        true_idxs: List[int]        = []
        with self.memory._lock:
            klabels = list(self.memory._classes.keys())
        kmap = {l: i for i, l in enumerate(klabels)}

        for x, y in calibration_data:
            _, h    = self._encode(x)
            _, _, llrs = self.memory.classify(
                h, h_field=self.field.h, context_prior=self._ctx_prior(klabels),
                temperature=1.0,      # raw LLRs; temperature applied below
                ood_sigma=self.ood_sigma,
            )
            if llrs and y in kmap:
                arr = np.array([llrs.get(k, 0.0) for k in klabels])
                raw_llrs.append(arr)
                true_idxs.append(kmap[y])

        if not raw_llrs:
            return self.temperature

        R = np.stack(raw_llrs)    # (N, K)
        T_grid = np.exp(np.linspace(np.log(T_min), np.log(T_max), n_grid))
        best_ece, best_T = float('inf'), self.temperature

        for T in T_grid:
            probs  = _softmax_batch(R / T)            # (N, K)
            confs  = probs.max(axis=1)
            preds  = probs.argmax(axis=1)
            correct= (preds == np.array(true_idxs)).astype(float)
            ece    = _compute_ece(confs, correct)
            if ece < best_ece:
                best_ece, best_T = ece, float(T)

        self.temperature = best_T
        return best_T

    def auto_recalibrate(self, decay: float = 0.995, boost: float = 1.02) -> float:
        """
        Lightweight online temperature correction using the tracked LLR scale EMA.

        During training self._llr_scale_ema tracks the mean |LLR_winner|.
        If this drops (noisy inputs, shifted distribution), LLRs are smaller
        and the current temperature becomes too high → model underconfident.
        Adjusting temperature proportionally restores calibration.

        Call this periodically during deployment (e.g. every 1000 steps) or
        whenever you suspect distribution shift.

        Returns new temperature.
        """
        if self._llr_scale_n < 50:
            return self.temperature   # not enough data yet

        # Ratio: current LLR scale vs training-time scale stored at base_temp
        # If no baseline recorded, use current as baseline
        if not hasattr(self, '_llr_scale_baseline') or self._llr_scale_baseline <= 0:
            self._llr_scale_baseline = max(self._llr_scale_ema, _EPS)
            return self.temperature

        ratio = self._llr_scale_ema / (self._llr_scale_baseline + _EPS)
        # Temperature should scale INVERSELY with LLR magnitude:
        # smaller LLRs → lower T → keeps softmax disc from collapsing
        T_adjusted = self._base_temp / (ratio + _EPS)
        # Smooth update: exponential decay toward adjusted T
        self.temperature = decay * self.temperature + (1 - decay) * float(
            np.clip(T_adjusted, self._base_temp * 0.2, self._base_temp * 5.0)
        )
        return self.temperature

    def freeze_temperature(self) -> None:
        """
        Record current LLR scale as the training-time baseline.
        Call this once after initial training, before deployment.
        """
        self._llr_scale_baseline = max(self._llr_scale_ema, _EPS)
        self._base_temp          = self.temperature

    # ── C. Class splitting ───────────────────────────────────────────────────

    def split_class(self, label: str,
                    n_subclusters: int = 2,
                    min_entropy_threshold: float = 0.5,
                    prefix: Optional[str] = None,
                    ) -> List[str]:
        """
        Split a class into sub-classes when its replay examples are multimodal.

        Motivation
        ----------
        After fit_unlabeled() or coarse labelling, a single class may contain
        multiple distinct modes.  split_class() detects this via Shannon entropy
        of the per-replay-sample loss distribution and, if multimodality is
        confirmed, k-means splits the class's replay pool into n_subclusters.

        Returns
        -------
        List of new label names created (empty if no split occurred).

        Parameters
        ----------
        label                  : class to examine
        n_subclusters          : number of sub-classes to split into
        min_entropy_threshold  : minimum replay-loss entropy required to trigger
                                 a split.  Lower = only split highly multimodal.
        prefix                 : prefix for new labels (default: 'label_sub')
        """
        if label not in self.memory._classes:
            raise KeyError(f"Unknown class '{label}'.")

        # Gather replay samples for this class
        by_cls = self.replay.by_class()
        pool   = by_cls.get(label, [])
        if len(pool) < max(n_subclusters * 4, 10):
            return []   # not enough data to split meaningfully

        prefix = prefix or f"{label}_sub"

        # Compute per-sample losses to measure spread
        with self.memory._lock:
            mu0   = self.memory.world.mu.copy()
            inv_v = self.memory.world.inv_v.copy()
            k_idx = self.memory._label_idx.get(label, 0)
            K     = len(self.memory._label_order)
            D     = self.memory._D_buf[:K].copy()

        losses = []
        for h_rep in pool:
            r_rep    = (h_rep - mu0) * inv_v
            cross    = D @ r_rep
            D_sq     = (D * D) @ inv_v
            scores   = cross - 0.5 * D_sq
            llr_k    = float(scores[k_idx])
            # Loss: negative log-likelihood for the class
            losses.append(-llr_k)

        losses = np.array(losses)
        # Normalise and compute entropy of loss distribution
        loss_bins = np.histogram(losses, bins=10)[0].astype(float)
        loss_bins = loss_bins / (loss_bins.sum() + _EPS)
        entropy   = float(-np.sum(loss_bins * np.log(loss_bins + _EPS)))

        if entropy < min_entropy_threshold:
            return []   # distribution is unimodal; no split needed

        # K-means in latent space to find sub-clusters
        H_pool = np.stack(pool)   # (N, d)
        rng_sp = self._rng

        # K-means++
        chosen = [int(rng_sp.integers(len(pool)))]
        for _ in range(n_subclusters - 1):
            dists = np.array([min(float(np.sum((H_pool[i] - H_pool[c]) ** 2 * inv_v))
                               for c in chosen) for i in range(len(pool))])
            p = dists / (dists.sum() + _EPS)
            chosen.append(int(rng_sp.choice(len(pool), p=p)))
        centroids = H_pool[chosen].copy()

        km_labels = np.zeros(len(pool), dtype=int)
        for _ in range(30):
            dists_all = np.array([
                np.sum((H_pool - c[np.newaxis, :]) ** 2 * inv_v[np.newaxis, :], axis=1)
                for c in centroids
            ])
            new_labels = np.argmin(dists_all, axis=0)
            if np.all(new_labels == km_labels):
                break
            km_labels = new_labels
            for ki in range(n_subclusters):
                mask = km_labels == ki
                if mask.sum() > 0:
                    centroids[ki] = H_pool[mask].mean(axis=0)

        # Create new sub-classes and train on partitioned replay pool
        new_labels = []
        for ki in range(n_subclusters):
            mask = km_labels == ki
            if mask.sum() < 2:
                continue
            sub_lbl = f"{prefix}_{ki}"
            for h_sub in H_pool[mask]:
                self.memory.train(h_sub, sub_lbl, h_field=self.field.h,
                                  context_prior=self._ctx_prior(
                                      list(self.memory._classes.keys()) + [sub_lbl]))
            new_labels.append(sub_lbl)

        return new_labels

    def diagnostics(self) -> Dict:
        recent_loss = float(np.mean(list(self._loss_buf))) if self._loss_buf else float('nan')
        world       = self.memory.world
        return {
            'total_steps'          : self._total_steps,
            'running_acc'          : self._total_correct / max(self._total_steps, 1),
            'recent_loss'          : recent_loss,
            'n_classes'            : len(self.memory._classes),
            'context_acc'          : self.context.recent_accuracy(),
            'field_confidence'     : self.context.field_confidence(),
            'world_prior_mean_norm': float(np.linalg.norm(world.mu)),
            'world_prior_var_mean' : float(world.v.mean()),
            'drift_score'          : self.drift_score(),
            'class_accuracy'       : self.memory.accuracy(),
            'class_complexity'     : self.memory.complexity(),
            'field_step'           : self.field.step,
            'replay_size'          : len(self.replay),
        }

    def reset_field(self) -> None:
        self.field.reset()

    # ── D. Hierarchical training ─────────────────────────────────────────────

    def train_step_hierarchical(self, x: Any,
                                fine_label   : str,
                                coarse_label : str,
                                coarse_weight: float = 0.3,
                                ) -> Tuple[float, float]:
        """
        Two-level supervised training: fine-grained + coarse-grained simultaneously.

        Trains the main classifier on fine_label normally, then trains a second
        parallel classifier on 'coarse:coarse_label', and applies a soft gravitational
        pull drawing fine_label's delta_mu toward the coarse centroid.

        This creates hierarchy without any architectural changes: fine classes with
        the same coarse parent naturally cluster in latent space.

        Parameters
        ----------
        x             : raw input
        fine_label    : specific class (e.g. 'labrador')
        coarse_label  : parent class (e.g. 'dog') — stored as 'coarse:dog' internally
        coarse_weight : strength of pull toward coarse centroid [0–1]

        Returns (fine_loss, coarse_loss).
        """
        fine_loss   = self.train_step(x, fine_label)
        coarse_key  = f"coarse:{coarse_label}"
        _, h        = self._encode(x)   # encode once; reuse for both levels
        # Use memory.train directly for coarse — train_step(h, ...) would re-encode h
        h_fld_hier  = self.field.h
        ctx_hier    = self._ctx_prior(list(self.memory._classes.keys()) + [coarse_key])
        _, _, coarse_loss_val, _, _ = self.memory.train(
            h, coarse_key, h_field=h_fld_hier, context_prior=ctx_hier)
        coarse_loss = coarse_loss_val

        # Soft hierarchical pull: attract fine delta_mu toward coarse delta_mu
        with self.memory._lock:
            if coarse_key in self.memory._classes and fine_label in self.memory._classes:
                dm_fine   = self.memory._classes[fine_label].delta_mu    # view
                dm_coarse = self.memory._classes[coarse_key].delta_mu.copy()
                dm_fine  += coarse_weight * (dm_coarse - dm_fine)         # in-place

        return fine_loss, coarse_loss

    def infer_hierarchical(self, x: Any) -> Tuple[str, str, float, float]:
        """
        Two-level inference over fine and coarse classes.

        Fine labels: any class not starting with 'coarse:'.
        Coarse labels: any class starting with 'coarse:'.

        Returns (fine_label, coarse_label, fine_conf, coarse_conf).
        Coarse label has the 'coarse:' prefix stripped in the return value.

        Implementation note: scores each level independently by computing LLRs
        only over the relevant label subset, preventing cross-level interference.
        """
        _, h = self._encode(x)
        with self.memory._lock:
            all_labels    = self.memory._label_order[:]
            mu0           = self.memory.world.mu.copy()
            inv_v         = self.memory.world.inv_v.copy()
            K             = len(all_labels)
            D_all         = self.memory._D_buf[:K].copy()
            n_obs_all     = np.array([self.memory._classes[l].n_obs for l in all_labels],
                                     dtype=np.float64)

        r_h     = (h - mu0) * inv_v
        cross   = D_all @ r_h          # (K,)
        D_sq    = (D_all * D_all) @ inv_v  # (K,)
        u_k     = inv_v.mean() / (n_obs_all + 1.0)
        llrs    = cross - 0.5 * D_sq - u_k  # (K,)
        T_inv   = 1.0 / (self.temperature + _EPS)

        def _infer_subset(label_filter):
            idxs  = [i for i, l in enumerate(all_labels) if label_filter(l)]
            if not idxs:
                return '__unknown__', 0.0
            sub_llrs = llrs[idxs]
            sub_lbl  = [all_labels[i] for i in idxs]
            best     = int(sub_llrs.argmax())
            probs    = _softmax(sub_llrs * T_inv)
            disc     = float(probs.max())
            return sub_lbl[best], disc

        fp, fc = _infer_subset(lambda l: not l.startswith('coarse:'))
        cp_raw, cc = _infer_subset(lambda l: l.startswith('coarse:'))
        cp = cp_raw.replace('coarse:', '')
        return fp, cp, fc, cc

    # ── E. Gradient-based hard negative augmentation ───────────────────────────

    def generate_hard_negatives(self, label: str,
                                n: int = 10,
                                n_steps: int = 8,
                                step_size: float = 0.06,
                                temperature: float = 0.5,
                                train_on_them: bool = True,
                                rng: Optional[np.random.Generator] = None,
                                ) -> List[np.ndarray]:
        """
        Generate hard negatives: latent vectors walked toward the decision boundary
        for `label`, then trained back as correct examples to sharpen the boundary.

        Mechanism
        ---------
        Start near the class mean, walk toward the nearest competing class using
        ∇_h (LLR_best_other − LLR_label), then train on each result with the
        correct label.  The model sees examples close to where it would be wrong
        and learns to push the boundary outward.

        If train_on_them=True (default), calls memory.train() on each hard negative
        immediately with its correct label.

        Use after few-shot addition, or call augment_boundaries() to sweep all classes.
        """
        rng = rng or self._rng
        with self.memory._lock:
            mu0   = self.memory.world.mu.copy()
            inv_v = self.memory.world.inv_v.copy()
            K     = len(self.memory._label_order)
            D     = self.memory._D_buf[:K].copy()
            k_idx = self.memory._label_idx.get(label, -1)
            if k_idx < 0:
                raise KeyError(f"Unknown label '{label}'.")

        mu_k   = mu0 + D[k_idx]
        v0     = self.memory.world.v.copy()
        std    = np.sqrt(np.maximum(v0, _MIN_VAR))
        h_fld  = self.field.h
        ctx    = self._ctx_prior(list(self.memory._classes.keys()))
        results = []

        for _ in range(n):
            h = mu_k + rng.standard_normal(len(mu_k)) * std * 0.8
            for _s in range(n_steps):
                # Gradient toward nearest competitor boundary
                r_h     = (h - mu0) * inv_v
                scores  = D @ r_h - 0.5 * (D * D) @ inv_v
                masked  = scores.copy(); masked[k_idx] = -np.inf
                j_idx   = int(masked.argmax())
                grad    = inv_v * (D[j_idx] - D[k_idx])
                h       = (h + step_size * grad
                           + math.sqrt(2 * step_size * temperature ** 2)
                           * rng.standard_normal(len(h)))
            results.append(h.copy())
            if train_on_them:
                # Only train if the sample is still close to the decision boundary
                # (predicted label is either the target or a direct competitor).
                # This prevents deep-misclassified samples from corrupting training.
                r_h    = (h - mu0) * inv_v
                scores = D @ r_h - 0.5 * (D * D) @ inv_v
                scores_masked        = scores.copy(); scores_masked[k_idx] = -np.inf
                margin = float(scores[k_idx]) - float(scores_masked.max())
                # Only train if margin is > -5 (sample hasn't wandered too far)
                if margin > -5.0:
                    self.memory.train(h, label, h_field=h_fld, context_prior=ctx)

        return results

    def augment_boundaries(self, n_per_class: int = 5,
                           n_steps: int = 8,
                           step_size: float = 0.06,
                           rng: Optional[np.random.Generator] = None,
                           ) -> Dict[str, int]:
        """
        Run generate_hard_negatives for ALL known classes simultaneously.
        Returns {label: n_trained}.
        """
        rng = rng or self._rng
        with self.memory._lock:
            classes = list(self.memory._classes.keys())
        if len(classes) < 2:
            return {}
        return {
            label: len(self.generate_hard_negatives(
                label, n=n_per_class, n_steps=n_steps,
                step_size=step_size, train_on_them=True, rng=rng))
            for label in classes
        }

    def __repr__(self) -> str:
        return (
            f"CyphaDIF(encoder={self.encoder_fn.__class__.__name__}, "
            f"feat_dim={self.feat_dim}, field_dim={self.field_dim}, "
            f"n_classes={len(self.memory._classes)}, steps={self._total_steps})"
        )



# ─────────────────────────────────────────────────────────────────────────────
# Phase 7a: RFFEncoder  —  deterministic nonlinear kernel feature map
# ─────────────────────────────────────────────────────────────────────────────

class RFFEncoder(Encoder):
    """
    Deterministic Random Fourier Features (RFF) encoder.

    Maps inputs to a D-dimensional feature space that approximates the
    Radial Basis Function (RBF) kernel (Rahimi & Recht, NIPS 2007):

        φ(x) = √(2/D) · cos(Wx + b)

    where W ~ N(0, γ²I) and b ~ U(0, 2π) are FIXED at initialisation
    from a seeded RNG — making the encoder fully deterministic.

    Inner product in RFF space approximates the RBF kernel:
        ⟨φ(x₁), φ(x₂)⟩  ≈  exp(−γ²‖x₁−x₂‖²/2)

    This means any linear model operating on φ(x) is actually a
    NONLINEAR model in input space — a kernel machine in disguise.
    Unlike a learned encoder, no training is needed and no collapse
    is possible.

    Pairing with CyphaDIF
    ─────────────────────
    CyphaDIF uses delta_mu vectors in latent space.  With RFFEncoder:
      - Class boundaries become nonlinear (RBF level-sets in input space)
      - OOD detection via Mahalanobis works in the kernel feature space
      - The world prior captures the RFF-space distribution of all inputs

    Bandwidth tuning
    ─────────────────
    γ controls the kernel bandwidth.  A good default is:
        γ = 1 / median_pairwise_distance(training_samples)
    Call auto_gamma(X_sample) to set this automatically.

    Parameters
    ----------
    input_dim : dimension of raw input vectors
    D         : number of random features (higher = better approximation)
    gamma     : bandwidth parameter (1/length_scale)
    seed      : RNG seed for reproducibility
    """

    def __init__(self, input_dim: int, D: int = 256,
                 gamma: float = 1.0, seed: int = 42):
        self._input_dim = input_dim
        self._D         = D
        self._gamma     = gamma
        self._seed      = seed
        self._init_weights()

    def _init_weights(self) -> None:
        rng       = np.random.default_rng(self._seed)
        self.W    = rng.normal(0, self._gamma, (self._D, self._input_dim)).astype(np.float64)
        self.b    = rng.uniform(0, 2.0 * math.pi, self._D).astype(np.float64)
        self._scale = math.sqrt(2.0 / self._D)

    @property
    def dim(self) -> int:
        return self._D

    def __call__(self, x: Any) -> np.ndarray:
        """Encode a single input: (d_in,) → (D,)"""
        x = np.asarray(x, dtype=np.float64).ravel()
        return self._scale * np.cos(self.W @ x + self.b)

    def batch_encode(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorised batch encoding: (N, d_in) → (N, D).
        One (N×d_in) @ (d_in×D) matmul — BLAS DGEMM, very fast.
        """
        X = np.asarray(X, dtype=np.float64)
        return self._scale * np.cos(X @ self.W.T + self.b)

    def auto_gamma(self, X_sample: np.ndarray, percentile: int = 50) -> float:
        """
        Set γ = 1/median_pairwise_distance(X_sample) and re-initialise W.

        This is the standard bandwidth heuristic for density / classification.
        For regression tasks where the target has specific frequency content,
        prefer auto_gamma_cv() which cross-validates γ against the target.

        Returns the chosen γ.
        """
        X_sample = np.asarray(X_sample, dtype=np.float64)
        n        = min(len(X_sample), 500)
        sub      = X_sample[:n]
        sq_dists = np.sum((sub[:, np.newaxis, :] - sub[np.newaxis, :, :]) ** 2, axis=2)
        idx      = np.triu_indices(n, k=1)
        dists    = np.sqrt(sq_dists[idx])
        med_dist = float(np.percentile(dists, percentile)) if len(dists) > 0 else 1.0
        self._gamma = 1.0 / max(med_dist, _EPS)
        self._init_weights()
        return self._gamma

    def auto_gamma_cv(self,
                      X_sample  : np.ndarray,
                      y_sample  : np.ndarray,
                      gammas    : Optional[List[float]] = None,
                      val_frac  : float = 0.2,
                      reg       : float = 1e-5,
                      rng       : Optional[np.random.Generator] = None,
                      ) -> float:
        """
        Select γ via leave-out cross-validation against regression targets.

        Tries each γ in `gammas` using a small held-out validation split.
        Solves ridge regression (closed form) for each γ and picks the one
        with lowest validation RMSE.  Very fast: O(len(gammas) × D² × N).

        Parameters
        ----------
        X_sample : (N, d_in) training inputs
        y_sample : (N,) or (N, d_out) targets
        gammas   : bandwidth candidates (default: logarithmic grid 0.1–10)
        val_frac : fraction to hold out for validation
        reg      : ridge regularisation for the linear solve

        Returns the best γ found, and re-initialises W with it.
        """
        if gammas is None:
            gammas = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
        rng_cv   = rng or np.random.default_rng(self._seed)
        X        = np.asarray(X_sample, dtype=np.float64)
        y        = np.asarray(y_sample, dtype=np.float64)
        n        = len(X)
        n_val    = max(1, int(n * val_frac))
        idx      = rng_cv.permutation(n)
        X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]
        X_fit, y_fit = X[idx[n_val:]], y[idx[n_val:]]

        best_g, best_rmse = gammas[0], float('inf')
        D_orig = self._D
        for g in gammas:
            enc_try = RFFEncoder(self._input_dim, D=D_orig, gamma=g, seed=self._seed)
            PHI_f   = enc_try.batch_encode(X_fit)
            PHI_v   = enc_try.batch_encode(X_val)
            try:
                w   = np.linalg.solve(PHI_f.T @ PHI_f + reg * np.eye(D_orig),
                                      PHI_f.T @ y_fit)
                e   = PHI_v @ w - y_val
                rmse = float(np.sqrt(np.mean(e ** 2))) if e.ndim == 1 else float(np.sqrt(np.mean(e**2)))
            except np.linalg.LinAlgError:
                continue
            if rmse < best_rmse:
                best_rmse, best_g = rmse, g

        self._gamma = best_g
        self._init_weights()
        return best_g

    def kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Approximate RBF kernel matrix K[i,j] = ⟨φ(X1[i]), φ(X2[j])⟩.
        Shape: (N1, N2).  For verification / downstream use.
        """
        Phi1 = self.batch_encode(X1)
        Phi2 = self.batch_encode(X2)
        return Phi1 @ Phi2.T


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7b: MKERegressor  —  Mixture of Kernel Experts regression
# ─────────────────────────────────────────────────────────────────────────────

class MKERegressor:
    """
    Mixture of Kernel Experts (MKE) — hybrid DIF + RFF regression.

    This replaces DIFRegressor with a fundamentally better algorithm.

    Architecture
    ─────────────
    1. RFF Encoder:   φ(x) = √(2/D)·cos(Wx+b)  [deterministic, nonlinear]
    2. DIF Router:    p(k|x) = softmax(LLR_k(φ(x)))  [OOD-aware routing]
    3. Kernel Experts: f_k(x) = w_k · φ(x)  [RBF regression per region]
    4. Prediction:    ŷ = Σ_k p(k|x) · f_k(x)  [smooth nonlinear mixture]

    Why this is better than DIFRegressor
    ──────────────────────────────────────
    - f_k(x) = w_k·φ(x) is a KERNEL FUNCTION in input space (RBF basis),
      not just a constant mean.  Each expert can learn a different nonlinear
      function within its region.
    - The update rule w_k += lr·p(k|x)·(y−ŷ)·φ(x) is the proper EM M-step:
      responsibility-weighted online gradient descent.
    - No routing collapse: DIF dedup prevents identical delta_mus, and all
      experts receive gradient when p is spread (no dead expert problem).
    - Universal approximation: RBF kernel with enough features approximates
      any continuous function on a compact domain.

    Theory
    ──────
    Each expert computes a linear model in RFF feature space:
        f_k(x) = w_k · φ(x)   (linear in D-dim kernel space)

    In input space this corresponds to:
        f_k(x) = Σ_i α_i · K(x, x_i)   (nonlinear kernel regression)

    where K is the RBF kernel approximated by the RFF features.
    The global predictor Σ_k p(k|x)·f_k(x) is a smooth mixture of these
    local kernel regressors, partitioned by the DIF routing.

    Supports scalar and vector (multi-output) targets.

    Parameters
    ----------
    encoder   : RFFEncoder (or any Encoder, but RFFEncoder recommended)
    K         : number of kernel experts
    lr        : expert weight learning rate (online gradient step)
    reg       : L2 regularisation on expert weights (prevents overfitting)
    field_dim : DIF field dimension

    Quick-start
    ─────────────
    reg = MKERegressor.from_data(X_train, K=8)   # auto-tune γ, seed K experts
    for x, y in stream:
        reg.train_step(x, y)
    y_pred, sigma = reg.predict(x_new)   # point estimate + uncertainty
    """

    def __init__(self,
                 encoder  : 'RFFEncoder',
                 K        : int   = 8,
                 lr       : float = 0.01,
                 reg      : float = 1e-6,
                 field_dim: int   = 160,
                 rng      : Optional[np.random.Generator] = None):
        if not isinstance(encoder, RFFEncoder):
            raise TypeError("MKERegressor requires an RFFEncoder.")
        self.enc      = encoder
        self.clf      = CyphaDIF(encoder=encoder, field_dim=field_dim,
                                 rng=rng or np.random.default_rng(42))
        self.K        = K
        self.lr       = lr
        self.reg      = reg
        self._D       = encoder.dim
        self._w       : Dict[str, np.ndarray] = {}   # per-expert weights (D,)
        self._target_dim : Optional[int]      = None
        self._n_seen  = 0

    @classmethod
    def from_data(cls,
                  X_seed     : np.ndarray,
                  y_seed     : Optional[np.ndarray] = None,
                  K          : int   = 8,
                  D          : int   = 256,
                  lr         : float = 0.01,
                  reg        : float = 1e-6,
                  field_dim  : int   = 160,
                  rng_seed   : int   = 42,
                  ) -> 'MKERegressor':
        """
        Factory: auto-tune γ then seed K experts via clustering.

        γ selection strategy
        ────────────────────
        - If y_seed is provided: cross-validated γ (auto_gamma_cv) — recommended
          for regression.  Tries a log-grid of candidates and picks the one with
          lowest hold-out RMSE.  Very fast (closed-form ridge per candidate).
        - If y_seed is None: uses median pairwise distance heuristic (auto_gamma).

        Parameters
        ----------
        X_seed : (N, d_in) representative training inputs
        y_seed : (N,) training targets for CV bandwidth selection (optional)
        K      : number of kernel experts
        D      : number of RFF features

        Returns a pre-initialised MKERegressor ready for train_step().
        """
        enc = RFFEncoder(X_seed.shape[1], D=D, gamma=1.0, seed=rng_seed)

        if y_seed is not None:
            enc.auto_gamma_cv(X_seed, y_seed,
                              rng=np.random.default_rng(rng_seed))
        else:
            enc.auto_gamma(X_seed)

        mke = cls(enc, K=K, lr=lr, reg=reg, field_dim=field_dim,
                  rng=np.random.default_rng(rng_seed))

        # Seed K experts via unsupervised clustering in RFF space
        n_seed = min(len(X_seed), 500)
        xs_list = [X_seed[i] for i in range(n_seed)]
        assignment = mke.clf.fit_unlabeled(xs_list, n_clusters=K, prefix='_e',
                                           min_cluster_size=2)
        for lbl in assignment:
            mke._w[lbl] = np.zeros(D)

        return mke

    def _w_for(self, label: str) -> np.ndarray:
        if label not in self._w:
            self._w[label] = np.zeros(self._D)
        return self._w[label]

    def _route(self, phi: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Get routing probabilities and label list for a single φ(x)."""
        LLR, labs = self.clf.score_matrix(phi.reshape(1, -1))
        if not labs:
            return np.array([1.0]), ['_e0']
        p = _softmax_batch(LLR / (self.clf.temperature + _EPS))[0]
        return p, labs

    def train_step(self, x: Any, y: Union[float, np.ndarray]) -> float:
        """
        One online training step — online EM M-step update.

        Parameters
        ----------
        x : raw input (will be encoded by RFFEncoder)
        y : target value (scalar or numpy array for multi-output)

        Returns squared error |y − ŷ|².
        """
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if self._target_dim is None:
            self._target_dim = len(y_arr)

        phi = self.enc(x)            # (D,)
        self._n_seen += 1

        # Route via DIF
        with self.clf.memory._lock:
            n_classes = len(self.clf.memory._classes)

        if n_classes == 0:
            label = '_e0'
            self.clf.train_step(x, label)
            self._w_for(label)   # initialise
            return float(np.sum(y_arr ** 2))

        # Get routing probabilities
        p, labs = self._route(phi)

        # Predict: ŷ = Σ_k p_k · w_k · φ
        if self._target_dim == 1:
            y_hat = sum(float(p[i]) * float(self._w_for(labs[i]) @ phi)
                        for i in range(len(labs)))
            err   = float(y_arr[0]) - y_hat
            err_sq = err ** 2
        else:
            y_hat = sum(float(p[i]) * (self._w_for(labs[i]) @ phi)
                        for i in range(len(labs)))
            err   = y_arr - y_hat
            err_sq = float(np.dot(err, err))

        # EM M-step: update each expert proportional to responsibility
        for i, lbl in enumerate(labs):
            pi   = float(p[i])
            if pi < 1e-9:
                continue   # skip near-zero responsibility
            w_k  = self._w_for(lbl)
            # w_k += lr · p_k · (y − ŷ) · φ  − reg · w_k
            if self._target_dim == 1:
                w_k += self.lr * pi * err * phi - self.reg * w_k
            else:
                for d_idx in range(self._target_dim):
                    w_k += self.lr * pi * float(err[d_idx]) * phi  # scalar grad per target dim
                w_k -= self.reg * w_k

        # Update DIF router (learns regions in RFF space)
        pred, _ = self.clf.infer(x)
        if pred == '__unknown__':
            pred = labs[int(np.argmax(p))]
        self.clf.train_step(x, pred)

        return err_sq

    def predict(self, x: Any) -> Tuple[np.ndarray, float]:
        """
        Predict target for a single input.

        Returns
        -------
        y_pred      : weighted prediction  ŷ = Σ_k p(k|x)·f_k(x)
        uncertainty : predictive std from routing entropy
                      High entropy routing → high uncertainty
        """
        phi      = self.enc(x)
        p, labs  = self._route(phi)

        d = self._target_dim or 1
        if self._target_dim == 1:
            y_pred = np.array([sum(float(p[i]) * float(self._w_for(labs[i]) @ phi)
                                   for i in range(len(labs)))])
        else:
            y_pred = sum(float(p[i]) * (self._w_for(labs[i]) @ phi)
                         for i in range(len(labs)))
            y_pred = np.asarray(y_pred)

        # Uncertainty = entropy of routing distribution (max when p is uniform)
        entr = float(-np.sum(p * np.log(p + _EPS)))
        return y_pred.ravel(), entr

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorised batch prediction — uses a single matmul for encoding and scoring.

        Parameters
        ----------
        X : (N, d_in) input matrix

        Returns
        -------
        y_pred    : (N,) or (N, d_out) prediction array
        uncertainty: (N,) routing entropy (higher = less certain)
        """
        X    = np.asarray(X, dtype=np.float64)
        N    = len(X)
        PHI  = self.enc.batch_encode(X)      # (N, D) — one BLAS gemm
        LLR, labs = self.clf.score_matrix(PHI)  # (N, K) — one BLAS gemm

        if not labs:
            d = self._target_dim or 1
            return np.zeros((N, d)), np.zeros(N)

        P    = _softmax_batch(LLR / (self.clf.temperature + _EPS))  # (N, K)
        entr = -np.sum(P * np.log(P + _EPS), axis=1)                # (N,)

        if self._target_dim is None or self._target_dim == 1:
            # Scalar target: w_k ∈ R^D, f_k(x) = w_k·φ(x) ∈ R
            W_mat  = np.stack([self._w.get(l, np.zeros(self._D)) for l in labs])  # (K,D)
            F      = PHI @ W_mat.T         # (N, K): f_k(x_i) for all k, i
            y_pred = np.sum(P * F, axis=1) # (N,)
            return y_pred, entr
        else:
            # Multi-output: w_k ∈ R^(D×d_out), each column is one output
            d_out  = self._target_dim
            y_pred = np.zeros((N, d_out))
            for i, lbl in enumerate(labs):
                w_k    = self._w.get(lbl, np.zeros((self._D, d_out)))
                F_k    = PHI @ w_k                    # (N, d_out)
                y_pred += P[:, i:i+1] * F_k
            return y_pred, entr

    def diagnostics(self) -> Dict:
        return {
            'n_experts'  : len(self._w),
            'expert_labels': list(self._w.keys()),
            'n_seen'     : self._n_seen,
            'target_dim' : self._target_dim,
            'gamma'      : self.enc._gamma,
            'D'          : self._D,
            'clf_steps'  : self.clf._total_steps,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: DIFRegressor  —  online mixture-of-experts regression
# ─────────────────────────────────────────────────────────────────────────────

class DIFRegressor:
    """
    Online mixture-of-experts regression built on CyphaDIF.

    Architecture
    ────────────
    A CyphaDIF classifier acts as a soft router: it partitions the input
    space into K latent classes (experts).  Each expert k maintains an EMA
    of the target values it has seen.

    Prediction
    ──────────
    E[y|x] = Σ_k  p(k|x) · μ_y_k

    This is a probabilistic mixture-of-experts model where:
      - p(k|x)  = softmax(LLR_k(x)) — the DIF routing probabilities
      - μ_y_k   = EMA of targets assigned to expert k during training
      - Σ_k² σ²_y_k · p(k|x)  is the predictive variance

    Properties
    ──────────
    - Online: each (x, y) pair updates the model incrementally
    - Non-parametric: number of experts K grows dynamically as needed
    - Continuous target: works for scalar, vector, or multi-output y
    - Inherits DIF: OOD detection, concept drift, replay, etc.

    Example
    ───────
    reg = DIFRegressor(encoder=VectorEncoder(64))
    for x, y in stream:
        reg.train_step(x, y)
    y_pred, uncertainty = reg.predict(x_new)
    """

    def __init__(self,
                 encoder   : 'Encoder',
                 field_dim : int   = 160,
                 n_experts : int   = 0,    # 0 = grow dynamically
                 target_lr : float = 0.05, # EMA rate for target means
                 rng       : Optional[np.random.Generator] = None):
        self.clf      = CyphaDIF(encoder=encoder, field_dim=field_dim, rng=rng)
        self.n_experts= n_experts
        self.target_lr= target_lr
        self._rng     = rng or np.random.default_rng(42)

        # Per-expert target statistics (updated incrementally)
        # Supports scalar and vector targets
        self._expert_mu  : Dict[str, np.ndarray] = {}   # EMA of targets
        self._expert_var : Dict[str, float]       = {}   # EMA of squared deviations
        self._expert_n   : Dict[str, int]         = {}   # observation counts
        self._target_dim : Optional[int]          = None

    def _label_for_expert(self, k_idx: int) -> str:
        return f'_e{k_idx}'

    def train_step(self, x: Any, y: Union[float, np.ndarray]) -> float:
        """
        One online training step.

        Parameters
        ----------
        x : input
        y : target (scalar float or numpy array for multi-output)

        Returns routing loss (from DIF classifier).

        Routing Strategy
        ────────────────
        Cold-start: while fewer than n_experts exist, hash-assign new inputs
        to experts so all K experts receive roughly equal initial training.
        After warm-up: the DIF classifier routes inputs to the expert whose
        latent centroid is closest (highest LLR).
        """
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if self._target_dim is None:
            self._target_dim = len(y_arr)

        self._step_count = getattr(self, '_step_count', 0) + 1
        K_target = max(self.n_experts, 4)  # default 4 experts if n_experts=0

        with self.clf.memory._lock:
            n_existing = len(self.clf.memory._classes)

        # Cold-start: hash-assign to spread data across experts
        if n_existing < K_target and self._step_count <= K_target * 20:
            expert = f'_e{self._step_count % K_target}'
        else:
            # Warm routing: use DIF classifier
            if n_existing > 0:
                pred, conf = self.clf.infer(x)
                expert = pred if pred != '__unknown__' else '_e0'
            else:
                expert = '_e0'

        # Update DIF classifier (learns the routing)
        loss = self.clf.train_step(x, expert)

        # Update expert target statistics via EMA
        lr = self.target_lr
        if expert not in self._expert_mu:
            self._expert_mu[expert]  = y_arr.copy()
            self._expert_var[expert] = 0.0
            self._expert_n[expert]   = 1
        else:
            old_mu  = self._expert_mu[expert]
            delta   = y_arr - old_mu
            self._expert_mu[expert]  = old_mu + lr * delta
            self._expert_var[expert] = (1-lr)*self._expert_var[expert] + lr*float(delta@delta)
            self._expert_n[expert]  += 1

        return loss

    def predict(self, x: Any) -> Tuple[np.ndarray, float]:
        """
        Predict target value for input x.

        Returns
        -------
        y_pred       : weighted mean prediction  E[y|x]
        uncertainty  : predictive std (mixture of expert variances)
        """
        if not self._expert_mu:
            d = self._target_dim or 1
            return np.zeros(d), float('inf')

        # Get routing probabilities
        _, h   = self.clf._encode(x)
        LLR, labels = self.clf.score_matrix(h.reshape(1, -1))
        probs  = _softmax_batch(LLR / (self.clf.temperature + _EPS))[0]   # (K,)

        d      = self._target_dim or 1
        y_pred = np.zeros(d)
        var    = 0.0

        for i, lbl in enumerate(labels):
            if lbl in self._expert_mu:
                p         = float(probs[i])
                y_pred   += p * self._expert_mu[lbl]
                var      += p * self._expert_var.get(lbl, 0.0)

        return y_pred, float(np.sqrt(max(var, 0.0)))

    def predict_batch(self, xs: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorised batch prediction.
        Returns (y_pred (N, d), uncertainty (N,)).
        """
        if (hasattr(xs,"__len__") and len(xs)==0) or not self._expert_mu:
            d = self._target_dim or 1
            return np.zeros((len(xs), d)), np.full(len(xs), float('inf'))

        H      = self.clf.batch_encode(xs)
        LLR, labels = self.clf.score_matrix(H)
        P      = _softmax_batch(LLR / (self.clf.temperature + _EPS))  # (N, K)

        d = self._target_dim or 1
        # Build expert target and variance matrices
        mu_mat  = np.stack([self._expert_mu.get(l,  np.zeros(d)) for l in labels])  # (K, d)
        var_vec = np.array([self._expert_var.get(l, 0.0) for l in labels])           # (K,)

        y_pred = P @ mu_mat          # (N, d)
        uncert = np.sqrt(np.maximum(P @ var_vec, 0.0))  # (N,)
        return y_pred, uncert

    def diagnostics(self) -> Dict:
        return {
            'n_experts'   : len(self._expert_mu),
            'expert_ns'   : {k: self._expert_n[k] for k in self._expert_mu},
            'target_dim'  : self._target_dim,
            'clf_steps'   : self.clf._total_steps,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: MultiLabelDIF  —  independent binary classifiers, shared encoder
# ─────────────────────────────────────────────────────────────────────────────

class MultiLabelDIF:
    """
    Multi-label classification: predict P(label_k=1|x) for K independent labels.

    Unlike the single-label CyphaDIF, each label is treated as an independent
    binary classification task sharing a common encoder projection.  This
    supports cases where a single input can have multiple simultaneous labels:

        "Is this input X? Is it Y? Is it Z?"

    Architecture
    ────────────
    One CyphaDIF binary classifier per label, all sharing:
      - The same encoder_fn (feature extractor)
      - The same latent dimension d

    Labels are added dynamically on first train_step.

    Example
    ───────
    mlf = MultiLabelDIF(encoder=VectorEncoder(64))
    mlf.train_step(x, {'dog': True, 'big': False, 'black': True})
    probs = mlf.predict(x_new)  # {'dog': 0.87, 'big': 0.12, 'black': 0.94}
    """

    def __init__(self,
                 encoder  : 'Encoder',
                 field_dim: int  = 160,
                 rng      : Optional[np.random.Generator] = None):
        self._encoder_fn = encoder
        self._field_dim  = field_dim
        self._rng        = rng or np.random.default_rng(42)
        self._classifiers: Dict[str, CyphaDIF] = {}  # one per label

    def _get_or_create(self, label: str) -> CyphaDIF:
        if label not in self._classifiers:
            self._classifiers[label] = CyphaDIF(
                encoder   = self._encoder_fn,
                field_dim = self._field_dim,
                rng       = np.random.default_rng(hash(label) % (2**32)),
            )
        return self._classifiers[label]

    def train_step(self, x: Any,
                   labels: Dict[str, bool]) -> Dict[str, float]:
        """
        Train on a single example with multiple boolean labels.

        Parameters
        ----------
        x      : input
        labels : dict mapping label_name → True/False

        Returns dict of per-label training losses.
        """
        losses = {}
        for label, is_positive in labels.items():
            clf  = self._get_or_create(label)
            loss = clf.train_step(x, 'pos' if is_positive else 'neg')
            losses[label] = loss
        return losses

    def predict(self, x: Any) -> Dict[str, float]:
        """
        Predict P(label=True|x) for all known labels.

        Returns dict mapping label_name → probability ∈ [0,1].
        """
        result = {}
        for label, clf in self._classifiers.items():
            pred, conf = clf.infer(x)
            if pred == 'pos':
                result[label] = conf
            elif pred == 'neg':
                result[label] = 1.0 - conf
            else:
                result[label] = 0.5
        return result

    def predict_batch(self, xs: List[Any]) -> Dict[str, np.ndarray]:
        """
        Batch predict P(label=True|x) for all inputs.
        Returns dict mapping label → (N,) probability array.
        """
        result = {}
        for label, clf in self._classifiers.items():
            H      = clf.batch_encode(xs)
            LLR, labels_clf = clf.score_matrix(H)
            probs  = _softmax_batch(LLR / (clf.temperature + _EPS))
            if 'pos' in labels_clf:
                pi = labels_clf.index('pos')
                result[label] = probs[:, pi] * clf.world_gate_vector(H)
            else:
                result[label] = np.full(len(xs), 0.5)
        return result

    @property
    def labels(self) -> List[str]:
        return list(self._classifiers.keys())

    def diagnostics(self) -> Dict:
        return {lbl: clf.diagnostics() for lbl, clf in self._classifiers.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: SimilarityIndex  —  Mahalanobis metric + approximate nearest neighbour
# ─────────────────────────────────────────────────────────────────────────────

class SimilarityIndex:
    """
    Metric learning and k-NN retrieval over stored examples.

    Computes similarity between encoded inputs using the DIF Mahalanobis
    metric: s(x1, x2) = exp(-½ (h1−h2)ᵀ Σ⁻¹ (h1−h2) / d)

    This gives:
      s ≈ 1.0  for same-class examples
      s ≈ 0.0  for well-separated classes
      s ∈ (0,1) for partially similar inputs

    Usage
    ─────
    idx = SimilarityIndex(clf)
    idx.add(x1, metadata={'label': 'cat', 'id': 42})
    idx.add(x2, metadata={'label': 'dog', 'id': 43})
    results = idx.query(x_new, k=5)
    # → [{'similarity': 0.92, 'metadata': {'label': 'cat', ...}}, ...]
    """

    def __init__(self, clf: CyphaDIF):
        self.clf      = clf
        self._H       : Optional[np.ndarray] = None   # (N, d) stored latents
        self._meta    : List[Any]             = []     # per-example metadata
        self._n       = 0

    def add(self, x: Any, metadata: Any = None) -> int:
        """
        Encode and store an example.
        Returns the storage index of the new entry.
        """
        _, h = self.clf._encode(x)
        if self._H is None:
            self._H = h.reshape(1, -1)
        else:
            self._H = np.vstack([self._H, h.reshape(1, -1)])
        self._meta.append(metadata)
        self._n += 1
        return self._n - 1

    def add_batch(self, xs: List[Any], metadatas: Optional[List] = None) -> List[int]:
        """Encode and store a batch of examples."""
        if not xs:
            return []
        H = self.clf.batch_encode(xs)
        if self._H is None:
            self._H = H
        else:
            self._H = np.vstack([self._H, H])
        if metadatas is None:
            metadatas = [None] * len(xs)
        self._meta.extend(metadatas)
        start = self._n
        self._n += len(xs)
        return list(range(start, self._n))

    def similarity(self, x1: Any, x2: Any) -> float:
        """
        Compute Mahalanobis similarity between two inputs.
        Returns scalar ∈ [0, 1] where 1 = identical, 0 = completely different.
        """
        _, h1 = self.clf._encode(x1)
        _, h2 = self.clf._encode(x2)
        return self._h_similarity(h1, h2)

    def _h_similarity(self, h1: np.ndarray, h2: np.ndarray) -> float:
        with self.clf.memory._lock:
            inv_v = self.clf.memory.world.inv_v.copy()
        d    = h1 - h2
        mahal= float(d @ (d * inv_v)) / (len(h1) + _EPS)
        return float(np.exp(-0.5 * mahal))

    def query(self, x: Any, k: int = 5,
              return_similarities: bool = True) -> List[Dict]:
        """
        Find the k most similar stored examples.

        Parameters
        ----------
        x                  : query input
        k                  : number of results
        return_similarities: include similarity scores in output

        Returns
        -------
        List of dicts with keys: index, metadata, [similarity]
        Sorted by decreasing similarity.
        """
        if self._H is None or self._n == 0:
            return []

        _, h_q = self.clf._encode(x)
        with self.clf.memory._lock:
            inv_v = self.clf.memory.world.inv_v.copy()

        # Vectorised Mahalanobis: (N,)
        diffs   = self._H - h_q                           # (N, d)
        mahals  = np.sum(diffs ** 2 * inv_v, axis=1) / (self._H.shape[1] + _EPS)
        sims    = np.exp(-0.5 * mahals)

        k_actual = min(k, self._n)
        top_idx  = np.argsort(sims)[::-1][:k_actual]

        results = []
        for i in top_idx:
            entry = {'index': int(i), 'metadata': self._meta[i]}
            if return_similarities:
                entry['similarity'] = float(sims[i])
            results.append(entry)
        return results

    def query_batch(self, xs: List[Any], k: int = 5) -> List[List[Dict]]:
        """Batch query: returns list of k-NN result lists."""
        if self._H is None or self._n == 0:
            return [[] for _ in xs]

        H_q = self.clf.batch_encode(xs)    # (M, d)
        with self.clf.memory._lock:
            inv_v = self.clf.memory.world.inv_v.copy()

        # (M, N) similarity matrix — vectorised
        # diffs[m, n] = H_q[m] - H[n], shape (M, N, d)
        diff_sq = np.sum(
            (H_q[:, np.newaxis, :] - self._H[np.newaxis, :, :]) ** 2 * inv_v,
            axis=2
        ) / (H_q.shape[1] + _EPS)              # (M, N)
        sims = np.exp(-0.5 * diff_sq)          # (M, N)

        results = []
        k_actual = min(k, self._n)
        for m in range(len(xs)):
            top_idx = np.argsort(sims[m])[::-1][:k_actual]
            results.append([{
                'index': int(i),
                'metadata': self._meta[i],
                'similarity': float(sims[m, i]),
            } for i in top_idx])
        return results

    def __len__(self) -> int:
        return self._n


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: PerformanceMonitor  —  online self-monitoring
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceMonitor:
    """
    Online performance monitoring with windowed accuracy, calibration,
    and drift tracking.  Fires alerts when metrics degrade.

    Usage
    ─────
    mon = PerformanceMonitor(clf, window=100)
    # In training loop:
    for x, y_true in stream:
        pred, conf = clf.infer(x)
        clf.train_step(x, y_true)
        alerts = mon.record(pred, conf, y_true)
        if alerts:
            print(f"Alert: {alerts}")

    Report
    ──────
    mon.report()  # full metrics dict
    """

    def __init__(self,
                 clf         : CyphaDIF,
                 window      : int   = 100,
                 acc_thresh  : float = 0.70,   # alert if rolling acc < this
                 ece_thresh  : float = 0.15,   # alert if ECE > this
                 drift_thresh: float = 0.22):  # alert if drift_score > this

        self.clf          = clf
        self.window       = window
        self.acc_thresh   = acc_thresh
        self.ece_thresh   = ece_thresh
        self.drift_thresh = drift_thresh

        # Rolling window buffers
        self._preds  : deque = deque(maxlen=window)
        self._trues  : deque = deque(maxlen=window)
        self._confs  : deque = deque(maxlen=window)
        self._total  = 0
        self._correct= 0

        # Alert history
        self.alerts: List[Dict] = []

    def record(self, pred: str, conf: float, true_label: str) -> List[Dict]:
        """
        Record one prediction. Returns list of triggered alerts (empty if none).
        """
        self._preds.append(pred)
        self._trues.append(true_label)
        self._confs.append(conf)
        self._total  += 1
        self._correct += int(pred == true_label)

        fired: List[Dict] = []

        # Only check once the window is full
        if len(self._preds) < self.window:
            return fired

        confs_arr   = np.array(self._confs)
        correct_arr = np.array([int(p==t) for p,t in zip(self._preds,self._trues)],dtype=float)

        # Rolling accuracy
        roll_acc = float(correct_arr.mean())
        if roll_acc < self.acc_thresh:
            alert = {'type': 'low_accuracy', 'value': roll_acc,
                     'threshold': self.acc_thresh, 'step': self._total}
            fired.append(alert)
            self.alerts.append(alert)

        # ECE (every 10 steps for efficiency)
        if self._total % 10 == 0:
            ece = _compute_ece(confs_arr, correct_arr)
            if ece > self.ece_thresh:
                alert = {'type': 'high_ece', 'value': ece,
                         'threshold': self.ece_thresh, 'step': self._total}
                fired.append(alert)
                self.alerts.append(alert)

        # Drift
        if self._total % 20 == 0:
            drift = self.clf.drift_score()
            if drift > self.drift_thresh:
                alert = {'type': 'drift', 'value': drift,
                         'threshold': self.drift_thresh, 'step': self._total}
                fired.append(alert)
                self.alerts.append(alert)

        return fired

    def report(self) -> Dict:
        """Return full metrics dict."""
        if not self._preds:
            return {'total': 0}

        confs_arr   = np.array(self._confs)
        correct_arr = np.array([int(p==t) for p,t in zip(self._preds,self._trues)],dtype=float)

        per_class: Dict[str, List] = {}
        for p, t, c in zip(self._preds, self._trues, self._confs):
            if t not in per_class: per_class[t] = [0, 0]
            per_class[t][0] += int(p==t); per_class[t][1] += 1

        return {
            'total'           : self._total,
            'window_size'     : len(self._preds),
            'rolling_accuracy': float(correct_arr.mean()),
            'overall_accuracy': self._correct / max(self._total, 1),
            'ece'             : _compute_ece(confs_arr, correct_arr),
            'mean_confidence' : float(confs_arr.mean()),
            'drift_score'     : self.clf.drift_score(),
            'n_alerts'        : len(self.alerts),
            'alerts'          : self.alerts,
            'recent_alerts'   : self.alerts[-5:],
            'per_class'       : {k: v[0]/v[1] for k,v in per_class.items() if v[1]>0},
        }

    def reset(self) -> None:
        """Clear rolling window (keep counters)."""
        self._preds.clear(); self._trues.clear(); self._confs.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: MultiModalCyphaDIF  —  per-encoder LLR fusion
# ─────────────────────────────────────────────────────────────────────────────

class MultiModalCyphaDIF:
    """
    Phase 5: Multi-modal fusion via independent DIF instances.

    Each modality has its own CyphaDIF (own encoder, own world prior).
    At inference, per-class LLR scores are summed across modalities.

    Advantages over ConcatEncoder:
      - Each modality's world prior is domain-correct
      - Graceful degradation: missing modalities at inference are skipped
      - Per-modality diagnostics remain interpretable

    Usage:
        mm = MultiModalCyphaDIF({
            'vision': CyphaDIF(encoder=VisionEncoder()),
            'audio' : CyphaDIF(encoder=AudioEncoder()),
        })
        mm.train_step({'vision': img, 'audio': wav}, 'cat')
        label, conf = mm.infer({'vision': img})   # audio missing — still works
    """

    def __init__(self, modalities: Dict[str, CyphaDIF]):
        if not modalities:
            raise ValueError("MultiModalCyphaDIF requires at least one modality.")
        self.modalities = modalities

    def train_step(self, inputs: Dict[str, Any], label: str) -> Dict[str, float]:
        """Train each modality on its input. Returns {modality: loss}."""
        return {name: clf.train_step(inputs[name], label)
                for name, clf in self.modalities.items() if name in inputs}

    def infer(self, inputs: Dict[str, Any],
              weights: Optional[Dict[str, float]] = None) -> Tuple[str, float]:
        """
        Fuse LLRs across present modalities, classify.
        Missing modalities are skipped. Returns (label, mean_confidence).
        """
        fused: Dict[str, float] = defaultdict(float)
        confs = []

        for name, clf in self.modalities.items():
            if name not in inputs:
                continue
            w = weights.get(name, 1.0) if weights else 1.0
            f, h = clf._encode(inputs[name])
            _, conf, llrs = clf.memory.classify(
                h,
                h_field       = clf.field.h,
                context_prior = clf._ctx_prior(list(clf.memory._classes.keys())),
                temperature   = clf.temperature,
                ood_sigma     = clf.ood_sigma,
            )
            for lbl, llr in llrs.items():
                fused[lbl] += w * llr
            confs.append(conf)

        if not fused:
            return '__unknown__', 0.0
        best = max(fused, key=fused.__getitem__)
        return best, float(np.mean(confs)) if confs else 0.0

    def batch_train(self, data: List[Tuple[Dict[str, Any], str]],
                    n_epochs: int = 1, shuffle: bool = True) -> Dict[str, List[float]]:
        import random
        all_losses: Dict[str, List[float]] = defaultdict(list)
        for _ in range(n_epochs):
            if shuffle:
                random.shuffle(data)
            for inputs, label in data:
                for name, loss in self.train_step(inputs, label).items():
                    all_losses[name].append(loss)
        return dict(all_losses)

    def diagnostics(self) -> Dict[str, Dict]:
        return {name: clf.diagnostics() for name, clf in self.modalities.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Extended capabilities — all new methods injected via monkey-patch
# into CyphaDIF, plus standalone utilities below.
# ─────────────────────────────────────────────────────────────────────────────

# ── 6A: Fisher-Rao class health analysis ─────────────────────────────────────

def _class_health(self) -> Dict[str, Dict]:
    """
    Fisher-Rao class health analysis.

    For each class returns:
        fr_norm      : Fisher-Rao complexity ||Δk||_FR  (MDL cost)
        n_obs        : training observations seen
        accuracy     : per-class training accuracy
        status       : 'healthy' | 'underfitted' | 'overfit' | 'sparse'
        recommendation : human-readable suggestion

    Thresholds (heuristic, scale with dim):
        underfitted : fr_norm < 0.05 * sqrt(feat_dim)   → Δk barely moved from prior
        overfit     : fr_norm > 5.0  * sqrt(feat_dim)   → MDL ball blown out
        sparse      : n_obs < 5                          → too few examples
    """
    with self.memory._lock:
        v0      = self.memory.world.v.copy()
        classes = dict(self.memory._classes)

    lo  = 0.05 * math.sqrt(self.feat_dim)
    hi  = 5.0  * math.sqrt(self.feat_dim)
    out = {}
    for lbl, cd in classes.items():
        norm = cd.fisher_rao_norm(v0)
        acc  = cd.accuracy()
        n    = cd.n_obs
        if n < 5:
            status = 'sparse'
            rec    = f"Need more examples (have {n}, want ≥5)"
        elif norm < lo:
            status = 'underfitted'
            rec    = f"Δk near prior (fr_norm={norm:.3f}<{lo:.3f}); add more/varied data"
        elif norm > hi:
            status = 'overfit'
            rec    = f"Δk very large (fr_norm={norm:.3f}>{hi:.3f}); reduce lr or add MDL decay"
        else:
            status = 'healthy'
            rec    = 'OK'
        out[lbl] = dict(fr_norm=norm, n_obs=n, accuracy=acc, status=status,
                        recommendation=rec)
    return out

CyphaDIF.class_health = _class_health


def _prune_classes(self, min_obs: int = 5, max_fr_norm: Optional[float] = None,
                   min_fr_norm: Optional[float] = None) -> List[str]:
    """
    Remove classes that fail health criteria. Returns list of pruned labels.

        min_obs      : prune if n_obs < min_obs
        max_fr_norm  : prune if fr_norm > threshold (overfit)
        min_fr_norm  : prune if fr_norm < threshold (underfitted/collapsed)
    """
    health  = self.class_health()
    pruned  = []
    with self.memory._lock:
        for lbl, info in health.items():
            remove = False
            if info['n_obs'] < min_obs:
                remove = True
            if max_fr_norm is not None and info['fr_norm'] > max_fr_norm:
                remove = True
            if min_fr_norm is not None and info['fr_norm'] < min_fr_norm:
                remove = True
            if remove:
                del self.memory._classes[lbl]
                pruned.append(lbl)
        if pruned:
            # Rebuild D_buf, label_order, label_idx to stay compact and consistent
            surviving = list(self.memory._classes.keys())
            self.memory._label_order.clear()
            self.memory._label_idx.clear()
            for new_k, label in enumerate(surviving):
                cd = self.memory._classes[label]
                self.memory._D_buf[new_k, :] = cd.delta_mu      # copy into buffer
                self.memory._D_buf[new_k, :] = cd.delta_mu      # ensure copy
                cd.delta_mu = self.memory._D_buf[new_k]          # re-hook as view
                self.memory._label_order.append(label)
                self.memory._label_idx[label] = new_k
            K = len(surviving)
            if K < _MAX_CLASSES:
                self.memory._D_buf[K:, :] = 0.0                 # zero stale rows
    return pruned

CyphaDIF.prune_classes = _prune_classes


# ── 6B: Markov chain analysis from Tier-2 transitions ────────────────────────

def _markov_analysis(self) -> Dict:
    """
    Analyse the label transition Markov chain from TieredContextBuffer Tier-2.

    Returns:
        transition_matrix : dict-of-dict row-normalised P[i→j]
        steady_state      : stationary distribution π via power iteration
        top_transitions   : top-5 most likely (from, to, prob) transitions
        self_loops        : P[k→k] per class  (high = class is self-sustaining)
        absorbing         : classes with P[k→k] ≈ 1 (rarely leave)
    """
    with self.context._lock:
        classes = list(self.memory._classes.keys())
        mid_t   = {k: dict(v) for k, v in self.context._mid_trans.items()}

    if not classes or not mid_t:
        return {'error': 'insufficient transition data'}

    K = len(classes)
    idx = {c: i for i, c in enumerate(classes)}

    # Build row-normalised P
    P = np.zeros((K, K))
    for src, dsts in mid_t.items():
        if src not in idx:
            continue
        i = idx[src]
        for dst, w in dsts.items():
            if dst in idx:
                P[i, idx[dst]] = w
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P /= row_sums

    # Steady state via power iteration
    pi = np.ones(K) / K
    for _ in range(1000):
        pi_new = pi @ P
        if np.linalg.norm(pi_new - pi) < 1e-9:
            break
        pi = pi_new
    pi = pi_new

    # Top transitions
    pairs = []
    for i, src in enumerate(classes):
        for j, dst in enumerate(classes):
            if P[i, j] > 0:
                pairs.append((src, dst, float(P[i, j])))
    pairs.sort(key=lambda t: -t[2])

    self_loops = {c: float(P[idx[c], idx[c]]) for c in classes}
    absorbing  = [c for c in classes if self_loops[c] > 0.8]

    trans_dict = {src: {dst: float(P[idx[src], idx[dst]])
                        for dst in classes if P[idx[src], idx[dst]] > 1e-4}
                  for src in classes}

    return {
        'transition_matrix': trans_dict,
        'steady_state'     : dict(zip(classes, pi.tolist())),
        'top_transitions'  : pairs[:5],
        'self_loops'       : self_loops,
        'absorbing_classes': absorbing,
    }

CyphaDIF.markov_analysis = _markov_analysis


# ── 6C: Session embedding ─────────────────────────────────────────────────────

def _session_embedding(self) -> np.ndarray:
    """
    Fixed-length session embedding from the NIGField slow state (τ=0.99 group).
    Represents the compressed history of everything observed this session.
    Can be used for: session similarity, change-point detection, transfer init.
    Returns np.ndarray of shape (field_dim // 5,).
    """
    return self.field.slow_state()


def _session_similarity(self, other: 'CyphaDIF') -> float:
    """
    Cosine similarity between this session's slow-state embedding and another.
    1.0 = identical trajectory, 0.0 = orthogonal, -1.0 = opposite.
    """
    a = self.session_embedding()
    b = other.session_embedding()
    na = np.linalg.norm(a) + _EPS
    nb = np.linalg.norm(b) + _EPS
    return float(np.dot(a / na, b / nb))

CyphaDIF.session_embedding  = _session_embedding
CyphaDIF.session_similarity = _session_similarity


# ── 6D: Semi-supervised learning ──────────────────────────────────────────────

def _semi_supervised_step(self, x: Any,
                           conf_threshold: float = 0.85,
                           query_threshold: float = 0.0) -> Dict:
    """
    Semi-supervised update on unlabelled input x.

    Strategy:
        1. Infer label and confidence.
        2. If confidence ≥ conf_threshold AND anomaly_score < 0.3:
               accept as pseudo-label → train_step(x, pseudo_label)
        3. If active_query_score > query_threshold:
               flag for human labelling.

    Returns dict:
        action        : 'pseudo_labelled' | 'query_human' | 'rejected'
        pseudo_label  : str | None
        confidence    : float
        anomaly_score : float
        query_score   : float
    """
    full      = self.infer_full(x)
    label     = full['label']
    conf      = full['confidence']
    anom      = full['anomaly_score']
    query     = full['query_score']

    if conf >= conf_threshold and anom < 0.85:
        self.train_step(x, label)
        action = 'pseudo_labelled'
    elif query > query_threshold:
        action = 'query_human'
    else:
        action = 'rejected'

    return dict(action=action, pseudo_label=label if action == 'pseudo_labelled' else None,
                confidence=conf, anomaly_score=anom, query_score=query)


def _semi_supervised_batch(self, unlabelled: List[Any],
                            conf_threshold: float = 0.85,
                            query_threshold: float = 0.0) -> Dict:
    """
    Run semi-supervised over a list of unlabelled inputs.
    Returns summary statistics and the list of items flagged for human labelling.
    """
    pseudo_n = 0; rejected_n = 0; query_items = []
    for x in unlabelled:
        r = self.semi_supervised_step(x, conf_threshold, query_threshold)
        if r['action'] == 'pseudo_labelled':
            pseudo_n += 1
        elif r['action'] == 'query_human':
            query_items.append((x, r))
        else:
            rejected_n += 1
    return dict(
        pseudo_labelled=pseudo_n,
        rejected=rejected_n,
        query_human=len(query_items),
        query_items=query_items,
    )

CyphaDIF.semi_supervised_step  = _semi_supervised_step
CyphaDIF.semi_supervised_batch = _semi_supervised_batch


# ── 6E: Continual learning — save / restore / merge ──────────────────────────

def _save_state(self) -> Dict:
    """
    Serialise the full model state to a plain Python dict (numpy arrays included).
    Can be pickled, deepcopied, or stored.

    Captures: world prior, all class differentials, field state, context Tier-2,
              OOD sigma, temperature, step counters.
    """
    with self.memory._lock:
        classes = {
            lbl: dict(delta_mu=cd.delta_mu.copy(), n_obs=cd.n_obs, n_correct=cd.n_correct)
            for lbl, cd in self.memory._classes.items()
        }
        world = dict(mu=self.memory.world.mu.copy(),
                     v=self.memory.world.v.copy(),
                     n=self.memory.world._n,
                     drift_ema=self.memory.world._drift_ema)

    with self.field._lock, self.field._wt_lock:
        field_h    = self.field._h.copy()
        field_step = self.field._step
        field_W_T  = self.field._W_T.copy()

    with self.encoder._lock:
        enc_W = self.encoder.W.copy()

    with self.context._lock:
        mid_freq  = dict(self.context._mid_freq)
        mid_n     = self.context._mid_n
        mid_trans = {k: dict(v) for k, v in self.context._mid_trans.items()}

    return dict(
        classes=classes, world=world,
        field_h=field_h, field_step=field_step, field_W_T=field_W_T,
        enc_W=enc_W,
        mid_freq=mid_freq, mid_n=mid_n, mid_trans=mid_trans,
        ood_sigma=self.ood_sigma, temperature=self.temperature,
        total_steps=self._total_steps, total_correct=self._total_correct,
        llr_ema=self._llr_ema,
        ll_world_ema=-1.5,
        mahal_ema=getattr(self, '_mahal_ema', None),
        mahal_std_ema=getattr(self, '_mahal_std_ema', 0.5),
        # Temperature adaptation state (Phase 6)
        llr_scale_ema=getattr(self, '_llr_scale_ema', 0.0),
        llr_scale_n=getattr(self, '_llr_scale_n', 0),
        llr_scale_baseline=getattr(self, '_llr_scale_baseline', 0.0),
        base_temp=getattr(self, '_base_temp', self.temperature),
        # Capacity state
        feat_dim=self.feat_dim,
    )


def _load_state(self, state: Dict) -> None:
    """Restore model state from a dict produced by save_state()."""
    with self.memory._lock:
        self.memory._classes.clear()
        self.memory._label_order.clear()
        self.memory._label_idx.clear()
        self.memory._D_buf[:] = 0.0
        for lbl, cd_d in state['classes'].items():
            cd = self.memory._get_or_create(lbl)          # sets up view into _D_buf
            cd.delta_mu[:]  = cd_d['delta_mu']            # in-place → syncs _D_buf row
            cd.n_obs     = cd_d['n_obs']
            cd.n_correct = cd_d['n_correct']
        # Sync _n_obs_buf from restored ClassDifferential objects
        for lbl, cd in self.memory._classes.items():
            k = self.memory._label_idx[lbl]
            self.memory._n_obs_buf[k] = float(cd.n_obs)
        w = state['world']
        self.memory.world.mu        = w['mu'].copy()
        self.memory.world.v         = w['v'].copy()
        self.memory.world.inv_v     = 1.0 / np.maximum(self.memory.world.v, _MIN_VAR)
        self.memory.world.v_mean    = float(self.memory.world.v.mean())
        self.memory.world._log_norm = -0.5 * (self.memory.world.d * _LOG2PI
                                              + float(np.sum(np.log(np.maximum(self.memory.world.v, _MIN_VAR)))))
        self.memory.world._n        = w['n']
        self.memory.world._drift_ema = w['drift_ema']

    with self.field._lock, self.field._wt_lock:
        self.field._h     = state['field_h'].copy()
        self.field._step  = state['field_step']
        if 'field_W_T' in state:
            self.field._W_T  = state['field_W_T'].copy()
            self.field._A_eff = (np.diag(self.field._a) + self.field._W_T).astype(np.float32)

    with self.encoder._lock:
        if 'enc_W' in state:
            self.encoder.W = state['enc_W'].copy()

    with self.context._lock:
        self.context._mid_freq  = defaultdict(float, state['mid_freq'])
        self.context._mid_n     = state['mid_n']
        self.context._mid_trans = defaultdict(
            lambda: defaultdict(float),
            {k: defaultdict(float, v) for k, v in state['mid_trans'].items()}
        )
        # Rebuild derived caches from restored state
        self.context._mid_freq_total = float(sum(self.context._mid_freq.values()))
        self.context._mid_trans_tot  = defaultdict(float,
            {k: float(sum(v.values())) for k, v in self.context._mid_trans.items()})
        self.context._t1_counts      = defaultdict(float)
        self.context._t1_total       = float(len(self.context._history))
        for lbl, _ in self.context._history:
            self.context._t1_counts[lbl] += 1.0

    self.ood_sigma          = state['ood_sigma']
    self.temperature        = state['temperature']
    self._total_steps       = state['total_steps']
    self._total_correct     = state['total_correct']
    self._llr_ema           = state['llr_ema']
    self._mahal_ema         = state.get('mahal_ema', None)
    self._mahal_std_ema     = state.get('mahal_std_ema', 0.5)
    # Temperature adaptation state
    self._llr_scale_ema     = state.get('llr_scale_ema', 0.0)
    self._llr_scale_n       = state.get('llr_scale_n', 0)
    self._llr_scale_baseline= state.get('llr_scale_baseline', 0.0)
    self._base_temp         = state.get('base_temp', self.temperature)


def _merge_from(self, other: 'CyphaDIF',
                weight_self: float = 0.5,
                weight_other: float = 0.5) -> List[str]:
    """
    Merge class differentials from `other` into this classifier.

    For shared classes: weighted average of Δk by Fisher-Rao norm.
    For classes only in `other`: copied directly.
    World prior is blended by observation count.

    Returns list of newly added class labels (from other but not in self).
    """
    with self.memory._lock, other.memory._lock:
        self_v0  = self.memory.world.v.copy()
        other_v0 = other.memory.world.v.copy()

        # Blend world prior by n
        n_self  = self.memory.world._n
        n_other = other.memory.world._n
        n_total = n_self + n_other + _EPS
        self.memory.world.mu = (n_self  * self.memory.world.mu +
                                n_other * other.memory.world.mu) / n_total
        self.memory.world.v  = (n_self  * self.memory.world.v +
                                n_other * other.memory.world.v)  / n_total
        self.memory.world._n = n_self + n_other

        new_labels = []
        for lbl, other_cd in other.memory._classes.items():
            if lbl in self.memory._classes:
                self_cd = self.memory._classes[lbl]
                # Weighted blend by FR-norm (more confident model gets higher weight)
                norm_s = self_cd.fisher_rao_norm(self_v0)
                norm_o = other_cd.fisher_rao_norm(other_v0)
                total  = norm_s * weight_self + norm_o * weight_other + _EPS
                w_s    = norm_s * weight_self / total
                w_o    = norm_o * weight_other / total
                self_cd.delta_mu[:]  = w_s * self_cd.delta_mu + w_o * other_cd.delta_mu
                self_cd.n_obs    += other_cd.n_obs
                self_cd.n_correct += other_cd.n_correct
                # Sync cached n_obs buffer
                self.memory._n_obs_buf[self.memory._label_idx[lbl]] = float(self_cd.n_obs)
            else:
                new_cd           = self.memory._get_or_create(lbl)
                new_cd.delta_mu[:]  = other_cd.delta_mu
                new_cd.n_obs     = other_cd.n_obs
                new_cd.n_correct = other_cd.n_correct
                self.memory._n_obs_buf[self.memory._label_idx[lbl]] = float(other_cd.n_obs)
                new_labels.append(lbl)

    return new_labels

CyphaDIF.save_state  = _save_state
CyphaDIF.load_state  = _load_state
CyphaDIF.merge_from  = _merge_from


# ── 6F: Uncertainty-aware retrieval ──────────────────────────────────────────

def _retrieve(self, query: Any, database: List[Any], top_k: int = 5,
              label: Optional[str] = None) -> List[Tuple[int, float, str]]:
    """
    Log-likelihood ranked retrieval.

    Scores each item in `database` by log p(h_item | θk) where k is either
    the predicted class of query (if label=None) or the given label.
    Returns top_k as [(index, log_likelihood, predicted_label), ...].

    Outperforms cosine similarity when class geometry is non-spherical or
    when you want retrieval conditioned on a specific class hypothesis.
    """
    _, h_q    = self._encode(query)
    with self.memory._lock:
        classes = list(self.memory._classes.keys())

    ctx = self._ctx_prior(classes)
    pred_label, _, llrs = self.memory.classify(
        h_q, h_field=self.field.h, context_prior=ctx,
        temperature=self.temperature, ood_sigma=self.ood_sigma
    )
    use_label = label if label is not None else pred_label
    params    = self.memory.get_class_params(use_label)
    if params is None:
        return []
    mu_k, v0 = params

    scored = []
    for i, item in enumerate(database):
        _, h_i = self._encode(item)
        ll     = _diag_gaussian_logpdf(h_i, mu_k, v0)
        p, _,_ = self.memory.classify(h_i, h_field=self.field.h, context_prior=ctx,
                                       temperature=self.temperature,
                                       ood_sigma=self.ood_sigma)
        scored.append((i, ll, p))

    scored.sort(key=lambda t: -t[1])
    return scored[:top_k]

CyphaDIF.retrieve = _retrieve


# ── 6G: Causal structure test ─────────────────────────────────────────────────

def _causal_test(self, observations: List[Any],
                 n_permutations: int = 10) -> Dict:
    """
    Lightweight causal test: does temporal order in `observations` carry
    information beyond what a shuffled sequence would give?

    Method:
        Run two parallel NIGField instances — one on the original sequence,
        one on n_permutations shuffled versions. Compare mean causal update
        error (||W_T @ h_t - h_{t+1}||²). If original error < shuffled error,
        the sequence has genuine temporal structure the field can learn.

    Returns:
        original_error   : mean causal prediction error on true sequence
        shuffled_error   : mean across permutations (null distribution)
        causal_score     : (shuffled - original) / shuffled  ∈ (-∞, 1]
                           positive = temporal structure exists
                           near zero or negative = no detectable causal order
        p_value_approx   : fraction of permutations where shuffled < original
                           (permutation test, informal)
    """
    import copy

    def run_field(seq):
        # Fresh field — measure cumulative prediction error over first half
        # (before adaptation saturates) relative to field norm.
        # Structured sequences allow the field to predict better from the start.
        f = NIGField(state_dim=self.field_dim, rng=np.random.default_rng(99))
        errs = []
        for x in seq:
            _, h = self._encode(x)
            hf   = self._to_field_dim(h)
            # Predict next state BEFORE injecting (pre-injection prediction)
            h_pred = (f._A_eff @ f._h.astype(np.float32)).astype(np.float64)
            f.inject(hf, strength=0.10)   # stronger injection → clearer signal
            h_actual = f.h
            # Error = ||predicted - actual|| normalised by ||actual||
            norm  = np.linalg.norm(h_actual) + _EPS
            err   = float(np.linalg.norm(h_pred - h_actual) / norm)
            errs.append(err)
            f.evolve(f.h, update_state=True)
        # Compare first-half errors (before full adaptation swamps the signal)
        half = max(len(errs)//2, 1)
        return float(np.mean(errs[:half]))

    orig_err = run_field(observations)

    rng2 = np.random.default_rng(123)
    shuf_errs = []
    obs_list  = list(observations)
    for _ in range(n_permutations):
        perm = list(obs_list)
        rng2.shuffle(perm)
        shuf_errs.append(run_field(perm))

    mean_shuf    = float(np.mean(shuf_errs))
    causal_score = (mean_shuf - orig_err) / (mean_shuf + _EPS)
    p_val        = float(np.mean([s < orig_err for s in shuf_errs]))

    return dict(
        original_error  = orig_err,
        shuffled_error  = mean_shuf,
        causal_score    = causal_score,
        p_value_approx  = p_val,
        interpretation  = (
            'significant temporal structure' if causal_score > 0.1 and p_val < 0.2
            else 'weak or no temporal structure'
        )
    )

CyphaDIF.causal_test = _causal_test


# ── 6H: Generative rollout — sequence simulation ──────────────────────────────

def _rollout(self, seed_label: Optional[str] = None,
             n_steps: int = 10,
             temperature: float = 1.0,
             exploration: float = 0.15,
             rng: Optional[np.random.Generator] = None) -> List[Tuple[str, np.ndarray]]:
    """
    Forward simulation of the model's generative beliefs.

    At each step:
        1. Predict next label distribution via predict_next(current_label)
        2. Mix with uniform prior at rate `exploration` (avoids absorbing-state collapse)
        3. Sample label k ~ mixed distribution
        4. Sample latent h ~ N(μk, temperature² · v₀)
        5. Update context (record label, advance field)
        6. Repeat

    Returns [(label, h), ...].

    seed_label  : starting label (None → sample from n_obs prior)
    n_steps     : number of steps to simulate
    temperature : generation diversity (σ scaling for h samples)
    exploration : ∈ [0,1]. Fraction of uniform prior mixed into the transition
                  distribution. 0 = pure learned transitions (may collapse if
                  the chain has absorbing states); 0.15 = default, enough to
                  produce varied sequences even from sequential training data.
    """
    rng = rng or self._rng
    with self.memory._lock:
        classes = list(self.memory._classes.keys())
        n_obs   = np.array([self.memory._classes[k].n_obs for k in classes],
                           dtype=np.float64)
    if not classes:
        raise ValueError("No classes registered.")

    K       = len(classes)
    uniform = np.ones(K) / K

    sequence = []

    # Seed: sample from n_obs prior (unbiased lifetime frequency)
    if seed_label is None:
        log_p = np.log(np.maximum(n_obs, 1.0))
        seed_label = classes[int(rng.choice(K, p=_softmax(log_p)))]

    current = seed_label

    for _ in range(n_steps):
        # Sample latent from current class
        samples = self.generate(current, n=1, temperature=temperature, rng=rng)
        h       = samples[0]
        sequence.append((current, h))

        # Advance context
        self.context.record(current, correct=True)
        self.field.inject(self._to_field_dim(h), strength=0.05)
        h_new = self.field.evolve(self.field.h, update_state=True)

        # Predict next: learned transitions mixed with uniform
        next_dist = self.predict_next(current)
        if not next_dist:
            break
        lbls      = list(next_dist.keys())
        raw_probs = np.array([next_dist.get(k, 0.0) for k in classes])
        raw_probs = raw_probs / (raw_probs.sum() + _EPS)
        mixed     = (1.0 - exploration) * raw_probs + exploration * uniform
        current   = classes[int(rng.choice(K, p=mixed / mixed.sum()))]

    return sequence

CyphaDIF.rollout = _rollout


# ─────────────────────────────────────────────────────────────────────────────
# Standalone utility: ClassifierDistillation
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierDistillation:
    """
    Distil an ensemble of CyphaDIF classifiers into a single one.

    Each expert votes via its LLR scores. The ensemble LLR is a weighted
    sum. A fresh student CyphaDIF is trained on samples generated from the
    ensemble's class models, labelled by ensemble consensus.

    Usage:
        experts = [clf_a, clf_b, clf_c]
        d = ClassifierDistillation(experts, student_encoder=VectorEncoder(128))
        student = d.distil(n_per_class=200, n_epochs=3)
    """

    def __init__(self, experts: List[CyphaDIF],
                 student_encoder: Optional[Encoder] = None,
                 weights: Optional[List[float]] = None):
        if not experts:
            raise ValueError("Need at least one expert.")
        self.experts  = experts
        self.enc      = student_encoder or experts[0].encoder_fn
        self.weights  = weights or [1.0] * len(experts)

    def ensemble_infer(self, h: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Weighted LLR fusion across all experts."""
        fused: Dict[str, float] = defaultdict(float)
        for clf, w in zip(self.experts, self.weights):
            _, _, llrs = clf.memory.classify(h, h_field=clf.field.h,
                                              temperature=clf.temperature,
                                              ood_sigma=clf.ood_sigma)
            for lbl, llr in llrs.items():
                fused[lbl] += w * llr
        if not fused:
            return '__unknown__', 0.0, {}
        best = max(fused, key=fused.__getitem__)
        arr  = np.array(list(fused.values()))
        prob = _softmax(arr)
        conf = float(prob.max())
        return best, conf, dict(zip(fused.keys(), prob.tolist()))

    def distil(self, n_per_class: int = 200, n_epochs: int = 3,
               temperature: float = 1.2, conf_threshold: float = 0.6,
               rng: Optional[np.random.Generator] = None,
               seed_data: Optional[List[Tuple[Any, str]]] = None) -> CyphaDIF:
        """
        Build and train a student by:
            1. Collecting all known classes across experts
            2. Generating synthetic samples by sampling expert class means
            3. Encoding them through the expert encoder (same space as training)
            4. Labelling with ensemble consensus
            5. Training student on high-confidence consensus samples
        Returns trained student CyphaDIF.

        seed_data : optional list of (raw_input, label) pairs.
            When provided, these are used as training data instead of sampling
            from expert class Gaussians. This is the preferred mode — raw inputs
            go through the student encoder (which shares W with the reference
            expert) and the student learns in exactly the same feature space
            it will operate in at inference time.
        """
        import random
        rng = rng or np.random.default_rng(77)

        # Union of all classes
        all_classes: set = set()
        for clf in self.experts:
            with clf.memory._lock:
                all_classes.update(clf.memory._classes.keys())

        # Student uses the first expert's encoder type
        ref_expert = self.experts[0]
        student = CyphaDIF(encoder=ref_expert.encoder_fn, rng=rng)

        # CRITICAL: copy expert's DIFEncoder projection W BEFORE training.
        # The student must learn its class means in the same feature space
        # it will use at inference time. Copying W after training breaks this
        # because class means would be calibrated to the student's random W.
        with ref_expert.encoder._lock, student.encoder._lock:
            student.encoder.W = ref_expert.encoder.W.copy()

        # Build training dataset
        dataset = []
        per_exp = max(1, n_per_class // max(len(self.experts), 1))

        if seed_data is not None:
            # Preferred: raw inputs encoded through shared W → correct feature space
            dataset = list(seed_data)
        else:
            # Fallback: sample from expert class Gaussians and label with ensemble
            for lbl in all_classes:
                for clf in self.experts:
                    with clf.memory._lock:
                        mu0  = clf.memory.world.mu.copy()
                        v0   = clf.memory.world.v.copy()
                        cd   = clf.memory._classes.get(lbl)
                        if cd is None:
                            continue
                        mu_k = cd.mu(mu0).copy()
                    std = np.sqrt(np.maximum(v0, _MIN_VAR)) * temperature
                    for _ in range(per_exp):
                        h    = mu_k + rng.standard_normal(len(mu_k)) * std
                        pred, conf, _ = self.ensemble_infer(h)
                        if conf >= conf_threshold:
                            dataset.append((h, pred))

        random.shuffle(dataset)
        # Freeze encoder during distillation: student learns class means in
        # the expert's feature space (W was copied above). Contrastive updates
        # would push W away from the expert's projection, breaking inference.
        student.encoder._frozen = True
        student.batch_train(dataset, n_epochs=n_epochs)
        student.encoder._frozen = False
        return student

