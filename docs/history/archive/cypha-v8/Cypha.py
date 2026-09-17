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

Update (natural gradient, Cramér-Rao optimal):
  Attraction (true class k):   Δk  +=  η · residual(h, θk)
  Repulsion  (wrong class j):  Δj  −=  η · wj · residual(h, θj)
  MDL decay  (all classes):    Δk  *=  (1 − λ)
  World prior (all obs):       θ₀  updated via Welford

Encoder feedback (contrastive):
  When pred ≠ true:  W_enc  +=  η_enc · (r_wrong − r_right) ⊗ raw_features
  where r_k = (h − μk) / vk  is the Fisher-Rao score residual

Structural parser:
  3-scale position-indexed features (character, word, structural)
  + cross-scale statistics
  → breaks bag-of-bytes fallacy: position 0 = '4d' ≠ position 10 = '4d'
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_EPS          = 1e-8
_FEAT_DIM     = 128      # structural parser output dim
_STAT_DIM     = 256      # sufficient statistics dim (2 × _FEAT_DIM)
_FIELD_DIM    = 128      # temporal field state dim
_MIN_VAR      = 1e-4     # minimum variance floor
_MDL_LAMBDA   = 0.002    # MDL decay base rate (adaptive per class)
_REPULSE_CAP  = 0.5      # max repulsion weight
_ENC_LR       = 0.002    # encoder contrastive learning rate
_DELTA_LR     = 0.08     # differential offset learning rate
_WORLD_LR     = 0.02     # world prior learning rate (poll: ema_alpha_store=0.2, scaled for online)
_CONTEXT_WIN  = 64       # context buffer window size

# Calibration constants
_TEMP_INIT    = 2.5    # initial softmax temperature (poll: temperature_init=2.5)
_TEMP_FLOOR   = 2.5    # temperature is fixed — poll proves decay=1.0 is optimal
_OOD_SIGMA    = 15.0   # OOD gate width (nats); adapts via EMA during training
_OOD_EMA      = 0.01   # EMA rate for tracking in-distribution LLR percentile


# ─────────────────────────────────────────────────────────────────────────────
# Math utilities
# ─────────────────────────────────────────────────────────────────────────────

def _diag_gaussian_logpdf(h: np.ndarray, mu: np.ndarray, v: np.ndarray) -> float:
    """
    Log-likelihood of h under N(mu, diag(v)).
    v must be positive (use _MIN_VAR floor).
    """
    v_safe = np.maximum(v, _MIN_VAR)
    return float(-0.5 * np.sum(np.log(v_safe) + (h - mu) ** 2 / v_safe))


def _fisher_rao_residual(h: np.ndarray, mu: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Score residual = ∂ log p(h | θ) / ∂μ = (h − μ) / v.
    This is the natural gradient direction for the mean parameter.
    """
    return (h - mu) / np.maximum(v, _MIN_VAR)


def _mean_residual(h: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Mean update residual (natural gradient, Welford-equivalent)."""
    return h - mu


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + _EPS)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    return x - np.log(np.sum(np.exp(x)) + _EPS)


# ─────────────────────────────────────────────────────────────────────────────
# StructuralParser  —  position-indexed 3-scale feature extraction
# ─────────────────────────────────────────────────────────────────────────────

# Keyword patterns: (name, expected_start_position, byte_pattern)
_KEYWORDS: List[Tuple[str, int, bytes]] = [
    # HTTP method — position-exact (strongest discriminator)
    ('GET',        0,  b'GET '),
    ('POST',       0,  b'POST'),
    # C2 beacon paths — appear at position 5 in "POST /beacon" etc.
    ('C2_BEACON',  5,  b'/bea'),
    ('C2_CHECK',   5,  b'/che'),
    ('C2_UPDATE',  5,  b'/upd'),
    ('C2_HEART',   5,  b'/hea'),
    ('C2_PING',    5,  b'/pin'),
    ('C2_TASK',    5,  b'/tas'),
    # Normal HTTP paths
    ('PATH_API',   5,  b'/api'),
    ('PATH_STAT',  5,  b'/sta'),
    ('PATH_IDX',   5,  b'/ind'),
    # Network protocols
    ('TCP',        0,  b'TCP '),
    ('UDP',        0,  b'UDP '),
    ('DNS',        0,  b'DNS '),
    ('HTTP',       0,  b'HTTP'),
    # Log severity — position-exact
    ('LOG_I',      0,  b'[INF'),
    ('LOG_W',      0,  b'[WAR'),
    ('LOG_E',      0,  b'[ERR'),
    # Scan / DDoS markers
    ('SYN',        4,  b'SYN'),
    ('FLOOD',      4,  b'floo'),
    ('PPS',        4,  b'pps'),
    ('SRC',        4,  b'src='),
    ('ARROW',      4,  b'->'),
    # DNS exfil
    ('DNS_TXT',    4,  b'TXT '),
    ('DNS_A',      4,  b'A '),
    # Exfil domain suffixes (searched in full string)
    ('EVIL_DOM',   0,  b'evil'),
    ('BADACTOR',   0,  b'bada'),
    ('EXFIL_D',    0,  b'exfi'),
    # Log body keywords
    ('PID',        0,  b'pid='),
    ('DISK',       0,  b'disk'),
    ('PCT',        0,  b'%'),
]

# Length bands (log-scale)
_LEN_BANDS = [8, 16, 32, 64, 128, 256, 512, 1024]


class StructuralParser:
    """
    Extracts _FEAT_DIM = 128 position-indexed structural features from raw bytes.

    Feature layout:
      [0:8]    Character scale — raw byte values at positions 0-7 (/ 255)
      [8:16]   Character scale — raw byte values at positions 8-15 (/ 255)
      [16:24]  Positional bigrams — hash of (pos, byte0, byte1) pairs at pos 0..7
      [24:56]  Word scale — 32 soft keyword match scores
      [56:72]  Global statistics — 16 dims
      [72:80]  Header statistics (first 16 bytes) — 8 dims
      [80:88]  Body statistics (remaining bytes) — 8 dims
      [88:104] Cross statistics — 8 header categories × 2 body stats
      [104:112] Length encoding — 8 dims (one-hot log-scale)
      [112:128] Transition statistics — 16 dims (run lengths, byte transitions)
    """

    def __call__(self, x: Any) -> np.ndarray:
        # Normalise input to bytes
        if   isinstance(x, str):                 raw = x.encode('utf-8', 'replace')
        elif isinstance(x, (bytes, bytearray)):  raw = bytes(x)
        elif isinstance(x, np.ndarray):          raw = x.astype(np.uint8).tobytes()
        else:                                    raw = str(x).encode('utf-8', 'replace')
        if not raw:
            raw = b'\x00'

        b   = np.frombuffer(raw, dtype=np.uint8)
        n   = len(b)
        fn  = 1.0 / max(n, 1)
        out = np.zeros(_FEAT_DIM, dtype=np.float32)

        # ── Block 1 & 2: positional byte values [0:16] ─────────────────────
        for i in range(min(8, n)):
            out[i] = b[i] / 255.0
        for i in range(min(8, max(0, n - 8))):
            out[8 + i] = b[8 + i] / 255.0

        # ── Block 3: positional bigrams [16:24] ────────────────────────────
        for i in range(min(8, n - 1)):
            # Map (position, byte_a, byte_b) → [0,1] deterministically
            h = (i * 256 * 256 + int(b[i]) * 256 + int(b[i + 1])) % 65521
            out[16 + i] = h / 65521.0

        # ── Block 4: keyword scores [24:56] ────────────────────────────────
        rl = raw.lower()
        for ki, (_, pos, pat) in enumerate(_KEYWORDS):
            if ki >= 32:
                break
            # Check at expected position (hard match = 1.0)
            if pos + len(pat) <= n and raw[pos:pos + len(pat)] == pat:
                out[24 + ki] = 1.0
            # Soft match — 64-byte window covers domain suffixes in typical payloads
            elif pat in raw[:64]:
                out[24 + ki] = 0.5
            elif pat in rl[:64]:
                out[24 + ki] = 0.3
            # Extended search for domain/body keywords beyond 64 bytes
            elif pat in raw:
                out[24 + ki] = 0.15
            elif pat in rl:
                out[24 + ki] = 0.1

        # ── Block 5: global statistics [56:72] ─────────────────────────────
        bm = float(b.mean())
        bv = float(b.var()) + _EPS
        p = np.bincount(b, minlength=256).astype(np.float32) * fn
        ph = p[p > 0]
        entropy = float(-np.dot(ph, np.log2(ph + _EPS)))
        diff = np.diff(b.view(np.int8)) if n > 1 else np.zeros(1, np.int8)
        out[56] = bm / 255.0
        out[57] = bv / 65025.0
        out[58] = entropy / 8.0
        out[59] = float(np.count_nonzero(b == 0)) * fn
        out[60] = float(np.count_nonzero(b > 127)) * fn
        out[61] = float(np.count_nonzero((b >= 32) & (b < 127))) * fn
        out[62] = float(np.count_nonzero((b >= 48) & (b <= 57))) * fn
        out[63] = float(np.count_nonzero(((b >= 65) & (b <= 90)) | ((b >= 97) & (b <= 122)))) * fn
        out[64] = raw.count(b' ')  * fn
        out[65] = raw.count(b'\n') * fn
        out[66] = raw.count(b':')  * fn
        out[67] = raw.count(b'/')  * fn
        out[68] = raw.count(b'.')  * fn
        out[69] = float(np.count_nonzero(diff)) / max(n - 1, 1)
        out[70] = float(len(set(raw))) / 256.0
        out[71] = min(n / 256.0, 1.0)

        # ── Block 6: header stats (first 16 bytes) [72:80] ─────────────────
        hdr = b[:min(16, n)]
        if len(hdr) > 0:
            ph2 = np.bincount(hdr, minlength=256).astype(np.float32) / len(hdr)
            ph2 = ph2[ph2 > 0]
            h_ent = float(-np.dot(ph2, np.log2(ph2 + _EPS)))
            out[72] = float(hdr.mean()) / 255.0
            out[73] = float(hdr.var() + _EPS) / 65025.0
            out[74] = h_ent / 8.0
            out[75] = float(np.count_nonzero(hdr == 0)) / len(hdr)
            out[76] = float(np.count_nonzero(hdr > 127)) / len(hdr)
            out[77] = float(np.count_nonzero((hdr >= 32) & (hdr < 127))) / len(hdr)
            out[78] = float(np.count_nonzero((hdr >= 48) & (hdr <= 57))) / len(hdr)
            out[79] = float(np.count_nonzero(((hdr >= 65) & (hdr <= 90)) | ((hdr >= 97) & (hdr <= 122)))) / len(hdr)

        # ── Block 7: body stats (bytes 16+) [80:88] ────────────────────────
        body = b[16:] if n > 16 else np.array([], dtype=np.uint8)
        if len(body) > 0:
            pb2 = np.bincount(body, minlength=256).astype(np.float32) / len(body)
            pb2 = pb2[pb2 > 0]
            b_ent = float(-np.dot(pb2, np.log2(pb2 + _EPS)))
            out[80] = float(body.mean()) / 255.0
            out[81] = float(body.var() + _EPS) / 65025.0
            out[82] = b_ent / 8.0
            out[83] = float(np.count_nonzero(body == 0)) / len(body)
            out[84] = float(np.count_nonzero(body > 127)) / len(body)
            out[85] = float(np.count_nonzero((body >= 32) & (body < 127))) / len(body)
            out[86] = float(np.count_nonzero((body >= 48) & (body <= 57))) / len(body)
            out[87] = float(np.count_nonzero(((body >= 65) & (body <= 90)) | ((body >= 97) & (body <= 122)))) / len(body)

        # ── Block 8: cross-scale statistics [88:104] ───────────────────────
        # 8 header category indicators × 2 body stats
        # Path slice used to distinguish normal POST from C2 POST
        _post_path = raw[5:9] if len(raw) > 9 else b''
        _c2_paths  = (b'/bea', b'/che', b'/upd', b'/hea', b'/pin', b'/tas')
        hdr_cats = [
            raw[:4] == b'GET ',                                   # GET = normal HTTP
            raw[:4] == b'POST' and _post_path not in _c2_paths,  # POST to normal path
            raw[:4] == b'POST' and _post_path in _c2_paths,      # POST to C2 path
            raw[:4] in (b'TCP ', b'UDP '),                        # network protocols
            raw[:4] == b'DNS ',                                   # DNS
            raw[:1] == b'[',                                      # log lines
            raw[:2] in (b'MZ', b'4d') or raw[:4] in (b'\x7fELF', b'7f45'),  # binary
            b'flood' in raw or b'pps' in raw,                     # DDoS
        ]
        body_ent  = float(out[82]) if len(body) > 0 else float(out[58])
        body_mean = float(out[80]) if len(body) > 0 else float(out[56])
        for ci, cat in enumerate(hdr_cats):
            out[88 + 2*ci]     = body_ent  if cat else 0.0
            out[88 + 2*ci + 1] = body_mean if cat else 0.0

        # ── Block 9: length encoding [104:112] ─────────────────────────────
        for li, threshold in enumerate(_LEN_BANDS):
            out[104 + li] = 1.0 if n >= threshold else float(n) / threshold

        # ── Block 10: transition statistics [112:128] ──────────────────────
        if n > 1:
            d8 = np.abs(diff).astype(np.float32)
            out[112] = float(d8.mean()) / 255.0              # mean abs transition
            out[113] = float(d8.max())  / 255.0              # max transition
            out[114] = float(np.count_nonzero(d8 == 0)) / (n - 1)  # run length frac
            out[115] = float(np.count_nonzero(d8 > 64)) / (n - 1)  # large jumps
            # Bigram entropy (16 buckets)
            pairs = b[:-1].astype(np.uint32) * 256 + b[1:].astype(np.uint32)
            bg = np.bincount((pairs * 16) >> 16, minlength=16).astype(np.float32)
            bg /= (bg.sum() + _EPS)
            bg_ent = float(-np.dot(bg[bg > 0], np.log2(bg[bg > 0] + _EPS)))
            out[116] = bg_ent / 4.0
            # Byte position entropy (first vs last half)
            if n >= 4:
                h1 = b[:n//2]; h2 = b[n//2:]
                p1 = np.bincount(h1, minlength=256).astype(np.float32) / len(h1)
                p2 = np.bincount(h2, minlength=256).astype(np.float32) / len(h2)
                p1_h = p1[p1 > 0]; p2_h = p2[p2 > 0]
                out[117] = float(-np.dot(p1_h, np.log2(p1_h + _EPS))) / 8.0
                out[118] = float(-np.dot(p2_h, np.log2(p2_h + _EPS))) / 8.0
                # Entropy difference: positive = first half more structured
                out[119] = (out[118] - out[117])

        return out


# ─────────────────────────────────────────────────────────────────────────────
# EncoderProjection  —  trainable W_enc with contrastive Fisher-Rao feedback
# ─────────────────────────────────────────────────────────────────────────────

class EncoderProjection:
    """
    Linear projection W ∈ R^{D×D}: raw_features → latent h.

    Initialised via PCA-like random orthogonal matrix (better than pure random).
    Updated by contrastive Fisher-Rao gradient:
      When true=k, pred=j (wrong):
        W += η · (r_j − r_k) ⊗ raw_features
      where r_k = (h − μk) / vk  is the Fisher-Rao score at class k.

    This pushes W to produce h that is far from wrong-class means and
    close to the true-class mean, in the natural geometry of the model.
    """

    def __init__(self, dim: int = _FEAT_DIM, rng: Optional[np.random.Generator] = None):
        self.d   = dim
        self._rng = rng or np.random.default_rng(42)
        # Initialise as random orthogonal matrix (QR decomposition of random Gaussian)
        Q, _ = np.linalg.qr(self._rng.normal(0, 1.0, (dim, dim)))
        self.W = (Q * 0.5).astype(np.float64)
        self._lock = threading.Lock()
        self._update_count = 0

    def project(self, f: np.ndarray) -> np.ndarray:
        """Project raw features to latent space. h = W @ f"""
        with self._lock:
            return self.W @ f

    def contrastive_update(self,
                           f      : np.ndarray,
                           h      : np.ndarray,
                           mu_k   : np.ndarray, v_k   : np.ndarray,
                           mu_j   : np.ndarray, v_j   : np.ndarray,
                           weight : float = 1.0) -> None:
        """
        Update W when true class = k but predicted class = j.
        Gradient: (r_j − r_k) ⊗ f, where r_k = (h − μk) / vk.
        """
        r_k = _fisher_rao_residual(h, mu_k, v_k)   # should be small (h near μk)
        r_j = _fisher_rao_residual(h, mu_j, v_j)   # should be large (h far from μj)
        # We want r_k small and r_j large → gradient pushes W in direction (r_j − r_k) ⊗ f
        grad = np.outer(r_j - r_k, f)
        with self._lock:
            self.W += _ENC_LR * weight * grad
            # Spectral normalisation: keep max singular value ≤ 1.5
            self._update_count += 1
            if self._update_count % 50 == 0:
                sv = np.linalg.norm(self.W, ord=2)  # largest singular value
                if sv > 1.5:
                    self.W *= 1.5 / sv

    def align_to_offsets(self, delta_mus: List[np.ndarray]) -> None:
        """
        Periodic alignment: rotate W toward the subspace spanned by the
        differential offsets {Δμk}. This ensures the projection allocates
        dimensions to class-discriminative directions.
        Only called every N steps (expensive: SVD).
        """
        if len(delta_mus) < 2:
            return
        D = np.stack([d for d in delta_mus if np.linalg.norm(d) > _EPS], axis=0)
        if len(D) < 2:
            return
        # SVD of delta matrix to find top discriminant directions
        try:
            U, s, _ = np.linalg.svd(D, full_matrices=False)
            top_k = min(len(s), self.d // 4)
            # Gentle nudge toward top discriminant directions (not a full replacement)
            with self._lock:
                for i in range(top_k):
                    v = D[i] / (np.linalg.norm(D[i]) + _EPS)
                    # Project W so that W^T @ v has high magnitude
                    wv = self.W.T @ v
                    if np.linalg.norm(wv) < 0.1:
                        # This direction is under-represented: nudge W
                        self.W += 0.01 * np.outer(v, v / (np.linalg.norm(v) + _EPS))
        except np.linalg.LinAlgError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# WorldPrior  —  shared NIG base distribution θ₀
# ─────────────────────────────────────────────────────────────────────────────

class WorldPrior:
    """
    The world prior θ₀: a diagonal Gaussian model of the typical input,
    fitted online via Welford's algorithm from all observations regardless
    of class label.

    All class models are defined relative to θ₀:
        θk = θ₀ ⊕ Δk  →  μk = μ₀ + Δμk,  vk = v₀ · exp(Δlogvk)

    The world prior provides:
      1. A strong inductive bias for few-shot classification (1-shot Δk is
         meaningful because it's measured against a well-fitted background).
      2. Domain drift handling: as the domain shifts (network → binary),
         θ₀ drifts, and all class models shift coherently.
      3. MDL reference: ||Δk||_F is measured in the Fisher-Rao geometry
         induced by θ₀, giving a proper complexity measure for each class.

    Optional field conditioning:
        θ₀(t) = θ_global + F_field @ h_field(t)
      where F_field ∈ R^{D × field_dim} maps temporal field state to
      world prior corrections.
    """

    def __init__(self, dim: int = _FEAT_DIM,
                 field_dim: int = _FIELD_DIM,
                 rng: Optional[np.random.Generator] = None):
        self.d = dim
        self._rng = rng or np.random.default_rng(0)
        # Welford accumulators
        self._n    = 0
        self.mu    = np.zeros(dim)
        self.v     = np.ones(dim)      # running variance
        self._M2   = np.ones(dim)      # Welford M2

        # Field conditioning: small learned map from field state to world prior shift
        self.F_field = self._rng.normal(0, 0.001, (dim, field_dim))
        self._lock   = threading.Lock()

    def update(self, h: np.ndarray, lr: float = _WORLD_LR) -> None:
        """
        Online update of world prior via Welford algorithm.
        Burn-in for first 20 observations (exact), then EMA with forgetting.
        """
        with self._lock:
            self._n += 1
            if self._n <= 20:
                delta     = h - self.mu
                self.mu  += delta / self._n
                self._M2 += delta * (h - self.mu)
                if self._n > 1:
                    self.v = np.maximum(self._M2 / (self._n - 1), _MIN_VAR)
            else:
                delta    = h - self.mu
                self.mu += lr * delta
                self.v   = np.maximum((1 - lr) * self.v + lr * delta * delta, _MIN_VAR)

    def condition_on_field(self, h_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return context-conditioned world prior:
          μ₀(t) = μ₀ + F_field @ h_field(t)
          v₀(t) = v₀  (variance unchanged for stability)
        """
        with self._lock:
            mu_cond = self.mu + self.F_field @ h_field
            return mu_cond, self.v.copy()

    def update_field_map(self, h_field: np.ndarray, residual: np.ndarray,
                         lr: float = 0.0001) -> None:
        """Update F_field to reduce world prior residual given field state."""
        with self._lock:
            self.F_field += lr * np.outer(residual, h_field)


# ─────────────────────────────────────────────────────────────────────────────
# ClassDifferential  —  per-class differential offset Δk
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassDifferential:
    """
    Differential offset for one class k.

    The class model is θk = θ₀ ⊕ Δk:
        μk  =  μ₀ + Δμ   (mean offset from world prior)
        vk  =  v₀         (variance owned by world prior — not per-class)

    Updating per-class variance in online learning drives collapse: the
    log-variance residual always pushes toward tighter distributions as
    the model converges, eventually making every off-distribution input
    score catastrophically low. Fixing vk = v₀ avoids this entirely.

    Natural gradient update:
        Δμ    += η · (h − μk)       [attraction]
        Δμ    *= (1 − λ_eff)        [MDL decay]

    Fisher-Rao norm:
        ||Δk||² = ||Δμ||²_v₀ = Σ_d Δμd²/v₀d
    """
    label    : str
    dim      : int
    delta_mu : np.ndarray = field(default_factory=lambda: np.zeros(_FEAT_DIM))
    n_obs    : int  = 0
    n_correct: int  = 0

    def mu(self, mu0: np.ndarray) -> np.ndarray:
        return mu0 + self.delta_mu

    def v(self, v0: np.ndarray) -> np.ndarray:
        return v0

    def log_likelihood(self, h: np.ndarray, mu0: np.ndarray, v0: np.ndarray) -> float:
        return _diag_gaussian_logpdf(h, self.mu(mu0), v0)

    def attract(self, h: np.ndarray, mu0: np.ndarray, v0: np.ndarray,
                lr: float = _DELTA_LR) -> None:
        mu_k = self.mu(mu0)
        self.delta_mu += lr * _mean_residual(h, mu_k)
        self.n_obs += 1

    def repel(self, h: np.ndarray, mu0: np.ndarray, v0: np.ndarray,
              weight: float = 1.0, lr: float = _DELTA_LR) -> None:
        mu_k = self.mu(mu0)
        self.delta_mu -= lr * min(weight, _REPULSE_CAP) * _mean_residual(h, mu_k)

    def mdl_decay(self, lam: float = _MDL_LAMBDA) -> None:
        """
        MDL regularisation: shrink offset toward zero (world prior).
        Adaptive: decay rate scales down for data-rich classes but never
        below lam/8 — ensures equilibrium point stays bounded even for
        many observations.
        """
        lam_eff = lam * max(0.125, 1.0 / (1.0 + self.n_obs / 16.0))
        self.delta_mu *= (1.0 - lam_eff)

    def fisher_rao_norm(self, v0: np.ndarray) -> float:
        return float(np.sum(self.delta_mu ** 2 / np.maximum(v0, _MIN_VAR)))

    def accuracy(self) -> float:
        if self.n_obs == 0:
            return 0.0
        return self.n_correct / self.n_obs


# ─────────────────────────────────────────────────────────────────────────────
# DIFMemory  —  manages world prior + all class differentials
# ─────────────────────────────────────────────────────────────────────────────

class DIFMemory:
    """
    Core DIF statistical memory.

    Owns:
      - WorldPrior θ₀
      - Dict[label → ClassDifferential Δk]

    Classification:
      score(k) = log p(h | θ₀ ⊕ Δk) + log p(k | context)
      y* = argmax_k score(k)

    Training (one step):
      1. Predict y_pred
      2. Attract true class k
      3. Repel competing classes weighted by their posterior probability
      4. Apply MDL decay to all classes
      5. Update world prior
      6. If mismatch: signal encoder contrastive update
    """

    def __init__(self, dim: int = _FEAT_DIM, field_dim: int = _FIELD_DIM,
                 rng: Optional[np.random.Generator] = None):
        self.d          = dim
        self.world      = WorldPrior(dim=dim, field_dim=field_dim, rng=rng)
        self._classes   : Dict[str, ClassDifferential] = {}
        self._lock      = threading.Lock()
        self._step      = 0

    def _get_or_create(self, label: str) -> ClassDifferential:
        if label not in self._classes:
            cd = ClassDifferential(label=label, dim=self.d,
                                   delta_mu=np.zeros(self.d))
            self._classes[label] = cd
        return self._classes[label]

    def classify(self,
                 h            : np.ndarray,
                 h_field      : Optional[np.ndarray] = None,
                 context_prior: Optional[Dict[str, float]] = None,
                 temperature  : float = _TEMP_INIT,
                 ood_sigma    : float = _OOD_SIGMA,
                 ) -> Tuple[str, float, Dict[str, float]]:
        """
        Calibrated classification via log-likelihood ratios (LLRs).

        LLR_k(h) = log p(h | theta0 + Dk) - log p(h | theta0) - U_k

        confidence = sigmoid(max_k LLR_k / ood_sigma)   [OOD gate]
                   * max_k softmax(LLR / temperature)    [class gate]

        OOD gate  : collapses toward 0 when all classes score below
                    the world prior (max_llr < 0 => genuinely novel input).
        Class gate : temperature-scaled discrimination over LLRs.
        Epistemic  : U_k = mean(v_k) / (n_obs_k + 1) penalises poorly-fit classes.
        Both gates must be high for confidence to be high.
        """
        with self._lock:
            if not self._classes:
                return '__unknown__', 0.0, {}

            if h_field is not None:
                mu0, v0 = self.world.condition_on_field(h_field)
            else:
                mu0, v0 = self.world.mu.copy(), self.world.v.copy()

            ll_world = _diag_gaussian_logpdf(h, mu0, v0)

            llrs = {}
            for lbl, cd in self._classes.items():
                ll_k = cd.log_likelihood(h, mu0, v0)
                u_k  = float(np.mean(v0)) / (cd.n_obs + 1)
                ctx  = context_prior.get(lbl, 0.0) if context_prior else 0.0
                llrs[lbl] = (ll_k - ll_world) - u_k + ctx

        if not llrs:
            return '__unknown__', 0.0, {}

        best    = max(llrs, key=llrs.__getitem__)
        llr_arr = np.array(list(llrs.values()))
        max_llr = float(llr_arr.max())

        # OOD gate: sigmoid(max_llr / ood_sigma)
        # max_llr > 0 => best class beats world prior => in-distribution
        # max_llr < 0 => world prior beats all classes => OOD
        ood_gate = float(1.0 / (1.0 + math.exp(-max_llr / (ood_sigma + _EPS))))

        # Class discrimination: temperature-scaled softmax over LLRs
        probs = _softmax(llr_arr / (temperature + _EPS))
        disc  = float(probs[list(llrs.keys()).index(best)])

        return best, ood_gate * disc, llrs

    def train(self,
              h          : np.ndarray,
              label      : str,
              h_field    : Optional[np.ndarray] = None,
              context_prior: Optional[Dict[str, float]] = None
              ) -> Tuple[str, bool, float]:
        """
        One training step.
        Returns (predicted_label, was_correct, loss).
        """
        with self._lock:
            self._step += 1
            # Get world prior
            if h_field is not None:
                mu0, v0 = self.world.condition_on_field(h_field)
            else:
                mu0, v0 = self.world.mu.copy(), self.world.v.copy()

            # Ensure true class exists
            cd_k = self._get_or_create(label)

            # Score all classes
            scores = {}
            for lbl, cd in self._classes.items():
                ll = cd.log_likelihood(h, mu0, v0)
                ctx = context_prior.get(lbl, 0.0) if context_prior else 0.0
                scores[lbl] = ll + ctx

            pred = max(scores, key=scores.__getitem__)
            correct = (pred == label)
            if correct:
                cd_k.n_correct += 1

            # Compute posterior weights for repulsion
            s_arr = np.array(list(scores.values()))
            probs = _softmax(s_arr)
            prob_map = dict(zip(scores.keys(), probs))

            # Attraction: pull true class toward h
            cd_k.attract(h, mu0, v0)

            # Repulsion: push all other classes away, weighted by their posterior
            for lbl, cd in self._classes.items():
                if lbl != label:
                    cd.repel(h, mu0, v0, weight=float(prob_map.get(lbl, 0.0)))

            # MDL decay on ALL classes every step
            for cd in self._classes.values():
                cd.mdl_decay()

            # Update world prior
            self.world.update(h)

            # Compute loss = negative log-likelihood of true class
            loss = -cd_k.log_likelihood(h, mu0, v0)

            # Return pred and context for encoder update
            pred_cd = self._classes.get(pred)
            true_cd = cd_k

        return pred, correct, float(loss)

    def get_class_params(self, label: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (μk, vk) for class label, or None if not registered."""
        with self._lock:
            if label not in self._classes:
                return None
            mu0, v0 = self.world.mu.copy(), self.world.v.copy()
            cd = self._classes[label]
            return cd.mu(mu0), v0.copy()

    def complexity(self) -> Dict[str, float]:
        """Fisher-Rao complexity (MDL cost) of each class."""
        with self._lock:
            v0 = self.world.v.copy()
            return {lbl: cd.fisher_rao_norm(v0) for lbl, cd in self._classes.items()}

    def accuracy(self) -> Dict[str, float]:
        with self._lock:
            return {lbl: cd.accuracy() for lbl, cd in self._classes.items()}


# ─────────────────────────────────────────────────────────────────────────────
# ContextBuffer  —  episodic history + context prior computation
# ─────────────────────────────────────────────────────────────────────────────

class ContextBuffer:
    """
    Maintains episodic history and computes the context prior
        p(k | context) = log π_k + log p_relational(k | recent_context)

    Two components:
      1. Frequency prior: log proportion of class k in recent window
      2. Relational prior: class co-occurrence given most recent predicted class

    Global state feeds directly into classification via this prior.
    """

    def __init__(self, window: int = _CONTEXT_WIN):
        self._window     = window
        self._history    : deque  = deque(maxlen=window)   # (label, was_correct)
        self._cooccur    : Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._last_label : Optional[str] = None
        self._lock       = threading.Lock()

    def record(self, label: str, correct: bool) -> None:
        with self._lock:
            if self._last_label is not None:
                self._cooccur[self._last_label][label] += 1
            self._history.append((label, correct))
            self._last_label = label

    def context_prior(self, classes: List[str]) -> Dict[str, float]:
        """
        Returns log prior probability for each class given recent context.
        Smoothed Dirichlet (pseudocount=1).
        """
        if not classes:
            return {}
        with self._lock:
            # Frequency prior from recent history
            counts = defaultdict(float)
            for lbl, _ in self._history:
                counts[lbl] += 1.0
            total = sum(counts.values()) + len(classes)  # Dirichlet smoothing
            freq  = {k: (counts[k] + 1.0) / total for k in classes}
            freq_logprob = {k: math.log(freq[k] + _EPS) for k in classes}

            # Co-occurrence prior from last observed label
            if self._last_label and self._last_label in self._cooccur:
                co = self._cooccur[self._last_label]
                co_total = sum(co.values()) + len(classes)
                co_logprob = {k: math.log((co.get(k, 0) + 1.0) / co_total + _EPS)
                              for k in classes}
            else:
                co_logprob = {k: 0.0 for k in classes}

            # Combine: 0.3 frequency + 0.2 co-occurrence
            combined = {k: 0.3 * freq_logprob[k] + 0.2 * co_logprob[k]
                        for k in classes}
        return combined

    def recent_accuracy(self) -> float:
        with self._lock:
            if not self._history:
                return 0.0
            return sum(c for _, c in self._history) / len(self._history)


# ─────────────────────────────────────────────────────────────────────────────
# NIGField  —  temporal state evolution (reused from CyphaOmega)
# ─────────────────────────────────────────────────────────────────────────────

class NIGField:
    """
    Diagonal-A temporal field: h_{t+1} = A_eff @ h_t + injection.
    4 timescale groups: τ = 0.30, 0.60, 0.85, 0.95.
    Spectral radius of W_T maintained ≤ 0.85 via power iteration.
    """

    def __init__(self, state_dim: int = _FIELD_DIM,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.d    = state_dim
        self._rng = rng or np.random.default_rng(2)

        g = state_dim // 4
        r = state_dim - 3 * g
        self._a = np.concatenate([
            np.full(g, 0.30), np.full(g, 0.60),
            np.full(g, 0.85), np.full(r, 0.95),
        ])

        W = self._rng.normal(0, 0.01, (state_dim, state_dim))
        v = self._rng.normal(0, 1, state_dim)
        for _ in range(20):
            v = W @ v; v /= (np.linalg.norm(v) + _EPS)
        sr = float(np.linalg.norm(W @ v))
        self._W_T     = W / (sr + _EPS) * 0.05
        self._wt_lock = threading.Lock()

        self._A_eff  = (np.diag(self._a) + self._W_T).astype(np.float32)
        self._h      = np.zeros(state_dim)
        self._step   = 0
        self._lock   = threading.Lock()
        self._sr_vec = v / (np.linalg.norm(v) + _EPS)

    def evolve(self, h: np.ndarray, update_state: bool = True) -> np.ndarray:
        h_new = (self._A_eff @ h.astype(np.float32)).astype(np.float64)
        if update_state:
            with self._lock:
                self._h    = h_new
                self._step += 1
        return h_new

    def inject(self, signal: np.ndarray, strength: float = 0.15) -> None:
        """Inject an external signal into the field state."""
        with self._lock:
            h_mag = float(np.linalg.norm(self._h)) + _EPS
            s_mag = float(np.linalg.norm(signal)) + _EPS
            self._h += strength * (h_mag / s_mag) * signal

    def update_causal(self, h_t: np.ndarray, h_target: np.ndarray,
                      lr: float = 0.0002) -> float:
        with self._wt_lock:
            err = self._W_T @ h_t - h_target
            W_new = self._W_T - lr * np.outer(err, h_t) / self.d
            v = self._sr_vec
            for _ in range(5):
                v = W_new @ v
                nv = np.linalg.norm(v)
                if nv < _EPS: break
                v /= nv
            sr = float(np.linalg.norm(W_new @ v))
            if sr > 0.85:
                W_new *= 0.85 / sr
            self._W_T    = W_new
            self._sr_vec = v
            self._A_eff  = (np.diag(self._a) + W_new).astype(np.float32)
        return float(err.dot(err) / self.d)

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


# Poll-derived constants (from 27,524-config brute-force sweep, top_fit=0.6206)
_DEDUP_THRESH    = 0.60   # cosine similarity above which two Δk offsets are "too similar"
_REPLAY_RATIO    = 0.30   # fraction of training steps that sample from replay buffer
_ALIGN_EVERY     = 500    # encoder alignment interval (poll: consolidate_every=500)
_DELIBERATE_LO   = 0.25   # deliberate mode: lower confidence bound
_DELIBERATE_HI   = 0.40   # deliberate mode: upper confidence bound (poll: deliberate_thresh=0.4)


# ─────────────────────────────────────────────────────────────────────────────
# ReplayBuffer  —  experience replay (poll: replay_ratio = 0.30)
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-size FIFO experience replay.
    Stores (encoded_h, raw_f, label) tuples.
    On sample(), returns a random subset of size int(replay_ratio * batch_size).

    Motivation: the poll found replay_ratio=0.30 is the 3rd most sensitive
    parameter. Replaying past experience prevents catastrophic forgetting
    when training is sequential (new class seen → old classes degraded).
    """

    def __init__(self, capacity: int = 2000,
                 rng: Optional[np.random.Generator] = None):
        self._cap  = capacity
        self._buf  : deque = deque(maxlen=capacity)
        self._rng  = rng or np.random.default_rng(5)
        self._lock = threading.Lock()

    def push(self, h: np.ndarray, f: np.ndarray, label: str) -> None:
        with self._lock:
            self._buf.append((h.copy(), f.copy(), label))

    def sample(self, n: int) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        with self._lock:
            if len(self._buf) < n:
                return list(self._buf)
            idx = self._rng.choice(len(self._buf), size=n, replace=False)
            buf = list(self._buf)
            return [buf[i] for i in idx]

    def by_class(self) -> Dict[str, List[np.ndarray]]:
        """Return all stored h vectors grouped by label."""
        with self._lock:
            groups: Dict[str, List[np.ndarray]] = defaultdict(list)
            for h, f, label in self._buf:
                groups[label].append(h)
            return dict(groups)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# ─────────────────────────────────────────────────────────────────────────────
# CyphaDIF  —  Differential Information Field main system
# ─────────────────────────────────────────────────────────────────────────────

class CyphaDIF:
    """
    Differential Information Field Classifier.

    A universal online classifier derived from first principles:
      ─ World prior θ₀ (all observations, Welford)
      ─ Class differentials Δk (per-class, natural parameter space)
      ─ Natural gradient updates (Cramér-Rao efficient)
      ─ MDL decay (Occam prior over class complexity)
      ─ Context prior (episodic frequency + co-occurrence)
      ─ Contrastive encoder feedback (Fisher-Rao gradient)
      ─ Temporal field (NIGField, integrates global state)
      ─ Position-indexed structural parser (breaks bag-of-bytes)

    Usage:
        clf = CyphaDIF()
        loss = clf.train_step("GET /index.html HTTP/1.1", "net_normal")
        label, conf = clf.infer("TCP SYN 1234->80")
    """

    def __init__(self,
                 feat_dim    : int = _FEAT_DIM,
                 field_dim   : int = _FIELD_DIM,
                 enc_lr      : float = _ENC_LR,
                 delta_lr    : float = _DELTA_LR,
                 world_lr    : float = _WORLD_LR,
                 mdl_lambda  : float = _MDL_LAMBDA,
                 context_win : int   = _CONTEXT_WIN,
                 rng         : Optional[np.random.Generator] = None,
                 n_workers   : int = 1):

        self._rng     = rng or np.random.default_rng(42)
        self.feat_dim = feat_dim
        self.field_dim = field_dim

        # Learning rates (overrideable)
        self.enc_lr    = enc_lr
        self.delta_lr  = delta_lr
        self.world_lr  = world_lr
        self.mdl_lambda = mdl_lambda

        # Core components
        self.parser  = StructuralParser()
        self.encoder = EncoderProjection(dim=feat_dim, rng=self._rng)
        self.memory  = DIFMemory(dim=feat_dim, field_dim=field_dim, rng=self._rng)
        self.context = ContextBuffer(window=context_win)
        self.field   = NIGField(state_dim=field_dim, rng=self._rng)

        # Statistics
        self._total_steps  = 0
        self._total_correct = 0
        self._loss_buf     : deque = deque(maxlen=100)
        self._align_every  = _ALIGN_EVERY    # poll: consolidate_every=500

        # Replay buffer (poll: replay_ratio=0.30)
        self.replay = ReplayBuffer(capacity=2000, rng=self._rng)

        # Calibration state — temperature and OOD sigma adapt online
        self.temperature = _TEMP_INIT   # softmax temperature; decays toward _TEMP_FLOOR
        self.ood_sigma   = _OOD_SIGMA   # OOD gate width; tracks in-distribution LLR median
        self._llr_ema    = 0.0          # EMA of max_k LLR for in-dist samples (training)

        # Thread pool for batch ops
        self._n_workers = n_workers

    def _encode(self, x: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        x → (raw_features, latent_h)
        raw_features: structural parser output, _FEAT_DIM
        latent_h:     encoded representation W_enc @ raw_features
        """
        f = self.parser(x).astype(np.float64)
        h = self.encoder.project(f)
        return f, h

    def infer(self, x: Any,
              use_field : bool = True) -> Tuple[str, float]:
        """
        Classify input x.
        Returns (predicted_label, calibrated_confidence).
        """
        f, h = self._encode(x)

        h_field = self.field.h if use_field else None

        with self.memory._lock:
            classes = list(self.memory._classes.keys())
        ctx_prior = self.context.context_prior(classes) if classes else {}

        pred, conf, _ = self.memory.classify(
            h, h_field=h_field, context_prior=ctx_prior,
            temperature=self.temperature, ood_sigma=self.ood_sigma)

        return pred, conf

    def _dedup_check(self, label: str) -> None:
        """
        Deduplication (poll: dedup_threshold=0.60, most sensitive parameter).
        If two class differential offsets are cosine-similar > 0.6, apply
        mutual repulsion proportional to the overlap. Prevents two classes
        from collapsing onto the same subspace of the latent space.
        """
        with self.memory._lock:
            classes = list(self.memory._classes.items())
            if len(classes) < 2:
                return
            cd_k = self.memory._classes.get(label)
            if cd_k is None:
                return
            nk = np.linalg.norm(cd_k.delta_mu) + _EPS
            for lbl_j, cd_j in classes:
                if lbl_j == label:
                    continue
                nj = np.linalg.norm(cd_j.delta_mu) + _EPS
                cos_sim = float(np.dot(cd_k.delta_mu, cd_j.delta_mu) / (nk * nj))
                if cos_sim > _DEDUP_THRESH:
                    # Mutual repulsion proportional to overlap above threshold
                    overlap = cos_sim - _DEDUP_THRESH
                    push = overlap * 0.5 * cd_j.delta_mu / nj
                    cd_k.delta_mu -= push
                    cd_j.delta_mu -= push

    def train_step(self, x: Any, label: str) -> float:
        """
        One online training step.
        Returns scalar loss (negative log-likelihood of true class).

        Incorporates poll findings:
          - replay_ratio=0.30: 30% of updates come from replay buffer
          - dedup_threshold=0.60: deduplicate overlapping class differentials
          - deliberate mode [0.10, 0.35]: extra contrastive update in uncertain zone
          - consolidate_every=500: encoder alignment at 500-step intervals
        """
        f, h = self._encode(x)

        # Push to replay before training (store fresh encoding)
        self.replay.push(h, f, label)

        # Context prior
        with self.memory._lock:
            classes = list(self.memory._classes.keys())
        if label not in classes:
            classes.append(label)
        ctx_prior = self.context.context_prior(classes)

        # Field state
        h_field = self.field.h

        # DIF update
        pred, correct, loss = self.memory.train(
            h, label,
            h_field      = h_field,
            context_prior = ctx_prior
        )

        # Contrastive encoder update on misclassification
        if not correct and pred != '__unknown__':
            params_k = self.memory.get_class_params(label)
            params_j = self.memory.get_class_params(pred)
            if params_k is not None and params_j is not None:
                mu_k, v_k = params_k
                mu_j, v_j = params_j
                self.encoder.contrastive_update(f, h, mu_k, v_k, mu_j, v_j)

        # Deliberate mode (poll: deliberate_lo=0.10, deliberate_hi=0.35)
        # When confidence is in the uncertain zone, apply a second softer
        # contrastive update even if the prediction was technically correct.
        # This sharpens the boundary in the regions where the model is unsure.
        _, conf, scores = self.memory.classify(h, h_field=h_field, context_prior=ctx_prior)
        if _DELIBERATE_LO < conf < _DELIBERATE_HI and len(scores) >= 2:
            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
            top_label  = sorted_scores[0][0]
            sec_label  = sorted_scores[1][0]
            params_top = self.memory.get_class_params(top_label)
            params_sec = self.memory.get_class_params(sec_label)
            if params_top is not None and params_sec is not None:
                mu_top, v_top = params_top
                mu_sec, v_sec = params_sec
                # Soft update: push top and second apart at half strength
                self.encoder.contrastive_update(
                    f, h, mu_top, v_top, mu_sec, v_sec, weight=0.3
                )

        # Deduplication (poll: most sensitive parameter)
        if self._total_steps % 5 == 0:
            self._dedup_check(label)

        # Replay step (poll: replay_ratio=0.30)
        if len(self.replay) >= 10 and np.random.random() < _REPLAY_RATIO:
            replay_samples = self.replay.sample(n=min(4, len(self.replay)))
            for rh, rf, rlabel in replay_samples:
                rctx = self.context.context_prior(
                    list(self.memory._classes.keys()))
                self.memory.train(rh, rlabel,
                                  h_field       = self.field.h,
                                  context_prior = rctx)

        # Update statistics
        self._total_steps += 1
        if correct:
            self._total_correct += 1
        self._loss_buf.append(loss)

        # Encoder alignment (poll: consolidate_every=500)
        if self._total_steps % self._align_every == 0:
            with self.memory._lock:
                deltas = [cd.delta_mu.copy()
                          for cd in self.memory._classes.values()
                          if np.linalg.norm(cd.delta_mu) > _EPS]
            if len(deltas) >= 2:
                self.encoder.align_to_offsets(deltas)

        # Field evolution
        inject_signal = h / (np.linalg.norm(h) + _EPS)
        self.field.inject(inject_signal, strength=0.05)
        h_old = self.field.h
        h_new = self.field.evolve(h_old, update_state=True)
        if self._total_steps % 10 == 0:
            self.field.update_causal(h_old, h_new, lr=0.0002)

        # Context record
        self.context.record(label, correct)

        # OOD sigma adapts to in-distribution LLR scale; temperature is fixed (poll: decay=1.0)
        if self._total_steps % 20 == 0:
            _, _, llrs_train = self.memory.classify(
                h, h_field=self.field.h,
                temperature=self.temperature, ood_sigma=self.ood_sigma)
            if llrs_train:
                max_llr = max(llrs_train.values())
                self._llr_ema = (1 - _OOD_EMA) * self._llr_ema + _OOD_EMA * max_llr
                self.ood_sigma = max(1.0, abs(self._llr_ema))

        return float(loss)

    def batch_train(self, data: List[Tuple[Any, str]],
                    n_epochs: int = 1, shuffle: bool = True) -> List[float]:
        """Train on a list of (input, label) pairs."""
        import random
        losses = []
        for ep in range(n_epochs):
            if shuffle:
                random.shuffle(data)
            for x, y in data:
                losses.append(self.train_step(x, y))
        return losses

    def macro_accuracy(self, data: List[Tuple[Any, str]]) -> Tuple[float, Dict[str, float]]:
        """Evaluate macro accuracy on held-out data."""
        from collections import defaultdict
        per = defaultdict(lambda: [0, 0])
        for x, y in data:
            pred, _ = self.infer(x)
            per[y][0] += int(pred == y)
            per[y][1] += 1
        per_cls = {k: v[0]/v[1] for k, v in per.items() if v[1] > 0}
        macro = sum(per_cls.values()) / len(per_cls) if per_cls else 0.0
        return macro, per_cls

    def diagnostics(self) -> Dict:
        """Return a diagnostic summary of the current model state."""
        acc = self.memory.accuracy()
        complexity = self.memory.complexity()
        recent_loss = float(np.mean(list(self._loss_buf))) if self._loss_buf else float('nan')
        n_classes = len(self.memory._classes)
        world = self.memory.world
        return {
            'total_steps'    : self._total_steps,
            'running_acc'    : self._total_correct / max(self._total_steps, 1),
            'recent_loss'    : recent_loss,
            'n_classes'      : n_classes,
            'context_acc'    : self.context.recent_accuracy(),
            'world_prior_mean_norm' : float(np.linalg.norm(world.mu)),
            'world_prior_var_mean'  : float(world.v.mean()),
            'class_accuracy' : acc,
            'class_complexity': complexity,
            'field_step'     : self.field.step,
        }

    def reset_field(self) -> None:
        """Reset temporal field state (use when domain changes dramatically)."""
        self.field.reset()

    def generate(self, label: str, n: int = 1,
                 rng: Optional[np.random.Generator] = None
                 ) -> List[np.ndarray]:
        """
        Sample n latent vectors from the classifier's own generative model.

        The classifier already defines p(h | k) = N(μk, v₀).
        Generation is just sampling from that — the same distribution
        used for classification. Returns h vectors directly.

            h ~ N(μk, v₀)

        These h vectors can be passed directly to memory.classify(),
        used for augmentation, anomaly scoring, or boundary probing.
        No reconstruction, no round-trip through W^{-1}.
        """
        if label not in self.memory._classes:
            raise KeyError(f"Unknown class '{label}'. "
                           f"Known: {list(self.memory._classes.keys())}")

        rng = rng or self._rng

        with self.memory._lock:
            mu0  = self.memory.world.mu.copy()
            v0   = self.memory.world.v.copy()
            mu_k = self.memory._classes[label].mu(mu0)

        std = np.sqrt(v0)
        return [mu_k + rng.standard_normal(len(mu_k)) * std for _ in range(n)]

    def __repr__(self) -> str:
        n = len(self.memory._classes)
        return (f"CyphaDIF(feat_dim={self.feat_dim}, field_dim={self.field_dim}, "
                f"n_classes={n}, steps={self._total_steps})")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark — real-world network/log/binary classification test
# ─────────────────────────────────────────────────────────────────────────────

def _make_templates():
    """Return dict of class_label → generator function."""
    import random as _r
    return {
        'net_normal' : lambda: "GET /index.html HTTP/1.1\r\nHost: example.com\r\n",
        'net_scan'   : lambda: f"TCP SYN {_r.randint(1,65535)}->{_r.randint(1,1024)}",
        'net_ddos'   : lambda: f"UDP flood {_r.randint(1000,9999)} pps src={_r.randint(1,254)}.x",
        'net_exfil'  : lambda: f"DNS TXT {''.join(_r.choices('abcdef0123456789',k=32))}.evil.com",
        'net_c2'     : lambda: f"POST /beacon HTTP/1.1\r\nContent-Length:{_r.randint(10,200)}\r\n",
        'log_info'   : lambda: f"[INFO] {time.time():.3f} process started pid={_r.randint(1000,9999)}",
        'log_warn'   : lambda: f"[WARN] disk usage {_r.randint(70,95)}% threshold exceeded",
        'log_error'  : lambda: f"[ERROR] connection refused {_r.randint(1,254)}.0.0.1:443",
        'bin_malware': lambda: bytes([0x4d,0x5a,0x90,0x00]+[_r.randint(0,255) for _ in range(28)]).hex(),
        'bin_benign' : lambda: bytes([0x7f,0x45,0x4c,0x46]+[_r.randint(0,255) for _ in range(28)]).hex(),
    }


def run_benchmark(n_train: int = 30, n_test: int = 30, n_epochs: int = 5,
                  verbose: bool = True) -> Dict:
    """Full benchmark comparing CyphaDIF to baseline."""
    import random
    random.seed(42)
    np.random.seed(42)

    TMPLS   = _make_templates()
    CLASSES = list(TMPLS.keys())

    train_data = [(TMPLS[c](), c) for c in CLASSES for _ in range(n_train)]
    test_data  = [(TMPLS[c](), c) for c in CLASSES for _ in range(n_test)]
    random.shuffle(train_data)

    clf = CyphaDIF()

    # Training with per-epoch accuracy tracking
    epoch_accs = []
    for ep in range(n_epochs):
        random.shuffle(train_data)
        losses = []
        for x, y in train_data:
            losses.append(clf.train_step(x, y))

        macro, per_cls = clf.macro_accuracy(test_data)
        epoch_accs.append(macro)
        if verbose:
            print(f"  Epoch {ep+1:2d}  macro={macro:.4f}  loss={np.mean(losses):.3f}")

    macro, per_cls = clf.macro_accuracy(test_data)

    if verbose:
        print(f"\nFinal per-class accuracy:")
        for c in CLASSES:
            a   = per_cls.get(c, 0.0)
            bar = '█' * int(a * 20) + '░' * (20 - int(a * 20))
            print(f"  {c:<15} {bar} {a:.2f}")

        diag = clf.diagnostics()
        print(f"\nDiagnostics:")
        print(f"  World prior ||μ₀||   = {diag['world_prior_mean_norm']:.3f}")
        print(f"  World prior E[v₀]    = {diag['world_prior_var_mean']:.4f}")
        print(f"  Running accuracy     = {diag['running_acc']:.4f}")
        print(f"  Recent loss          = {diag['recent_loss']:.3f}")
        print(f"\n  Class complexity (Fisher-Rao ||Δk||):")
        for k, v in sorted(diag['class_complexity'].items(), key=lambda x: -x[1]):
            print(f"    {k:<15} {v:.4f}")

    return {
        'macro'      : macro,
        'per_class'  : per_cls,
        'epoch_accs' : epoch_accs,
        'model'      : clf,
    }


def run_few_shot_benchmark(verbose: bool = True) -> None:
    """Test DIF on 1, 2, 5 shot scenarios — where few-shot matters most."""
    import random
    # Generate test data with a fixed seed BEFORE any training seeds are touched
    random.seed(9999)
    np.random.seed(9999)
    TMPLS   = _make_templates()
    CLASSES = list(TMPLS.keys())
    test_data = [(TMPLS[c](), c) for c in CLASSES for _ in range(20)]

    print("\n=== FEW-SHOT BENCHMARK ===")
    print(f"  {'shots':>6}  {'macro':>8}")
    print(f"  {'-'*20}")

    for shots in [1, 2, 3, 5, 10, 20, 30]:
        random.seed(77)
        np.random.seed(77)
        clf = CyphaDIF()
        train = [(TMPLS[c](), c) for c in CLASSES for _ in range(shots)]

        for ep2 in range(3):
            random.shuffle(train)
            for x, y in train:
                clf.train_step(x, y)

        macro, _ = clf.macro_accuracy(test_data)
        print(f"  {shots:>6}   {macro:>7.4f}")


def run_repeat_benchmark(verbose: bool = True) -> None:
    """Test DIF stability on repeated identical samples — the v3 failure mode."""
    import random
    random.seed(7)
    np.random.seed(7)

    TMPLS   = _make_templates()
    CLASSES = list(TMPLS.keys())
    test_data = [(TMPLS[c](), c) for c in CLASSES for _ in range(20)]

    print("\n=== REPEATED DATA STABILITY TEST ===")
    print("(Same 10 samples repeated N times — tests self-regulation)")
    print(f"  {'N_repeats':>10}  {'macro':>8}")
    print(f"  {'-'*25}")

    fixed_train = [(TMPLS[c](), c) for c in CLASSES]  # 1 fixed sample per class

    for n_reps in [1, 3, 5, 10, 20, 50]:
        random.seed(42)
        np.random.seed(42)
        clf = CyphaDIF()
        for _ in range(n_reps):
            for x, y in fixed_train:
                clf.train_step(x, y)
        macro, _ = clf.macro_accuracy(test_data)
        print(f"  {n_reps:>10}   {macro:>7.4f}")


if __name__ == '__main__':
    print("=" * 60)
    print("CyphaDIF — Differential Information Field Classifier")
    print("=" * 60)

    print("\n>>> Standard benchmark (30-shot, 5 epochs):")
    results = run_benchmark(n_train=30, n_test=30, n_epochs=5, verbose=True)

    run_few_shot_benchmark()
    run_repeat_benchmark()

    print(f"\n>>> Final macro accuracy: {results['macro']:.4f}")
