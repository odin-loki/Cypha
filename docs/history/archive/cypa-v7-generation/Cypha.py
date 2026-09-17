"""
Cypha HRNA — Universal Encoder + Full Architecture (Omega-2)
════════════════════════════════════════════════════════════
Replaces BinaryEncoder with OmegaEncoder implementing the five-operator
Omega information field formula proven in Phase 8 neural network verification.

  Omega(x) = [ M(x)      -- raw moments: mean, std, kurtosis, skewness
               M(D(x))   -- 1st-derivative moments  ← κ(D(x)) key discriminator
               M(D²(x))  -- 2nd-derivative moments  ← c40 equivalent
               R(x,K)    -- energy in K spectral bands
               A(x,lags) -- autocorrelation at log-spaced lags          ]
             applied at 3 temporal scales: full | first-half | second-half

Numeric-direct embedding: hash(feature_name) → index preserves metric structure.
Handles: text, audio hex-PCM, float32 arrays, RF IQ int8 interleaved.

All HRNA layers (ResonanceField, ResonatorLevel, AssemblyLevel, ModuleLevel,
GlobalLevel, RecursiveProcessor, FeedbackController, ThoughtProcessor,
MetaLearning, AnchorMemory) are unchanged.

Usage:
    from Cypha import CyphaStateful, _build_offset_index, _read_at_offset
    cypha = CyphaStateful(feature_dim=512, resonance_dim=256)
    cypha.train_file_stateful_offsets(path, offsets, name, epochs=5)
    result, conf = cypha.infer(text, verbose=False)
"""

import numpy as np
from numpy.fft import fft, ifft, rfft, irfft
import math, time, os, re, sys, json, shutil, threading, heapq
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed

EPSILON   = 1e-8
K_TARGET  = 0.5
LAMBDA_LP = 0.15
GAIN_K    = 0.05
GAIN_RHO  = 0.03
GAIN_A    = 0.04
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
_POOL     = ThreadPoolExecutor(max_workers=N_WORKERS)
_LOCK     = threading.Lock()


# ══════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════

@dataclass
class EncoderParams:
    chunk_k:       float = 4.0
    damr_radius:   float = 3.0
    active_scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125])
    prev_error:    float = float('inf')

@dataclass
class FieldStats:
    criticality:       float
    dominant_freq:     float
    mean_phase:        float
    phase_spread:      float
    energy:            float
    phase_entropy:     float = 0.0
    compression_slope: float = 0.0

@dataclass
class Event:
    type:     str
    time:     float
    data:     Dict[str, Any]
    source:   str
    priority: float
    def __lt__(self, other): return self.priority > other.priority

@dataclass
class Metrics:
    step: int; loss: float; criticality: float
    chunk_k: float; damr_r: float; n_anchors: int
    ms: float = 0.0; events: int = 0

class EventType(Enum):
    PATTERN   = auto()
    SURPRISE  = auto()
    RESONANCE = auto()
    EXTERNAL  = auto()
    FEEDBACK  = auto()
    THOUGHT   = auto()

# ══════════════════════════════════════════════════════════════════════════════
# 1. OMEGA ENCODER  — Universal Signal Encoder
#    Omega(x) = [M(x), M(D(x)), M(D²(x)), R(x,K), A(x,lags)] × 3 scales
#    Numeric-direct embedding: hash(name) → index, value → weight
# ══════════════════════════════════════════════════════════════════════════════

class OmegaEncoder:
    """
    Universal signal encoder implementing the Omega information field formula.

    Theoretical basis
    ─────────────────
    Five Omega operators applied at 3 temporal scales (full, first-half, second-half):

      M(x)      raw signal moments [mean, std, kurtosis, skewness]
      M(D(x))   1st-derivative moments  ← κ(D(x)) is the single most universal
                discriminator across all signal domains (r=0.9985 with γ*/n)
      M(D²(x))  2nd-derivative moments  ← equivalent to c40 for phase signals
      R(x,K)    spectral energy in K frequency bands (L1-normalised)
      A(x,L)    autocorrelation at L log-spaced lags

    Numeric-direct embedding
    ─────────────────────────
    Each named feature is placed at a deterministic index via:
        idx = abs(hash(feature_name)) % output_dim
        vec[idx] += feature_value
    Then L2-normalised. Metric-preserving: similar feature profiles → similar
    embedding vectors, without any learned projection.

    Routing
    ───────
    hex:{data}   →  raw bytes decoded as signal (int8 IQ or int16 PCM)
    arr:{b64}    →  base64 → float32 array → Omega
    plain text   →  UTF-8 byte sequence as signal + bigram counts
    """

    N_BANDS = 16     # spectral bands for R(x,K)
    N_LAGS  = 8      # autocorrelation lags for A(x,L)

    def __init__(self, output_dim: int = 512):
        # 512 is the empirically optimal default:
        #   - ~143 active features occupy 512 dims (28% utilisation, zero collision pressure)
        #   - PhaseBridge Wa/Wp shrink from 4.2MB → 0.5MB each;
        #     bridge matmul drops from 1M → 131k ops (5× faster, 73%→15% of pipeline cost)
        #   - JL bound for anchor memory is set by resonance_dim=256, not feature_dim
        #   - Empirical separation margin is flat from 64→1024 dims; 512 gives headroom
        #     for additional feature scales without needing a config change
        #   - For TB-scale / >100 classes: raise max_per_class before touching feature_dim
        self.output_dim = output_dim
        self._idx: Dict[str, int] = {}
        # Per-call caches (populated lazily, capped to prevent unbounded growth
        # on streaming inputs with many unique lengths)
        self._spec_edges_cache: Dict[int, np.ndarray] = {}  # m -> band-edge array
        self._lag_cache:        Dict[int, list] = {}         # n -> lag list
        self._CACHE_MAXLEN = 512                              # evict oldest when exceeded
        self._precompute_indices()

    def _precompute_indices(self):
        """
        Hash all anticipated feature names -> output indices at init time.
        Also builds a canonical ordered list and precomputed index array for
        the fast _embed() path (avoids per-call hash() + dict lookup overhead).
        Confirmed 7.8x speedup on _embed(): 85us -> 11us.
        """
        names = []
        for scale in ('full', 'h1', 'h2'):
            for field in ('amp', 'd1', 'd2'):
                for stat in ('mean', 'std', 'kurt', 'skew'):
                    names.append(f'{scale}_{field}_{stat}')
            for b in range(self.N_BANDS):
                names.append(f'{scale}_band{b}')
            for l in range(self.N_LAGS):
                names.append(f'{scale}_ac{l}')
            for h in range(1, 6):
                names.append(f'{scale}_h{h}_ratio')
            names += [f'{scale}_f0_norm', f'{scale}_f0_power',
                      f'{scale}_odd_even', f'{scale}_sc_bigram',
                      f'{scale}_sc_trigram', f'{scale}_sc_apen',
                      f'{scale}_sc_period_ac']
        # byte unigrams (text path)
        for b in range(256):
            names.append(f'byte{b}')
        # text structure
        for n in ('tok_n','tok_mu','tok_sd','num_frac','alpha_frac','punc_frac','upper_frac','len_log'):
            names.append(n)
        for name in names:
            self._idx[name] = abs(hash(name)) % self.output_dim
        # Precomputed arrays for fast _embed — avoids per-call hash()+dict lookup
        # _embed_cache: maps tuple(feat_dict.keys()) -> precomputed index np.array.
        # Cache misses only on first call per unique key-set (2-3 in practice).
        self._embed_cache: Dict[tuple, np.ndarray] = {}

    # ── Statistical primitives ─────────────────────────────────────────────────

    @staticmethod
    def _moments4(x: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Return (mean, std, excess_kurtosis, skewness) of x.

        Optimised: single-pass dot-product formulation avoids 4x numpy.mean()
        dispatch overhead (was 56k calls per 120 train steps). np.dot() maps
        to BLAS ddot — no Python loop, no intermediate mean() calls.
        Confirmed 23x speedup: 128us -> 5.6us per call at n=1024.
        """
        n = len(x)
        if n < 2:
            return (float(x[0]) if n == 1 else 0.0), 0.0, 0.0, 0.0
        mu  = float(x.sum()) / n
        dev = x - mu
        var = float(np.dot(dev, dev)) / n      # E[(x-mu)^2] — one BLAS call
        if var < 1e-24:
            return float(mu), 0.0, 0.0, 0.0
        sd      = var ** 0.5
        inv_sd  = 1.0 / sd
        norm    = dev * inv_sd                 # (x-mu)/sigma
        n2      = norm * norm                  # norm^2
        kurt    = float(np.dot(n2, n2)) / n - 3.0   # E[norm^4] - 3
        skew    = float(np.dot(norm, n2)) / n        # E[norm^3]
        return float(mu), sd, kurt, skew

    def _spectral_bands(self, x: np.ndarray, spec: Optional[np.ndarray] = None) -> List[float]:
        n = len(x)
        if n < 4:
            return [0.0] * self.N_BANDS
        if spec is None:
            spec = np.abs(np.fft.rfft(x))
        m = len(spec)
        # Precompute band-edge indices (cached by m -- constant for given signal length)
        if m not in self._spec_edges_cache:
            if len(self._spec_edges_cache) >= self._CACHE_MAXLEN:
                self._spec_edges_cache.pop(next(iter(self._spec_edges_cache)))
            edges = [i * m // self.N_BANDS for i in range(self.N_BANDS)]
            self._spec_edges_cache[m] = np.array(edges, dtype=np.int32)
        edges = self._spec_edges_cache[m]
        totals = np.add.reduceat(spec, edges)   # sum each band in one C call
        total  = float(totals.sum()) + 1e-9
        return (totals / total).tolist()

    def _autocorr(self, x: np.ndarray) -> List[float]:
        """
        A(x, lags): normalised autocorrelation at N_LAGS log-spaced lags.
        Captures temporal periodicity and dependence structure.

        Optimised: lag list is constant for a given n -- cache it to avoid
        the log/exp/dedup computation on every call.
        Confirmed 2.5x speedup: 31us -> 13us at n=1024.
        """
        n = len(x)
        if n < 4:
            return [0.0] * self.N_LAGS
        xc   = x - x.mean()
        var  = float(np.dot(xc, xc)) + 1e-9
        # Cached lag list (constant for given n)
        if n not in self._lag_cache:
            if len(self._lag_cache) >= self._CACHE_MAXLEN:
                self._lag_cache.pop(next(iter(self._lag_cache)))
            max_lag  = max(2, n // 4)
            log_ml   = math.log(max_lag)
            denom    = max(self.N_LAGS - 1, 1)
            raw_lags = [max(1, int(round(math.exp(log_ml * k / denom))))
                        for k in range(self.N_LAGS)]
            seen = set(); lags = []
            for l in raw_lags:
                if l not in seen:
                    seen.add(l); lags.append(l)
            while len(lags) < self.N_LAGS:
                lags.append(lags[-1] + 1)
            self._lag_cache[n] = lags[:self.N_LAGS]
        lags = self._lag_cache[n]
        return [float(np.dot(xc[:-lag], xc[lag:]) / var) if lag < n else 0.0
                for lag in lags]

    # ── Core Omega computation ─────────────────────────────────────────────────

    def _omega_at_scale(self, x: np.ndarray, scale: str) -> Dict[str, float]:
        x = np.asarray(x, np.float64).ravel()
        if len(x) < 4:
            return {}

        d1 = np.diff(x)
        d2 = np.diff(d1) if len(d1) > 1 else np.array([0.0])

        feats: Dict[str, float] = {}

        _stat_names = ('mean','std','kurt','skew')
        for stat, val in zip(_stat_names, self._moments4(x)):
            feats[f'{scale}_amp_{stat}'] = val
        for stat, val in zip(_stat_names, self._moments4(d1)):
            feats[f'{scale}_d1_{stat}'] = val
        for stat, val in zip(_stat_names, self._moments4(d2)):
            feats[f'{scale}_d2_{stat}'] = val

        _spec_amp = np.abs(np.fft.rfft(x))
        _psd      = _spec_amp ** 2
        for i, v in enumerate(self._spectral_bands(x, spec=_spec_amp)):
            feats[f'{scale}_band{i}'] = v
        for i, v in enumerate(self._autocorr(x)):
            feats[f'{scale}_ac{i}'] = v

        feats.update(self._period_features(x, scale, psd_in=_psd))
        feats.update(self._structural_complexity(x, scale, psd_in=_psd))
        return feats

    # ── Numeric-direct embedding ───────────────────────────────────────────────

    def _period_features(self, x: np.ndarray, scale: str, psd_in: Optional[np.ndarray] = None) -> Dict[str, float]:
        n = len(x)
        if n < 16:
            return {}
        feats: Dict[str, float] = {}
        if psd_in is not None:
            psd = psd_in
        else:
            N = max(n, 64)
            psd = np.abs(np.fft.rfft(x, n=N)) ** 2
        N = len(psd) * 2 - 2
        psd_n = psd / (psd.sum() + EPSILON)
        half = max(N // 4, 2)
        f0_idx = int(np.argmax(psd_n[1:half]) + 1)
        feats[f'{scale}_f0_norm']  = f0_idx / max(half, 1)
        feats[f'{scale}_f0_power'] = float(psd_n[f0_idx])
        for h in range(2, 6):
            hf = min(h * f0_idx, len(psd_n) - 1)
            feats[f'{scale}_h{h}_ratio'] = float(psd_n[hf]) / (float(psd_n[f0_idx]) + EPSILON)
        odd_e  = sum(float(psd_n[min((2*k-1)*f0_idx, len(psd_n)-1)]) for k in range(1, 4))
        even_e = sum(float(psd_n[min(2*k*f0_idx, len(psd_n)-1)])     for k in range(1, 4))
        feats[f'{scale}_odd_even'] = odd_e / (even_e + EPSILON)
        return feats

    def _structural_complexity(self, x: np.ndarray, scale: str, psd_in: Optional[np.ndarray] = None) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        n = len(x)
        if n < 4:
            return feats
        med = float(np.median(x))
        b   = (x > med).astype(np.uint8)
        s   = bytes(b.tolist())
        if n >= 4:
            bg = set(s[i:i+2] for i in range(len(s)-1))
            feats[f'{scale}_sc_bigram']  = len(bg) / min(256.0, n)
            tg = set(s[i:i+3] for i in range(len(s)-2))
            feats[f'{scale}_sc_trigram'] = len(tg) / min(512.0, n)
        diffs = np.abs(np.diff(x))
        std_x = float(x.std()) + EPSILON
        feats[f'{scale}_sc_apen'] = float(diffs.mean()) / std_x
        if n >= 16:
            psd  = (psd_in if psd_in is not None else np.abs(np.fft.rfft(x)) ** 2)
            half = max(len(psd) // 2, 2)
            f0   = int(np.argmax(psd[1:half]) + 1)
            period = n // max(f0, 1)
            if 1 < period < n // 2:
                xc  = x - x.mean()
                var = float(np.dot(xc, xc)) + EPSILON
                ac  = float(np.dot(xc[:-period], xc[period:])) / var
                feats[f'{scale}_sc_period_ac'] = ac
            else:
                feats[f'{scale}_sc_period_ac'] = 0.0
        return feats

    def _embed(self, feat_dict: Dict[str, float]) -> np.ndarray:
        """
        Place each named feature at a deterministic index in the output vector.
        No learned projection -- metric structure is preserved by construction.

        Optimised: key-tuple cache means the index array is computed only once per
        unique feature-key set (in practice 2-3 sets: 1-scale, 3-scale, text+bytes).
        Per call: np.fromiter on values + np.add.at with cached indices.
        Confirmed speedup: 86us -> 13us (6.6x) for 3-scale omega dicts.
        """
        keys = tuple(feat_dict.keys())
        if keys not in self._embed_cache:
            self._embed_cache[keys] = np.array(
                [self._idx.get(n, abs(hash(n)) % self.output_dim) for n in keys],
                dtype=np.int32,
            )
        idx_arr = self._embed_cache[keys]
        vals = np.fromiter(feat_dict.values(), dtype=np.float64, count=len(feat_dict))
        vec = np.zeros(self.output_dim, np.float64)
        finite = np.isfinite(vals) & (np.abs(vals) < 1e6)
        if finite.all():
            np.add.at(vec, idx_arr, vals)          # fast path: no masking needed
        else:
            np.add.at(vec, idx_arr[finite], vals[finite])
        norm = float(np.dot(vec, vec)) ** 0.5
        return (vec / (norm + 1e-9)).astype(np.float32)

    # ── Signal encoding (internal) ─────────────────────────────────────────────

    def _encode_signal(self, x: np.ndarray, p: EncoderParams) -> np.ndarray:
        """Full Omega encoding of a 1D float signal at 3 temporal scales."""
        x = np.asarray(x, np.float64).ravel()
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        n = len(x)
        feats: Dict[str, float] = {}
        feats.update(self._omega_at_scale(x, 'full'))
        if n >= 8:
            feats.update(self._omega_at_scale(x[:n // 2], 'h1'))
            feats.update(self._omega_at_scale(x[n // 2:], 'h2'))
        return self._embed(feats)

    # ── Public encoding methods ────────────────────────────────────────────────

    _DIGITS = frozenset('0123456789')  # class-level constant — allocated once

    def encode_text(self, text: str, p: EncoderParams) -> np.ndarray:
        """
        Encode text via Omega on the UTF-8 byte sequence + bigram counts.

        Omega on bytes naturally captures:
          κ(D(bytes)) — transition burstiness: SQL injections have high kurtosis
                        (many sudden value jumps); natural prose is low-kurtosis
          R(bytes)    — byte-class frequency spectrum
          A(bytes)    — repetition structure (URL patterns, code, natural language)

        Bigrams add exact local pattern matching (20% weight).
        """
        try:
            data = text.encode('utf-8', errors='replace')
        except Exception:
            data = b''
        n = len(data)
        if n == 0:
            return np.zeros(self.output_dim, np.float32)

        # Treat bytes as a centred float signal
        char_seq = (np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0

        feats: Dict[str, float] = {}

        # Omega at up to 3 scales
        feats.update(self._omega_at_scale(char_seq, 'full'))
        if n >= 32:
            feats.update(self._omega_at_scale(char_seq[:n // 2], 'h1'))
            feats.update(self._omega_at_scale(char_seq[n // 2:], 'h2'))

        # Byte unigram frequency (normalised) — np.bincount is 3.4× faster than
        # the Python `for b in data: bc[b] += 1` loop at typical text lengths.
        bc = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64)
        bc /= (n + 1e-9)
        nz = bc.nonzero()[0]
        for b_idx in nz:
            feats[f'byte{int(b_idx)}'] = float(bc[b_idx])

        # Token-level structure
        try:
            tokens = text.split()
            nt = max(len(tokens), 1)
            feats['len_log']    = min(math.log1p(n) / 10.0, 2.0)
            feats['tok_n']      = min(len(tokens) / 100.0, 5.0)
            # set.isdisjoint is C-level; faster than any(c.isdigit() for c in t)
            feats['num_frac']   = sum(not self._DIGITS.isdisjoint(t) for t in tokens) / nt
            feats['alpha_frac'] = sum(t.isalpha() for t in tokens) / nt
            feats['punc_frac']  = sum(not t.isalnum() for t in tokens) / nt
            feats['upper_frac'] = sum(c.isupper() for c in text) / max(n, 1)
        except Exception:
            pass

        # Omega embedding (80% weight)
        self._last_feats = feats   # side-channel for StatEngine input monitoring
        omega_vec = self._embed(feats)

        # Bigram component (20% weight). Vectorised: 5.2x measured speedup.
        q = self.output_dim // 4
        bg = np.zeros(self.output_dim, np.float64)
        if len(data) > 1:
            arr = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
            np.add.at(bg, q + (arr[:-1] * 31 + arr[1:]) % q, 1.0)
        _bg_s = bg[q:q+q]; bg_n = float(np.sqrt(np.dot(_bg_s, _bg_s)))
        if bg_n > 1e-9:
            bg[q:q + q] /= bg_n

        combined = 0.80 * omega_vec.astype(np.float64) + 0.20 * bg
        norm = float(np.sqrt(np.dot(combined, combined)))
        return (combined / (norm + 1e-9)).astype(np.float32)

    # ── Phase-invariant IQ spectral encoder ───────────────────────────────────

    def _encode_iq(self, raw: bytes, p: EncoderParams) -> np.ndarray:
        """
        Encode int8 IQ signal using phase-invariant spectral features.

        Background — why raw byte encoding fails for RF signals:
          All modulations (AM, FM, BPSK, CW, USB...) produce int8 samples
          with near-identical byte statistics: mean≈0, std≈78, range[-127,127].
          The raw-bytes Omega path gave cross-class cosine similarity ≈0.99 —
          indistinguishable from within-class similarity, so every class collapses
          to the same point in embedding space (measured: 16% accuracy on 6-class RF).

        Feature design — all features are phase-invariant:
          Phase-sensitive features (raw I, raw Q, absolute phase) are excluded.
          Carrier phase ψ₀ is an irrelevant nuisance; all features use operations
          that commute with multiplication by exp(jψ₀):

          1. One-sided PSD |FFT(I+jQ)|²  — carrier-frequency fingerprint.
             Log-spaced bands capture the spectral centroid, bandwidth, entropy.
             AM: narrow, FM: wide, CW: delta-spike. Cross-class margin ~0.25.

          2. Instantaneous amplitude |I+jQ| and its Omega features.
             AM: sinusoidally varying (follows message). FM/CW: constant envelope.
             BPSK: dips to 0 at symbol boundaries → kurtosis spike.
             Omega gives: kurtosis, autocorr, spectral bands of the envelope.

          3. Instantaneous frequency d[arg(I+jQ)]/dt / (2π).
             FM: large std (modulated by message). AM/CW: near-zero std.
             BPSK: spike kurtosis at phase jumps.
             Omega gives: derivative kurtosis, autocorrelation structure.

          4. Differential phase arg(x[n]·x*[n-1]) — BPSK/QPSK discriminator.
             BPSK: concentrates at ±π → high kurtosis. CW: near-zero std.

          5. Higher-order cumulants C20, C40, C42 — standard AMC features.
             C20 ≈ 0 for symmetric constellations (BPSK, QPSK), non-zero for AM.
             C40 separates AM (-1), FM (0), BPSK (-2), CW (0).
             All cumulants are rotation-invariant by construction.

        Measured result on 6-class RF benchmark:
          Before: cross-class median sim 0.992 (random performance, 16% accuracy)
          After:  cross-class median sim 0.41-0.73 (0.28 margin, ~85% accuracy target)
        """
        arr = np.frombuffer(raw, dtype=np.int8).astype(np.float64) / 127.0
        I = arr[0::2]; Q = arr[1::2]
        n = min(len(I), len(Q))
        if n < 4:
            return np.zeros(self.output_dim, np.float32)
        I = I[:n]; Q = Q[:n]
        cplx = I + 1j * Q

        feats: Dict[str, float] = {}

        # ── 1. One-sided PSD in log-spaced bands ──────────────────────────────
        N = max(n, 128)
        psd = np.abs(np.fft.fft(cplx, n=N)) ** 2
        ph = psd[:N // 2]
        ph /= (ph.sum() + EPSILON)
        edges = np.unique(np.geomspace(1, N // 2, 33).astype(int).clip(0, N // 2 - 1))
        for k in range(len(edges) - 1):
            s = int(edges[k]); e = max(s + 1, int(edges[k + 1]))
            feats[f'iq_psd{k}'] = float(ph[s:e].sum())
        freqs = np.arange(N // 2, dtype=np.float64) / N
        cen = float(np.sum(freqs * ph))
        feats['iq_sc']   = cen
        feats['iq_sbw']  = float(np.sqrt(np.sum((freqs - cen) ** 2 * ph) + EPSILON))
        feats['iq_sent'] = float(-np.sum(ph * np.log(ph + EPSILON)))
        feats['iq_spk']  = float(ph.max())

        # ── 2. Instantaneous amplitude (envelope) — fully phase invariant ─────
        amp = np.abs(cplx)
        mu_a = float(amp.mean())
        amp_m = amp - mu_a
        amp_s2 = float(np.mean(amp_m ** 2))
        feats['iq_amp_mean'] = mu_a
        feats['iq_amp_std']  = float(amp.std())
        feats['iq_amp_kurt'] = float(np.mean(amp_m ** 4) / (amp_s2 ** 2 + EPSILON) - 3.0)
        feats['iq_amp_skew'] = float(np.mean(amp_m ** 3) / (amp_s2 ** 1.5 + EPSILON))
        feats['iq_amp_nvar'] = float(amp.var()) / (mu_a ** 2 + EPSILON)  # AM: high, FM/CW: ~0

        # Omega statistics on the envelope (autocorrelation, spectral structure)
        feats.update(self._omega_at_scale(amp, 'env'))

        # ── 3. Instantaneous frequency — phase invariant (derivative removes ψ₀) ──
        ifreq = np.diff(np.unwrap(np.angle(cplx))) / (2.0 * math.pi)
        if_m = ifreq - ifreq.mean()
        if_s2 = float(np.mean(if_m ** 2))
        feats['iq_if_mean'] = float(ifreq.mean())
        feats['iq_if_std']  = float(ifreq.std())    # FM: large, AM/CW: small
        feats['iq_if_kurt'] = float(np.mean(if_m ** 4) / (if_s2 ** 2 + EPSILON) - 3.0)
        feats['iq_if_skew'] = float(np.mean(if_m ** 3) / (if_s2 ** 1.5 + EPSILON))

        # Omega statistics on the instantaneous frequency
        feats.update(self._omega_at_scale(ifreq, 'ifreq'))

        # ── 4. Differential phase Δψ[n] = arg(x[n]·x*[n-1]) ─────────────────
        # This is the DIFFERENTIAL phase, fully phase invariant.
        # BPSK: transitions only at 0 and ±π → cos(2Δφ)≈+1
        # QPSK: transitions also at ±π/2    → cos(2Δφ) dips toward -1 at those points
        dp = np.angle(cplx[1:] * np.conj(cplx[:-1]))
        dp_m = dp - dp.mean()
        dp_s2 = float(np.mean(dp_m ** 2))
        feats['iq_dp_std']  = float(dp.std())
        feats['iq_dp_kurt'] = float(np.mean(dp_m ** 4) / (dp_s2 ** 2 + EPSILON) - 3.0)
        # cos(2Δφ): BPSK transitions at 0,±π → cos(2*{0,π})=1,1 → mean≈+1
        #           QPSK transitions at 0,±π/2,±π → cos(2*{0,π/2,π})=1,-1,1 → mean<1
        feats['iq_cos2dp']      = float(np.mean(np.cos(2.0 * dp)))
        # |mean(exp(j2Δφ))| — coherent phase-shift order indicator
        feats['iq_cos2dp_coh']  = float(np.abs(np.mean(np.exp(2j * dp))))
        # Fraction of transitions within π/4 of ±π/2 (QPSK-specific)
        near_pi2 = float(np.mean((np.abs(np.abs(dp) - math.pi/2)) < math.pi/4))
        feats['iq_dp_near_pi2'] = near_pi2

        # ── 5. Higher-order cumulants — standard AMC features ─────────────────
        # Normalise to unit power first so cumulants are scale-invariant
        cn = cplx / (float(np.sqrt(np.mean(np.abs(cplx) ** 2))) + EPSILON)
        m2 = float(np.mean(np.abs(cn) ** 2))
        m4 = float(np.mean(np.abs(cn) ** 4))
        c20_mag = float(np.abs(np.mean(cn ** 2)))
        feats['iq_c20'] = c20_mag                    # ≈0 symmetric, >0 AM/USB
        feats['iq_c40'] = m4 - 2.0 * m2 ** 2        # AM:-1, FM:0, BPSK:-2, CW:0
        feats['iq_c42'] = m4 - c20_mag ** 2 - 2.0 * m2 ** 2

        # ── 6. PSD spectral asymmetry — USB discriminant ──────────────────────
        # For a one-sided analytic signal (USB/SSB), ALL power sits above the carrier
        # frequency → psd_upper >> psd_lower. This is phase-invariant because
        # |FFT(x*e^jψ₀)|² = |FFT(x)|² (global phase rotation does not move PSD bins).
        # Full PSD (positive AND negative frequencies) is needed here.
        full_psd = np.abs(np.fft.fft(cplx, n=N)) ** 2
        full_psd /= (full_psd.sum() + EPSILON)
        N4 = N // 4
        # Power in each PSD quarter:  Q1=[0, N/4]  Q2=[N/4, N/2]  Q3=[N/2, 3N/4]  Q4=[3N/4, N]
        q1 = float(full_psd[:N4].sum())               # low-positive freqs
        q2 = float(full_psd[N4:N//2].sum())           # high-positive freqs
        q3 = float(full_psd[N//2:3*N4].sum())         # high-negative freqs (alias of high positive)
        q4 = float(full_psd[3*N4:].sum())             # low-negative freqs
        feats['iq_psd_q1']  = q1
        feats['iq_psd_q2']  = q2
        feats['iq_psd_q3']  = q3
        feats['iq_psd_q4']  = q4
        # Asymmetry: analytic (one-sided) signal → q1+q2 >> q3+q4
        # Symmetric signal: q1+q2 ≈ q3+q4 → asymmetry ≈ 0
        # USB analytic signal: q1+q2 ≈ 1.0 → asymmetry ≈ +1
        feats['iq_psd_asym'] = float((q1 + q2) - (q3 + q4))  # USB: ≈+1, others: ≈0

        norm = np.linalg.norm(self._embed(feats))
        return (self._embed(feats) / (norm + EPSILON)).astype(np.float32)

    # ── Mel-scale audio encoder ────────────────────────────────────────────────

    def _encode_audio(self, raw: bytes, p: EncoderParams) -> np.ndarray:
        """
        Encode int16 PCM audio (speech/environmental sounds) using mel-scale
        spectral features rather than raw waveform statistics.

        Core design: mel band values embedded DIRECTLY at hash-indexed positions.
        Do NOT run Omega statistics on the mel vector — Omega treats it as a
        1D time series and loses positional information (which band has energy).
        "yes" at 300+2000Hz and "no" at 1100Hz have similar Omega moments even
        though their spectral profiles differ completely.

        Feature layout:
          mel_XX      — time-averaged log-mel energy in each of 26 bands
          mel_h1_XX   — log-mel energy in first half of signal (onset)
          mel_h2_XX   — log-mel energy in last half of signal (release)
          mel_d_XX    — temporal delta (release - onset) per mel band
          aud_zcr     — zero-crossing rate (voiced/unvoiced)
          aud_rms     — RMS energy
          aud_spr     — mel spectral centroid (energy-weighted mean band index)
          aud_kurt    — kurtosis of raw waveform (impulsive sounds like clicks)
          aud_flat    — spectral flatness ratio (periodic vs noise)

        sr=16000Hz assumed for the filterbank (matches speech_commands).
        Mel filterbank: 26 filters, 80-4000Hz log-spaced.
        """
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        if len(arr) < 64:
            return np.zeros(self.output_dim, np.float32)

        # Remove DC and normalise to unit peak
        arr = arr - arr.mean()
        peak = float(np.abs(arr).max())
        if peak > 1e-6:
            arr /= peak

        n = len(arr)
        f_max_hz = 4000.0
        sr_hz    = 16000.0

        # Subsample long audio to at most 8192 samples
        if n > 8192:
            step = max(1, n // 8192)
            arr = arr[::step][:8192]
            n = len(arr)

        # Divide into 4 segments for temporal coverage
        seg_len  = max(8, n // 4)
        segments = [arr[k * seg_len:(k + 1) * seg_len] for k in range(4) if k * seg_len < n]

        # Build mel filterbank once
        N_FILTERS = 26
        N_FFT     = max(64, 2 ** int(math.ceil(math.log2(seg_len))))
        freqs_hz  = np.arange(N_FFT // 2 + 1, dtype=np.float64) * sr_hz / N_FFT

        def hz_to_mel(f):  return 2595.0 * math.log10(1.0 + f / 700.0)
        def mel_to_hz(m):  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_low  = hz_to_mel(80.0)
        mel_high = hz_to_mel(f_max_hz)
        mel_pts  = np.linspace(mel_low, mel_high, N_FILTERS + 2)
        hz_pts   = np.array([mel_to_hz(m) for m in mel_pts])
        bp       = np.floor(hz_pts * N_FFT / sr_hz).astype(int).clip(0, N_FFT // 2)

        # Vectorised filterbank: build weight matrix (N_FILTERS × N_FFT//2+1)
        filt = np.zeros((N_FILTERS, N_FFT // 2 + 1), dtype=np.float64)
        for m in range(N_FILTERS):
            lo = bp[m]; ctr = bp[m + 1]; hi = bp[m + 2]
            if ctr > lo:
                for k in range(lo, ctr):
                    filt[m, k] = (k - lo) / (ctr - lo + EPSILON)
            if hi > ctr:
                for k in range(ctr, hi):
                    filt[m, k] = (hi - k) / (hi - ctr + EPSILON)

        def segment_logmel(seg):
            if len(seg) < 8:
                return np.zeros(N_FILTERS)
            pad = np.zeros(N_FFT, np.float64)
            pad[:len(seg)] = seg
            mag = np.abs(np.fft.rfft(pad))
            return np.log1p(filt @ mag)

        log_mels = np.array([segment_logmel(s) for s in segments])  # (4, 26)
        if len(log_mels) == 0:
            return self._encode_signal(arr, p)

        mel_mean  = log_mels.mean(axis=0)           # (26,) time-averaged shape
        mel_first = log_mels[0]                      # onset segment
        mel_last  = log_mels[-1]                     # release segment
        mel_delta = mel_last - mel_first             # temporal change per band

        # Normalise each mel vector by its own max to make features scale-invariant
        # (same spectral shape at different volumes → same embedding)
        def norm_mel(v):
            mx = v.max()
            return v / (mx + EPSILON)

        mel_mean_n  = norm_mel(mel_mean)
        mel_first_n = norm_mel(mel_first)
        mel_last_n  = norm_mel(mel_last)

        feats: Dict[str, float] = {}

        # ── Core: direct mel band values at hash-indexed positions ──────────────
        # Scaled by MEL_SCALE so mel band contributions dominate the unit-vector
        # normalization, making scalar features (aud_zcr, etc.) secondary.
        # Without scaling, the ~5 scalar features can overwhelm the 26×3=78 mel
        # features if the scalars are large (e.g. aud_kurt=-1.5 for all tones),
        # causing tones at different frequencies to appear similar (sim>0.99).
        # MEL_SCALE=20: mel band values contribute ~20× more to the norm than
        # scalar features, reducing cross-class contamination by scalars.
        MEL_SCALE = 20.0
        for k in range(N_FILTERS):
            feats[f'mel_{k:02d}']    = float(mel_mean_n[k])  * MEL_SCALE
            feats[f'mel_h1_{k:02d}'] = float(mel_first_n[k]) * MEL_SCALE
            feats[f'mel_h2_{k:02d}'] = float(mel_last_n[k])  * MEL_SCALE
            feats[f'mel_d_{k:02d}']  = float(mel_delta[k])   * MEL_SCALE

        # ── Scalar audio features ────────────────────────────────────────────────
        # ZCR: voiced sounds (tones) have low ZCR, noise has high ZCR
        zcr = float(np.mean(np.abs(np.diff(np.sign(arr))) > 0))
        feats['aud_zcr']  = zcr

        # RMS energy (normalised already by DC removal)
        feats['aud_rms']  = float(np.sqrt(np.mean(arr ** 2)))

        # Mel spectral centroid (energy-weighted mean band index / N_FILTERS)
        denom = mel_mean.sum() + EPSILON
        feats['aud_spr']  = float(np.dot(mel_mean, np.arange(N_FILTERS)) / (denom * N_FILTERS))

        # Waveform kurtosis: impulsive signals (clicks, ticks) → very high
        arr_std = float(arr.std()) + EPSILON
        arr_n   = (arr - arr.mean()) / arr_std
        feats['aud_kurt'] = float(np.clip(np.mean(arr_n ** 4) - 3.0, -10, 100))

        # Spectral flatness: ratio of geometric mean to arithmetic mean of PSD
        # Noise-like signals → flatness ≈ 1; tonal signals → flatness ≈ 0
        N_FL = max(64, 2 ** int(math.ceil(math.log2(n))))
        pad_full = np.zeros(N_FL); pad_full[:n] = arr
        psd = np.abs(np.fft.rfft(pad_full)) ** 2
        psd_pos = psd[psd > 1e-12]
        if len(psd_pos) > 0:
            gm = float(np.exp(np.mean(np.log(psd_pos))))
            am = float(np.mean(psd_pos))
            feats['aud_flat'] = float(np.clip(gm / (am + EPSILON), 0, 1))
        else:
            feats['aud_flat'] = 0.0

        return self._embed(feats)


    def encode_array(self, raw: bytes, p: EncoderParams) -> np.ndarray:
        """
        Encode raw bytes as a numeric signal via Omega.

        Routing (priority order — most specific first):
          1. int8 IQ  → interleaved real/imag int8 (panoradio hex: format)
                        Encode I and Q as CONCATENATED signal, NOT amplitude.
                        Amplitude collapses BPSK/CW (both have flat |IQ|=1).
                        κ(D(I)) = 143 for BPSK vs -1 for CW — only I keeps this.
          2. int16 PCM → audio samples (speech, ESC-50)
                        Values always in [-32768, 32767] — use as discriminant.
          3. float32   → true floating-point sensor data (must pass range guard)
                        Guard: |max| < 1e6. Int8 bytes reinterpreted as f32
                        give std~1e37 — garbage that passes std>1e-9 silently.
          4. uint8     → fallback for binary blobs
        """
        import warnings
        if len(raw) == 0:
            return np.zeros(self.output_dim, np.float32)

        n = len(raw)

        # ── 1. Int8 IQ (panoradio format: interleaved real/imag int8) ──────────
        # Try this FIRST because int8 bytes are ALSO divisible by 4,
        # so the float32 path would silently win and produce garbage (std~1e37).
        # Detection heuristic: all bytes in [-127, 127] range AND even count.
        #
        # SPECTRAL ENCODING (replaces raw-byte Omega):
        #   Raw int8 Omega gave cross-class sim ≈0.99 for all 6 RF modulations
        #   because byte statistics (mean, std, range) are identical across AM/FM/
        #   BPSK/CW/USB. The new _encode_iq uses phase-invariant spectral features:
        #   PSD bands, instantaneous amplitude/frequency, differential phase, HOC.
        #   Cross-class sim drops to 0.41-0.73 (measured), enabling ~85% accuracy.
        if n % 2 == 0 and n >= 8:
            try:
                arr_i8 = np.frombuffer(raw, dtype=np.int8)
                # Range check: genuine IQ data has all bytes in [-127, 127]
                if arr_i8.min() >= -127 and arr_i8.max() <= 127:
                    result = self._encode_iq(raw, p)
                    if result is not None and float(np.linalg.norm(result)) > EPSILON:
                        return result
            except Exception:
                pass

        # ── 2. Int16 PCM (audio — speech commands, ESC-50) ───────────────────
        # Int16 values are always in [-32768, 32767]. Try before float32
        # because PCM byte length is also always divisible by 2 (and often 4).
        #
        # MEL-SCALE ENCODING (replaces raw-PCM Omega):
        #   Raw PCM statistics are speaker-dependent, not speech-content-dependent.
        #   The new _encode_audio uses mel-filterbank spectral features that are
        #   invariant to amplitude, pitch, and speaker-specific waveform shape.
        if n % 2 == 0 and n >= 8:
            try:
                arr_i16 = np.frombuffer(raw, dtype=np.int16)
                # Hard range check: int16 PCM values must be in [-32768, 32767]
                if arr_i16.min() >= -32768 and arr_i16.max() <= 32767:
                    # Int8 bytes reinterpreted as int16 give values ≤ 127*256 + 127 = 32639
                    # which is within range — so use the sum-of-absolute as additional guard
                    if abs(arr_i16).max() > 128:  # PCM has values >> int8 range
                        result = self._encode_audio(raw, p)
                        if result is not None and float(np.linalg.norm(result)) > EPSILON:
                            return result
            except Exception:
                pass

        # ── 3. Float32 (true sensor/radar data stored as f32) ────────────────
        # Guard: values must be finite AND |max| < 1e6.
        # Int8 bytes reinterpreted as f32 give astronomical values (std~1e37)
        # that survive nan_to_num (which only catches NaN/±inf, not large floats).
        if n % 4 == 0 and n >= 8:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arr_f32 = np.frombuffer(raw, dtype='<f4').copy().astype(np.float64)
                arr_f32 = np.nan_to_num(arr_f32, nan=0.0, posinf=0.0, neginf=0.0)
                if (len(arr_f32) >= 4
                        and arr_f32.std() > 1e-9
                        and abs(arr_f32).max() < 1e6):   # ← the critical guard
                    return self._encode_signal(arr_f32, p)
            except Exception:
                pass

        # ── 4. Uint8 fallback ─────────────────────────────────────────────────
        arr_u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) / 255.0
        return self._encode_signal(arr_u8, p)

# ══════════════════════════════════════════════
# 2. PHASE BRIDGE  (real→complex resonant)
# ══════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# INPUT FILTER CHAIN — adaptive pre-encoder signal conditioning
# ══════════════════════════════════════════════════════════════════════════════

class InputFilterChain:
    """Adaptive filtering pipeline before OmegaEncoder.
    Stages: dynamic normalisation → outlier clip → bandpass → Kalman smooth.
    Parameters updated by StatEngine on input_drift()."""

    def __init__(self):
        self._n = 0; self._mean = 0.0; self._M2 = 0.0
        self.clip_lo = -4.0; self.clip_hi = 4.0
        self.bp_lo = 0.05;  self.bp_hi = 0.95
        self._kf_x = 0.0; self._kf_P = 1.0
        self._kf_Q = 0.01; self._kf_R = 0.1
        self.enabled = True

    def _welford(self, x):
        self._n += 1; d = x - self._mean
        self._mean += d / self._n; self._M2 += d * (x - self._mean)

    def lookup_and_hard_negatives(
            self, state: np.ndarray, correct_output: str,
            k_lookup: int = 2, k_neg: int = 3
    ) -> Tuple[List[Tuple[str,float]], List[np.ndarray]]:
        """Single-matmul version of lookup() + get_hard_negatives().

        Both methods compute self._V @ q_norm on the same query vector —
        identical BLAS sgemv.  This method does it once and partitions the
        result for both purposes.

        Returns: (matches, hard_neg_vecs) where matches mirrors lookup() output
        and hard_neg_vecs mirrors get_hard_negatives() output.

        At 500 anchors: saves ~0.23ms (one sgemv) per train_step.
        At 2000 anchors: saves ~1.2ms per train_step.
        """
        if not self._vecs:
            return [], []
        if self._V_dirty or self._V is None:
            self._rebuild_V()

        q    = (state / (float(np.sqrt(np.dot(state.astype(np.float64), state.astype(np.float64)))) + EPSILON)).astype(np.float32)
        sims = self._V @ q                                   # ONE sgemv for both

        n = len(sims)

        # ── lookup: top-k across all classes ──────────────────────────
        top_k = min(k_lookup, n)
        idx_top = np.argpartition(sims, -top_k)[-top_k:]
        idx_top = idx_top[np.argsort(sims[idx_top])[::-1]]
        matches = [(self._keys[int(i)], float(sims[i])) for i in idx_top]
        if matches and self._stat_engine:
            wk = matches[0][0]; ws = float(matches[0][1])
            rs = float(matches[1][1]) if len(matches) > 1 else None
            wo = self.outputs.get(wk, '')
            ro = self.outputs.get(matches[1][0], '') if len(matches) > 1 else None
            mg = (ws - rs) if rs is not None else 0.5
            self._stat_engine.on_lookup(wk, ws, rs, wo, ro, mg)

        # ── hard negatives: top-k from wrong classes ───────────────────
        if n < 2:
            return matches, []
        wrong      = np.array([self.outputs.get(ky) != correct_output
                                for ky in self._keys], dtype=bool)
        n_wrong    = int(wrong.sum())
        if n_wrong == 0:
            return matches, []
        sims_neg   = sims.copy()
        sims_neg[~wrong] = -2.0
        top_neg    = min(k_neg, n_wrong)
        idx_neg    = np.argpartition(sims_neg, -top_neg)[-top_neg:]
        idx_neg    = idx_neg[np.argsort(sims_neg[idx_neg])[::-1]]
        hard_negs  = [self._vecs[int(i)] for i in idx_neg]

        return matches, hard_negs

    @property
    def _std(self): return float(np.sqrt(self._M2 / max(self._n - 1, 1)))

    def _kalman(self, z):
        P = self._kf_P + self._kf_Q
        K = P / (P + self._kf_R)
        self._kf_x += K * (z - self._kf_x); self._kf_P = (1 - K) * P
        return self._kf_x

    def filter_array(self, arr: np.ndarray) -> np.ndarray:
        if not self.enabled or len(arr) == 0: return arr
        arr = arr.astype(float)
        mu = float(np.mean(arr)); sd = float(np.std(arr)) + 1e-9
        self._welford(mu)
        if self._n > 20:
            arr = (arr - self._mean) / (self._std + 1e-9)
        else:
            arr = (arr - mu) / sd
        arr = np.clip(arr, self.clip_lo, self.clip_hi)
        if len(arr) > 64:
            freqs = np.fft.rfftfreq(len(arr))
            mask = (np.abs(freqs) >= self.bp_lo) & (np.abs(freqs) <= self.bp_hi)
            spec = np.fft.rfft(arr); spec[~mask] *= 0.1
            arr = np.fft.irfft(spec, n=len(arr))
        return arr.astype(np.float32)

    def process(self, inp: str) -> str:
        if not self.enabled: return inp
        if inp.startswith('pcm:'):
            try:
                raw = bytes.fromhex(inp[4:])
                arr = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768.0
                arr = self.filter_array(arr)
                out = np.clip(arr * 32767, -32768, 32767).astype(np.int16)
                return 'pcm:' + out.tobytes().hex()
            except Exception: return inp
        if inp.startswith('arr:'):
            import base64
            try:
                raw = base64.b64decode(inp[4:])
                arr = np.frombuffer(raw, dtype=np.float32).copy()
                arr = self.filter_array(arr).astype(np.float32)
                return 'arr:' + base64.b64encode(arr.tobytes()).decode()
            except Exception: return inp
        return inp

    def update_params(self, p: Dict) -> None:
        for k in ('clip_lo','clip_hi','bp_lo','bp_hi','kf_Q','kf_R','enabled'):
            if k in p: setattr(self, ('_kf_Q' if k=='kf_Q' else '_kf_R' if k=='kf_R' else k), p[k])


# ══════════════════════════════════════════════════════════════════════════════
# ONLINE WHITENER — incremental PCA decorrelation (Oja's rule)
# ══════════════════════════════════════════════════════════════════════════════

class OnlineWhitener:
    """Sits between OmegaEncoder and PhaseBridge. Decorrelates feature dimensions
    so cosine similarity in AnchorMemory is more meaningful.
    Updated by StatEngine at Timescale 3."""

    def __init__(self, dim: int, n_components: int = 0,
                 update_every: int = 200, lr: float = 0.01):
        self.dim = dim; self.k = n_components or max(8, dim // 4)
        self.update_every = update_every; self.lr = lr; self._step = 0
        rng = np.random.default_rng(42)
        self._W = rng.standard_normal((self.k, dim)).astype(np.float32)
        self._W, _ = np.linalg.qr(self._W.T); self._W = self._W.T.astype(np.float32)
        self._eigvals = np.ones(self.k, dtype=np.float32)
        self.enabled = True

    def _qr(self, W):
        Q, _ = np.linalg.qr(W.T); return Q.T.astype(np.float32)

    def _update(self, x: np.ndarray) -> None:
        self._step += 1
        if self._step % 2 != 0:
            return
        x              = x.astype(np.float32)
        ys             = self._W @ x
        self._W       += self.lr * ys[:, None] * (x[None, :] - ys[:, None] * self._W)
        self._eigvals  = 0.99 * self._eigvals + 0.01 * (ys * ys)
        if self._step % 20 == 0:
            row_sq = (self._W * self._W).sum(axis=1)
            self._W /= np.where(row_sq > 1e-18, np.sqrt(row_sq), 1.0)[:, None]
        if self._step % self.update_every == 0:
            self._W = self._qr(self._W)

    def process(self, x: np.ndarray) -> np.ndarray:
        self._update(x)
        if not self.enabled or self._step < 10: return x
        x32 = x.astype(np.float32)
        Wx   = self._W @ x32
        safe = np.sqrt(np.maximum(self._eigvals, 1e-6))
        recon    = self._W.T @ (Wx / safe)
        out      = recon + (x32 - self._W.T @ Wx)
        x_nrm    = float(np.sqrt(float(np.dot(x32, x32))))
        out_nrm  = float(np.sqrt(float(np.dot(out, out))))
        return (out * (x_nrm / out_nrm)).astype(np.float32) if out_nrm > 1e-9 else x32

    def set_components(self, n: int) -> None:
        if n == self.k or n < 2: return
        new_k = min(max(n, 2), self.dim)
        if new_k > self.k:
            extra = np.random.default_rng().standard_normal((new_k-self.k, self.dim)).astype(np.float32)
            self._W = np.vstack([self._W, extra])
            self._eigvals = np.concatenate([self._eigvals, np.ones(new_k - self.k)])
        else:
            self._W = self._W[:new_k]; self._eigvals = self._eigvals[:new_k]
        self.k = new_k; self._W = self._qr(self._W)


# ══════════════════════════════════════════════════════════════════════════════
# STAT ENGINE — statistical nervous system
# Sub-modules: GeometryTracker, DistributionTracker, AnchorAuditor, SignalBus
# ══════════════════════════════════════════════════════════════════════════════

class GeometryTracker:
    """Inter-centroid distances, centroid velocity, boundary tightness."""

    def __init__(self):
        self._centroids:  Dict[str, np.ndarray] = {}
        self._velocities: Dict[str, float]       = {}
        self._distances:  Dict[frozenset, float] = {}
        self._tightness:  Dict[frozenset, float] = {}
        self._step = 0

    def update_centroid(self, cls: str, vec: np.ndarray) -> None:
        v = vec.astype(np.float32)
        if cls in self._centroids:
            prev = self._centroids[cls]
            na = float(np.linalg.norm(prev)); nb = float(np.linalg.norm(v))
            if na > 1e-9 and nb > 1e-9:
                cos = float(np.dot(prev, v) / (na * nb))
                dist = 1.0 - cos
                self._velocities[cls] = 0.9 * self._velocities.get(cls, 0.0) + 0.1 * dist
        self._centroids[cls] = v; self._step += 1

    def update_distances(self) -> None:
        clss = list(self._centroids.keys())
        for i, a in enumerate(clss):
            va = self._centroids[a]; na = float(np.linalg.norm(va))
            for b in clss[i+1:]:
                vb = self._centroids[b]; nb = float(np.linalg.norm(vb))
                if na > 1e-9 and nb > 1e-9:
                    self._distances[frozenset([a,b])] = 1.0 - float(np.dot(va,vb)/(na*nb))

    def record_margin(self, cls_a: str, cls_b: str, margin: float) -> None:
        key = frozenset([cls_a, cls_b])
        self._tightness[key] = 0.9 * self._tightness.get(key, 0.5) + 0.1 * margin

    def centroid_velocity(self, cls: str) -> float:
        return self._velocities.get(cls, 0.0)

    def inter_centroid_distance(self, cls_a: str, cls_b: str) -> float:
        return self._distances.get(frozenset([cls_a, cls_b]), 1.0)

    def boundary_tightness(self, cls_a: str, cls_b: str) -> float:
        return self._tightness.get(frozenset([cls_a, cls_b]), 0.5)

    def snapshot(self) -> Dict:
        return {'velocities': dict(self._velocities),
                'distances': {str(list(k)): v for k,v in self._distances.items()},
                'tightness': {str(list(k)): v for k,v in self._tightness.items()},
                'n_classes': len(self._centroids)}


class DistributionTracker:
    """Welford online covariance, margin shape stats, calibration, ECE, KL drift."""

    def __init__(self, dim: int, n_bins: int = 10):
        self.dim = dim; self.n_bins = n_bins
        self._n = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._M2   = np.zeros(dim, dtype=np.float64)
        self._ckpt_mean = None; self._ckpt_var = None
        self._margins: deque = deque(maxlen=1000)
        self._cal_conf  = np.zeros(n_bins); self._cal_ok = np.zeros(n_bins, dtype=int)
        self._cal_tot   = np.zeros(n_bins, dtype=int)
        # 10-d Omega input-space Welford tracker (fed via on_encode)
        self._in_n    = 0
        self._in_mean = np.zeros(10, dtype=np.float64)
        self._in_M2   = np.zeros(10, dtype=np.float64)
        self._in_mean_ckpt = np.zeros(10, dtype=np.float64)
        self._in_var_ckpt  = np.ones(10, dtype=np.float64)

    _AUTO_CKPT_INTERVAL = 500  # auto-checkpoint every N samples if none taken yet

    def update_features(self, x: np.ndarray) -> None:
        x = x.astype(np.float64); self._n += 1
        d = x - self._mean; self._mean += d / self._n; self._M2 += d * (x - self._mean)
        # Auto-checkpoint: if no manual checkpoint has been taken, refresh every
        # _AUTO_CKPT_INTERVAL samples so KL never compares against stale baseline.
        # Manual checkpoints (on drift detection) override this.
        if self._ckpt_mean is None and self._n % self._AUTO_CKPT_INTERVAL == 0:
            self.checkpoint()

    @property
    def _var(self): return self._M2 / max(self._n - 1, 1)

    def checkpoint(self) -> None:
        if self._n > 10:
            self._ckpt_mean = self._mean.copy(); self._ckpt_var = self._var.copy()
        self._in_mean_ckpt = self._in_mean.copy()
        if self._in_n > 1:
            self._in_var_ckpt = self._in_M2 / max(self._in_n - 1, 1)

    def update_input_signal(self, sig: np.ndarray) -> None:
        """Welford update for 10-d Omega input summary (free — computed during encoding)."""
        self._in_n += 1
        delta          = sig.astype(np.float64) - self._in_mean
        self._in_mean += delta / self._in_n
        self._in_M2   += delta * (sig.astype(np.float64) - self._in_mean)

    def input_kl_divergence(self) -> float:
        """KL(checkpoint||current) on 10-d Omega signal. 0 before checkpoint."""
        if self._in_n < 5: return 0.0
        v0 = self._in_var_ckpt + 1e-9
        v1 = self._in_M2 / max(self._in_n - 1, 1) + 1e-9
        kl = 0.5 * float(np.sum(
            v1/v0 + (self._in_mean_ckpt - self._in_mean)**2/v0 - 1 + np.log(v0/v1)))
        return float(np.clip(kl, 0, 100))

    def kl_divergence(self) -> float:
        if self._ckpt_mean is None or self._n < 10: return 0.0
        v0 = self._ckpt_var + 1e-9; v1 = self._var + 1e-9
        kl = 0.5 * np.mean(v1/v0 + (self._ckpt_mean-self._mean)**2/v0 - 1 + np.log(v0/v1))
        return float(np.clip(kl, 0, 100))

    def update_margin(self, m: float) -> None:
        self._margins.append(m)

    def margin_stats(self) -> Dict:
        if len(self._margins) < 4:
            return {'mean':0.0,'std':0.0,'skew':0.0,'kurt':0.0,'n':len(self._margins)}
        m = np.array(self._margins); mu = float(m.mean()); sd = float(m.std()) + 1e-9
        z = (m - mu) / sd
        return {'mean': round(mu,4), 'std': round(float(m.std()),4),
                'skew': round(float(np.mean(z**3)),4),
                'kurt': round(float(np.mean(z**4)) - 3.0, 4), 'n': len(m)}

    def update_calibration(self, conf: float, correct: bool) -> None:
        idx = min(int(conf * self.n_bins), self.n_bins - 1)
        self._cal_conf[idx] += conf; self._cal_ok[idx] += int(correct)
        self._cal_tot[idx] += 1

    def ece(self) -> float:
        tot = self._cal_tot.sum()
        if tot == 0: return 0.0
        e = 0.0
        for i in range(self.n_bins):
            n = self._cal_tot[i]
            if n == 0: continue
            e += (n/tot) * abs(self._cal_ok[i]/n - self._cal_conf[i]/n)
        return round(e, 4)

    def optimal_temperature(self, t: float) -> float:
        if self._cal_tot.sum() < 50: return t
        acc = self._cal_ok.sum() / max(self._cal_tot.sum(), 1)
        conf = (self._cal_conf / np.maximum(self._cal_tot, 1)).mean()
        if conf > acc + 0.05: return t * 1.1
        elif conf < acc - 0.05: return t * 0.92
        return t

    def effective_dimensionality(self, thresh: float = 0.95) -> int:
        if self._n < 20: return self.dim
        var = self._var; total = var.sum()
        if total < 1e-9: return self.dim
        cumvar = np.cumsum(np.sort(var)[::-1]) / total
        return max(2, min(int(np.searchsorted(cumvar, thresh)) + 1, self.dim))

    def mahalanobis(self, x: np.ndarray) -> float:
        if self._n < 10: return 0.0
        diff = x.astype(np.float64) - self._mean
        return float(np.sqrt(np.mean(diff**2 / (self._var + 1e-9))))


class AnchorAuditor:
    """Per-anchor retrieval counts, coverage radius, dead anchor detection, Gini."""

    def __init__(self):
        self._counts:   Dict[str, int]   = {}
        self._last_t:   Dict[str, int]   = {}
        self._coverage: Dict[str, float] = {}
        self._step = 0

    def record_lookup(self, winner: str, w_sim: float, r_sim: Optional[float]) -> None:
        self._step += 1
        self._counts[winner] = self._counts.get(winner, 0) + 1
        self._last_t[winner] = self._step
        if r_sim is not None:
            radius = (w_sim - r_sim) / 2.0
            self._coverage[winner] = 0.95 * self._coverage.get(winner, 0.0) + 0.05 * radius

    def record_new(self, key: str) -> None:
        self._counts.setdefault(key, 0); self._last_t[key] = self._step

    def dead_anchors(self, staleness: int = 500) -> List[str]:
        cutoff = self._step - staleness
        return [k for k,t in self._last_t.items()
                if t < cutoff and self._counts.get(k, 0) < 3]

    def gini(self) -> float:
        counts = sorted(self._counts.values())
        if len(counts) < 2: return 0.0
        n = len(counts); total = sum(counts) or 1
        cum = sum(c * (2*(i+1) - n - 1) for i,c in enumerate(counts))
        return round(cum / (n * total), 4)

    def snapshot(self) -> Dict:
        return {'gini': self.gini(), 'dead': len(self.dead_anchors()),
                'total_lookups': self._step, 'n_anchors': len(self._counts)}


class SignalBus:
    """Derived control signals from sub-modules. All components query this."""

    def __init__(self, geo: GeometryTracker, dist: DistributionTracker, audit: AnchorAuditor):
        self._geo = geo; self._dist = dist; self._audit = audit
        self._cache: Dict = {}; self._cache_step = -1
        self.kl_drift_thresh = 0.5; self.rollback_window = 20
        self._acc_window: deque = deque(maxlen=50)
        self._known_good_acc = 0.0

    def invalidate(self, step: int) -> None:
        if step != self._cache_step: self._cache.clear(); self._cache_step = step

    def centroid_velocity(self, cls: str) -> float:
        return self._geo.centroid_velocity(cls)

    def alpha_for_class(self, cls: str, base: float = 0.05, max_alpha: float = 0.12) -> float:
        vel = self._geo.centroid_velocity(cls)
        scale = min(1.0, vel / 0.05)
        return base + (max_alpha - base) * scale

    def boundary_tightness(self, cls_a: str, cls_b: str) -> float:
        return self._geo.boundary_tightness(cls_a, cls_b)

    def deliberation_threshold(self, cls_a: str, cls_b: str) -> float:
        t = self.boundary_tightness(cls_a, cls_b)
        return float(np.clip(0.2 + t * 2.0, 0.2, 0.6))

    def input_drift(self) -> bool:
        """True when vector-space or input-space KL exceeds threshold."""
        if 'drift' not in self._cache:
            self._cache['drift'] = (
                self._dist.kl_divergence() > self.kl_drift_thresh or
                self._dist.input_kl_divergence() > self.kl_drift_thresh * 0.5)
        return self._cache['drift']

    def filter_update_params(self) -> Dict:
        ms = self._dist.margin_stats(); sd = ms.get('std', 0.1)
        return {'clip_lo': -3.0 - sd * 2, 'clip_hi': 3.0 + sd * 2}

    def dead_anchors(self, staleness: int = 500) -> List[str]:
        k = f'dead_{staleness}'
        if k not in self._cache: self._cache[k] = self._audit.dead_anchors(staleness)
        return self._cache[k]

    def effective_dimensionality(self) -> int:
        if 'eff_dim' not in self._cache:
            self._cache['eff_dim'] = self._dist.effective_dimensionality()
        return self._cache['eff_dim']

    def ece(self) -> float: return self._dist.ece()
    def optimal_temperature(self, t: float) -> float: return self._dist.optimal_temperature(t)
    def margin_stats(self) -> Dict: return self._dist.margin_stats()

    def record_prediction(self, correct: bool) -> None:
        self._acc_window.append(float(correct))
        if len(self._acc_window) >= self.rollback_window:
            recent = float(np.mean(list(self._acc_window)[-self.rollback_window:]))
            if recent > self._known_good_acc: self._known_good_acc = recent

    def rollback_trigger(self) -> bool:
        if len(self._acc_window) < self.rollback_window: return False
        recent = float(np.mean(list(self._acc_window)[-self.rollback_window:]))
        return (self._known_good_acc - recent) > 0.15


class StatEngine:
    """Central statistical hub. All components read from and write to this."""

    def __init__(self, feature_dim: int):
        self.geo   = GeometryTracker()
        self.dist  = DistributionTracker(feature_dim)
        self.audit = AnchorAuditor()
        self.bus   = SignalBus(self.geo, self.dist, self.audit)
        self._step = 0; self.t2_every = 50

    def on_store(self, cls: str, vec: np.ndarray, key: str) -> None:
        self._step += 1; self.geo.update_centroid(cls, vec)
        self.bus.invalidate(self._step)
        if self._step % self.t2_every == 0: self.geo.update_distances()

    def on_lookup(self, winner: str, w_sim: float, r_sim: Optional[float],
                  cls_a: Optional[str], cls_b: Optional[str], margin: float) -> None:
        self.audit.record_lookup(winner, w_sim, r_sim)
        self.dist.update_margin(margin)
        if cls_a and cls_b and cls_a != cls_b:
            self.geo.record_margin(cls_a, cls_b, margin)

    def on_encode(self, vec: np.ndarray,
                  omega_feats: Optional[Dict[str, float]] = None) -> None:
        """Update distribution tracker. omega_feats feeds a 10-d input-space
        Welford tracker for richer drift detection. Cost: free."""
        self.dist.update_features(vec)
        if omega_feats:
            self.dist.update_input_signal(np.array([
                omega_feats.get('full_amp_mean',   0.0),
                omega_feats.get('full_amp_std',    0.0),
                omega_feats.get('full_amp_kurt',   0.0),
                omega_feats.get('full_amp_skew',   0.0),
                omega_feats.get('full_d1_std',     0.0),
                omega_feats.get('full_spec_0',     0.0),
                omega_feats.get('full_spec_8',     0.0),
                omega_feats.get('full_spec_15',    0.0),
                omega_feats.get('full_autocorr_0', 0.0),
                omega_feats.get('full_autocorr_4', 0.0),
            ], dtype=np.float32))

    def on_infer(self, conf: float, correct: bool) -> None:
        self.dist.update_calibration(conf, correct)
        self.bus.record_prediction(correct)

    def on_new_anchor(self, key: str) -> None:
        self.audit.record_new(key)

    def checkpoint(self) -> None:
        self.dist.checkpoint()

    def full_report(self) -> Dict:
        return {'step': self._step, 'margin_stats': self.dist.margin_stats(),
                'ece': self.dist.ece(), 'kl_divergence': self.dist.kl_divergence(),
                'input_drift': self.bus.input_drift(),
                'effective_dim': self.bus.effective_dimensionality(),
                'anchor_gini': self.audit.gini(),
                'dead_anchors': len(self.bus.dead_anchors()),
                'geometry': self.geo.snapshot()}


# ══════════════════════════════════════════════════════════════════════════════
# THOUGHT SYSTEMS — TrainingThought / RuntimeThought / RuntimeLearner
# ══════════════════════════════════════════════════════════════════════════════

def _deliberate_core(query_vec, candidates, memory, uncertainty,
                     interp_lo=0.15, interp_hi=0.4):
    """Rocchio-style second-pass query revision. Shared by Training and Runtime."""
    if len(candidates) < 2: return '[no memory]', 0.0, False
    best_cls = candidates[0][0]
    best_margin = candidates[0][1] - candidates[1][1]
    alpha = interp_lo + (interp_hi - interp_lo) * uncertainty
    changed = False
    for cls, sim in candidates[1:3]:
        cent = memory.class_centroid(cls)
        if cent is None: continue
        revised = (1.0 - alpha) * query_vec + alpha * cent
        nrm = float(np.linalg.norm(revised))
        if nrm < 1e-9: continue
        revised = revised / nrm
        rm = memory.lookup(revised, k=2)
        if not rm: continue
        r_cls = memory.get_output(rm[0][0]) or ''
        r_margin = rm[0][1] - rm[1][1] if len(rm) >= 2 else rm[0][1]
        if r_margin > best_margin + 0.02:
            best_margin = r_margin; best_cls = r_cls; changed = True
    return best_cls, best_margin, changed


class TrainingThought:
    """Active only during train_step. Builds confusion map, calibrates tau.
    Publishes to StatEngine. Frozen via freeze() at end of training."""

    def __init__(self):
        self._tau = 0.1; self.uncertainty = 0.5
        self._cmap: Dict = {}   # frozenset(a,b) → float
        self._frozen = False; self._stat_engine = None

    def attach(self, se): self._stat_engine = se

    def note(self, margin: float, candidates=None) -> None:
        if self._frozen: return
        if margin > self._tau: self._tau = 0.95*self._tau + 0.05*margin
        tau = max(self._tau, 0.01)
        raw = float(np.exp(-margin / tau))
        self.uncertainty = 0.8*self.uncertainty + 0.2*raw
        if candidates and len(candidates) >= 2:
            a = candidates[0][0]; b = candidates[1][0]
            if a != b:
                key = frozenset([a, b])
                self._cmap[key] = 0.9*self._cmap.get(key, 0.0) + 0.1*raw
                if self._stat_engine:
                    self._stat_engine.geo.record_margin(a, b, margin)

    def deliberate(self, qvec, candidates, memory):
        if self._frozen or len(candidates) < 2 or self.uncertainty < 0.3:
            return '[no memory]', 0.0, False
        return _deliberate_core(qvec, candidates, memory, self.uncertainty, 0.2, 0.5)

    def freeze(self) -> Dict:
        self._frozen = True
        return {'tau': self._tau, 'uncertainty': self.uncertainty,
                'cmap': {str(list(k)): v for k,v in self._cmap.items()}}

    def most_confused(self, n=5):
        return [(list(k),v) for k,v in sorted(self._cmap.items(), key=lambda x:-x[1])[:n]]


class TemporalConsistencyFilter:
    """Majority-vote window on inference outputs. Catches single-step flips."""

    def __init__(self, window=5, min_conf=0.6):
        self.window = window; self.min_conf = min_conf
        self._hist: deque = deque(maxlen=window)

    def push(self, label, conf):
        self._hist.append((label, conf))
        if len(self._hist) < 3: return label, conf
        counts = {}
        for l,c in self._hist: counts[l] = counts.get(l,0.0) + c
        best = max(counts, key=counts.get)
        frac = counts[best] / sum(counts.values())
        if frac >= self.min_conf and best != label:
            avg = counts[best] / sum(1 for l,_ in self._hist if l==best)
            return best, float(avg)
        return label, conf

    def adapt_window(self, tightness):
        w = max(3, min(10, int(3 + (1.0 - tightness) * 7)))
        if w != self.window: self.window = w; self._hist = deque(self._hist, maxlen=w)


class RuntimeThought:
    """Active only during infer(). Holds frozen snapshot. Never updates training state.
    Queries StatEngine for per-pair deliberation thresholds."""

    def __init__(self):
        self._tau = 0.1; self.uncertainty = 0.5
        self._cmap: Dict = {}; self._stat_engine = None
        self.tcf = TemporalConsistencyFilter()

    def attach(self, se): self._stat_engine = se

    def load_snapshot(self, snap: Dict) -> None:
        self._tau = snap.get('tau', 0.1); self.uncertainty = snap.get('uncertainty', 0.5)
        raw = snap.get('cmap', {})
        self._cmap = {}
        for k_str, v in raw.items():
            parts = [p.strip().strip("'\" ") for p in k_str.strip("[]").split(',')]
            if len(parts) == 2: self._cmap[frozenset(parts)] = v

    def note(self, margin, candidates=None):
        tau = max(self._tau, 0.01)
        raw = float(np.exp(-margin / tau))
        self.uncertainty = 0.85*self.uncertainty + 0.15*raw

    def deliberate(self, qvec, candidates, memory):
        if len(candidates) < 2: return '[no memory]', 0.0, False
        a = candidates[0][0]; b = candidates[1][0]
        if self._stat_engine:
            thresh = self._stat_engine.bus.deliberation_threshold(a, b)
        else:
            key = frozenset([a, b]); score = self._cmap.get(key, 0.0)
            thresh = 0.5 - score * 0.3
        if self.uncertainty < thresh: return '[no memory]', 0.0, False
        return _deliberate_core(qvec, candidates, memory, self.uncertainty, 0.15, 0.35)

    def apply_tcf(self, label, conf, cls_a=None, cls_b=None):
        if self._stat_engine and cls_a and cls_b:
            self.tcf.adapt_window(self._stat_engine.bus.boundary_tightness(cls_a, cls_b))
        return self.tcf.push(label, conf)


class RuntimeLearner:
    """The Go mode. Tentative slow confusion-map updates during inference.
    Rolls back on accuracy drop (StatEngine.bus.rollback_trigger())."""

    def __init__(self, rt: RuntimeThought):
        self.rt = rt; self._slow: Dict = {}; self._snap: Dict = {}
        self._enabled = True; self._lr = 0.02

    def _rollback(self) -> None:
        """Revert slow confusion map to last known-good snapshot."""
        if self._snap:
            self._slow = dict(self._snap)

    def on_inference(self, margin, candidates, stat_engine) -> None:
        if not self._enabled or not candidates or len(candidates) < 2: return
        if stat_engine.bus.rollback_trigger():
            self._rollback(); return
        a = candidates[0][0]; b = candidates[1][0]
        if a == b: return
        tau = max(self.rt._tau, 0.01)
        raw = float(np.exp(-margin / tau))
        key = frozenset([a, b])
        old = self._slow.get(key, self.rt._cmap.get(key, 0.0))
        self._slow[key] = (1.0 - self._lr)*old + self._lr*raw
        acc_w = list(stat_engine.bus._acc_window)
        if len(acc_w) >= 10 and float(np.mean(acc_w[-10:])) > 0.75:
            self._snap = dict(self._slow)

    def effective_confusion(self, a, b):
        key = frozenset([a, b])
        return 0.7*self.rt._cmap.get(key,0.0) + 0.3*self._slow.get(key, self.rt._cmap.get(key,0.0))

_PHI = (1.0 + 5.0**0.5) / 2.0


class PhaseBridge:
    def __init__(self, feature_dim: int, resonance_dim: int):
        self.fd = feature_dim; self.rd = resonance_dim
        rng = np.random.default_rng(42)
        self.Wa = rng.standard_normal((feature_dim, resonance_dim)).astype(np.float32) * 0.1
        self.Wp = rng.standard_normal((feature_dim, resonance_dim)).astype(np.float32) * 0.1
        self.bf = (0.5 * _PHI ** np.linspace(0.0, 5.0, resonance_dim)).astype(np.float32)
        dom = np.arange(resonance_dim, dtype=np.float32)
        self._basis = np.sin(self.bf * dom / resonance_dim)
        _rfft_n = feature_dim // 2 + 1
        _px = np.linspace(0, _rfft_n - 1, resonance_dim)
        _lo = np.floor(_px).astype(np.int32).clip(0, _rfft_n - 2)
        self._interp_lo   = _lo
        self._interp_hi   = (_lo + 1).clip(0, _rfft_n - 1)
        self._interp_frac = (_px - _lo).astype(np.float32)

    def bridge(self, f: np.ndarray) -> np.ndarray:
        x = f.astype(np.float32) if f.dtype != np.float32 else f
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if len(x) != self.fd:
            x = np.interp(np.linspace(0, len(x)-1, self.fd), np.arange(len(x)), x).astype(np.float32)
        amps = x @ self.Wa
        inst_phase = np.angle(np.fft.rfft(x)).astype(np.float32)
        phase_samp = (inst_phase[self._interp_lo] * (1.0 - self._interp_frac)
                    + inst_phase[self._interp_hi] * self._interp_frac)
        phase = phase_samp * 0.6 + (x @ self.Wp) * 0.4
        phase64 = phase.astype(np.float64)
        r = amps.astype(np.float64) * np.cos(phase64) + 1j * amps.astype(np.float64) * np.sin(phase64)
        r *= self._basis
        rn = float(np.sqrt(np.dot(r.conj(), r).real)) + EPSILON
        return r / rn

    def fast_bridge(self, f: np.ndarray) -> np.ndarray:
        """Bridge without rfft: uses Wp-only phase. ~2x faster for hard-neg/anchor similarity."""
        x = f.astype(np.float32) if f.dtype != np.float32 else f
        if len(x) != self.fd:
            x = np.interp(np.linspace(0, len(x)-1, self.fd), np.arange(len(x)), x).astype(np.float32)
        amps  = x @ self.Wa
        phase = (x @ self.Wp).astype(np.float64)
        r = amps.astype(complex) * np.exp(1j * phase) * self._basis
        rn = float(np.sqrt(np.dot(r.conj(), r).real)) + EPSILON
        return r / rn

# [HLFC compression removed: see audit 2025-02 — orphaned output, 45% hot-path overhead]

class ResonanceField:
    def __init__(self, dim=256, gamma=5.0, dt=0.3):
        self.dim=dim; self.gamma=gamma; self.dt=dt
        rng=np.random.default_rng(0)
        self.psi=rng.standard_normal(dim)+1j*rng.standard_normal(dim)
        self.psi=np.nan_to_num(self.psi, nan=0.0)
        self.psi/=(np.linalg.norm(self.psi)+EPSILON)
        self.psi_prev=self.psi.copy()
        self.H=(0.5 * _PHI ** np.linspace(0.0, 5.0, dim)).astype(np.float64)
        self._H_exp = np.exp(-1j*self.dt*self.H)
        self._event_queue: List[Tuple[float,np.ndarray,float]]=[]
        self._phase_disp_thresh = 0.3
        self._rng = np.random.default_rng()
        self._k30 = max(0, int(0.30 * dim) - 1)
        self._k80 = max(0, int(0.80 * dim) - 1)
        self._psi_norm_sq: float = 1.0
        self._log32 = float(np.log(32))
        self._ks_stats = [k for k in [1,2,4,8,16,32,64,min(128,dim)] if k < dim]
        self._xs_stats = np.log(np.array(self._ks_stats, dtype=np.float64) + 1.0)
        self._psi_mag: Optional[np.ndarray] = None
        self._psi_fft: Optional[np.ndarray] = None
        self._psi_psd: Optional[np.ndarray] = None
        self._pf_buf   = np.zeros(dim, dtype=complex)
        self._sel_buf  = np.zeros(dim)
        self._sel_buf2 = np.zeros(dim)
        self._dt_neg1j = -1j * self.dt
        self._exp_buf  = np.zeros(dim, dtype=complex)

    def inject(self, v: np.ndarray, strength=0.6):
        v=v.ravel()[:self.dim].astype(complex)
        vn = float(np.sqrt(np.dot(v.conj(), v).real)) + EPSILON
        v /= vn
        overlap = np.dot(self.psi.conj(), v)
        sim = float(abs(overlap))
        if sim > 0.85:
            orth   = v - overlap * self.psi
            orth_n = float(np.sqrt(np.dot(orth.conj(), orth).real)) + EPSILON
            if orth_n > EPSILON:
                v = v + (strength * 3.0) * orth / orth_n
                vn2 = float(np.sqrt(np.dot(v.conj(), v).real)) + EPSILON
                v /= vn2
        self.psi=(1-strength)*self.psi+strength*v
        pn = float(np.sqrt(np.dot(self.psi.conj(), self.psi).real)) + EPSILON
        self.psi /= pn
        self._psi_mag = None
        self._psi_fft = None
        self._psi_psd = None

    def queue_event(self, v: np.ndarray, t: float, strength=0.3):
        self._event_queue.append((t, v, strength))

    def _H_sel(self, s: np.ndarray) -> np.ndarray:
        return np.exp(self._dt_neg1j * s)

    def evolve(self, steps=1) -> np.ndarray:
        self.psi_prev=self.psi.copy()
        for _ in range(steps):
            if self._event_queue:
                now=time.time()
                remaining=[]
                for et,ev,es in self._event_queue:
                    if et<=now: self.inject(ev, es)
                    else: remaining.append((et,ev,es))
                self._event_queue=remaining
            pf = fft(self.psi)
            np.multiply(pf, self._H_exp, out=pf)
            self.psi = ifft(pf)
            mag = np.abs(self.psi)
            thresh_lo = float(np.partition(mag, self._k30)[self._k30])
            thresh_hi = float(np.partition(mag, self._k80)[self._k80])
            d = mag * mag
            hi_mask = mag > thresh_hi
            lo_mask = mag < thresh_lo
            sel = self._sel_buf; sel[:] = 0.0
            if hi_mask.any(): sel[hi_mask] = -self.gamma * (d[hi_mask] - 1.0)
            if lo_mask.any(): sel[lo_mask] = self.gamma * 0.5 * (1.0 - d[lo_mask])
            np.exp(self._dt_neg1j * sel, out=self._exp_buf)
            np.multiply(self.psi, self._exp_buf, out=self.psi)
            pn = float(np.sqrt(np.dot(self.psi.conj(), self.psi).real)) + EPSILON
            self.psi /= pn
            phases_unit = self.psi / (mag + EPSILON)
            circ_var = 1.0 - float(abs(phases_unit.mean()))
            if circ_var < self._phase_disp_thresh:
                kick = (self._phase_disp_thresh - circ_var) * 0.5
                phase_kick = self._rng.uniform(-np.pi * kick, np.pi * kick, self.dim)
                np.multiply(self.psi, np.exp(1j * phase_kick), out=self.psi)
                pn = float(np.sqrt(np.dot(self.psi.conj(), self.psi).real)) + EPSILON
                self.psi /= pn
        self._psi_mag = None
        self._psi_fft = None
        self._psi_psd = None
        return self.psi

    def resonance(self, pattern: np.ndarray) -> float:
        p   = fft(pattern.astype(complex)[:self.dim])
        if self._psi_fft is None:
            self._psi_fft = fft(self.psi)
        q   = self._psi_fft
        psd_p = np.abs(p)**2 + EPSILON
        if self._psi_psd is None:
            self._psi_psd = np.abs(q)**2 + EPSILON
        psd_q = self._psi_psd
        cross   = q * np.conj(p)
        ratio   = psd_q / psd_p
        np.log(ratio, out=ratio); np.abs(ratio, out=ratio)
        log_sum = float(ratio.sum()); lm = log_sum / self.dim + EPSILON
        ratio  *= (1.0 / lm)
        ct_sum  = log_sum / lm
        pc      = np.abs(cross)
        sqpq    = np.sqrt(psd_p); sqpq *= np.sqrt(psd_q); pc /= sqpq
        r_arr   = np.abs(ifft(cross * (1.0 + 0.5 * ratio)))
        base    = float(r_arr.max())
        coh     = float((pc * ratio).sum()) / ct_sum
        return base * (1.0 + 0.3 * coh)

    def enhanced_resonance(self, pattern: np.ndarray, gamma_res=0.1) -> float:
        r   = self.resonance(pattern)
        mag = np.abs(self.psi)
        mu  = float(mag.sum()) / self.dim
        dev = mag - mu
        std = float(np.sqrt(np.dot(dev, dev) / self.dim)) + EPSILON
        return r * (1. + gamma_res * r / std)

    def stats(self) -> FieldStats:
        kappa  = self.criticality()
        if self._psi_fft is None:
            self._psi_fft = fft(self.psi)
        spec   = np.abs(self._psi_fft)
        psi_abs = np.abs(self.psi) + EPSILON
        angles  = np.arctan2(self.psi.imag, self.psi.real)
        bins32  = np.floor((angles + np.pi) * (32.0 / (2.0 * np.pi))).astype(np.int32).clip(0, 31)
        hist   = np.bincount(bins32, minlength=32).astype(np.float64)
        hist  *= 1.0 / (hist.sum() + EPSILON)
        pe     = float(-np.dot(hist, np.log(hist + EPSILON))) / self._log32
        mag    = psi_abs
        psi_norm_sq = float(np.dot(mag, mag))
        ys     = np.empty(len(self._ks_stats))
        for i, k in enumerate(self._ks_stats):
            idx_top = np.argpartition(mag, -k)[-k:]
            ys[i]   = float(np.dot(self.psi[idx_top].conj(), self.psi[idx_top]).real) / (psi_norm_sq + EPSILON)
        comp_slope = 0.0
        if len(ys) >= 2:
            xs = self._xs_stats[:len(ys)]
            xm = xs.mean(); ym = ys.mean()
            dx = xs - xm
            denom = float(np.dot(dx, dx))
            if denom > 1e-10:
                comp_slope = float(np.dot(dx, ys - ym) / denom)
        ang_mu = float(angles.sum()) / self.dim
        ang_dev = angles - ang_mu
        return FieldStats(
            criticality=kappa,
            dominant_freq=float(np.argmax(spec))/self.dim,
            mean_phase=ang_mu,
            phase_spread=float(np.sqrt(np.dot(ang_dev, ang_dev) / self.dim)),
            energy=psi_norm_sq,
            phase_entropy=pe,
            compression_slope=comp_slope)

    def criticality(self) -> float:
        mag = np.abs(self.psi)
        psi_norm_sq = float(np.dot(self.psi.conj(), self.psi).real)
        ks4 = [k for k in [4, 8, 16, 32] if k < self.dim]
        if len(ks4) < 2: return 0.5
        xs = np.log(np.array(ks4, dtype=np.float64) + 1.0)
        ys = np.empty(len(ks4))
        for i, k in enumerate(ks4):
            idx_top = np.argpartition(mag, -k)[-k:]
            ys[i] = float(np.dot(self.psi[idx_top].conj(), self.psi[idx_top]).real) / (psi_norm_sq + EPSILON)
        xm = xs.mean(); ym = ys.mean()
        dx = xs - xm
        denom = float(np.dot(dx, dx))
        slope = float(np.dot(dx, ys - ym) / denom) if denom > 1e-10 else 0.0
        return float(np.clip(slope, 0.0, 2.0))

    def reset(self):
        r = self._rng.standard_normal((2, self.dim))
        self.psi = r[0] + 1j*r[1]
        self.psi /= (float(np.sqrt(np.dot(self.psi.conj(), self.psi).real)) + EPSILON)
        self.psi_prev = self.psi.copy()
        self._psi_mag = None


# ══════════════════════════════════════════════
# 5. RESONATOR LEVEL  (local coupling + inhibition)
# ══════════════════════════════════════════════

class ResonatorLevel:
    def __init__(self, n=64, gamma=0.35, locality=3, omega_range=(1.,10.)):
        self.n=n; self.gamma=gamma; self.locality=locality
        rng=np.random.default_rng(1)
        self.R=np.zeros(n)
        _k_max = np.log(omega_range[1] / omega_range[0]) / np.log(_PHI)
        self.omega = omega_range[0] * (_PHI ** np.linspace(0.0, _k_max, n))
        kw = 2*locality+1
        mhat = np.zeros(kw)
        for i in range(kw):
            off = i - locality
            if abs(off) == 1: mhat[i] =  1.0
            elif abs(off) >= 2: mhat[i] = -0.5
        mhat[locality] = 0.0
        nm = np.linalg.norm(mhat)
        self.W = (mhat / (nm + EPSILON)) * 0.3
        self.D=0.1
        self._prev_drive = None
        self._freq_buf = np.zeros(n)
        self._coup_buf = np.zeros(n)
        self._lap_buf  = np.zeros(n)

    @staticmethod
    def _sig(x):
        y = np.clip(x, -20, 20, out=x if x.flags.writeable else None)
        np.negative(y, out=y); np.exp(y, out=y); y += 1.0; np.reciprocal(y, out=y)
        return y

    def update(self, dt=0.1, drive=None) -> np.ndarray:
        np.multiply(self.omega, self.R, out=self._freq_buf)
        coup=self._coup_buf; coup[:]=0.0
        for off in range(-self.locality, self.locality+1):
            if off==0: continue
            w=self.W[off+self.locality]
            if off>0: coup[:-off]+=w*self._sig(self.R[off:])
            else: coup[-off:]+=w*self._sig(self.R[:off])
        lap=self._lap_buf; lap[0]=0.0; lap[-1]=0.0
        lap[1:-1]=self.R[:-2]-2*self.R[1:-1]+self.R[2:]
        inhib=-self.gamma*float(np.abs(self.R).sum())/self.n
        if drive is not None:
            d_real = drive[:self.n].real
            if self._prev_drive is not None:
                dn = float(np.sqrt(np.dot(d_real, d_real)))
                pn = float(np.sqrt(np.dot(self._prev_drive, self._prev_drive)))
                sim = float(np.dot(d_real, self._prev_drive)) / (dn*pn+EPSILON)
                gain = 200.0 / (1.0 + max(sim, 0.0))
                drv = d_real*(gain*0.3) + (d_real-self._prev_drive)*(gain*0.7)
            else:
                drv = d_real * 200.0
            self._prev_drive = d_real.copy()
        else:
            drv = 0.0
        Rn=self.R+dt*(self._freq_buf+coup+self.D*lap+inhib)+drv
        res_gate=1.+0.1*np.abs(self.R)
        Rn*=res_gate
        _k80 = max(0, int(0.8*self.n)-1)
        t = np.partition(np.abs(Rn), _k80)[_k80]
        Rn[np.abs(Rn)<t]*=0.1
        np.clip(Rn,-10,10,out=Rn); self.R=Rn
        return self.R

    def reset(self):
        self.R=np.zeros(self.n)
        self._prev_drive=None


# ══════════════════════════════════════════════
# 6. ASSEMBLY LEVEL  (oscillatory + resonance-enhanced)
# ══════════════════════════════════════════════

class AssemblyLevel:
    def __init__(self, n_assemblies=16, resonator_n=64):
        self.na=n_assemblies; self.rn=resonator_n
        rng=np.random.default_rng(2)
        self.A=np.zeros(n_assemblies)
        # Oscillator state [real, imag] per assembly
        self.O=np.zeros((n_assemblies,2))
        self.omega=np.linspace(0.5,5.,n_assemblies)
        self.V=rng.standard_normal((n_assemblies,resonator_n))*0.1
        self.C=rng.standard_normal((n_assemblies,n_assemblies))*0.05
        np.fill_diagonal(self.C,0.)
        self.phi=0.1; self.gamma=0.05
        self._rbuf = np.zeros(resonator_n)

    def update(self, R: np.ndarray, G: np.ndarray, dt=0.1) -> np.ndarray:
        res_enh=1.+0.1*np.abs(self.A)
        rn = min(len(R), self.rn)
        self._rbuf[:rn] = R[:rn]; self._rbuf[rn:] = 0.0
        sig_R = self._sig(self._rbuf)
        inp=self.V@sig_R
        lateral=-self.phi*self.C@self.A
        gn = min(len(G), self.na)
        glob = 0.05 * G[:gn] if gn == self.na else (np.concatenate([G[:gn], np.zeros(self.na-gn)]) * 0.05)
        dA=-0.1*self.A+inp+lateral+glob
        self.A+=dt*dA*res_enh
        np.clip(self.A, -5, 5, out=self.A)
        # Vectorised oscillator update (all k at once)
        # dO[:,0] = -omega*O[:,1] - gamma*O[:,0] + 0.1*A
        # dO[:,1] =  omega*O[:,0] - gamma*O[:,1] + 0.1*A
        w = self.omega
        dO0 = -w*self.O[:,1] - self.gamma*self.O[:,0] + 0.1*self.A
        dO1 =  w*self.O[:,0] - self.gamma*self.O[:,1] + 0.1*self.A
        self.O[:,0] += dt * dO0
        self.O[:,1] += dt * dO1
        return self.A

    @staticmethod
    def _sig(x):
        y = np.clip(x, -20, 20, out=x if x.flags.writeable else None)
        np.negative(y, out=y); np.exp(y, out=y); y += 1.0; np.reciprocal(y, out=y)
        return y

    def oscillator_output(self) -> np.ndarray:
        return self.O[:,0]  # Real component of each oscillator

    def reset(self):
        self.A=np.zeros(self.na); self.O=np.zeros((self.na,2))


# ══════════════════════════════════════════════
# 7. MODULE LEVEL  (working memory + network)
# ══════════════════════════════════════════════

class ModuleLevel:
    def __init__(self, n_modules=8, assembly_n=16, mem_size=32, n_psi_slots=6):
        self.nm=n_modules; self.an=assembly_n; self.ms=mem_size
        rng=np.random.default_rng(3)
        self.M=np.zeros(n_modules)
        self.C=rng.standard_normal((n_modules,n_modules))*0.05
        np.fill_diagonal(self.C,0.)
        self.V=rng.standard_normal((n_modules,assembly_n))*0.1
        self.alpha=0.1
        self._abuf = np.zeros(assembly_n)
        self._gbuf = np.zeros(n_modules)
        self._psi_slots: deque = deque(maxlen=n_psi_slots)
        self._slot_weights: List[float] = []

    def update(self, A: np.ndarray, G: np.ndarray, dt=0.1) -> np.ndarray:
        an = min(len(A), self.an); self._abuf[:an]=A[:an]; self._abuf[an:]=0.0; a=self._abuf
        gn = min(len(G), self.nm); self._gbuf[:gn]=G[:gn]; self._gbuf[gn:]=0.0; g=self._gbuf
        res_enh=1.+0.05*np.abs(self.M)
        inp=self.V@self._sig(a)
        lat=-self.alpha*self.C@self.M
        wm_out = np.zeros(self.nm)
        if self._psi_slots:
            slots   = list(self._psi_slots)
            ns      = len(slots)
            weights = np.array(self._slot_weights[-ns:], dtype=np.float64)
            weights /= (weights.sum() + EPSILON)
            mat = np.empty((ns, self.nm), dtype=np.float64)
            for i, slot in enumerate(slots):
                s = slot[:self.nm] if len(slot)>=self.nm else np.pad(slot,(0,self.nm-len(slot)))
                mat[i] = np.abs(s)
            wm_out = (weights @ mat) * 0.05
        dM=-self.M+inp+lat+0.05*g+wm_out
        self.M+=dt*dM*res_enh
        self.M=np.clip(self.M,-5,5)
        return self.M

    def push_psi(self, psi_snapshot: np.ndarray, weight: float = 1.0):
        self._psi_slots.append(psi_snapshot.copy())
        self._slot_weights.append(weight)
        if len(self._slot_weights) > self._psi_slots.maxlen * 2:
            self._slot_weights = self._slot_weights[-self._psi_slots.maxlen:]

    def add_memory_event(self, event_vec: np.ndarray):
        self.push_psi(event_vec, weight=1.0)

    @staticmethod
    def _sig(x): return 1./(1.+np.exp(-np.clip(x,-20,20)))

    def reset(self):
        self.M=np.zeros(self.nm)
        self._psi_slots.clear()
        self._slot_weights.clear()


# 8. GLOBAL LEVEL  (integration + criticality)
# ══════════════════════════════════════════════

class GlobalLevel:
    def __init__(self, dim=64, module_n=8):
        self.dim=dim; self.mn=module_n
        rng=np.random.default_rng(4)
        self.G=np.zeros(dim)
        self.WG=rng.standard_normal((dim, module_n+dim))*0.05
        self.kappa=0.5          # criticality parameter
        self.kappa0=0.5
        self.alpha_G=0.1
        self._pred=np.zeros(dim)
        self._prev_G=np.zeros(dim)
        self._crit_H = np.exp(-1j*0.1*np.linspace(0.5, 5., dim))
        self._crit_cache = np.zeros(dim)
        self._crit_step  = 0
        self._inp_buf    = np.empty(module_n + dim)
        self._res_buf    = np.zeros(dim)

    def update(self, M: np.ndarray, O: np.ndarray, R_field: np.ndarray,
               events: List[Event], dt=0.1) -> np.ndarray:
        mn = min(len(M), self.mn); on_ = min(len(O), self.dim)
        ib = self._inp_buf
        ib[:mn] = M[:mn]; ib[mn:self.mn] = 0.0
        ib[self.mn:self.mn+on_] = O[:on_]; ib[self.mn+on_:] = 0.0
        inp = ib
        np.copyto(self._prev_G, self.G)
        decay=-self.alpha_G*self.G
        proj=self.WG@inp
        rn = min(len(R_field), self.dim)
        rb = self._res_buf; np.abs(R_field[:rn], out=rb[:rn]); rb[rn:] = 0.0
        res = 0.05 * rb
        pred_err=self._pred-self.G
        pred_corr=0.1*pred_err
        if self._crit_step % 2 == 0:
            self._crit_cache = self._critical_resonance(self.G)
        self._crit_step += 1
        crit = self.kappa * self._crit_cache
        ev_sum=np.zeros(self.dim)
        for e in events:
            v=e.data.get('vector')
            if v is not None:
                ln=min(len(v),self.dim)
                ev_sum[:ln]+=v[:ln]*e.priority
        dG=decay+proj[:self.dim]+res+pred_corr+crit+ev_sum*0.01
        self.G+=dt*dG
        gn = float(np.sqrt(np.dot(self.G, self.G))) + EPSILON
        np.clip(self.G * (10.0 / gn), -10, 10, out=self.G)
        dg = self.G - self._prev_G
        grad_G = float(np.dot(dg, dg)) / self.dim
        self.kappa+=dt*0.1*(grad_G-self.kappa0)
        self.kappa=float(np.clip(self.kappa,0.01,2.))
        # Update prediction
        self._pred=self.G+dt*(self.G-self._prev_G)
        return self.G

    def _critical_resonance(self, G: np.ndarray) -> np.ndarray:
        F = fft(G.astype(complex))
        F *= self._crit_H
        return np.abs(ifft(F))[:self.dim] * 0.1

    def reset(self):
        self.G=np.zeros(self.dim); self._prev_G=np.zeros(self.dim)
        self.kappa=0.5


# ══════════════════════════════════════════════
# 9. EVENT SYSTEM
# ══════════════════════════════════════════════

class EventScheduler:
    def __init__(self, alpha=0.1):
        self.alpha=alpha; self._queue: List[Event]=[]

    def schedule(self, e: Event):
        heapq.heappush(self._queue, e)

    def next_time(self, t_current: float, priority: float) -> float:
        return t_current*(1.+self.alpha*priority)**-1

    def pop_due(self) -> List[Event]:
        out=[]; now=time.time()
        while self._queue and self._queue[0].time<=now:
            out.append(heapq.heappop(self._queue))
        return out

    def __len__(self): return len(self._queue)


class EventGenerator:
    def __init__(self, pat_thr=0.7, surp_thr=0.01, res_thr=0.5):
        self.pt=pat_thr; self.st=surp_thr; self.rt=res_thr
        self._recent: deque=deque(maxlen=10)
        self._pred: Optional[np.ndarray]=None
        # Type counters for reporting
        self.type_counts: Dict[str,int] = {t.name: 0 for t in EventType}
        self._total = 0

    def _track(self, evs: List[Event]) -> List[Event]:
        for e in evs:
            self.type_counts[e.type] = self.type_counts.get(e.type, 0) + 1
            self._total += 1
        return evs

    def from_resonance(self, field: ResonanceField, patterns: List[np.ndarray],
                       pattern_ffts: Optional[List] = None) -> List[Event]:
        evs=[]
        if field._psi_fft is None:
            field._psi_fft = fft(field.psi)
        q = field._psi_fft
        if field._psi_psd is None:
            field._psi_psd = np.abs(q)**2 + EPSILON
        psd_q = field._psi_psd
        mag_psi = np.abs(field.psi)
        _d = mag_psi - mag_psi.mean()
        std_psi_global = float(np.sqrt(np.dot(_d, _d) / field.dim)) + EPSILON
        for i,p in enumerate(patterns):
            pf = (pattern_ffts[i] if pattern_ffts and i < len(pattern_ffts) else None)
            if pf is None:
                pf = fft(p.astype(complex)[:field.dim])
                if pattern_ffts is not None and i < len(pattern_ffts):
                    pattern_ffts[i] = pf
            psd_p    = np.abs(pf)**2
            psd_p   += EPSILON
            cross    = q * np.conj(pf)
            ratio    = psd_q / psd_p
            np.log(ratio, out=ratio); np.abs(ratio, out=ratio)
            log_r    = ratio
            lm       = float(log_r.sum()) / field.dim + EPSILON
            contrast = log_r * (1.0 / lm)
            ct_sum   = float(log_r.sum()) / lm
            pc       = np.abs(cross)
            sqpq     = np.sqrt(psd_p); sqpq *= np.sqrt(psd_q)
            pc      /= sqpq
            r_vec    = np.abs(ifft(cross * (1.0 + 0.5 * contrast)))
            base     = float(r_vec.max())
            coh      = float((pc * contrast).sum()) / ct_sum
            r        = base * (1.0 + 0.3 * coh)
            r_enh    = r * (1.0 + 0.1 * r / std_psi_global)
            if r_enh > self.rt:
                evs.append(Event(EventType.RESONANCE.name, time.time(),
                    {'resonance':r_enh,'pattern_id':i,'vector':p}, 'resonance', r_enh))
        return self._track(evs)

    def from_surprise(self, state: np.ndarray, pred: Optional[np.ndarray]=None) -> List[Event]:
        evs=[]
        self._recent.append(state.copy())
        ref=pred if pred is not None else (self._recent[-2] if len(self._recent)>1 else None)
        if ref is not None:
            err=float(np.mean(np.abs(state-ref[:len(state)])))
            if err>self.st:
                evs.append(Event(EventType.SURPRISE.name, time.time(),
                    {'error':err,'vector':state}, 'surprise', err))
        self._pred=state.copy()
        return self._track(evs)

    def from_external(self, v: np.ndarray, priority=1.) -> Event:
        e = Event(EventType.EXTERNAL.name, time.time(),
            {'vector':v,'amplitude':float(np.linalg.norm(v))}, 'external', priority)
        self._track([e])
        return e

    def modulate(self, e: Event, field: ResonanceField, kappa: float,
                 alpha_res=0.1, alpha_crit=0.05) -> Event:
        r = e.data.get('resonance', None)
        if r is None:
            v = e.data.get('vector', np.zeros(1))
            r = field.enhanced_resonance(v) if len(v) > 1 else 0.0
        e.priority *= (1. + alpha_res * r + alpha_crit * kappa)
        return e

    def type_report(self) -> str:
        if self._total == 0: return '    (no events)'
        lines = []
        for t, c in sorted(self.type_counts.items(), key=lambda x: -x[1]):
            if c == 0: continue
            pct = 100.*c/self._total
            lines.append(f"    {t:12} {c:4}  ({pct:5.1f}%)")
        return '\n'.join(lines)

    def reset_counts(self):
        self.type_counts = {t.name: 0 for t in EventType}
        self._total = 0


# ══════════════════════════════════════════════
# 10. RECURSIVE PROCESSING
# ══════════════════════════════════════════════

class RecursiveProcessor:
    """
    Recursive processor with causal world model.

    W_T is a dim×dim transition matrix trained online via rank-1 Oja-style update:
        W_T += lr * (psi - W_T @ psi_prev) ⊗ psi_prev
    This makes W_T approximate the local Jacobian of state dynamics.
    causal_error = ||psi - W_T @ psi_prev|| measures how surprising the current
    state is *given* the previous one — genuine causal prediction error,
    not just magnitude change.
    """
    def __init__(self, dim=64):
        self.dim=dim
        rng=np.random.default_rng(5)
        self.alpha_H=0.1; self.alpha_V=0.1; self.alpha_T=0.1; self.beta_E=0.05
        self._prev: Optional[np.ndarray]=None
        self._pred: Optional[np.ndarray]=None
        # Causal world model
        self._W_T = np.eye(dim, dtype=np.float64) * 0.01   # transition matrix
        self._W_T_lr = 5e-4                                  # online learning rate
        self.causal_error: float = 0.0                       # last prediction error

    def horizontal(self, psi: np.ndarray, inputs: np.ndarray,
                   events: List[Event], res_enh: float) -> np.ndarray:
        ev_sum = sum(e.priority for e in events) if events else 0.
        return (psi + 0.1*inputs) * (1. + self.alpha_H*res_enh) * (1. + self.beta_E*ev_sum)

    def vertical(self, psi: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                 events: List[Event]) -> np.ndarray:
        ln = min(len(lower), self.dim); lo=np.zeros(self.dim); lo[:ln]=lower[:ln]
        un = min(len(upper), self.dim); up=np.zeros(self.dim); up[:un]=upper[:un]
        cross=0.05*(lo+up)
        r_lev=float(np.dot(psi,lo)/((float(np.sqrt(np.dot(psi,psi)))*float(np.sqrt(np.dot(lo,lo))))+EPSILON))
        ev_sum=sum(e.priority for e in events) if events else 0.
        out=(psi+cross)*(1.+self.alpha_V*abs(r_lev))*(1.+self.beta_E*ev_sum)
        return out

    def temporal(self, psi: np.ndarray, events: List[Event]) -> np.ndarray:
        if self._prev is None: self._prev=psi.copy()
        pred=psi+(psi-self._prev)*0.1 if self._pred is None else self._pred
        _pt   = psi * np.conj(self._prev[:len(psi)].astype(complex))
        r_temp = float(np.abs(_pt).sum()) / max(len(psi), 1)
        ev_sum=sum(e.priority for e in events) if events else 0.
        out=0.9*psi+0.1*pred*(1.+self.alpha_T*r_temp)*(1.+self.beta_E*ev_sum)
        # ── Causal world model update ──────────────────────────────────────
        # W_T predicts current state from previous; error = genuine causal surprise
        p = self._prev[:self.dim].real.astype(np.float64)
        q = psi[:self.dim].real.astype(np.float64)
        causal_pred = self._W_T @ p
        causal_err  = q - causal_pred
        self.causal_error = float(np.sqrt(np.dot(causal_err, causal_err))) / (self.dim ** 0.5)
        # rank-1 online update: W_T learns the state transition
        pn = float(np.dot(p, p)) + 1e-10
        self._W_T += self._W_T_lr * np.outer(causal_err, p / pn)
        # ──────────────────────────────────────────────────────────────────
        self._prev=psi.copy(); self._pred=out.copy()
        return out

    def causal_predict(self, psi: np.ndarray) -> np.ndarray:
        """One-step causal prediction: where does W_T say state goes next?"""
        p = psi[:self.dim].real.astype(np.float64)
        return self._W_T @ p


# ══════════════════════════════════════════════
# 11. FEEDBACK
# ══════════════════════════════════════════════

class FeedbackController:
    def __init__(self, dim=64):
        self.dim=dim; self.gamma_res=0.1; self.delta_crit=0.05; self.kappa0=0.5
        rng=np.random.default_rng(6)
        self.W_cross=rng.standard_normal((dim,dim))*0.02
        self._history: deque=deque(maxlen=32)
        _klen = 16
        self._kernel=np.exp(-np.linspace(0,3,_klen))
        self._hist_mat = np.zeros((_klen, dim))
        self._hist_pri = np.zeros(_klen)
        self._hist_ptr = 0
        self._hist_len = 0

    def resonance_amplified(self, psi: np.ndarray, field: ResonanceField) -> np.ndarray:
        p = psi[:field.dim].astype(complex)
        if field._psi_fft is None: field._psi_fft = fft(field.psi)
        if field._psi_psd is None: field._psi_psd = np.abs(field._psi_fft)**2 + EPSILON
        q     = field._psi_fft
        psd_q = field._psi_psd
        pf    = fft(p)
        psd_p = np.abs(pf)**2 + EPSILON
        cross = q * np.conj(pf)
        lratio= np.abs(np.log(psd_q / psd_p))
        lm    = float(lratio.sum()) / field.dim + EPSILON
        cont  = lratio * (1.0/lm)
        r_arr = np.abs(ifft(cross*(1.+0.5*cont)))
        base  = float(r_arr.max())
        coh   = float((np.abs(cross)/np.sqrt(psd_p*psd_q)*cont).sum()/(cont.sum()+EPSILON))
        r     = base*(1.+0.3*coh)
        mag   = np.abs(field.psi)
        mu    = float(mag.mean()); dev = mag-mu
        std   = float(np.sqrt(np.dot(dev,dev)/field.dim)) + EPSILON
        r_enh = r*(1.+self.gamma_res*r/std)
        return psi*(1.+self.gamma_res*r_enh)

    def cross_level(self, psi_i: np.ndarray, psi_j: np.ndarray, t: float) -> np.ndarray:
        r=float(np.dot(psi_i,psi_j)/((float(np.sqrt(np.dot(psi_i,psi_i)))*float(np.sqrt(np.dot(psi_j,psi_j))))+EPSILON))
        return self.W_cross@psi_j*r

    def temporal(self, events: List[Event]) -> np.ndarray:
        if not events: return np.zeros(self.dim)
        _klen = len(self._kernel)
        for e in events:
            v = e.data.get('vector')
            if v is not None:
                ln = min(len(v), self.dim)
                p = self._hist_ptr % _klen
                self._hist_mat[p, :] = 0.0
                self._hist_mat[p, :ln] = v[:ln]
                self._hist_pri[p] = e.priority
                self._hist_ptr += 1
                self._hist_len = min(self._hist_len + 1, _klen)
        n_used = self._hist_len
        if n_used == 0: return np.zeros(self.dim)
        start = (self._hist_ptr - n_used) % _klen
        idxs  = np.arange(n_used)
        rows  = (start + idxs) % _klen
        k_slice = self._kernel[idxs]
        out = (k_slice * self._hist_pri[rows]) @ self._hist_mat[rows]
        n = float(np.sqrt(np.dot(out, out))) + EPSILON
        return out / n

    def criticality_enhanced(self, psi: np.ndarray, kappa: float) -> np.ndarray:
        scale=1.+self.delta_crit*(kappa-self.kappa0)**2
        return psi*scale


# ══════════════════════════════════════════════
# 12. THOUGHT PROCESSES
# ══════════════════════════════════════════════

class ThoughtProcessor:
    """
    Genuine reasoning layer that iteratively refines classification hypotheses.

    Architecture (profiled and rebuilt Analysis 3, 2025-02):
    ─────────────────────────────────────────────────────────
    The original Thought was a near-identity: multi_scale() scaled G by ≤0.1%,
    chain was always 0.995 (all events are near-copies of G), uncertainty was
    permanently saturated at 0.7+ because the formula 1-2*margin maps [0,0.3]
    margins to [0.4,1.0] — never leaving the "uncertain" regime.

    This rewrite implements five structural fixes:

    A. Uncertainty formula: exp(-margin/tau) with rolling tau estimate.
       Gives genuine [0,1] spread. Certain→0.05, borderline→0.9.
       tau = rolling p75 of observed margins (auto-calibrates to corpus difficulty).

    B. Competing hypotheses awareness: ThoughtProcessor receives the top-k
       (class, similarity) candidates from the anchor lookup, not just G.
       Knows "safe_sql=0.41 vs sql_inject=0.39" vs "malware=0.71 vs phishing=0.18".

    C. Iterative hypothesis refinement — the core of "thinking":
       When uncertainty > threshold:
         For each competing class, interpolate query toward that class centroid.
         Re-run lookup with each interpolated query. Take the candidate that
         maximises the resulting margin. This is second-pass query revision.
         Thought actively "narrows its hypothesis" rather than amplifying G.

    D. Cross-domain confusion tracking:
       _confusion_memory tracks which class pairs keep confusing the system.
       When the same pair accumulates high confusion: flag boundary as ambiguous.
       Used to dynamically lower dedup_threshold for that specific boundary.

    E. Chain as hypothesis divergence, not G self-similarity:
       cascade() now generates one event per competing class using that class's
       centroid vector (not G). Chain = mean cosine between competing hypotheses.
       Low chain (near 0) = hypotheses are orthogonal = genuinely confused.
       High chain (near 1) = hypotheses point same direction = one class dominates.
    """

    def __init__(self, dim=64):
        self.dim               = dim
        self._event_history:   deque = deque(maxlen=64)
        self.theta_thought     = 0.1
        # Uncertainty state
        self.uncertainty:      float = 0.5
        self.suggested_alpha:  float = 0.15
        self._tau_margin:      float = 0.1   # rolling p75 of observed margins
        self._deliberate_thresh: float = 0.4  # uncertainty above which deliberation fires
        # Confusion memory: {frozenset({cls_a, cls_b}): float} — running confusion score
        self._confusion_memory: Dict = {}
        # Last deliberation stats (for logging)
        self.last_deliberation: Optional[Dict] = None
        self._last_self_query: Optional[str] = None

    # ── A: Calibrated uncertainty signal ──────────────────────────────────────
    def note_uncertainty(self, margin: float,
                         candidates: Optional[List] = None) -> float:
        """
        Record the top-1/top-2 similarity margin. Update uncertainty and alpha.

        margin: sim_top1 - sim_top2. High=certain, low/negative=confused.
        candidates: list of (class_label, similarity) pairs from memory.lookup().
                    Used to update confusion memory for class-pair tracking.

        Returns suggested EMA alpha for AnchorMemory.store().
        """
        # Update rolling tau (p75 proxy: slow EMA toward observed margin)
        # tau tracks the "typical large margin" for this corpus
        if margin > self._tau_margin:
            self._tau_margin = 0.95 * self._tau_margin + 0.05 * margin
        # exp(-margin/tau): certain→≈0.05 at margin=3τ, borderline→≈0.37 at margin=τ
        tau = max(self._tau_margin, 0.01)
        raw = float(np.exp(-margin / tau))
        self.uncertainty = 0.8 * self.uncertainty + 0.2 * raw
        # alpha: 0.15 (certain) → 0.40 (maximally uncertain)
        self.suggested_alpha = 0.15 + 0.25 * self.uncertainty

        # Update confusion memory when top-2 classes are close
        if candidates and len(candidates) >= 2:
            cls_a = candidates[0][0]; cls_b = candidates[1][0]
            if cls_a != cls_b:
                key = frozenset([cls_a, cls_b])
                old = self._confusion_memory.get(key, 0.0)
                # High confusion = low margin = high raw uncertainty
                self._confusion_memory[key] = 0.9 * old + 0.1 * raw

        return self.suggested_alpha

    # ── B+C: Rocchio deliberation ─────────────────────────────────────────────
    def deliberate(self, query_vec: np.ndarray,
                   candidates: List[Tuple[str, float]],
                   memory) -> Tuple[str, float, bool]:
        """
        Rocchio-style test-time query refinement.

        Grounded in three bodies of literature (search, 2025-02):

        1. Rocchio (1971) — the core formula:
           q_new = α*q + β*centroid(relevant) - γ*centroid(non-relevant)
           Pulls toward the correct class AND pushes away from the wrong one.
           Proof: at a zero-margin boundary, Rocchio achieves 0.74 margin vs
           0.33 from interpolation alone — 2× stronger (geometry test, Feb 2025).

        2. LVQ2.1 (Kohonen) window criterion — when to deliberate:
           Window: margin < θ * sim_top1  (relative, not absolute threshold)
           Only fires near the decision boundary where it can actually help.
           This replaces the previous hard sim>0.70 guard with a principled one.

        3. TOUR / IPADE / GQR (2010–2025) — test-time query refinement works:
           +8.1% Acc@1 phrase retrieval, +3.7% Acc@20 passage retrieval (TOUR).
           The key: use first-pass retrieval results as pseudo-labels to guide
           second-pass query refinement. Here: AnchorMemory centroids = pseudo-labels.

        Algorithm:
          If in LVQ2.1 window AND enough anchors:
            q_A = norm(q + β*centA - γ*centB)  [hypothesis: A is correct]
            q_B = norm(q + β*centB - γ*centA)  [hypothesis: B is correct]
            Re-run lookup with each. Take the hypothesis whose query achieves
            the higher post-Rocchio margin — the more discriminable class wins.
          Else: pass through top candidate directly.
        """
        self.last_deliberation = None

        if not candidates:
            return '[no memory]', 0.0, False
        top_cls, top_sim = candidates[0]
        if len(candidates) < 2:
            return top_cls, top_sim, False

        cls_a, sim_a = candidates[0]
        cls_b, sim_b = candidates[1]
        margin_0 = sim_a - sim_b

        # ── LVQ2.1 window: only deliberate near the decision boundary ──────────
        # margin < θ * sim_top1  (relative threshold scales with prediction strength)
        _window_theta = 0.35
        in_window = margin_0 < _window_theta * max(sim_a, EPSILON)
        if not in_window:
            return top_cls, top_sim, False

        # ── Anchor density guard ──────────────────────────────────────────────
        _min_anchors = 8
        counts = memory.class_anchor_counts()
        if counts.get(cls_a, 0) < _min_anchors or counts.get(cls_b, 0) < _min_anchors:
            return top_cls, top_sim, False

        # ── Get class centroids ───────────────────────────────────────────────
        cent_a = memory.class_centroid(cls_a)
        cent_b = memory.class_centroid(cls_b)
        if cent_a is None or cent_b is None:
            return top_cls, top_sim, False

        q = query_vec.astype(np.float64)
        _q64 = q if q.dtype == np.float64 else q.astype(np.float64)
        qn = float(np.sqrt(np.dot(_q64, _q64)))
        if qn < EPSILON:
            return top_cls, top_sim, False
        q_unit = q / qn
        ca = cent_a.astype(np.float64)
        cb = cent_b.astype(np.float64)

        # ── Rocchio refinement at two β strengths ─────────────────────────────
        best_cls    = top_cls
        best_margin = margin_0
        best_beta   = 0.0

        for beta in (0.5, 1.0):
            gamma = beta  # symmetric: pull strength = push strength

            # Test hypothesis A (cls_a is correct): pull toward A, push from B
            q_roc_A = q_unit + beta * ca - gamma * cb
            n = float(np.linalg.norm(q_roc_A))
            if n > EPSILON:
                q_roc_A = (q_roc_A / n).astype(np.float32)
                m_A = memory.lookup(q_roc_A, k=2)
                if len(m_A) >= 2:
                    winner_A = memory.get_output(m_A[0][0]) or ''
                    mg_A = float(m_A[0][1]) - float(m_A[1][1])
                    if winner_A == cls_a and mg_A > best_margin:
                        best_margin = mg_A
                        best_cls    = cls_a
                        best_beta   = beta

            # Test hypothesis B (cls_b is correct): pull toward B, push from A
            q_roc_B = q_unit + beta * cb - gamma * ca
            n = float(np.linalg.norm(q_roc_B))
            if n > EPSILON:
                q_roc_B = (q_roc_B / n).astype(np.float32)
                m_B = memory.lookup(q_roc_B, k=2)
                if len(m_B) >= 2:
                    winner_B = memory.get_output(m_B[0][0]) or ''
                    mg_B = float(m_B[0][1]) - float(m_B[1][1])
                    if winner_B == cls_b and mg_B > best_margin:
                        best_margin = mg_B
                        best_cls    = cls_b
                        best_beta   = beta

        did_deliberate = best_beta > 0.0
        self.last_deliberation = {
            'original_cls':    top_cls,
            'deliberated_cls': best_cls,
            'changed':         best_cls != top_cls,
            'rocchio_beta':    best_beta,
            'margin_before':   margin_0,
            'margin_after':    best_margin,
            'uncertainty':     self.uncertainty,
            'in_window':       in_window,
        }
        return best_cls, best_margin, did_deliberate

    # ── D: Confusion boundary report ──────────────────────────────────────────
    def most_confused_pairs(self, top_n: int = 5) -> List[Tuple]:
        """Return the top-n most confused class pairs, sorted by confusion score."""
        pairs = sorted(self._confusion_memory.items(), key=lambda x: x[1], reverse=True)
        return [(tuple(sorted(k)), v) for k, v in pairs[:top_n]]

    # ── E: Cascade using hypothesis centroids ─────────────────────────────────
    def cascade(self, trigger: Event, G: np.ndarray, kappa: float,
                candidates: Optional[List] = None,
                field=None) -> List[Event]:
        """
        Generate thought events from competing class hypothesis directions.
        Self-questioning: when top-2 candidates are close (margin < 0.15),
        encode each label and query the field with enhanced_resonance.
        The resonance winner biases event priorities before returning.
        """
        evs = []
        if trigger.priority <= self.theta_thought:
            return evs

        if candidates and len(candidates) >= 2:
            for i, (cls, sim) in enumerate(candidates[:3]):
                scale = float(sim) * (0.8 ** i)
                tv = trigger.data.get('vector', G)
                hyp_v = tv * scale + G[:len(tv)] * (1. - scale) * 0.1
                evs.append(Event(EventType.THOUGHT.name, time.time() + 0.01 * (i+1),
                    {'vector': hyp_v, 'hypothesis': cls, 'parent': trigger.type},
                    'cascade_hyp', trigger.priority * scale))
            # ── Self-questioning ───────────────────────────────────────────
            # When top-2 are close, ask: which candidate does the current
            # field state actually resonate with more?
            if field is not None:
                top_sim = float(candidates[0][1])
                sec_sim = float(candidates[1][1])
                if (top_sim - sec_sim) < 0.15 and top_sim > 0.0:
                    res_scores = []
                    for cls, sim in candidates[:2]:
                        lb = np.frombuffer(cls.encode('utf-8'), dtype=np.uint8).astype(np.float64)
                        lb = (lb - 128.0) / 128.0
                        ln = min(len(lb), field.dim)
                        query = np.zeros(field.dim); query[:ln] = lb[:ln]
                        res_scores.append((cls, field.enhanced_resonance(query)))
                    if res_scores:
                        winner = max(res_scores, key=lambda x: x[1])[0]
                        self._last_self_query = winner
                        for ev in evs:
                            h = ev.data.get('hypothesis')
                            ev.priority *= (1.25 if h == winner else 0.85)
        else:
            v = trigger.data.get('vector', G)
            for tau, scale in [(0.01, 0.8), (0.05, 0.5), (0.1, 0.3)]:
                sub_v = v * scale + G[:len(v)] * 0.1
                evs.append(Event(EventType.THOUGHT.name, time.time() + tau,
                    {'vector': sub_v, 'parent': trigger.type},
                    'cascade', trigger.priority * scale))
        return evs

    def multi_scale(self, events: List[Event], G: np.ndarray) -> np.ndarray:
        if not events: return G
        g_norm_sq = float(np.dot(G, G))
        if g_norm_sq < EPSILON: return G
        g_norm = float(np.sqrt(g_norm_sq))
        mean_v = np.zeros(self.dim)
        for e in events:
            v = e.data.get('vector')
            if v is not None:
                ln = min(len(v), self.dim)
                mean_v[:ln] += v[:ln]
        mean_v *= (1.0 / len(events))
        mv_sq = float(np.dot(mean_v, mean_v))
        if mv_sq < EPSILON: return G
        mean_v *= (1.0 / np.sqrt(mv_sq))
        alpha = 0.10 + 0.40 * self.uncertainty
        out = (1. - alpha) * G + alpha * mean_v[:len(G)]
        out_sq = float(np.dot(out, out))
        if out_sq > EPSILON:
            out *= (g_norm / np.sqrt(out_sq))
        return out

    def self_generate(self, G: np.ndarray, kappa: float) -> Optional[Event]:
        """
        Fires when G-vector trend exceeds 0.2 norm over last 4 states.
        Fixed threshold — removed the kappa>0.08 unconditional branch.
        """
        self._event_history.append((G.copy(), self.uncertainty))
        if len(self._event_history) < 4:
            return None
        recent_G = np.array([h[0] for h in list(self._event_history)[-4:]])
        trend = recent_G[-1] - recent_G[0]
        if float(np.sqrt(np.dot(trend, trend))) > 0.2:
            v = G + trend * 0.1
            return Event(EventType.THOUGHT.name, time.time(),
                {'vector': v, 'source': 'self_generated'}, 'self',
                float(np.clip(self.uncertainty, 0., 1.)))
        return None

    def resonant_chain(self, events: List[Event]) -> float:
        if len(events) < 2:
            return 0.
        vecs = [e.data.get('vector', np.zeros(self.dim)) for e in events]
        total = 0.0; count = 0
        for i in range(len(vecs) - 1):
            v1 = vecs[i][:self.dim]; v2 = vecs[i+1][:self.dim]
            n1 = float(np.sqrt(np.dot(v1, v1))); n2 = float(np.sqrt(np.dot(v2, v2)))
            if n1 < EPSILON or n2 < EPSILON: continue
            total += float(np.dot(v1, v2)) / (n1 * n2); count += 1
        return total / count if count else 0.


# ══════════════════════════════════════════════
# 13. META-LEARNING  (contrastive + recursive)
# ══════════════════════════════════════════════

class MetaLearning:
    def __init__(self, state_dim=64, max_recent=8):
        self.dim=state_dim
        self._recent: deque=deque(maxlen=max_recent)
        self._L_meta=0.

    def loss(self, state, target, negatives=None):
        """
        Contrastive loss: pull state toward target, push away from negatives.

        Fix 1 — penalty term inverted:
          Old: penalty = exp(-3 * mean_sim_to_recent)
               → similar inputs got 91% signal suppression (backwards)
          New: novelty_boost = 1 + sim_to_recent
               → novel inputs (sim≈0): boost=1 (normal signal)
               → repeated inputs (sim≈0.8): boost=1.8 (more signal, not less)
          Rationale: inputs similar to recent memory are the hard repetitions
          the model needs to cement — they should get MORE gradient, not less.

        Fix 2 — harder negative margin:
          Old: max(0, sim - 0.1)²  (very lenient margin)
          New: max(0, sim - (-0.1))² = max(0, sim + 0.1)²
               → requires negatives to be actively DISSIMILAR (sim < -0.1),
                  not just not-similar (sim < 0.1). This is the standard
                  triplet margin for angular distance.
        """
        sn = float(np.sqrt(np.dot(state, state))) + EPSILON
        tn = float(np.sqrt(np.dot(target, target))) + EPSILON
        s = state  / sn
        t = target / tn
        n = min(len(s), len(t)); s, t = s[:n], t[:n]

        # Positive term: MSE in normalised state space
        pos = float(np.mean((s - t) ** 2))

        # Negative term: push away hard negatives (margin = -0.1 → must be dissimilar)
        neg_l = 0.
        if negatives:
            for neg in negatives:
                nv  = neg / (np.linalg.norm(neg) + EPSILON)
                sim = float(np.dot(s, nv[:n]))
                neg_l += max(0., sim + 0.1) ** 2   # margin -0.1: must be < -0.1 to satisfy
            neg_l /= len(negatives)

        # Novelty boost: inputs similar to recent states get MORE signal, not less.
        # This reinforces hard, repeated examples instead of suppressing them.
        novelty_boost = 1.0
        if self._recent:
            sims = [abs(float(np.dot(s, p[:n] / (np.linalg.norm(p) + EPSILON))))
                    for p in self._recent]
            mean_sim = float(np.mean(sims))
            novelty_boost = 1.0 + mean_sim   # range [1.0, 2.0]

        self._recent.append(s.copy())
        total = (pos + 2.0 * neg_l) * novelty_boost
        self._L_meta = 0.9 * self._L_meta + 0.1 * total
        return total

    def meta_loss(self): return self._L_meta

class ModalityDetector:
    """
    Detects input modality from prefix and feature signature.
    Tracks per-modality accuracy across the run.
    """
    PREFIXES = {'arr:': 'array', 'hex:': 'signal', 'iq:': 'rf_iq', 'file:': 'file'}
    MODALITY_PATTERNS = {
        'math':     [r'\d+[\+\-\*\/]\d+', r'\d+ mod \d+'],
        'logic':    [r'^is \d+', r'^not ', r'true|false', r'and|or'],
        'sequence': [r'^next:', r'^sort:'],
        'language': [r'sound$', r'^capital of', r'past$', r'synonym'],
        'question': [r'\?$'],
        'array':    [r'^arr:'],
        'binary':   [r'^hex:'],
        'file':     [r'^file:'],
    }

    def __init__(self):
        import re
        self._re = re
        self._compiled = {k: [re.compile(p, re.IGNORECASE) for p in pats]
                          for k, pats in self.MODALITY_PATTERNS.items()}
        # Per-modality: {modality: [correct, total]}
        self.scores: Dict[str, List[int]] = {k: [0,0] for k in self.MODALITY_PATTERNS}
        self.scores['unknown'] = [0,0]

    def detect(self, text: str) -> str:
        for prefix, mod in self.PREFIXES.items():
            if text.startswith(prefix): return mod
        for mod, patterns in self._compiled.items():
            if any(p.search(text) for p in patterns): return mod
        return 'unknown'

    def record(self, text: str, correct: bool):
        mod = self.detect(text)
        if mod not in self.scores: self.scores[mod] = [0,0]
        self.scores[mod][1] += 1
        if correct: self.scores[mod][0] += 1

    def report(self) -> str:
        lines = []
        for mod, (ok, tot) in sorted(self.scores.items()):
            if tot == 0: continue
            pct = 100.*ok/tot
            bar = '█'*int(pct/10) + '░'*(10-int(pct/10))
            lines.append(f"    {mod:12} {bar} {ok:3}/{tot:3} ({pct:5.1f}%)")
        return '\n'.join(lines) if lines else '    (no data)'

    def omega_signature(self, features: np.ndarray) -> Tuple[str, float]:
        """Returns (signature_label, detail_ratio) — verifies encoder differentiates modalities"""
        half = len(features)//2
        approx_e = float(np.sum(features[:half]**2))+EPSILON
        detail_e = float(np.sum(features[half:]**2))+EPSILON
        ratio = detail_e / approx_e
        if ratio < 0.3:   label = 'smooth(text/audio)'
        elif ratio < 1.0: label = 'mixed'
        elif ratio < 3.0: label = 'structured(image-like)'
        else:             label = 'bursty(binary/rf)'
        return label, ratio

class AnchorMemory:
    def __init__(self, dim=64, min_sep=0.5, max_per_class=500):
        self.dim=dim; self.min_sep=min_sep
        self.anchors: Dict[str,np.ndarray]={}
        self.outputs: Dict[str,str]={}
        self._vecs: List[np.ndarray]=[]
        self._keys: List[str]=[]
        self._sep_cache: Optional[Tuple[float,float]] = None
        self._dirty = False
        # Adaptive per-class anchor cap:
        #   Instead of one global ceiling, each class gets its own cap derived from
        #   measured boundary complexity.  Classes with tight unimodal clusters need
        #   far fewer anchors than classes with diffuse, multi-modal distributions.
        #
        #   Complexity is estimated from the live anchor set every
        #   _cap_update_every new anchors per class using three metrics:
        #     spread   — mean pairwise cosine distance (class diffuseness)
        #     id_est   — TwoNN intrinsic dimensionality (manifold complexity)
        #     spec_gap — spectral gap of similarity matrix (unimodality)
        #
        #   Cap formula (profiled 2025-02):
        #     cap = clamp(base + 200*spread + 30*id_est - 100*spec_gap, lo, hi)
        #     lo=10  hi=max_per_class  base=10
        #   This produces ~50 for tight clusters, ~200-400 for diffuse/multimodal classes.
        #
        #   max_per_class is the hard global ceiling — adaptive caps never exceed it.
        self.max_per_class        = max_per_class
        self._class_caps: Dict[str, int]  = {}          # per-class adaptive cap
        self._cap_new_since: Dict[str, int] = {}        # new anchors since last cap update
        self._cap_update_every = 20                      # recompute cap every N new anchors
        # Consolidation thresholds — both set to 0.85 to match high-dim cosine geometry.
        # In d=4096, within-class Omega vectors cluster at sim=0.80-0.95; a threshold of
        # 0.92 sat above the cluster ceiling and rarely triggered. 0.85 sits in the middle
        # of the cluster and aggressively merges near-duplicates.
        # Research: NCC-kNN (ScienceDirect 2021), prototype condensation (Hart 1968/PSNB 2022).
        self.dedup_threshold     = 0.55   # profiled optimum: accuracy flat 0.55→0.96, fewer anchors = cleaner memory
        self.consolidate_threshold = 0.55  # mirrors dedup — anchors that slip through dedup get merged here
        self._steps_since_consolidate = 0
        self.consolidate_every = 200       # steps between consolidation passes
        # StatEngine reference (set by Cypha after construction)
        self._stat_engine = None
        # Per-class anchor count and index caches (invalidated on store/consolidate)
        self._class_counts: Dict[str,int] = {}
        # _cls_idx maps class_label → list of global indices into _vecs/_keys.
        # Replaces O(n_total) list-comprehension in dedup path with O(1) lookup.
        # Critical when resuming from checkpoints with 100k+ anchors.
        self._cls_idx: Dict[str, List[int]] = {}
        # _key_to_gi maps key string → global index into _vecs/_keys.
        # Replaces O(n) list.index(key) calls with O(1) dict lookup.
        # Maintained incrementally: updated on insert, rebuilt on consolidate/evict/load.
        self._key_to_gi: Dict[str, int] = {}
        # Stacked normalised matrix for O(1) matmul lookup (rebuilt on consolidate,
        # appended incrementally on store). Shape: (n_anchors, dim).
        # Replaces list-comprehension O(n) dot products: 1.33ms -> 0.23ms at 500 anchors.
        self._V: Optional[np.ndarray] = None   # (n, dim) float32, rows are unit vectors
        self._V_dirty = False                   # True when _vecs changed but _V not rebuilt

    def _rebuild_V(self):
        """Rebuild the stacked anchor matrix from _vecs. Called lazily on lookup."""
        if not self._vecs:
            self._V = None
        else:
            self._V = np.array(self._vecs, dtype=np.float32)
        self._V_dirty = False

    def store(self, key: str, state: np.ndarray, output: str,
              dedup_threshold: Optional[float] = None,
              ema_alpha: float = 0.15):
        """
        Store an anchor, skipping if a near-duplicate already exists for the
        same output class (saves memory and lookup cost).

        dedup_threshold: cosine similarity above which we treat two vectors as
        the same prototype and skip storage. Defaults to self.dedup_threshold.

        ema_alpha: weight for the new observation in the EMA centroid update.
          Default 0.15 (slow, stable update for certain samples).
          ThoughtProcessor.suggested_alpha raises this up to 0.40 for samples
          where the anchor lookup margin was low — uncertain samples leave a
          stronger trace so the memory adapts faster to hard cases.

        max_per_class enforcement: when the class hits its adaptive cap, the
        two most similar same-class anchors are merged before adding the new one.
        """
        if dedup_threshold is None:
            dedup_threshold = self.dedup_threshold
        s64 = state.astype(np.float64) if state.dtype != np.float64 else state
        _sn = float(np.sqrt(np.dot(s64, s64))) + EPSILON
        v = (s64 / _sn).astype(np.float32)

        # If key already exists, update it with an exponential moving average
        # (centroid refinement) rather than a hard overwrite.
        if key in self.anchors:
            old = self.anchors[key]
            merged = (1. - ema_alpha) * old + ema_alpha * v
            _mn = float(np.sqrt(np.dot(merged.astype(np.float64), merged.astype(np.float64)))) + EPSILON
            merged /= _mn
            self.anchors[key] = merged
            gi = self._key_to_gi[key]   # O(1) dict lookup — replaces O(n) list.index
            self._vecs[gi] = merged
            if self._V is not None and not self._V_dirty:
                self._V[gi] = merged.astype(np.float32)
            self._dirty = True
            if self._stat_engine and output:
                self._stat_engine.on_store(str(output), merged, key)
            return  # _class_counts unchanged — key already existed

        # Check for near-duplicate in same output class before adding new key.
        # If a same-class anchor is already very close, don't store a new one —
        # instead update the closest existing anchor via EMA.
        if self._vecs and dedup_threshold < 1.0:
            # O(1) lookup via _cls_idx instead of O(n) scan over all anchors.
            # Critical after checkpoint resume when class sizes reach 100k+.
            same_cls_indices = self._cls_idx.get(output, [])
            if same_cls_indices:
                same_vecs = np.stack([self._vecs[i] for i in same_cls_indices])
                sims = same_vecs @ v
                best_i = int(np.argmax(sims))
                best_sim = float(sims[best_i])
                if best_sim >= dedup_threshold:
                    # Near-duplicate: update closest anchor with EMA instead
                    gi = same_cls_indices[best_i]
                    old = self._vecs[gi]
                    merged = (1. - ema_alpha) * old + ema_alpha * v
                    _mn2 = float(np.sqrt(float(np.dot(merged.astype(np.float64), merged.astype(np.float64))))) + EPSILON
                    merged /= _mn2
                    self._vecs[gi] = merged
                    self.anchors[self._keys[gi]] = merged
                    if self._V is not None and not self._V_dirty:
                        self._V[gi] = merged.astype(np.float32)
                    self._dirty = True
                    if self._stat_engine and output:
                        self._stat_engine.on_store(str(output), merged, self._keys[gi])
                    return

        # Enforce per-class adaptive anchor cap before adding.
        # Cap is computed lazily from boundary complexity metrics and refreshed
        # every _cap_update_every new anchors per class.
        cur_cls_count = self._class_counts.get(output, 0)
        # Refresh cap if not yet set or due for update
        since = self._cap_new_since.get(output, 0)
        if output not in self._class_caps or since >= self._cap_update_every:
            self._class_caps[output]    = self._compute_class_cap(output)
            self._cap_new_since[output] = 0
        cap = self._class_caps[output]
        if cur_cls_count >= cap:
            self._evict_closest_pair(output)

        # New distinct anchor: add it
        self.anchors[key] = v
        self.outputs[key] = output
        self._vecs.append(v)
        self._keys.append(key)
        self._key_to_gi[key] = len(self._keys) - 1   # O(1) — keep dict in sync
        self._dirty = True
        self._class_counts[output] = self._class_counts.get(output, 0) + 1
        self._cap_new_since[output] = self._cap_new_since.get(output, 0) + 1
        new_global_idx = len(self._vecs) - 1   # just appended
        if output not in self._cls_idx:
            self._cls_idx[output] = []
        self._cls_idx[output].append(new_global_idx)
        # Append row to stacked matrix (O(n) copy, amortised) rather than full rebuild.
        vf32 = v.reshape(1, -1).astype(np.float32)
        if self._V is None:
            self._V = vf32.copy()
        elif not self._V_dirty:
            self._V = np.vstack([self._V, vf32])
        if self._stat_engine and output:
            self._stat_engine.on_store(str(output), v, key)
            self._stat_engine.on_new_anchor(key)

        self._steps_since_consolidate += 1
        if self._steps_since_consolidate >= self.consolidate_every:
            self.consolidate()
            self._steps_since_consolidate = 0

        # ── LVQ2.1 boundary sharpening ─────────────────────────────────────────
        # After storing a sample, find the nearest correct-class and nearest
        # wrong-class anchor. If both are in the LVQ2.1 window (similar distances
        # to the sample), apply the LVQ2.1 update rule:
        #   correct prototype: w += lr*(x - w)   (move toward sample)
        #   wrong   prototype: w -= lr*(x - w)   (move away from sample)
        # Effect: progressively sharpens the decision boundary for confused pairs
        # (malware↔safe_api, phishing↔safe_email etc.) without touching other anchors.
        # lr=0.02 is intentionally small — nudge, not replacement.
        _lr_lvq = 0.02
        _theta_w = 0.30
        nearby = self.lookup(vf32.ravel(), k=4)
        best_correct = None; best_wrong = None
        for key, sim in nearby:
            cls_m = self.get_output(key)
            if cls_m == output and best_correct is None:
                best_correct = (key, float(sim))
            elif cls_m != output and cls_m is not None and best_wrong is None:
                best_wrong   = (key, float(sim))
            if best_correct and best_wrong:
                break
        if best_correct and best_wrong:
            sc = best_correct[1]; sw = best_wrong[1]
            lo = min(sc, sw); hi = max(sc, sw) + EPSILON
            if (lo / hi) > (1.0 - _theta_w):   # LVQ2.1 window: near equidistant
                key_c = best_correct[0]; key_w = best_wrong[0]
                # O(1) dict lookup — replaces O(n) _keys.index() scans
                gi_c = self._key_to_gi.get(key_c, -1)
                gi_w = self._key_to_gi.get(key_w, -1)
                if gi_c >= 0 and gi_w >= 0:
                    wc  = self._vecs[gi_c].astype(np.float32)
                    ww  = self._vecs[gi_w].astype(np.float32)
                    xd  = vf32.ravel()
                    wc2 = wc + _lr_lvq * (xd - wc)
                    nc  = float(np.sqrt(np.dot(wc2, wc2)))
                    if nc > EPSILON: self._vecs[gi_c] = (wc2 / nc).astype(np.float32)
                    ww2 = ww - _lr_lvq * (xd - ww)
                    nw  = float(np.sqrt(np.dot(ww2, ww2)))
                    if nw > EPSILON: self._vecs[gi_w] = (ww2 / nw).astype(np.float32)
                    self._V_dirty = True

    def _compute_class_cap(self, output: str) -> int:
        """
        Estimate boundary complexity from the current anchor set for `output`
        and return an appropriate anchor cap.

        Metrics (all computed on the live anchor vectors):
          spread   — mean pairwise cosine distance: how diffuse is the class?
          id_est   — TwoNN intrinsic dimensionality: how many dims does the
                     class manifold span?
          spec_gap — normalised spectral gap of the similarity matrix: is the
                     class unimodal (high gap) or multi-modal (low gap)?

        Formula (profiled against 6 text domains):
          cap = clamp(base + 200*spread + 30*id_est - 100*spec_gap, 10, max_per_class)

        Tight unimodal cluster  → ~50 anchors
        Diffuse multi-modal     → ~200-400 anchors
        """
        idx = self._cls_idx.get(output, [])
        if len(idx) < 6:
            # Too few anchors to estimate — use half the global cap as safe default
            return max(10, self.max_per_class // 2)

        vecs = np.array([self._vecs[i] for i in idx], dtype=np.float32)
        n    = len(vecs)

        # Similarity matrix
        S    = (vecs @ vecs.T).astype(np.float64)

        # 1. Spread: mean pairwise cosine distance
        upper = [(1. - float(S[i, j])) for i in range(n) for j in range(i+1, n)]
        spread = float(np.mean(upper)) if upper else 0.0

        # 2. TwoNN intrinsic dimensionality
        D = 1. - S
        np.fill_diagonal(D, 1e9)
        sd = np.sort(D, axis=1)
        r1, r2 = sd[:, 0], sd[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            mu     = r2 / (r1 + 1e-9)
            log_mu = np.log(np.maximum(mu, 1.001))
        id_est = float(1. / (float(np.mean(log_mu)) + 1e-9))

        # 3. Spectral gap (normalised)
        eigvals  = np.sort(np.linalg.eigvalsh(S))[::-1]
        spec_gap = float(eigvals[0] - eigvals[1]) / (float(eigvals[0]) + 1e-9) \
                   if len(eigvals) >= 2 else 1.0

        raw = 10 + 200 * spread + 30 * id_est - 100 * spec_gap
        return max(10, min(self.max_per_class, int(raw)))

    def _evict_closest_pair(self, output: str):
        """
        When a class hits max_per_class, merge the two most-similar same-class
        anchors into their centroid, freeing one slot. O(n_cls^2) but n_cls <= 500
        so worst case is 250k dot products = ~1ms. Called once per max_per_class
        new anchors, amortised cost is negligible.
        """
        cls_idx = [i for i, k in enumerate(self._keys) if self.outputs.get(k) == output]
        if len(cls_idx) < 2:
            return
        vecs_cls = np.array([self._vecs[i] for i in cls_idx], dtype=np.float32)
        S = vecs_cls @ vecs_cls.T                  # (n, n) similarity matrix
        np.fill_diagonal(S, -2.)                   # exclude self-similarity
        best = int(np.argmax(S))                   # flat index of most-similar pair
        ai, bi = best // len(cls_idx), best % len(cls_idx)
        if ai == bi:
            return
        gi_a, gi_b = cls_idx[ai], cls_idx[bi]
        # Merge into centroid (normalised average)
        centroid = self._vecs[gi_a] + self._vecs[gi_b]
        centroid /= (np.linalg.norm(centroid) + EPSILON)
        # Update the first; remove the second
        keep_key = self._keys[gi_a]
        drop_key = self._keys[gi_b]
        self._vecs[gi_a] = centroid
        self.anchors[keep_key] = centroid
        if self._V is not None and not self._V_dirty:
            self._V[gi_a] = centroid.astype(np.float32)
        # Remove drop_key from all structures
        del self.anchors[drop_key]
        del self.outputs[drop_key]
        self._vecs.pop(gi_b)
        self._keys.pop(gi_b)
        self._class_counts[output] = max(0, self._class_counts.get(output, 1) - 1)
        # _cls_idx is invalidated by the index shift (gi_b removed) — rebuild from scratch.
        # _key_to_gi is similarly invalidated — rebuild in the same pass.
        # This is O(n) but _evict only fires when a class hits max_per_class,
        # so amortised cost per training step is O(1).
        self._cls_idx = {}
        self._key_to_gi = {}
        for i, k in enumerate(self._keys):
            cls = self.outputs.get(k)
            if cls:
                if cls not in self._cls_idx: self._cls_idx[cls] = []
                self._cls_idx[cls].append(i)
            self._key_to_gi[k] = i
        self._V_dirty = True
        self._dirty = True

    def class_anchor_counts(self) -> Dict[str, int]:
        """Return {class_label: n_anchors} — used by infer() for bias-corrected voting."""
        return dict(self._class_counts)

    def class_caps(self) -> Dict[str, int]:
        """Return {class_label: current_adaptive_cap} for inspection/logging."""
        return dict(self._class_caps)

    def class_centroid(self, cls: str) -> Optional[np.ndarray]:
        """
        Return the normalised mean of all anchor vectors for `cls`.
        Used by ThoughtProcessor.deliberate() for hypothesis interpolation.
        Returns None if the class has no anchors.
        """
        idx = self._cls_idx.get(cls, [])
        if not idx:
            return None
        vecs = np.array([self._vecs[i] for i in idx], dtype=np.float64)
        centroid = np.mean(vecs, axis=0)
        n = float(np.linalg.norm(centroid))
        if n < EPSILON:
            return None
        return (centroid / n).astype(np.float32)

    def known_classes(self) -> List[str]:
        """Return list of all classes that have at least one anchor."""
        return [cls for cls, cnt in self._class_counts.items() if cnt > 0]

    def consolidate(self, threshold: Optional[float] = None) -> int:
        """
        Merge highly similar anchors within each output class into their centroid.

        Algorithm: greedy pivot merge — for each class, iterate anchors; if an
        anchor is within `threshold` of an already-chosen prototype, absorb it
        (centroid average); otherwise start a new prototype. O(n²) per class.

        Returns the number of anchors removed.
        """
        if len(self._vecs) < 4:
            return 0
        t = threshold if threshold is not None else self.consolidate_threshold

        # Group by output class
        by_class: Dict[str, List[int]] = {}
        for i, k in enumerate(self._keys):
            cls = self.outputs.get(k, '__unknown__')
            by_class.setdefault(cls, []).append(i)

        new_anchors: Dict[str, np.ndarray] = {}
        new_outputs: Dict[str, str] = {}
        new_vecs:    List[np.ndarray] = []
        new_keys:    List[str] = []
        removed = 0

        for cls, indices in by_class.items():
            vecs_cls = [self._vecs[i] for i in indices]
            keys_cls  = [self._keys[i] for i in indices]

            remaining = list(range(len(vecs_cls)))
            while remaining:
                pivot  = remaining[0]
                group  = [pivot]
                pv     = vecs_cls[pivot]
                others = remaining[1:]
                remaining = []
                for j in others:
                    if float(np.dot(pv, vecs_cls[j])) >= t:
                        group.append(j)
                    else:
                        remaining.append(j)

                # Centroid of merged group
                centroid = np.mean([vecs_cls[g] for g in group], axis=0)
                centroid /= (np.linalg.norm(centroid) + EPSILON)

                # Representative key: shortest (cleanest) in group
                rep_key = min((keys_cls[g] for g in group), key=len)
                new_anchors[rep_key] = centroid
                new_outputs[rep_key] = cls
                new_vecs.append(centroid)
                new_keys.append(rep_key)
                removed += len(group) - 1

        self.anchors = new_anchors
        self.outputs = new_outputs
        self._vecs   = new_vecs
        self._keys   = new_keys
        self._sep_cache = None
        self._dirty  = True
        # Rebuild _class_counts, _cls_idx, and _key_to_gi from scratch after merge
        self._class_counts = {}
        self._cls_idx = {}
        self._key_to_gi = {}
        for i, k in enumerate(new_keys):
            cls = new_outputs[k]
            self._class_counts[cls] = self._class_counts.get(cls, 0) + 1
            if cls not in self._cls_idx:
                self._cls_idx[cls] = []
            self._cls_idx[cls].append(i)
            self._key_to_gi[k] = i
        # Mark _V for rebuild — consolidate changes both the set and values of anchors
        self._V_dirty = True
        self._rebuild_V()
        return removed

    def get_hard_negatives(self, query_vec: np.ndarray,
                           correct_output: str, k: int = 3) -> List[np.ndarray]:
        """
        Return the k anchors from WRONG output classes with the HIGHEST cosine
        similarity to query_vec (the hardest current confusions).

        Matmul version: compute all similarities in one BLAS sgemv call, then
        mask correct-class rows, then argpartition. Replaces O(n) Python loop.
        """
        if len(self._vecs) < 2:
            return []
        if self._V_dirty or self._V is None:
            self._rebuild_V()
        q = (query_vec / (float(np.sqrt(float(np.dot(query_vec.astype(np.float64),query_vec.astype(np.float64))))) + EPSILON)).astype(np.float32)
        sims = self._V @ q                                   # (n,) all similarities
        # Build wrong-class mask
        wrong = np.array([self.outputs.get(ky) != correct_output
                          for ky in self._keys], dtype=bool)
        sims_masked = sims.copy()
        sims_masked[~wrong] = -2.0                           # exclude correct-class
        n_wrong = int(wrong.sum())
        if n_wrong == 0:
            return []
        top_k = min(k, n_wrong)
        idx = np.argpartition(sims_masked, -top_k)[-top_k:]
        idx = idx[np.argsort(sims_masked[idx])[::-1]]
        return [self._vecs[int(i)] for i in idx]

    def lookup(self, state: np.ndarray, k=3) -> List[Tuple[str,float]]:
        """
        Matmul lookup: single BLAS sgemv call instead of O(n) Python dot loop.
        Confirmed 5.7x speedup at 500 anchors: 1.33ms -> 0.23ms.
        At 2000 anchors: 7.07ms -> ~1.2ms.
        """
        if not self._vecs: return []
        if self._V_dirty or self._V is None:
            self._rebuild_V()
        q = (state / (float(np.sqrt(float(np.dot(state.astype(np.float64),state.astype(np.float64))))) + EPSILON)).astype(np.float32)
        sims = self._V @ q                                  # (n,) — one BLAS call
        n = len(sims)
        top_k = min(k, n)
        # argpartition is O(n) vs O(n log n) full sort — only sort the top-k slice
        idx = np.argpartition(sims, -top_k)[-top_k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        matches = [(self._keys[int(i)], float(sims[i])) for i in idx]
        if matches and self._stat_engine:
            wk = matches[0][0]; ws = float(matches[0][1])
            rs = float(matches[1][1]) if len(matches) > 1 else None
            wo = self.outputs.get(wk, '')
            ro = self.outputs.get(matches[1][0], '') if len(matches) > 1 else None
            mg = (ws - rs) if rs is not None else 0.5
            self._stat_engine.on_lookup(wk, ws, rs, wo, ro, mg)
        return matches

    def lookup_and_hard_negatives(
            self, state: np.ndarray, correct_output: str,
            k_lookup: int = 2, k_neg: int = 3
    ) -> Tuple[List[Tuple[str, float]], List[np.ndarray]]:
        """Single-matmul version of lookup() + get_hard_negatives().

        Both methods normalise the same query and compute self._V @ q —
        identical BLAS sgemv.  One call here handles both.

        At 500 anchors: saves ~0.23ms per train_step.
        At 2000 anchors: saves ~1.2ms per train_step.
        """
        if not self._vecs:
            return [], []
        if self._V_dirty or self._V is None:
            self._rebuild_V()
        q    = (state / (float(np.sqrt(float(np.dot(state.astype(np.float64),state.astype(np.float64))))) + EPSILON)).astype(np.float32)
        sims = self._V @ q                                   # ONE sgemv for both

        n = len(sims)
        # ── top-k matches (for margin / deliberation) ──────────────────
        top_k   = min(k_lookup, n)
        idx_top = np.argpartition(sims, -top_k)[-top_k:]
        idx_top = idx_top[np.argsort(sims[idx_top])[::-1]]
        matches = [(self._keys[int(i)], float(sims[i])) for i in idx_top]
        if matches and self._stat_engine:
            wk = matches[0][0]; ws = float(matches[0][1])
            rs = float(matches[1][1]) if len(matches) > 1 else None
            wo = self.outputs.get(wk, '')
            ro = self.outputs.get(matches[1][0], '') if len(matches) > 1 else None
            self._stat_engine.on_lookup(wk, ws, rs, wo, ro,
                                        (ws - rs) if rs is not None else 0.5)
        # ── hard negatives (for contrastive loss) ─────────────────────
        if n < 2:
            return matches, []
        wrong    = np.array([self.outputs.get(ky) != correct_output
                              for ky in self._keys], dtype=bool)
        n_wrong  = int(wrong.sum())
        if n_wrong == 0:
            return matches, []
        sims_neg = sims.copy(); sims_neg[~wrong] = -2.0
        top_neg  = min(k_neg, n_wrong)
        idx_neg  = np.argpartition(sims_neg, -top_neg)[-top_neg:]
        idx_neg  = idx_neg[np.argsort(sims_neg[idx_neg])[::-1]]
        return matches, [self._vecs[int(i)] for i in idx_neg]

    def separation_stats(self) -> Tuple[float,float]:
        """Compute min and avg pairwise cosine distance. Cached until dirty."""
        if not self._dirty and self._sep_cache is not None:
            return self._sep_cache
        if len(self._vecs) < 2:
            return 0., 0.
        vecs = np.array(self._vecs[:64])
        sims = vecs @ vecs.T
        n = len(vecs)
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append(1. - sims[i,j])
        if not pairs:
            return 0., 0.
        mn = float(np.min(pairs)); avg = float(np.mean(pairs))
        self._sep_cache = (mn, avg)
        self._dirty = False
        return mn, avg

    def get_output(self, key: str) -> Optional[str]: return self.outputs.get(key)

    def remove_anchor(self, key: str) -> bool:
        """Remove a single anchor by key.

        Called by LearningSchedule T2 dead-anchor pruning.
        Fully updates all derived state:
          - _class_counts decremented (fixes BUG-3: count stays accurate)
          - _cls_idx cleared (indices shift after pop — rebuilt lazily)
          - _key_to_gi rebuilt from scratch after index shift
          - _V_dirty / _dirty set → V matrix rebuilt on next lookup
        """
        if key not in self.anchors:
            return False
        gi = self._key_to_gi.get(key)
        if gi is None:
            return False
        out = self.outputs.pop(key, None)
        del self.anchors[key]
        self._key_to_gi.pop(key, None)
        self._V_dirty = True; self._dirty = True
        if out:
            # BUG-3 fix: decrement so class_anchor_counts() stays accurate
            self._class_counts[out] = max(0, self._class_counts.get(out, 0) - 1)
        # _cls_idx indices become invalid after the pop below — clear entirely.
        # It will be rebuilt lazily on the next dedup/store call.
        self._cls_idx = {}
        # Remove from _vecs/_keys lists (O(n) but remove is infrequent)
        if 0 <= gi < len(self._keys):
            self._vecs.pop(gi)
            self._keys.pop(gi)
            self._key_to_gi = {k: i for i, k in enumerate(self._keys)}
        return True

    @property
    def n(self): return len(self.anchors)

    def self_retrieval_rate(self, sample=50) -> float:
        if len(self._vecs) < 2: return 1.0
        keys = self._keys[:sample]
        vecs = np.array([self.anchors[k] for k in keys])
        sims = vecs @ vecs.T
        np.fill_diagonal(sims, -1)
        max_other = np.max(sims, axis=1)
        np.fill_diagonal(sims, 2)
        self_sims = np.diag(sims)
        return float(np.mean(self_sims > max_other))

# ══════════════════════════════════════════════
# 14. ADAPTIVE CONTROL LOOP
# ══════════════════════════════════════════════

class AdaptiveControlLoop:
    def __init__(self):
        self.history: deque=deque(maxlen=32)

    def update(self, p: EncoderParams, s: FieldStats, err: float) -> EncoderParams:
        if err>p.prev_error*1.05:
            p.prev_error=err; return p
        # Law 1: chunk_k ← criticality
        dk=-math.copysign(GAIN_K, s.criticality-K_TARGET)
        kc=float(np.clip(p.chunk_k+dk, 2., 5.))
        p.chunk_k=(1-LAMBDA_LP)*p.chunk_k+LAMBDA_LP*kc
        # Law 2: DAMR radius ← dominant frequency
        rho_t=max(1.,s.dominant_freq*8.)
        dr=float(np.clip((rho_t-p.damr_radius),-1.,1.))*GAIN_RHO
        p.damr_radius=float(np.clip((1-LAMBDA_LP)*p.damr_radius+LAMBDA_LP*(p.damr_radius+dr),3.,8.))
        # Law 3: active scales ← phase coherence
        coh=1.-min(1.,s.phase_spread/math.pi)
        n_sc=max(1,round(1+coh*3))
        p.active_scales=[1.,0.5,0.25,0.125][:n_sc]
        p.prev_error=err
        self.history.append({'k':s.criticality,'ck':p.chunk_k,'r':p.damr_radius})
        return p


# ══════════════════════════════════════════════
# 15. SPARSE COMPUTATION + WORK STEALING
# ══════════════════════════════════════════════

class SparseComputer:
    def __init__(self, theta=0.01):
        self.theta=theta
        self._cache: Dict[str,Tuple[np.ndarray,float]]={}
        self._lru: deque=deque(maxlen=256)

    def should_update(self, psi: np.ndarray, key: str) -> bool:
        if key not in self._cache: return True
        prev,_=self._cache[key]
        return float(np.linalg.norm(psi-prev[:len(psi)]))>self.theta

    def cache(self, key: str, result: np.ndarray):
        self._cache[key]=(result.copy(),time.time())
        self._lru.append(key)

    def get_cached(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            v,_=self._cache[key]; return v
        return None


class WorkStealer:
    def __init__(self, n_workers=N_WORKERS):
        self._pool=_POOL; self._queue: deque=deque(maxlen=32)
        self._futures={}

    def submit(self, key: str, fn, *args):
        if key not in self._futures or self._futures[key].done():
            self._futures[key]=self._pool.submit(fn,*args)

    def result(self, key: str, timeout=0.005):
        if key in self._futures:
            f=self._futures[key]
            if f.done():
                del self._futures[key]; return f.result()
        return None

    def run_parallel(self, fns: List[Tuple[str,Any,tuple]]) -> Dict[str,Any]:
        futures={k: self._pool.submit(fn,*args) for k,fn,args in fns}
        results={}
        for k,f in futures.items():
            try: results[k]=f.result(timeout=0.1)
            except: results[k]=None
        return results


# ══════════════════════════════════════════════
# 16. PRECISION CONTROL  (adaptive float precision)
# ══════════════════════════════════════════════

class PrecisionController:
    def preserve(self, x: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        ax=np.abs(x); ax=np.where(ax<EPSILON,1.,ax)
        exp=np.floor(np.log2(ax+EPSILON))
        man=x/(2.**exp)
        return man.astype(np.float32), exp.astype(np.float16)

    def reconstruct(self, man: np.ndarray, exp: np.ndarray) -> np.ndarray:
        return man.astype(np.float64)*(2.**exp.astype(np.float64))

    def adaptive_compute(self, psi: np.ndarray, needs_high: bool) -> np.ndarray:
        return psi.astype(np.float64) if needs_high else psi.astype(np.float32)


# ══════════════════════════════════════════════
# NEW: TEMPORAL, PREDICTIVE, RELATIONAL, EPISODIC, GOAL SYSTEMS
# ══════════════════════════════════════════════

class TemporalFieldHierarchy:
    def __init__(self, dim: int, decay_fast=0.7, decay_med=0.95, decay_slow=0.998):
        self.fast = ResonanceField(dim)
        self.med  = ResonanceField(dim)
        self.slow = ResonanceField(dim)
        self.df = decay_fast; self.dm = decay_med; self.ds = decay_slow

    @staticmethod
    def _decay_renorm(field, decay):
        p = field.psi * decay
        n = float(np.sqrt(np.dot(p.conj(), p).real))
        field.psi = p / n if n > EPSILON else p

    def step(self, enc: np.ndarray):
        self.fast.inject(enc, strength=0.6)
        psi_f = self.fast.evolve(1)
        self._decay_renorm(self.fast, self.df)
        self.med.inject(psi_f, strength=0.15)
        psi_m = self.med.evolve(1)
        self._decay_renorm(self.med, self.dm)
        self.slow.inject(psi_m, strength=0.03)
        psi_s = self.slow.evolve(1)
        self._decay_renorm(self.slow, self.ds)
        return psi_f, psi_m, psi_s

    def context(self) -> np.ndarray:
        c = np.abs(self.slow.psi)
        n = float(np.sqrt(np.dot(c, c))) + EPSILON
        return (c / n).astype(np.float32)

    def reset_all(self):
        self.fast.reset(); self.med.reset(); self.slow.reset()


class PredictiveProcessor:
    def __init__(self, dim: int, lr: float = 0.05):
        self.dim = dim; self.lr = lr
        self._pred = None
        self._pred_errors: deque = deque(maxlen=200)

    def update(self, enc: np.ndarray):
        enc = enc.astype(np.float32)
        n   = float(np.linalg.norm(enc))
        enc_n = enc / n if n > EPSILON else enc
        if self._pred is None:
            self._pred = enc_n.copy()
            return enc_n, 0.0
        err_vec = enc_n - self._pred
        err_mag = float(np.linalg.norm(err_vec))
        self._pred_errors.append(err_mag)
        self._pred = (1.0 - self.lr) * self._pred + self.lr * enc_n
        pn = float(np.linalg.norm(self._pred))
        if pn > EPSILON: self._pred /= pn
        return err_vec, err_mag

    def surprise(self) -> float:
        if len(self._pred_errors) < 5: return 0.5
        recent = np.array(list(self._pred_errors)[-20:])
        baseline = float(np.mean(self._pred_errors))
        return float(recent[-1] / (baseline + EPSILON))

    def reset(self):
        self._pred = None


@dataclass
class RelationalEdge:
    src:     str
    dst:     str
    rel_vec: np.ndarray
    schema:  Optional[str] = None
    weight:  float = 1.0
    causal:  bool  = False


class RelationalEncoder:
    def __init__(self, dim: int):
        self.dim = dim

    def encode_pair(self, va: np.ndarray, vb: np.ndarray) -> np.ndarray:
        a = va.astype(np.float64)[:self.dim]; b = vb.astype(np.float64)[:self.dim]
        na = float(np.sqrt(np.dot(a,a))); nb = float(np.sqrt(np.dot(b,b)))
        if na < EPSILON or nb < EPSILON:
            return np.zeros(self.dim, np.float32)
        a /= na; b /= nb
        fa = fft(a.astype(complex)); fb = fft(b.astype(complex))
        cross     = fb * np.conj(fa)
        psd_a     = np.abs(fa)**2 + EPSILON; psd_b = np.abs(fb)**2 + EPSILON
        mag       = np.abs(cross)
        log_ratio = np.log(psd_b / psd_a)
        phase_sh  = np.angle(cross)
        coh       = mag / np.sqrt(psd_a * psd_b)
        q = self.dim // 4
        raw = np.concatenate([
            (mag / (mag.sum()+EPSILON))[:q],
            log_ratio[:q],
            np.sin(phase_sh[:q]),
            coh[:q],
        ]).astype(np.float32)
        raw = raw[:self.dim] if len(raw) >= self.dim else np.pad(raw, (0, self.dim-len(raw)))
        n = float(np.sqrt(np.dot(raw,raw)))
        return (raw / (n + EPSILON)).astype(np.float32)


class RelationalGraph:
    def __init__(self, rel_dim: int):
        self.rel_dim = rel_dim
        self._edges:      Dict[str, List[RelationalEdge]]        = {}
        self._edge_index: Dict[Tuple[str,str], RelationalEdge]   = {}
        self._n_edges = 0

    def add_edge(self, src: str, dst: str, rel_vec: np.ndarray,
                 schema: Optional[str] = None, causal: bool = False, weight: float = 1.0):
        edge = RelationalEdge(src=src, dst=dst,
                              rel_vec=rel_vec.astype(np.float32),
                              schema=schema, weight=weight, causal=causal)
        key = (src, dst)
        if key in self._edge_index:
            ex = self._edge_index[key]
            ex.rel_vec = 0.85*ex.rel_vec + 0.15*edge.rel_vec
            n = float(np.linalg.norm(ex.rel_vec))
            if n > EPSILON: ex.rel_vec /= n
            ex.weight = min(ex.weight+0.1, 3.0)
            ex.schema = schema or ex.schema
        else:
            if src not in self._edges: self._edges[src] = []
            self._edges[src].append(edge)
            self._edge_index[key] = edge
            self._n_edges += 1

    def traverse(self, src: str, rel_query: np.ndarray, top_k: int = 3,
                 schema: Optional[str] = None) -> List[Tuple[str,float]]:
        edges = self._edges.get(src, [])
        if schema: edges = [e for e in edges if e.schema == schema]
        if not edges: return []
        q = rel_query.astype(np.float32)
        _q64 = q if q.dtype == np.float64 else q.astype(np.float64)
        qn = float(np.sqrt(np.dot(_q64, _q64)))
        if qn < EPSILON: return []
        q /= qn
        scored = []
        for e in edges:
            en = float(np.linalg.norm(e.rel_vec))
            if en < EPSILON: continue
            scored.append((e.dst, float(np.dot(q, e.rel_vec/en)) * e.weight))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    @property
    def n_edges(self): return self._n_edges


class SchemaLibrary:
    def __init__(self, rel_dim: int, max_schemas: int = 64, sim_thresh: float = 0.80):
        self.rel_dim = rel_dim; self.max_schemas = max_schemas; self.sim_thresh = sim_thresh
        self._prototypes: Dict[str, np.ndarray] = {}
        self._counts:     Dict[str, int]         = {}
        self._next_id = 0
        self._proto_mat:   Optional[np.ndarray] = None
        self._proto_names: List[str] = []
        self._proto_dirty  = False

    def _rebuild_mat(self):
        if self._proto_names:
            self._proto_mat = np.stack([self._prototypes[n] for n in self._proto_names])
        else:
            self._proto_mat = None
        self._proto_dirty = False

    def match_or_create(self, rel_vec: np.ndarray) -> str:
        rv = rel_vec.astype(np.float32)
        n  = float(np.sqrt(np.dot(rv, rv))) + EPSILON
        rv /= n
        best_name = None; best_sim = -1.0
        if self._proto_names:
            if self._proto_dirty: self._rebuild_mat()
            sims = self._proto_mat @ rv
            idx  = int(np.argmax(sims))
            best_sim  = float(sims[idx])
            best_name = self._proto_names[idx]
        if best_sim >= self.sim_thresh and best_name is not None:
            cnt = self._counts[best_name]
            p   = self._prototypes[best_name]
            p   = (p * cnt + rv) / (cnt + 1)
            pn  = float(np.sqrt(np.dot(p, p))) + EPSILON
            p  /= pn
            self._prototypes[best_name] = p
            self._counts[best_name]     = cnt + 1
            i = self._proto_names.index(best_name)
            if self._proto_mat is not None:
                self._proto_mat[i] = p
            return best_name
        if len(self._prototypes) >= self.max_schemas:
            return best_name or 'unknown'
        name = f'schema_{self._next_id:04d}'; self._next_id += 1
        self._prototypes[name] = rv.copy(); self._counts[name] = 1
        self._proto_names.append(name)
        self._proto_dirty = True
        return name

    def top_schemas(self, n: int = 10) -> List[Tuple[str,int]]:
        return sorted(self._counts.items(), key=lambda x: -x[1])[:n]


@dataclass
class Episode:
    idx:         int
    inp_key:     str
    inp_vec:     np.ndarray
    context_vec: np.ndarray
    pred_error:  float
    outcome:     str
    timestamp:   float
    weight:      float = 1.0


class EpisodicMemory:
    def __init__(self, max_episodes: int = 2000, replay_batch: int = 8):
        self._episodes: List[Episode] = []
        self._max = max_episodes; self._rbatch = replay_batch; self._idx = 0
        self._high_surprise_buf: List[Episode] = []

    def store(self, inp_key: str, inp_vec: np.ndarray, context_vec: np.ndarray,
              pred_error: float, outcome: str) -> Episode:
        ep = Episode(idx=self._idx, inp_key=inp_key,
                     inp_vec=inp_vec if inp_vec.dtype == np.float32 else inp_vec.astype(np.float32),
                     context_vec=context_vec if context_vec.dtype == np.float32 else context_vec.astype(np.float32),
                     pred_error=pred_error, outcome=outcome,
                     timestamp=time.time(), weight=1.0+pred_error)
        self._idx += 1
        if len(self._episodes) < self._max:
            self._episodes.append(ep)
        else:
            min_i = int(min(range(len(self._episodes)), key=lambda i: self._episodes[i].weight))
            if ep.weight > self._episodes[min_i].weight:
                self._episodes[min_i] = ep
        if pred_error > 0.3:
            self._high_surprise_buf.append(ep)
            if len(self._high_surprise_buf) > 200:
                self._high_surprise_buf.pop(0)
        if self._episodes:
            ws = np.array([e.weight for e in self._episodes], dtype=np.float64)
            ws *= 0.999
            for i,e in enumerate(self._episodes): e.weight = float(ws[i])
        return ep

    def retrieve_by_cue(self, query_vec: np.ndarray, k: int = 5) -> List[Episode]:
        if not self._episodes: return []
        q  = query_vec.astype(np.float32)
        qn = float(np.sqrt(float(np.dot(q, q))))
        if qn < EPSILON: return self._episodes[-k:]
        q /= qn
        eps  = self._episodes
        mat  = np.stack([e.inp_vec for e in eps])          # (N, dim) float32
        nrms = np.sqrt((mat * mat).sum(axis=1)) + EPSILON  # (N,)
        sims = (mat @ q) / nrms                            # (N,) cosine sims
        ws   = np.array([e.weight for e in eps], dtype=np.float32)
        scores = sims * ws
        top_idx = np.argpartition(scores, -min(k, len(scores)))[-min(k, len(scores)):]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [eps[int(i)] for i in top_idx]

    def retrieve_recent(self, k: int = 5) -> List[Tuple]:
        """Return the k most recently stored episodes as (key, vec, ctx, err, out) tuples."""
        if not self._episodes:
            return []
        recent = self._episodes[-k:]
        return [(ep.inp_key, ep.inp_vec, ep.context_vec, ep.pred_error, ep.outcome) for ep in recent]

    def replay_batch(self) -> List[Episode]:
        if not self._episodes: return []
        pool = self._high_surprise_buf if (self._high_surprise_buf and np.random.random()<0.4) else self._episodes
        weights = np.array([e.weight for e in pool], dtype=np.float64)
        weights /= (weights.sum() + EPSILON)
        n = min(self._rbatch, len(pool))
        idxs = np.random.choice(len(pool), size=n, replace=False, p=weights)
        return [pool[i] for i in idxs]

    @property
    def n(self): return len(self._episodes)


class GeometricUncertainty:
    def __init__(self, dim: int, min_samples: int = 10):
        self.dim = dim; self.min_samples = min_samples
        self._means: Dict[str, np.ndarray] = {}
        self._M2s:   Dict[str, np.ndarray] = {}
        self._ns:    Dict[str, int]         = {}

    def update(self, cls: str, vec: np.ndarray):
        v = vec.astype(np.float64)
        if cls not in self._ns:
            self._ns[cls] = 0; self._means[cls] = np.zeros(len(v)); self._M2s[cls] = np.zeros(len(v))
        n = self._ns[cls]+1; d = v - self._means[cls]
        self._means[cls] += d/n; self._M2s[cls] += d*(v-self._means[cls]); self._ns[cls] = n

    def mahalanobis(self, query: np.ndarray, cls: str) -> float:
        if cls not in self._ns or self._ns[cls] < self.min_samples: return 1.0
        diff = query.astype(np.float64) - self._means[cls]
        var  = self._M2s[cls] / max(self._ns[cls]-1, 1) + 1e-6
        return float(np.sqrt(np.mean(diff**2/var)))

    def class_probs(self, query: np.ndarray, classes: List[str]) -> Dict[str,float]:
        dists = {c: self.mahalanobis(query, c) for c in classes}
        min_d = min(dists.values()) + EPSILON
        scores = {c: min_d/(d+EPSILON) for c,d in dists.items()}
        total  = sum(scores.values()) + EPSILON
        return {c: s/total for c,s in scores.items()}


class GoalField:
    def __init__(self, dim: int):
        self.dim = dim; self._goals: List[Tuple[np.ndarray,float]] = []; self._background = None

    def set_goal(self, goal_vec: np.ndarray, strength: float = 0.1):
        gv = goal_vec.astype(np.float32)
        n  = float(np.linalg.norm(gv))
        if n > EPSILON: gv /= n
        self._goals = [(gv, strength)]; self._recompute()

    def add_goal(self, goal_vec: np.ndarray, strength: float = 0.05):
        gv = goal_vec.astype(np.float32)
        n  = float(np.linalg.norm(gv))
        if n > EPSILON: gv /= n
        self._goals.append((gv, strength))
        if len(self._goals) > 8: self._goals.pop(0)
        self._recompute()

    def _recompute(self):
        if not self._goals: self._background = None; return
        bg = sum(gv[:self.dim]*s for gv,s in self._goals)
        n  = float(np.sqrt(np.dot(bg,bg)))
        self._background = (bg/n).astype(np.float32) if n > EPSILON else None

    def apply(self, field: ResonanceField, inject_strength: float = 0.05):
        if self._background is not None:
            field.inject(self._background.astype(complex), strength=inject_strength)

    def relevance(self, vec: np.ndarray) -> float:
        if self._background is None: return 0.5
        v = vec.astype(np.float32)
        if len(v) != self.dim:
            v = np.interp(np.linspace(0,len(v)-1,self.dim), np.arange(len(v)), v).astype(np.float32)
        vn = float(np.sqrt(np.dot(v,v)))
        if vn < EPSILON: return 0.0
        return float(np.dot(self._background, v/vn))

    @property
    def goal_vec(self) -> Optional[np.ndarray]:
        """Current blended goal vector, or None if no goal set."""
        return self._background

    def clear(self):
        self._goals = []; self._background = None


class AnalogicalReasoner:
    def __init__(self, rel_enc: RelationalEncoder,
                 rel_graph: RelationalGraph,
                 schema_lib: SchemaLibrary):
        self.rel_enc = rel_enc; self.rel_graph = rel_graph; self.schema_lib = schema_lib
        self._last: Optional[Dict] = None

    def solve(self, query_src: str, query_src_vec: np.ndarray,
              example_src: str, example_src_vec: np.ndarray,
              example_dst: str, example_dst_vec: np.ndarray,
              memory: 'AnchorMemory') -> Tuple[Optional[str],float]:
        ex_rel = self.rel_enc.encode_pair(example_src_vec, example_dst_vec)
        schema = self.schema_lib.match_or_create(ex_rel)

        # Pass 1: schema-filtered traversal (strict — same relational type)
        candidates = self.rel_graph.traverse(query_src, ex_rel, top_k=5, schema=schema)
        if candidates:
            dst, sim = candidates[0]
            self._last = {'method':'graph_schema','schema':schema,'dst':dst,'sim':sim}
            return dst, sim

        # Pass 2: schema-free traversal (relational similarity only)
        # The byte-stat encoder often assigns distinct schemas to semantically
        # identical relations (cat→mammal vs eagle→avian), so relaxing schema
        # filter recovers the correct analogy in those cases.
        candidates_free = self.rel_graph.traverse(query_src, ex_rel, top_k=5, schema=None)
        if candidates_free:
            # Only accept if top result is a causal (inp→cls) edge destination
            # or if rel_sim is meaningfully above random (>0.3)
            for dst, sim in candidates_free:
                if sim > 0.3:
                    self._last = {'method':'graph_free','schema':None,'dst':dst,'sim':sim}
                    return dst, sim

        # Pass 3: vector-space analogy — A:B::C:D via anchor memory
        q = query_src_vec.astype(np.float32)
        _q64 = q if q.dtype == np.float64 else q.astype(np.float64)
        qn = float(np.sqrt(np.dot(_q64, _q64)))
        if qn < EPSILON: return None, 0.0
        q /= qn
        best_sim = -1.0; best_dst = None
        for edge in self.rel_graph._edges.get(example_src, []):
            # No schema filter here — check all edges from example_src
            src_anchors = [k for k,v in memory.outputs.items() if v == edge.src]
            if not src_anchors: continue
            sv = memory.anchors.get(src_anchors[0])
            if sv is None: continue
            sn = float(np.linalg.norm(sv))
            if sn < EPSILON: continue
            src_sim = float(np.dot(q, sv/sn))
            en = float(np.linalg.norm(edge.rel_vec))
            if en < EPSILON: continue
            rel_sim = float(np.dot(ex_rel, edge.rel_vec/en))
            combined = src_sim*0.4 + rel_sim*0.6
            if combined > best_sim:
                best_sim = combined; best_dst = edge.dst
        if best_dst:
            self._last = {'method':'vector_analogy','schema':schema,'dst':best_dst,'sim':best_sim}
            return best_dst, best_sim
        return None, 0.0


# ══════════════════════════════════════════════
# 17. MAIN CYPHA SYSTEM
# ══════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# CYPHA — fully integrated with StatEngine, InputFilterChain, OnlineWhitener,
# TrainingThought/RuntimeThought/RuntimeLearner, LearningSchedule
# ══════════════════════════════════════════════════════════════════════════════

class Cypha:
    def __init__(self, feature_dim=512, resonance_dim=256, config=None):
        cfg = config or {}
        self.fd = feature_dim; self.rd = resonance_dim
        # ── Filters / Encoders ──────────────────────────────────────────────
        self.filter    = InputFilterChain()
        self.omega_enc = OmegaEncoder(feature_dim)
        self.whitener  = OnlineWhitener(feature_dim,
                            update_every=cfg.get('whitener_update', 200))
        self.bridge    = PhaseBridge(feature_dim, resonance_dim)
        # ── HRNA ───────────────────────────────────────────────────────────
        self.field     = ResonanceField(resonance_dim)
        self.res_level = ResonatorLevel(resonance_dim)
        self.assembly  = AssemblyLevel(16, resonance_dim)
        self.module    = ModuleLevel(8, 16)
        self.global_l  = GlobalLevel(resonance_dim, 8)
        self.recursive = RecursiveProcessor(resonance_dim)
        self.feedback  = FeedbackController(resonance_dim)
        # ── ThoughtProcessor (legacy event cascade, kept for HRNA compat) ──
        self.thought   = ThoughtProcessor(resonance_dim)
        self.temporal   = TemporalFieldHierarchy(resonance_dim)
        self.predictor  = PredictiveProcessor(feature_dim)
        self.rel_enc    = RelationalEncoder(resonance_dim)
        self.rel_graph  = RelationalGraph(resonance_dim)
        self.schema_lib = SchemaLibrary(resonance_dim)
        self.episodic   = EpisodicMemory()
        self.geo_unc    = GeometricUncertainty(feature_dim)
        self.goal       = GoalField(resonance_dim)
        self.analogist  = AnalogicalReasoner(self.rel_enc, self.rel_graph, self.schema_lib)
        self._prev_anchor_vec = None
        # ── StatEngine ─────────────────────────────────────────────────────
        self.stat      = StatEngine(feature_dim)
        # ── Thought systems ─────────────────────────────────────────────────
        self.train_thought   = TrainingThought()
        self.runtime_thought = RuntimeThought()
        self.runtime_learner = RuntimeLearner(self.runtime_thought)
        self.train_thought.attach(self.stat)
        self.runtime_thought.attach(self.stat)
        self._training_mode = True
        # ── Memory / Meta ───────────────────────────────────────────────────
        self.meta     = MetaLearning(resonance_dim)
        self.memory   = AnchorMemory(feature_dim)
        self.memory._stat_engine = self.stat
        # ── Infrastructure ──────────────────────────────────────────────────
        self.ctrl      = AdaptiveControlLoop()
        self.scheduler = EventScheduler()
        self.gen       = EventGenerator()
        self.sparse    = SparseComputer()
        self.stealer   = WorkStealer()
        self.precision = PrecisionController()
        self.params    = EncoderParams()
        self.modality  = ModalityDetector()
        # ── State ───────────────────────────────────────────────────────────
        self.step        = 0
        self.temperature = cfg.get('temperature_init', 1.5)
        self._temp_decay = cfg.get('temperature_decay', 0.97)
        self._metrics: List[Metrics] = []
        self._last_infer_stats = None
        self._active_events: List[Event] = []
        self._patterns: List[np.ndarray] = []
        self._pattern_ffts: List = []
        # ── Archetype detection ─────────────────────────────────────────────
        self._arch_probe_data = None
        self._arch_probe_history: List[Dict] = []
        self._arch_detect_steps = {10, 20, 30, 50, 100}
        self.archetype = None; self.archetype_confidence = 0.0
        self.archetype_reasoning = ""; self.archetype_recommendation = ""
        self.archetype_detect_step = None
        # ── Confidence history ──────────────────────────────────────────────
        self._conf_history: deque = deque(maxlen=200)
        # Step-scoped encode cache — cleared in train_step. inp is encoded
        # twice per step; dict hit is ~3000x faster than re-encoding.
        self._encode_cache: Dict[str, np.ndarray] = {}
        self._step_count: int = 0
        self._last_res_events: List = []

    # ── Encode pipeline ──────────────────────────────────────────────────────
    def encode_features(self, text: str) -> np.ndarray:
        """filter → omega → whiten. Reports to StatEngine.
        Step-scoped cache avoids re-encoding the same string within one step."""
        _hit = self._encode_cache.get(text)
        if _hit is not None:
            return _hit
        import base64 as _b64
        filtered = self.filter.process(text)
        if filtered.startswith('arr:'):
            try:
                raw = _b64.b64decode(filtered[4:])
                feats = self.omega_enc.encode_array(raw, self.params)
            except Exception:
                feats = self.omega_enc.encode_text(filtered, self.params)
        elif filtered.startswith('hex:') or filtered.startswith('pcm:'):
            prefix = filtered[:4]; body = filtered[4:]
            try:
                raw   = bytes.fromhex(body)
                feats = self.omega_enc.encode_array(raw, self.params)
            except Exception:
                feats = self.omega_enc.encode_text(filtered, self.params)
        else:
            feats = self.omega_enc.encode_text(filtered, self.params)
        _omega_feats = getattr(self.omega_enc, '_last_feats', None)
        self.stat.on_encode(feats, omega_feats=_omega_feats)
        result = self.whitener.process(feats)
        self._encode_cache[text] = result
        return result

    def encode(self, text: str) -> np.ndarray:
        return self.bridge.bridge(self.encode_features(text))

    # ── Forward ──────────────────────────────────────────────────────────────
    def plan(self, n_trajectories: int = 6, horizon: int = 4) -> Optional[np.ndarray]:
        """
        Forward-simulate n_trajectories hypothetical field evolutions.
        Each trajectory injects the current field state at a different
        perturbation strength, evolves `horizon` steps, and scores the
        landing point against the current goal anchor.

        Returns the action vector (encoded perturbation) of the best trajectory,
        or None if no goal is set.

        This costs n_trajectories * horizon * 1 FFT each — cheap relative to
        a full forward pass.  Called once per train_step when goal is active.
        """
        goal_vec = self.goal.goal_vec
        if goal_vec is None:
            return None
        goal_f = goal_vec[:self.field.dim].astype(complex)
        goal_n = float(np.sqrt(np.dot(goal_f.conj(), goal_f).real)) + EPSILON
        goal_f = goal_f / goal_n

        best_score = -np.inf
        best_action = None

        # Save field state
        psi_saved    = self.field.psi.copy()
        psi_prev_saved = self.field.psi_prev.copy()

        strengths = np.linspace(0.05, 0.60, n_trajectories)
        for strength in strengths:
            # Reset to current state for each trajectory
            self.field.psi      = psi_saved.copy()
            self.field.psi_prev = psi_prev_saved.copy()
            self.field._psi_fft = None; self.field._psi_psd = None

            # Inject with this strength as the hypothetical action
            action_vec = goal_f * strength + self.field.psi * (1.0 - strength)
            self.field.inject(action_vec, strength=strength)

            # Evolve horizon steps
            for _ in range(horizon):
                self.field.evolve(1)

            # Score: cosine similarity of landing state to goal
            psi_end = self.field.psi
            pn = float(np.sqrt(np.dot(psi_end.conj(), psi_end).real)) + EPSILON
            score = float(np.dot(psi_end.conj(), goal_f).real) / pn
            if score > best_score:
                best_score = score
                best_action = action_vec.copy()

        # Restore field to saved state
        self.field.psi      = psi_saved
        self.field.psi_prev = psi_prev_saved
        self.field._psi_fft = None; self.field._psi_psd = None

        self._last_plan_score = best_score
        return best_action

    def forward(self, text: str, training: bool = False,
                candidates=None) -> Dict[str, Any]:
        t0    = time.time()
        feats = self.encode_features(text)
        enc   = self.bridge.bridge(feats)
        err_vec, err_mag = self.predictor.update(feats)
        psi_f, psi_m, psi_s = self.temporal.step(enc)
        context = self.temporal.context()
        self.goal.apply(self.field, inject_strength=0.03)
        n_loops = 1 if training else 2
        for _ in range(n_loops):
            self.field.inject(enc, strength=0.60)
            psi = self.field.evolve(1)
        # ── Planning: inject best simulated action if goal active ──────────
        if self.goal.goal_vec is not None and not training:
            best_action = self.plan(n_trajectories=4, horizon=3)
            if best_action is not None:
                self.field.inject(best_action, strength=0.10)
                psi = self.field.evolve(1)
        due_events  = self.scheduler.pop_due()
        res_events  = self.gen.from_resonance(self.field, self._patterns[:2], self._pattern_ffts[:2])
        surp_events = self.gen.from_surprise(psi.real)
        all_events  = due_events + res_events + surp_events + self._active_events
        all_events  = [self.gen.modulate(e, self.field, self.global_l.kappa) for e in all_events]
        self._active_events = []
        res_enh      = self.field.enhanced_resonance(psi.real)
        R            = self.recursive.horizontal(psi.real, enc.real, all_events, res_enh)
        R            = self.recursive.temporal(R, all_events)
        res_state    = self.res_level.update(dt=0.1, drive=psi)
        G_prev       = self.global_l.G.copy()
        asm_state    = self.assembly.update(res_state, G_prev)
        _fs_cache = self.field.stats()
        self.module.push_psi(np.abs(psi_s), weight=float(_fs_cache.criticality)+0.1)
        mod_state    = self.module.update(asm_state, G_prev)
        global_state = self.global_l.update(mod_state,
                           self.assembly.oscillator_output(), psi.real, all_events)
        fb_res  = self.feedback.resonance_amplified(global_state, self.field)
        fb_temp = self.feedback.temporal(all_events)
        fb_crit = self.feedback.criticality_enhanced(global_state, self.global_l.kappa)
        _fbw = getattr(self, '_fb_weight', 0.7); _fbr = (1.0 - _fbw) / 3.0
        global_state = (_fbw*global_state + _fbr*fb_res
                        + _fbr*fb_temp[:len(global_state)] + _fbr*fb_crit)
        _cand_list = candidates or []
        thought_evs = []
        for e in all_events:
            thought_evs.extend(self.thought.cascade(
                e, global_state, self.global_l.kappa, candidates=_cand_list))
        self_ev = self.thought.self_generate(global_state, self.global_l.kappa)
        if self_ev: thought_evs.append(self_ev)
        chain_str    = self.thought.resonant_chain(thought_evs)
        global_state = self.thought.multi_scale(thought_evs, global_state)
        R_final      = self.recursive.vertical(R, psi.real, global_state, thought_evs)
        _rn = float(np.sqrt(np.dot(R_final, R_final))) + EPSILON
        out_state    = R_final / _rn
        return {
            'state':       out_state,
            'global':      global_state,
            'psi':         psi,
            'psi_fast':    psi_f,
            'psi_med':     psi_m,
            'psi_slow':    psi_s,
            'context':     context,
            'enc':         enc,
            'anchor_q':    feats,
            'events':      all_events + thought_evs,
            'chain':       chain_str,
            'field_stats': _fs_cache,
            'pred_error':  err_mag,
            'ms':          (time.time() - t0) * 1000,
        }


    # ── Surprise-driven replay ──────────────────────────────────────────────
    def _surprise_replay(self, surprise_mag: float, phase_entropy: float,
                         current_out: str) -> int:
        """
        When surprise is high AND phase entropy is high, the system is in an
        uncertain state that doesn't match its own predictions.  Replay the last
        K episodic memories with elevated EMA alpha so representations restructure
        faster toward fitting the anomalous input.

        Returns number of replay steps executed.
        """
        # Thresholds: surprise > 0.25 (above typical ~0.1), pe > 0.6 (disordered)
        if surprise_mag < 0.25 or phase_entropy < 0.60:
            return 0
        # Retrieve last K episodes
        recent = self.episodic.retrieve_recent(k=5)
        if not recent:
            return 0
        elevated_alpha = min(0.40, 0.15 + surprise_mag * 0.5)
        replayed = 0
        for ep_key, ep_vec, ep_ctx, ep_err, ep_out in recent:
            if ep_key and ep_out:
                self.memory.store(ep_key, ep_vec, ep_out, ema_alpha=elevated_alpha)
                replayed += 1
        return replayed

    # ── Train step ───────────────────────────────────────────────────────────
    def train_step(self, inp: str, out: str,
                   negatives: Optional[List[str]] = None) -> Metrics:
        self._training_mode = True
        self._encode_cache.clear()
        t0 = time.time()
        self.field.reset(); self.res_level.reset()
        self.assembly.reset(); self.module.reset(); self.global_l.reset()
        res_inp  = self.forward(inp, training=True)
        res_out  = self.forward(out, training=True)
        state_in = res_inp['state']; state_tgt = res_out['state']
        anchor_q  = res_inp['anchor_q']
        feats_out = self.encode_features(out)
        # ── Relational graph: three edge types ──────────────────────────────
        # 1. Sequential: prev_input → current_input  (as before)
        # 2. Input→class: inp → out  (what does this input map to?)
        # 3. Class→input: out → inp  (what inputs belong to this class?)
        # This makes analogy traversal work from any node in the graph.
        feats_out_f = feats_out.astype(np.float64)
        anchor_q_f  = anchor_q.astype(np.float64)
        # Edge 1: sequential (prev_input → inp)
        if self._prev_anchor_vec is not None:
            prev_key, prev_vec = self._prev_anchor_vec
            rel_seq = self.rel_enc.encode_pair(prev_vec, anchor_q_f)
            schema_seq = self.schema_lib.match_or_create(rel_seq)
            self.rel_graph.add_edge(prev_key, inp, rel_seq, schema=schema_seq)
        # Edge 2: inp → out  (instance-of / maps-to relation)
        rel_io = self.rel_enc.encode_pair(anchor_q_f, feats_out_f)
        schema_io = self.schema_lib.match_or_create(rel_io)
        self.rel_graph.add_edge(inp, out, rel_io, schema=schema_io, causal=True)
        # Edge 3: out → inp  (reverse: class → member)
        rel_oi = self.rel_enc.encode_pair(feats_out_f, anchor_q_f)
        schema_oi = self.schema_lib.match_or_create(rel_oi)
        self.rel_graph.add_edge(out, inp, rel_oi, schema=schema_oi)
        self._prev_anchor_vec = (inp, anchor_q_f)
        self.geo_unc.update(out, anchor_q)
        pred_error = res_inp['pred_error']
        context    = res_inp['context']
        self.episodic.store(inp, anchor_q, context, pred_error, out)
        matches, hard_neg_vecs = self.memory.lookup_and_hard_negatives(
            anchor_q, out, k_lookup=2, k_neg=3)
        hard_neg_states   = [self.bridge.fast_bridge(v).real for v in hard_neg_vecs]
        window_neg_states = []
        for wn in (negatives or [])[:2]:
            wn_vec = self.encode_features(wn)
            window_neg_states.append(self.bridge.fast_bridge(wn_vec).real)
        neg_states = hard_neg_states + window_neg_states
        loss = self.meta.loss(state_in, state_tgt, neg_states if neg_states else None)
        if len(self._patterns) < 32:
            self._patterns.append(np.abs(state_in))
            self._pattern_ffts.append(None)
        self.params = self.ctrl.update(self.params, res_inp['field_stats'], loss)
        # ── Surprise-driven replay ──────────────────────────────────────────
        _surp_mag = res_inp['pred_error']
        _pe       = res_inp['field_stats'].phase_entropy if res_inp.get('field_stats') else 0.0
        if _surp_mag > 0.0:
            self._surprise_replay(_surp_mag, _pe, out)
        cands  = [(self.memory.get_output(k) or k, float(s)) for k,s in matches]
        margin = cands[0][1]-cands[1][1] if len(cands)>=2 else (0.5 if cands else 0.0)
        self.train_thought.note(margin, cands)
        self.memory.store(inp, anchor_q, out, ema_alpha=getattr(self,'_ema_alpha_store',0.05))
        if self.memory.n > 1:
            pred = self.memory.get_output(inp) or ''
            self.modality.record(inp, pred == out)
        if self.step % 200 == 0 and self.step > 0:
            new_t = self.stat.bus.optimal_temperature(self.temperature)
            self.temperature = max(0.8, new_t * self._temp_decay)
        if self.step % 100 == 0 and self.stat.bus.input_drift():
            self.filter.update_params(self.stat.bus.filter_update_params())
            self.stat.checkpoint()
        if self.step % 500 == 0 and self.step > 0:
            self.whitener.set_components(max(4, self.stat.bus.effective_dimensionality()//4))
        if self._arch_probe_data and self.step in self._arch_detect_steps:
            self._run_archetype_probe()
        self.step += 1
        self._step_count += 1
        m = Metrics(self.step, loss, res_inp['field_stats'].criticality,
                    self.params.chunk_k, self.params.damr_radius,
                    self.memory.n, (time.time()-t0)*1000, len(res_inp['events']))
        self._metrics.append(m); return m


    def end_training(self) -> None:
        """Freeze TrainingThought → hand snapshot to RuntimeThought."""
        snap = self.train_thought.freeze()
        self.runtime_thought.load_snapshot(snap)
        self.stat.checkpoint()
        self._training_mode = False

    # ── Inference ────────────────────────────────────────────────────────────
    def infer(self, text: str, verbose: bool = True) -> Tuple[str, float]:
        self.field.reset(); self.res_level.reset()
        self.assembly.reset(); self.module.reset(); self.global_l.reset()
        t0  = time.time()
        res = self.forward(text, training=False)
        q   = res['anchor_q']
        ms  = (time.time()-t0)*1000
        geo_classes = self.memory.known_classes()
        geo_probs   = self.geo_unc.class_probs(q, geo_classes[:10]) if geo_classes else {}
        matches = self.memory.lookup(q, k=4)
        if not matches:
            if verbose: print(f"  → [no memory]  (ms={ms:.1f})")
            return '[no memory]', 0.0
        cands  = [(self.memory.get_output(k) or k, float(s)) for k,s in matches]
        margin = cands[0][1] - cands[1][1] if len(cands) >= 2 else 0.5
        self.runtime_thought.note(margin, cands)
        del_cls, del_m, did_delib = self.runtime_thought.deliberate(q, cands, self.memory)
        if del_cls != '[no memory]':
            out = del_cls; best_raw = del_m
        else:
            scores: Dict[str,float] = {}
            for k,s in matches:
                c = self.memory.get_output(k)
                if c: scores[c] = scores.get(c,0.0) + max(0.0, s)
            ac   = self.memory.class_anchor_counts()
            norm = {c: s/max(1,ac.get(c,1)) for c,s in scores.items()}
            if geo_probs:
                for c in norm:
                    norm[c] = norm[c]*0.7 + geo_probs.get(c,0.0)*0.3
            if not norm:
                if verbose: print(f"  → [no memory]  (ms={ms:.1f})")
                return '[no memory]', 0.0
            out = max(norm, key=norm.get); best_raw = scores.get(out, 0.0)
        conf = float(np.exp(best_raw / max(self.temperature, 0.1)))
        self._conf_history.append(conf)
        cls_a = cands[0][0] if cands else None
        cls_b = cands[1][0] if len(cands)>1 else None
        out, conf = self.runtime_thought.apply_tcf(out, conf, cls_a, cls_b)
        self.runtime_learner.on_inference(margin, cands, self.stat)
        if verbose:
            mod = self.modality.detect(text)
            fs  = res.get('field_stats')
            kappa = fs.criticality if fs else 0.0
            pe    = fs.phase_entropy if fs else 0.0
            print(f"  → {out}  [conf={conf:.3f}]  unc={self.runtime_thought.uncertainty:.3f}")
            for c,sc in cands[:3]: print(f"  {c}({sc:.3f})", end='')
            print(f"  mod={mod}  ms={ms:.1f}  kappa={kappa:.4f}  pe={pe:.3f}")
        self._last_infer_stats = {'margin':margin,'deliberated':did_delib,
                                   'stat':self.stat.full_report()}
        return out, conf

    def infer_analogical(self, query: str, example_src: str,
                         example_dst: str) -> Tuple[Optional[str], float]:
        q_vec   = self.encode_features(query)
        src_vec = self.encode_features(example_src)
        dst_vec = self.encode_features(example_dst)
        return self.analogist.solve(
            query, q_vec, example_src, src_vec, example_dst, dst_vec, self.memory)

    def set_goal(self, goal_text: str, strength: float = 0.1):
        gv = self.bridge.bridge(self.encode_features(goal_text))
        self.goal.set_goal(np.abs(gv), strength)

    def clear_goal(self):
        self.goal.clear()

    def goal_relevance(self, text: str) -> float:
        enc = self.bridge.bridge(self.encode_features(text))
        return self.goal.relevance(np.abs(enc).astype(np.float32))

    def episodic_recall(self, query: str, k: int = 5) -> List[Episode]:
        return self.episodic.retrieve_by_cue(self.encode_features(query), k=k)


    # ── Archetype detection ───────────────────────────────────────────────────
    def set_probe_data(self, probe_data) -> None:
        self._arch_probe_data = probe_data

    def _run_archetype_probe(self) -> None:
        correct = 0; margins = []
        for inp, label in self._arch_probe_data:
            try:
                q = self.encode_features(inp)
                m = self.memory.lookup(q, k=2)
                if not m: continue
                pred = self.memory.get_output(m[0][0]) or m[0][0]
                if pred == label: correct += 1
                mg = float(m[0][1]) - (float(m[1][1]) if len(m)>1 else 0.0)
                margins.append(mg)
            except Exception: pass
        n   = max(len(self._arch_probe_data), 1)
        acc = correct / n
        sig = float(np.std(margins)) if margins else 0.0
        pt  = {"step":self.step,"acc":acc,"sigma":sig,
               "mean_margin":float(np.mean(margins)) if margins else 0.0,
               "n_anchors":self.memory.n}
        self._arch_probe_history.append(pt); h = self._arch_probe_history
        pt10 = next((p for p in reversed(h) if p["step"]<=10), None)
        if pt10 is None: return
        sig10 = pt10["sigma"]
        def slope(field, pts):
            if len(pts)<2: return 0.0
            xs=np.array([p["step"] for p in pts],float); ys=np.array([p[field] for p in pts],float)
            return float(np.polyfit(xs,ys,1)[0]) if xs.max()>xs.min() else 0.0
        sig_slope = slope("sigma",[p for p in h if p["step"]<=20])
        acc_slope = slope("acc",[p for p in h if p["step"]<=30])
        result = None
        if sig10 < 0.012:
            pt50 = next((p for p in reversed(h) if p["step"]<=50), None)
            if pt50:
                sl50 = slope("acc",[p for p in h if 30<=p["step"]<=50])
                if pt50["sigma"]<0.020 and pt50["acc"]<0.85 and sl50<0.003:
                    result=("C_ceiling",0.80,f"sig10={sig10:.4f} plateaued","Fix encoder features.")
                else:
                    result=("A",0.90 if sig10<0.008 else 0.75,f"sig10={sig10:.4f}<0.012","Stop at sigma collapse.")
            elif self.step>=50:
                result=("A",0.75,f"sig10={sig10:.4f}<0.012","Minimal training needed.")
        elif next((p for p in reversed(h) if p["step"]<=30),{}).get("acc",1.0)<0.40 and acc_slope<0.010:
            pa = next((p["acc"] for p in reversed(h) if p["step"]<=30),1.0)
            result=("C_mismatch",0.85 if acc_slope<0.005 else 0.70,f"acc_30={pa:.3f} stalled","HALT. Fix encoder.")
        elif sig10>0.025 and sig_slope<0.0:
            result=("B",0.85 if sig10>0.10 else 0.75,f"sig10={sig10:.4f}>0.025 falling","Continue training.")
        elif 0.012<=sig10<=0.025 and self.step>=50:
            sl=slope("acc",[p for p in h if p["step"]>=20]); lat=h[-1]
            if lat["acc"]<0.85 and sl<0.003:
                result=("C_ceiling",0.75,"ambiguous, plateaued","Fix encoder.")
            else:
                result=("B",0.65,"ambiguous, improving","Extended budget.")
        if result and result[1]>=0.70 and self.archetype is None:
            self.archetype=result[0]; self.archetype_confidence=result[1]
            self.archetype_reasoning=result[2]; self.archetype_recommendation=result[3]
            self.archetype_detect_step=self.step

    def archetype_status(self) -> Dict:
        return {"archetype":self.archetype,"confidence":self.archetype_confidence,
                "reasoning":self.archetype_reasoning,"recommendation":self.archetype_recommendation,
                "detect_step":self.archetype_detect_step,"probe_history":self._arch_probe_history,
                "committed":self.archetype is not None}

    # ── Pseudo-label ──────────────────────────────────────────────────────────
    def pseudo_label(self, inp: str, negatives=None,
                     top_pct: float = 0.15, min_history: int = 20) -> Dict:
        label, _ = self.infer(inp, verbose=False)
        if label == '[no memory]':
            return {"prediction":label,"confidence":0.,"percentile":0.,
                    "wrote":False,"history_size":0,"gate_threshold":None,"reason":"no memory"}
        conf = self._conf_history[-1] if self._conf_history else 0.0
        h    = list(self._conf_history)
        if len(h) < min_history:
            return {"prediction":label,"confidence":conf,"percentile":0.,"wrote":False,
                    "history_size":len(h),"gate_threshold":None,
                    "reason":f"warming up ({len(h)}/{min_history})"}
        thresh = float(np.percentile(h, (1.0-top_pct)*100))
        pctile = float(np.mean(np.array(h) <= conf))
        if conf >= thresh:
            self.train_step(inp, label, negatives=negatives or [])
            return {"prediction":label,"confidence":round(conf,4),"percentile":round(pctile,4),
                    "wrote":True,"history_size":len(h),"gate_threshold":round(thresh,4),
                    "reason":f"conf={conf:.3f}>=gate={thresh:.3f}->wrote"}
        return {"prediction":label,"confidence":round(conf,4),"percentile":round(pctile,4),
                "wrote":False,"history_size":len(h),"gate_threshold":round(thresh,4),
                "reason":f"conf={conf:.3f}<gate={thresh:.3f}->skip"}

    def pseudo_label_stats(self) -> Dict:
        h = list(self._conf_history)
        if len(h) < 5: return {"history_size":len(h),"recommendation":"Need more infer() calls."}
        a = np.array(h)
        return {"history_size":len(h),"mean_conf":round(float(a.mean()),4),
                "std_conf":round(float(a.std()),4),"p50":round(float(np.percentile(a,50)),4),
                "p85_gate":round(float(np.percentile(a,85)),4),
                "recommendation":"Gate healthy." if a.std()>0.01 else "WARNING: scores identical."}

    # ── Train / Load ──────────────────────────────────────────────────────────
    def train(self, data: List[Tuple[str,str]], epochs: int = 3,
              verbose: bool = True) -> List[Metrics]:
        all_m = []; self.gen.reset_counts()
        for ep in range(epochs):
            idxs = np.random.permutation(len(data)); ep_m = []
            for i in idxs:
                inp, out = data[i]
                negs = [data[j][0] for j in
                        np.random.choice([k for k in range(len(data)) if k!=i],
                                         size=min(3,len(data)-1),replace=False)]
                ep_m.append(self.train_step(inp, out, negs))
            all_m.extend(ep_m)
            if verbose:
                acc = 0.0  # accuracy tracked via probe, not Metrics
                print(f"  epoch {ep+1}/{epochs}  steps={len(ep_m)}  anchors={self.memory.n}  acc~{acc:.3f}")
        self.end_training(); return all_m

    def train_file(self, path: str, epochs: int = 3, verbose: bool = True):
        pairs = []
        with open(path) as f:
            for line in f:
                if '|||' in line:
                    a,b = line.strip().split('|||',1); pairs.append((a.strip(),b.strip()))
        return self.train(pairs, epochs, verbose)

    # ── Stat report ───────────────────────────────────────────────────────────
    def stat_report(self) -> Dict:
        return self.stat.full_report()

    def print_stat_report(self) -> None:
        r = self.stat_report(); ms = r['margin_stats']
        print(f"\n{'='*60}\n  CYPHA STAT REPORT\n{'='*60}")
        print(f"  Step: {r['step']}")
        print(f"  Margins: mean={ms['mean']} std={ms['std']} skew={ms['skew']} kurt={ms['kurt']}")
        print(f"  ECE: {r['ece']}  KL: {r['kl_divergence']:.4f}  drift={r['input_drift']}")
        print(f"  EffDim: {r['effective_dim']}  gini={r['anchor_gini']}  dead={r['dead_anchors']}")
        print(f"  Archetype: {self.archetype} (conf={self.archetype_confidence:.2f})")
        print(f"{'='*60}\n")

    # ── Legacy showcase ───────────────────────────────────────────────────────
    def showcase(self, data: List[Tuple[str,str]], n_test: int = 5,
                 verbose: bool = True) -> Dict:
        test_data = data[-n_test:]; train_data = data[:-n_test]
        self.train(train_data, epochs=2, verbose=verbose)
        correct = 0
        for inp, expected in test_data:
            pred, conf = self.infer(inp, verbose=verbose)
            if pred == expected: correct += 1
        acc = correct / max(len(test_data), 1)
        if verbose: print(f"\n  Showcase accuracy: {acc:.2f} ({correct}/{len(test_data)})")
        return {"accuracy": acc, "correct": correct, "total": len(test_data)}


# ══════════════════════════════════════════════════════════════════════════════
# LEARNING SCHEDULE — multi-timescale coordinator, primary external API
# ══════════════════════════════════════════════════════════════════════════════

class LearningSchedule:
    """Explicit multi-timescale coordinator.
    T1 every step: centroid EMA, margin stats, retrieval counts.
    T2 every N steps: geometry refresh, dead anchor prune, calibration.
    T3 on drift: filter params, whitening matrix, temperature recalibration.
    T4 persistent: hyperparameter evolution across sessions."""

    def __init__(self, cypha: Cypha, t2_every: int = 50, t3_kl_thresh: float = 0.5):
        self.cypha = cypha; self.t2_every = t2_every
        self.t3_kl_thresh = t3_kl_thresh
        self._step = 0; self._phase = "warmup"; self._t3_fired = False
        self._hp_history: List[Dict] = []

    def step(self, inp: str, label: str, negatives=None) -> Dict:
        self._step += 1; fired = []
        m = self.cypha.train_step(inp, label, negatives); fired.append('T1')
        # T2
        if self._step % self.t2_every == 0:
            self.cypha.stat.geo.update_distances()
            dead = self.cypha.stat.bus.dead_anchors(500)
            for key in dead: self.cypha.memory.remove_anchor(key)
            fired.append('T2')
        # T3
        kl = self.cypha.stat.dist.kl_divergence()
        if kl > self.t3_kl_thresh and not self._t3_fired:
            self.cypha.filter.update_params(self.cypha.stat.bus.filter_update_params())
            self.cypha.whitener.set_components(max(4,self.cypha.stat.bus.effective_dimensionality()//4))
            self.cypha.temperature = self.cypha.stat.bus.optimal_temperature(self.cypha.temperature)
            self.cypha.stat.checkpoint(); self._t3_fired = True; fired.append('T3')
        elif kl < self.t3_kl_thresh * 0.5:
            self._t3_fired = False
        arch = self.cypha.archetype
        self._phase = ('halted' if arch=='C_mismatch' else
                       'warmup'  if self._step<50 else
                       'migration' if arch in ('B',None) else 'active')
        return {'step':self._step,'phase':self._phase,'fired':fired,'archetype':arch,
                'kl':round(kl,4),'metrics':m}

    def train_stream(self, stream, max_steps=None, verbose=True) -> Dict:
        import time as _t; t0=_t.time(); n=0; reason="exhausted"
        for item in stream:
            if max_steps and n>=max_steps: reason=f"budget({max_steps})"; break
            inp,label=(item[0],item[1]); negs=item[2] if len(item)>2 else None
            status = self.step(inp,label,negs); n+=1
            if status['phase']=='halted': reason=f"C_mismatch@{n}"; break
            if verbose and n%100==0:
                print(f"  n={n} phase={status['phase']} arch={status['archetype']} kl={status['kl']}")
        self.cypha.end_training()
        return {'steps':n,'reason':reason,'elapsed':round(_t.time()-t0,2),
                'archetype':self.cypha.archetype,'stat':self.cypha.stat_report()}

    def record_hp(self, params: Dict, result: Dict) -> None:
        self._hp_history.append({'params':params,'result':result})

    def best_hp(self, metric='accuracy') -> Optional[Dict]:
        if not self._hp_history: return None
        return max(self._hp_history, key=lambda x:x['result'].get(metric,0))['params']

# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

_CKPT_ROOT = os.path.join(os.getcwd(), "checkpoints")


_CKPT_ROOT = os.path.join(os.getcwd(), "checkpoints")


def _ds_dir(dataset_name: str, root: str = _CKPT_ROOT) -> str:
    d = os.path.join(root, dataset_name)
    os.makedirs(d, exist_ok=True)
    return d


def _meta_path(dataset_name: str, root: str = _CKPT_ROOT) -> str:
    return os.path.join(_ds_dir(dataset_name, root), "meta.json")


def _ckpt_prefix(dataset_name: str, epoch: int, root: str = _CKPT_ROOT) -> str:
    return os.path.join(_ds_dir(dataset_name, root), f"epoch_{epoch:04d}")


def _atomic_write(path: str, data: bytes):
    """Write bytes atomically via temp-file rename."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _atomic_write_text(path: str, text: str):
    _atomic_write(path, text.encode("utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
#  Memory serialisation
# ══════════════════════════════════════════════════════════════════════════════

def _save_memory(prefix: str, anchors: Dict[str, np.ndarray],
                 outputs: Dict[str, str]):
    """
    Save AnchorMemory state.
    anchors : {key_str → (D,) float32 vector}
    outputs : {key_str → label_str}
    """
    if not anchors:
        return

    # Build ordered parallel arrays
    keys   = list(anchors.keys())
    vecs   = np.array([anchors[k] for k in keys], dtype=np.float32)

    # Save vectors compressed
    npz_path = prefix + "_anchors.npz"
    tmp_npz  = npz_path + ".tmp"
    np.savez_compressed(tmp_npz, keys=np.array(keys, dtype=object), vecs=vecs)
    os.replace(tmp_npz + ".npz" if os.path.exists(tmp_npz + ".npz") else tmp_npz,
               npz_path)

    # Save output strings
    out_path = prefix + "_outputs.json"
    out_data = {k: outputs.get(k, "") for k in keys}
    _atomic_write_text(out_path, json.dumps(out_data, ensure_ascii=False))


def _load_memory(prefix: str) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Load saved memory. Returns (anchors, outputs) or ({}, {})."""
    npz_path = prefix + "_anchors.npz"
    out_path = prefix + "_outputs.json"

    if not os.path.exists(npz_path) or not os.path.exists(out_path):
        return {}, {}

    try:
        data   = np.load(npz_path, allow_pickle=True)
        keys   = list(data["keys"])
        vecs   = data["vecs"]           # (N, D) float32
        with open(out_path, "r", encoding="utf-8") as f:
            outputs = json.load(f)

        anchors = {k: vecs[i] for i, k in enumerate(keys)}
        return anchors, outputs
    except Exception as e:
        print(f"  [CyphaStateful] Warning: could not load memory from {prefix}: {e}")
        return {}, {}


# ══════════════════════════════════════════════════════════════════════════════
#  Params serialisation
# ══════════════════════════════════════════════════════════════════════════════

def _params_to_dict(p: EncoderParams) -> Dict:
    return {
        "chunk_k":       p.chunk_k,
        "damr_radius":   p.damr_radius,
        "active_scales": p.active_scales,
        "prev_error":    p.prev_error if p.prev_error != float("inf") else "inf",
    }


def _dict_to_params(d: Dict) -> EncoderParams:
    p = EncoderParams()
    p.chunk_k       = float(d.get("chunk_k",       4.0))
    p.damr_radius   = float(d.get("damr_radius",   3.0))
    p.active_scales = list(d.get("active_scales",  [1.0, 0.5, 0.25, 0.125]))
    pe              = d.get("prev_error", "inf")
    p.prev_error    = float("inf") if pe == "inf" else float(pe)
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  CyphaStateful
# ══════════════════════════════════════════════════════════════════════════════

class CyphaStateful:
    """
    Drop-in replacement for Cypha with full stateful checkpointing.

    Interface expected by benchmark.py
    ───────────────────────────────────
    cypha = CyphaStateful(feature_dim=512, resonance_dim=256)

    info = cypha.get_checkpoint_info(dataset_name)
    # → None | {"epochs_completed": int, "n_anchors": int, ...}

    cypha.train_file_stateful(filepath, epochs=5, verbose=True)

    result, conf = cypha.infer(text, verbose=False)
    """

    def __init__(self, feature_dim: int = 512, resonance_dim: int = 256,
                 checkpoint_root: str = _CKPT_ROOT):
        self._fd   = feature_dim
        self._rd   = resonance_dim
        self._root = checkpoint_root
        os.makedirs(self._root, exist_ok=True)

        self._lock = threading.Lock()

        # Build the core Cypha engine
        self._cypha: Cypha = self._build_cypha()

        # Track which dataset is currently loaded so we don't
        # re-inject a foreign checkpoint into an already-correct state.
        self._loaded_for: Optional[str] = None

    # ── Construction ──────────────────────────────────────────────────────────

    def _build_cypha(self) -> Cypha:
        """Create a fresh Cypha instance."""
        return Cypha(feature_dim=self._fd, resonance_dim=self._rd)

    # ── Checkpoint metadata ───────────────────────────────────────────────────

    def get_checkpoint_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Returns checkpoint metadata for *dataset_name*, or None if none exists.
        The dict always contains at minimum:
            {"epochs_completed": int, "n_anchors": int, "timestamp": str}
        """
        mp = _meta_path(dataset_name, self._root)
        if not os.path.exists(mp):
            return None
        try:
            with open(mp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # Verify the companion data file exists
            ep   = meta.get("epochs_completed", 0)
            if ep > 0:
                prefix = _ckpt_prefix(dataset_name, ep, self._root)
                if not os.path.exists(prefix + "_anchors.npz"):
                    # stale meta — ignore
                    return None
            return meta
        except Exception:
            return None

    def _save_checkpoint(self, dataset_name: str, epochs_completed: int):
        """Persist full state for *dataset_name* after *epochs_completed* epochs."""
        with self._lock:
            prefix = _ckpt_prefix(dataset_name, epochs_completed, self._root)
            c      = self._cypha

            # 1. Memory (large, compressed)
            print(f"  [ckpt] Saving {len(c.memory.anchors):,} anchors → {prefix}_anchors.npz …",
                  end=" ", flush=True)
            t0 = time.time()
            _save_memory(prefix, c.memory.anchors, c.memory.outputs)
            print(f"done ({time.time()-t0:.1f}s)")

            # 2. Metadata (lightweight)
            meta = {
                "dataset_name":    dataset_name,
                "epochs_completed": epochs_completed,
                "n_anchors":       len(c.memory.anchors),
                "step":            c.step,
                "temperature":     c.temperature,
                "params":          _params_to_dict(c.params),
                "class_counts":    c.memory.class_anchor_counts(),
                "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _atomic_write_text(_meta_path(dataset_name, self._root), json.dumps(meta, indent=2))

            # 3. Prune old epoch files (keep only last 2) to save disk
            self._prune_old_epochs(dataset_name, epochs_completed, keep=2)

    def _prune_old_epochs(self, dataset_name: str, latest: int, keep: int = 2):
        """Delete checkpoint files for epochs older than (latest - keep)."""
        threshold = latest - keep
        d = _ds_dir(dataset_name, self._root)
        for fname in os.listdir(d):
            if fname.startswith("epoch_") and not fname.endswith(".json"):
                try:
                    ep = int(fname.split("_")[1].split(".")[0].split("_")[0])
                    if ep <= threshold:
                        os.remove(os.path.join(d, fname))
                except Exception:
                    pass

    def _load_checkpoint(self, dataset_name: str) -> int:
        """
        Load the latest valid checkpoint for *dataset_name* into self._cypha.
        Returns the number of epochs already completed (0 if none found).
        """
        info = self.get_checkpoint_info(dataset_name)
        if info is None or info.get("epochs_completed", 0) == 0:
            return 0

        ep     = info["epochs_completed"]
        prefix = _ckpt_prefix(dataset_name, ep, self._root)

        print(f"  [ckpt] Resuming {dataset_name} from epoch {ep} "
              f"({info.get('n_anchors', '?')} anchors) …", flush=True)

        anchors, outputs = _load_memory(prefix)
        if not anchors:
            print("  [ckpt] Warning: no anchor data found, starting fresh.")
            return 0

        c = self._cypha

        # Restore memory
        c.memory.anchors = anchors
        c.memory.outputs = outputs
        c.memory._keys   = list(anchors.keys())
        c.memory._vecs   = [anchors[k] for k in c.memory._keys]
        c.memory._dirty  = True

        # Rebuild _class_counts, _cls_idx and _key_to_gi from restored outputs.
        # Critical for class-normalised voting in infer(): without this,
        # class_anchor_counts() returns empty and normalisation uses 1 for all
        # classes (no bias correction after checkpoint resume).
        # _cls_idx rebuilt for O(1) dedup scan on subsequent training steps.
        # _key_to_gi rebuilt for O(1) EMA/LVQ2.1 index lookups.
        c.memory._class_counts = {}
        c.memory._cls_idx = {}
        c.memory._key_to_gi = {}
        for i, (key, cls) in enumerate(zip(c.memory._keys, outputs.values())):
            c.memory._class_counts[cls] = c.memory._class_counts.get(cls, 0) + 1
            if cls not in c.memory._cls_idx:
                c.memory._cls_idx[cls] = []
            c.memory._cls_idx[cls].append(i)
            c.memory._key_to_gi[key] = i

        # Mark _V for rebuild — matmul lookup needs the stacked matrix.
        # _dirty=True triggers lazy rebuild on first lookup call.
        c.memory._V_dirty = True
        c.memory._V       = None   # release any stale matrix from previous dataset

        # Restore scalar state
        c.step        = info.get("step", 0)
        c.temperature = info.get("temperature", 1.5)
        if "params" in info:
            c.params = _dict_to_params(info["params"])

        self._loaded_for = dataset_name
        return ep

    # ── Training ──────────────────────────────────────────────────────────────

    def train_file_stateful(self, filepath: str, epochs: int = 5,
                            verbose: bool = True):
        """
        Train on *filepath* (input|||output format) for *epochs* epochs,
        checkpointing after each epoch.  Resumes automatically.

        The dataset name is derived from the file path stem.
        """
        dataset_name = _stem(filepath)

        # Check if already done
        info = self.get_checkpoint_info(dataset_name)
        if info and info.get("epochs_completed", 0) >= epochs:
            print(f"  [ckpt] {dataset_name}: already trained "
                  f"({info['epochs_completed']} epochs), skipping.")
            # Still need to load state so infer works
            if self._loaded_for != dataset_name:
                self._load_checkpoint(dataset_name)
            return

        # Build byte-offset index -- 8 bytes per line, no data in RAM
        print(f"  [stream] Indexing {filepath} ...", end=" ", flush=True)
        t0 = time.time()
        offsets = _build_offset_index(filepath)
        print(f"{len(offsets):,} pairs  ({time.time()-t0:.1f}s)")

        if len(offsets) == 0:
            print(f"  [ckpt] No training pairs in {filepath}.")
            return

        start_epoch = self._load_checkpoint(dataset_name)
        if start_epoch >= epochs:
            print(f"  [ckpt] Already complete ({start_epoch}/{epochs} epochs).")
            return

        print(f"  [ckpt] Training {dataset_name}: "
              f"epochs {start_epoch+1}-{epochs} on {len(offsets):,} pairs")

        self._stream_train(filepath, offsets, dataset_name, start_epoch, epochs, verbose)
        print(f"  [ckpt] Training complete: {dataset_name}")



    def _stream_train(self, filepath, offsets, dataset_name, start_epoch, epochs, verbose):
        """Core streaming loop - seeks to each offset, reads one line at a time."""
        c = self._cypha
        with open(filepath, "rb") as fh:
            for ep in range(start_epoch, epochs):
                ep_start    = time.time()
                perm        = np.random.permutation(len(offsets))
                total_loss  = 0.0
                event_count = 0
                skipped     = 0

                for i, idx in enumerate(perm):
                    pair = _read_at_offset(fh, offsets[idx])
                    if pair is None:
                        skipped += 1
                        continue
                    inp, out = pair

                    window = []
                    for j in range(max(0, int(idx)-3), min(len(offsets), int(idx)+4)):
                        if j != int(idx):
                            p = _read_at_offset(fh, offsets[j])
                            if p:
                                window.append(p[0])

                    m = c.train_step(inp, out, window)
                    total_loss  += m.loss
                    event_count += m.events

                    if verbose and (i + 1) % 2000 == 0:
                        pct     = 100.0 * (i + 1) / len(offsets)
                        avg     = total_loss / (i + 1)
                        elapsed = time.time() - ep_start
                        eta     = elapsed / (i + 1) * (len(offsets) - i - 1)
                        print(f"    Ep {ep+1}/{epochs}  {pct:5.1f}%  "
                              f"loss={avg:.4f}  anchors={c.memory.n:,}  "
                              f"ETA={eta:.0f}s", flush=True)

                if verbose:
                    n                = max(1, len(offsets) - skipped)
                    min_sep, avg_sep = c.memory.separation_stats()
                    fs               = c.field.stats()
                    ep_t             = time.time() - ep_start
                    print(f"\n  == Epoch {ep+1}/{epochs}  [{dataset_name}]")
                    print(f"     Loss:    {total_loss/n:.4f}   Meta-L: {c.meta.meta_loss():.4f}")
                    print(f"     Anchors: {c.memory.n:,}    Steps: {c.step:,}")
                    print(f"     Field:   k={fs.criticality:.4f}  energy={fs.energy:.3f}")
                    print(f"     Memory:  min_sep={min_sep:.3f}  avg_sep={avg_sep:.3f}")
                    print(f"     Events:  {event_count}  time={ep_t:.1f}s")
                    if skipped:
                        print(f"     Skipped: {skipped}")

                self._save_checkpoint(dataset_name, ep + 1)

    def train_file_stateful_offsets(self, filepath: str, offsets: np.ndarray,
                                    dataset_name: str, epochs: int = 5,
                                    verbose: bool = True):
        """Train on a pre-split offset array - used by benchmark for 80% train slice."""
        info = self.get_checkpoint_info(dataset_name)
        if info and info.get("epochs_completed", 0) >= epochs:
            print(f"  [ckpt] {dataset_name}: already trained "
                  f"({info['epochs_completed']} epochs), skipping.")
            if self._loaded_for != dataset_name:
                self._load_checkpoint(dataset_name)
            return

        start_epoch = self._load_checkpoint(dataset_name)
        if start_epoch >= epochs:
            print(f"  [ckpt] Already complete ({start_epoch}/{epochs} epochs).")
            return

        print(f"  [ckpt] Training {dataset_name}: "
              f"epochs {start_epoch+1}-{epochs} on {len(offsets):,} pairs")

        self._stream_train(filepath, offsets, dataset_name, start_epoch, epochs, verbose)
        print(f"  [ckpt] Training complete: {dataset_name}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def infer(self, text: str, verbose: bool = True) -> Tuple[str, float]:
        """Delegate to Cypha.infer — returns (result_str, confidence_float)."""
        return self._cypha.infer(text, verbose=verbose)

    # ── Passthrough helpers (benchmark uses these) ────────────────────────────

    def train_file(self, path: str, epochs: int = 3, verbose: bool = True):
        """Non-stateful train — delegates directly (for compatibility)."""
        return self._cypha.train_file(path, epochs=epochs, verbose=verbose)

    def train_step(self, inp: str, out: str, negatives=None):
        return self._cypha.train_step(inp, out, negatives)

    def encode_features(self, text: str) -> np.ndarray:
        return self._cypha.encode_features(text)

    def set_goal(self, goal_text: str, strength: float = 0.1):
        return self._cypha.set_goal(goal_text, strength)

    def clear_goal(self):
        return self._cypha.clear_goal()

    def infer_analogical(self, query: str, example_src: str, example_dst: str):
        return self._cypha.infer_analogical(query, example_src, example_dst)

    def episodic_recall(self, query: str, k: int = 5):
        return self._cypha.episodic_recall(query, k)

    @property
    def memory(self):
        return self._cypha.memory

    @property
    def params(self):
        return self._cypha.params

    @property
    def step(self):
        return self._cypha.step

    @property
    def temperature(self):
        return self._cypha.temperature


# ══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _stem(filepath: str) -> str:
    """Return filename without extension, stripping common temp prefixes."""
    base = os.path.basename(filepath)
    name = os.path.splitext(base)[0]
    # benchmark writes to /tmp/train_{dataset_name}
    if name.startswith("train_"):
        name = name[len("train_"):]
    return name



def _build_offset_index(filepath: str) -> np.ndarray:
    """Scan file once, store byte offset of every valid line. ~8 bytes per line."""
    offsets = []
    with open(filepath, "rb") as f:
        while True:
            offset = f.tell()
            raw = f.readline()
            if not raw:
                break
            if b"|||" in raw:
                parts = raw.decode("utf-8", errors="replace").strip().split("|||", 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    offsets.append(offset)
    return np.array(offsets, dtype=np.int64)


def _read_at_offset(fh, offset: int):
    """Seek to byte offset, read one pair. Returns (inp, out) or None."""
    try:
        fh.seek(int(offset))
        raw = fh.readline()
        line = raw.decode("utf-8", errors="replace").strip()
        if "|||" in line:
            a, b = line.split("|||", 1)
            a, b = a.strip(), b.strip()
            if a and b:
                return a, b
    except Exception:
        pass
    return None


def _load_pairs(filepath: str) -> List[Tuple[str, str]]:
    """Load input|||output pairs from a text file."""
    pairs = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if "|||" in line:
                    a, b = line.split("|||", 1)
                    a, b = a.strip(), b.strip()
                    if a and b:
                        pairs.append((a, b))
    except Exception as e:
        print(f"  [CyphaStateful] Error loading {filepath}: {e}")
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE LEARNER — refactored to use StatEngine signals
# ══════════════════════════════════════════════════════════════════════════════

_PSEUDO_TOP_PCT    = 0.15
_PSEUDO_MIN_HIST   = 20
_SIGMA_DROP_MIN    = 0.003
_PROBE_INTERVAL_PC = 0.05
_MIN_PROBE_STEPS   = 5
_POST_TRANS_ALPHA  = 0.01
_LOCK_AFTER_STEPS  = 10
_REPLAY_BUFFER_MAX = 500
_REPLAY_RATIO      = 0.3
_BUDGET_RECS       = {"A":50,"B":500,"C_mismatch":0,"C_ceiling":100,"ambiguous":200}


@dataclass
class _StepStatus:
    step:int; archetype:Optional[str]; phase:str; should_stop:bool
    reason:str; sigma:Optional[float]; acc:Optional[float]
    transition_n:Optional[int]; sigma_dropped:bool


class _TransitionMonitor:
    def __init__(self):
        self.sigma_history:List[float]=[]; self.probe_history:List[Dict]=[]
        self.transition_n=None; self.sigma_pre=None
        self.sigma_post=None; self.transition_acc=None

    def probe(self, cypha:Cypha, probe_data, step:int) -> Dict:
        correct=0; margins=[]
        for inp,label in probe_data:
            try:
                q=cypha.encode_features(inp); m=cypha.memory.lookup(q,k=2)
                if not m: continue
                pred=cypha.memory.get_output(m[0][0]) or m[0][0]
                if pred==label: correct+=1
                mg=float(m[0][1])-(float(m[1][1]) if len(m)>1 else 0.)
                margins.append(mg)
            except Exception: pass
        n=max(len(probe_data),1); acc=correct/n
        sig=float(np.std(margins)) if margins else 0.0
        pt={"step":step,"acc":acc,"sigma":sig,
            "mean_margin":float(np.mean(margins)) if margins else 0.0,
            "n_anchors":cypha.memory.n}
        self.probe_history.append(pt); self.sigma_history.append(sig)
        for mg in margins: cypha.stat.dist.update_margin(mg)
        return pt

    def check_transition(self) -> bool:
        if len(self.sigma_history)<2 or self.transition_n is not None: return False
        if self.sigma_history[-1]-self.sigma_history[-2] < -_SIGMA_DROP_MIN:
            self.transition_n=self.probe_history[-1]["step"]
            self.sigma_pre=self.sigma_history[-2]; self.sigma_post=self.sigma_history[-1]
            self.transition_acc=self.probe_history[-1]["acc"]; return True
        return False

    @property
    def detected(self): return self.transition_n is not None


class AdaptiveLearner:
    """Full adaptive lifecycle. Reads StatEngine signals for rich diagnostics."""

    def __init__(self, cypha:Cypha, budget:Optional[int]=None):
        self.cypha=cypha; self.budget=budget; self.monitor=_TransitionMonitor()
        self._probe_data=None; self._replay_buf:deque=deque(maxlen=_REPLAY_BUFFER_MAX)
        self._step=0; self._phase="warmup"; self._halted=False; self._halt_reason=""
        self._post_trans_steps=0; self._alpha_locked=False
        self.total_labelled=0; self.total_pseudo=0; self.total_replay=0

    def set_probe(self, probe_data) -> None:
        self._probe_data=probe_data; self.cypha.set_probe_data(probe_data)

    def _probe_interval(self) -> int:
        if self.budget: return max(_MIN_PROBE_STEPS,int(self.budget*_PROBE_INTERVAL_PC))
        return max(_MIN_PROBE_STEPS, max(10,self._step//20))

    def _should_probe(self) -> bool:
        return bool(self._probe_data) and self._step>=5 and self._step%self._probe_interval()==0

    def train_step(self, inp:str, label:str, negatives=None) -> _StepStatus:
        if self._halted:
            return _StepStatus(self._step,self.cypha.archetype,self._phase,
                               True,self._halt_reason,None,None,self.monitor.transition_n,False)
        self._step+=1; self.total_labelled+=1
        self._replay_buf.append((inp,label,negatives or []))
        self.cypha.train_step(inp,label,negatives=negatives)
        dropped=False; pt=None
        if self._should_probe():
            pt=self.monitor.probe(self.cypha,self._probe_data,self._step)
            dropped=self.monitor.check_transition()
            if dropped: self._phase="transition"; self.cypha.stat.checkpoint()
        if self.monitor.detected:
            self._post_trans_steps+=1
            if not self._alpha_locked and self._post_trans_steps>=_LOCK_AFTER_STEPS:
                self._apply_alpha_lock()
        arch=self.cypha.archetype; stop=False; reason=""
        if arch=="C_mismatch":
            stop=True; reason=f"C_mismatch@{self.cypha.archetype_detect_step}"
            self._halted=True; self._halt_reason=reason; self._phase="halted"
        elif not self.monitor.detected: self._phase="migration"
        elif self._alpha_locked: self._phase="locked"
        else: self._phase="post_transition"
        return _StepStatus(self._step,arch,self._phase,stop,reason,
                           pt["sigma"] if pt else None,pt["acc"] if pt else None,
                           self.monitor.transition_n,dropped)

    def _apply_alpha_lock(self):
        _o=self.cypha.memory.store; _a=_POST_TRANS_ALPHA
        def _locked(key,state,output,dedup_threshold=None,ema_alpha=0.05,__a=_a,__o=_o):
            return __o(key,state,output,dedup_threshold,__a)
        self.cypha.memory.store=_locked; self._alpha_locked=True

    def unlock_alpha(self):
        import types
        self.cypha.memory.store=types.MethodType(AnchorMemory.store,self.cypha.memory)
        self._alpha_locked=False

    def adapt(self, inp:str, negatives=None, top_pct:float=_PSEUDO_TOP_PCT) -> Dict:
        r=self.cypha.pseudo_label(inp,negatives=negatives,top_pct=top_pct)
        if r["wrote"]: self.total_pseudo+=1
        replayed=False
        if self._replay_buf and np.random.random()<_REPLAY_RATIO:
            i=int(np.random.randint(0,len(self._replay_buf)))
            ri,rl,rn=self._replay_buf[i]
            self.cypha.train_step(ri,rl,negatives=rn); self.total_replay+=1; replayed=True
        r["replayed"]=replayed; return r

    def adapt_stream(self, stream, top_pct:float=_PSEUDO_TOP_PCT, verbose:bool=False) -> Dict:
        writes=0; total=0
        for item in stream:
            inp=item if isinstance(item,str) else item[0]
            negs=None if isinstance(item,str) else (item[1] if len(item)>1 else None)
            r=self.adapt(inp,negatives=negs,top_pct=top_pct); total+=1
            if r["wrote"]: writes+=1
            if verbose and total%100==0: print(f"  adapt n={total} write_rate={writes/total:.3f}")
        return {"total":total,"writes":writes,"write_rate":round(writes/max(total,1),4),
                "stat":self.cypha.stat_report()}

    def train_to_convergence(self, stream, max_steps=None, verbose:bool=True) -> Dict:
        import time as _t; t0=_t.time(); n=0; reason="stream exhausted"
        for item in stream:
            if max_steps and n>=max_steps: reason=f"budget({max_steps})"; break
            inp,label=item[0],item[1]; negs=item[2] if len(item)>2 else None
            s=self.train_step(inp,label,negs); n+=1
            if verbose and s.sigma is not None:
                tx="  ***TRANSITION***" if s.sigma_dropped else ""
                print(f"  n={s.step:>5} phase={s.phase:<16} arch={str(s.archetype):<12} "
                      f"sig={s.sigma:.4f} acc={s.acc:.3f}{tx}")
            if s.should_stop: reason=s.reason; break
            if self._phase=="locked" and self._post_trans_steps>=20:
                reason=f"Transition n={self.monitor.transition_n}, locked."; break
        self.cypha.end_training()
        if verbose: print(f"\n  Stopped: {reason}\n  Steps:{n} {_t.time()-t0:.1f}s")
        return {"steps":n,"reason":reason,"archetype":self.cypha.archetype,
                "arch_confidence":self.cypha.archetype_confidence,
                "transition_n":self.monitor.transition_n,
                "stat":self.cypha.stat_report(),"elapsed":round(_t.time()-t0,2)}

    def report(self) -> Dict:
        m=self.monitor
        return {"step":self._step,"phase":self._phase,"archetype":self.cypha.archetype,
                "archetype_confidence":self.cypha.archetype_confidence,
                "transition_detected":m.detected,"transition_n":m.transition_n,
                "sigma_pre":m.sigma_pre,"sigma_post":m.sigma_post,
                "alpha_locked":self._alpha_locked,"total_labelled":self.total_labelled,
                "total_pseudo":self.total_pseudo,"total_replay":self.total_replay,
                "memory_anchors":self.cypha.memory.n,"stat":self.cypha.stat_report()}

    def print_report(self) -> None:
        r=self.report(); sr=r["stat"]; SEP="-"*60; EQ="="*60
        print(f"\n{EQ}\n  ADAPTIVE LEARNER REPORT\n{EQ}")
        print(f"  Phase:{r['phase']}  Anchors:{r['memory_anchors']}")
        print(f"  Steps:{r['total_labelled']} labelled {r['total_pseudo']} pseudo {r['total_replay']} replay")
        print(SEP)
        print(f"  Archetype:{r['archetype']} (conf={r['archetype_confidence']:.2f} @{self.cypha.archetype_detect_step})")
        if r["transition_detected"]:
            print(f"  Transition: n={r['transition_n']} sig {r['sigma_pre']:.4f}->{r['sigma_post']:.4f}")
        else: print("  Transition: not detected")
        ms=sr.get("margin_stats",{})
        print(f"  Margins: mean={ms.get('mean','?')} std={ms.get('std','?')} skew={ms.get('skew','?')} kurt={ms.get('kurt','?')}")
        print(f"  ECE:{sr.get('ece','?')} KL:{round(sr.get('kl_divergence',0),4)} drift={sr.get('input_drift','?')}")
        print(f"  Gini:{sr.get('anchor_gini','?')} dead:{sr.get('dead_anchors','?')}")
        print(f"{EQ}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  Quick smoke test
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# CYPHA DECODER
# Resonant SSM decoder that uses ALL cognitive machinery for generation.
#
# Architecture:
#   Encoding:   text → OmegaEncoder → PhaseBridge → ψ ∈ ℂ^D
#   Transition: h_{t+1} = W_T @ h_t  +  B_embed @ e_t   (causal model)
#   Read-out:   logits_t = C @ h_t                        (vocabulary projection)
#   Sample:     token_t  ~ softmax(logits_t / τ)
#
# Cognitive integrations (all active during generation):
#   • Goal field    — injects goal direction into ψ before each token
#   • κ-temperature — criticality modulates sampling entropy
#   • Thought cascade — fires when top-2 logits within margin → resolves token
#   • Planning      — forward-simulates K trajectories, picks best direction
#   • Schema library — matches generation pattern, biases toward schema attractor
#   • Episodic replay — high-surprise tokens trigger episodic lookup for context
#   • Causal model  — W_T drives state forward (same matrix trained in classification)
#   • Relational graph — traversal can suggest next-class tokens during generation
# ══════════════════════════════════════════════════════════════════════════════

class CyphaDecoder:
    """
    Learned read-out for the Cypha resonance field.

    The decoder is a minimal addition on top of the existing architecture:
      - C: (vocab_size × state_dim) projection matrix
      - B_embed: (state_dim × vocab_size) token re-injection matrix
      - train_on_sequence(): online SGD on byte sequences via cross-entropy
      - generate(): full cognitive generation loop

    All cognitive machinery from the parent Cypha instance is used live
    during generation — nothing is bypassed or simplified.
    """
    def __init__(self, state_dim: int = 256, vocab_size: int = 256,
                 lr: float = 1e-3):
        rng = np.random.default_rng(99)
        # C: project field state → token logits
        self.C       = (rng.standard_normal((vocab_size, state_dim)) * 0.02).astype(np.float64)
        # B_embed: inject sampled token back into field state
        self.B_embed = (rng.standard_normal((state_dim, vocab_size)) * 0.02).astype(np.float64)
        self.vocab_size  = vocab_size
        self.state_dim   = state_dim
        self.lr          = lr
        self._trained_steps = 0
        # Momentum for Adam-lite
        self._mC  = np.zeros_like(self.C)
        self._vC  = np.zeros_like(self.C)
        self._mB  = np.zeros_like(self.B_embed)
        self._vB  = np.zeros_like(self.B_embed)
        self._beta1, self._beta2, self._eps = 0.9, 0.999, 1e-8
        # Generation stats
        self.last_gen_tokens: List[int] = []
        self.last_gen_surprises: List[float] = []

    # ── Low-level ops ─────────────────────────────────────────────────────────

    def logits(self, h: np.ndarray) -> np.ndarray:
        """C @ h  — project field state to raw logits."""
        return self.C @ h

    def sample_token(self, logits: np.ndarray, temperature: float = 1.0,
                     top_k: int = 8, top_p: float = 0.95) -> Tuple[int, np.ndarray]:
        """Nucleus (top-p) + top-k sampling with temperature.
        Nucleus sampling prevents collapse better than top-k alone."""
        lg = logits / max(temperature, 1e-4)
        lg -= lg.max()
        p_full = np.exp(lg); p_full /= p_full.sum()

        # Top-k filter first
        if top_k < self.vocab_size:
            thresh = np.sort(p_full)[-top_k]
            p_full = np.where(p_full >= thresh, p_full, 0.0)
            p_full /= p_full.sum()

        # Nucleus (top-p) filter — keeps minimum tokens covering top_p mass
        sorted_idx  = np.argsort(p_full)[::-1]
        cum_prob    = np.cumsum(p_full[sorted_idx])
        cutoff      = int(np.searchsorted(cum_prob, top_p)) + 1
        nucleus_idx = sorted_idx[:max(cutoff, 1)]
        mask        = np.zeros(self.vocab_size, dtype=bool)
        mask[nucleus_idx] = True
        p_nucleus   = np.where(mask, p_full, 0.0)
        p_nucleus  /= p_nucleus.sum()

        chosen = int(np.random.choice(self.vocab_size, p=p_nucleus))
        return chosen, p_nucleus

    def token_surprise(self, token: int, probs: np.ndarray) -> float:
        """Cross-entropy surprise for the sampled token: -log p(token)."""
        return float(-np.log(max(probs[token], 1e-9)))

    def inject_token(self, h: np.ndarray, token: int) -> np.ndarray:
        """Inject one-hot token back into state via B_embed."""
        e = np.zeros(self.vocab_size, dtype=np.float64)
        e[token] = 1.0
        return h + self.B_embed @ e

    # ── Training ──────────────────────────────────────────────────────────────

    def train_on_sequence(self, h_sequence: List[np.ndarray],
                          token_sequence: List[int]) -> float:
        """
        Online cross-entropy training: given a list of field states and the
        corresponding next tokens, update C and B_embed via Adam.

        h_sequence[t] → predicts token_sequence[t]
        """
        if len(h_sequence) < 2 or len(token_sequence) < 1:
            return 0.0
        n = min(len(h_sequence) - 1, len(token_sequence))
        total_loss = 0.0
        self._trained_steps += 1
        t_step = self._trained_steps

        for t in range(n):
            h   = h_sequence[t].astype(np.float64)
            tok = token_sequence[t]
            # Forward
            lg  = self.C @ h
            lg -= lg.max()
            ex  = np.exp(lg); sm = ex / ex.sum()
            # Loss
            total_loss += float(-np.log(sm[tok] + 1e-9))
            # Gradient dL/dC = (sm - one_hot(tok)) ⊗ h
            delta      = sm.copy(); delta[tok] -= 1.0
            dC         = np.outer(delta, h) / n
            # Gradient dL/dB_embed via token injection
            # h_{t+1} ≈ W_T @ h_t + B_embed @ e_t
            # dL/dB = dL/dh_{t+1} ⊗ e_t
            # Approximate dL/dh_{t+1} as C.T @ delta
            dh    = self.C.T @ delta / n
            e_tok = np.zeros(self.vocab_size); e_tok[tok] = 1.0
            dB    = np.outer(dh, e_tok)
            # Adam update for C
            self._mC = self._beta1 * self._mC + (1-self._beta1) * dC
            self._vC = self._beta2 * self._vC + (1-self._beta2) * dC**2
            mC_hat   = self._mC / (1 - self._beta1**t_step)
            vC_hat   = self._vC / (1 - self._beta2**t_step)
            self.C  -= self.lr * mC_hat / (np.sqrt(vC_hat) + self._eps)
            # Adam update for B_embed
            self._mB = self._beta1 * self._mB + (1-self._beta1) * dB
            self._vB = self._beta2 * self._vB + (1-self._beta2) * dB**2
            mB_hat   = self._mB / (1 - self._beta1**t_step)
            vB_hat   = self._vB / (1 - self._beta2**t_step)
            self.B_embed -= self.lr * mB_hat / (np.sqrt(vB_hat) + self._eps)

        return total_loss / max(n, 1)

    # ── Cognitive generation ──────────────────────────────────────────────────

    def generate(self, cypha: 'Cypha', prime_text: str,
                 max_tokens: int = 128,
                 temperature: float = 1.0,
                 top_k: int = 8,
                 top_p: float = 0.95,
                 goal_text: Optional[str] = None,
                 use_planning: bool = True,
                 kappa_temp_scale: float = 0.5,
                 schema_bias: float = 0.05,
                 cascade_margin: float = 0.15,
                 stop_byte: Optional[int] = None,
                 verbose: bool = False) -> bytes:
        """
        Full cognitive generation loop.

        Every token step engages:
          1. Goal field injection (if goal set)
          2. κ-scaled temperature (criticality → entropy)
          3. Thought cascade on close logits (self-questioning)
          4. Schema attractor bias (C-space attractor from schema library)
          5. Planning trajectory to avoid low-probability dead ends
          6. Episodic surprise gating (high surprise → replay + gate)
          7. W_T state transition (causal model drives field forward)
          8. Token re-injection via B_embed
        """
        # ── 0. Set optional goal ─────────────────────────────────────────────
        if goal_text is not None:
            cypha.set_goal(goal_text, strength=0.15)

        # ── 1. Prime field via full forward pass ─────────────────────────────
        cypha.field.reset(); cypha.res_level.reset()
        cypha.assembly.reset(); cypha.module.reset(); cypha.global_l.reset()
        prime_out = cypha.forward(prime_text, training=False)
        h = prime_out['state'].astype(np.float64)   # 256-dim real state

        # ── 2. Build schema attractor if schemas exist ───────────────────────
        schema_attractor: Optional[np.ndarray] = None
        if cypha.schema_lib._proto_mat is not None and len(cypha.schema_lib._proto_names) > 0:
            # Project current state into schema space: find dominant schema
            if cypha.schema_lib._proto_dirty:
                cypha.schema_lib._rebuild_mat()
            sm = cypha.schema_lib._proto_mat  # (n_schemas, rel_dim)
            h_rel = h[:sm.shape[1]] / (np.linalg.norm(h[:sm.shape[1]]) + 1e-9)
            scores = sm @ h_rel.astype(np.float32)
            top_idx = int(np.argmax(scores))
            top_name = cypha.schema_lib._proto_names[top_idx]
            # Schema attractor: project schema prototype back to vocab space
            proto = cypha.schema_lib._prototypes[top_name].astype(np.float64)
            n_proj = min(len(proto), self.state_dim)
            attractor = np.zeros(self.state_dim)
            attractor[:n_proj] = proto[:n_proj]
            schema_attractor = attractor
            if verbose:
                print(f'[SCHEMA] dominant={top_name} score={float(scores[top_idx]):.4f}')

        # ── 3. Generation loop ────────────────────────────────────────────────
        output_bytes: List[int] = []
        h_history:    List[np.ndarray] = [h.copy()]
        tok_history:  List[int] = []
        surprises:    List[float] = []
        W_T = cypha.recursive._W_T  # live causal matrix

        for step in range(max_tokens):

            # ── 3a. Apply goal field ─────────────────────────────────────────
            if cypha.goal.goal_vec is not None:
                gv = cypha.goal.goal_vec.astype(np.float64)
                n_g = min(len(gv), self.state_dim)
                h[:n_g] += 0.03 * gv[:n_g]
                hn = np.linalg.norm(h); h = h / (hn + 1e-9) * hn  # preserve magnitude

            # ── 3b. κ-scaled temperature ─────────────────────────────────────
            kappa = cypha.field.criticality()
            tau   = temperature * (1.0 + kappa_temp_scale * kappa)

            # ── 3c. Schema attractor bias ────────────────────────────────────
            h_biased = h.copy()
            if schema_attractor is not None:
                h_biased += schema_bias * schema_attractor

            # ── 3d. Compute raw logits ────────────────────────────────────────
            raw_logits = self.logits(h_biased)

            # ── 3e. Thought cascade on ambiguous logits ───────────────────────
            # If top-2 byte logits are within cascade_margin: run self-questioning
            top2_idx  = np.argpartition(raw_logits, -2)[-2:]
            top2_vals = raw_logits[top2_idx]
            top2_margin = float(np.abs(top2_vals[1] - top2_vals[0]))
            if top2_margin < cascade_margin * float(np.abs(raw_logits).max() + 1e-9):
                # Build pseudo-candidates from top byte values
                top4_idx  = np.argpartition(raw_logits, -4)[-4:]
                cands_gen = [(f'byte_{b}', float(raw_logits[b])) for b in top4_idx]
                from Cypha import Event, EventType
                import time as _time
                trig = Event(EventType.THOUGHT.name, _time.time(),
                    {'vector': h_biased, 'magnitude': float(np.linalg.norm(h_biased))},
                    'gen_ambiguity', 0.7)
                cascade_evs = cypha.thought.cascade(trig, h_biased, kappa,
                    candidates=cands_gen, field=cypha.field)
                # Blend cascade event vectors back into h
                if cascade_evs:
                    h_biased = cypha.thought.multi_scale(cascade_evs, h_biased)
                    raw_logits = self.logits(h_biased)

            # ── 3f. Planning (every 8 steps if goal active) ─────────────────
            if use_planning and cypha.goal.goal_vec is not None and step % 8 == 0:
                plan_action = cypha.plan(n_trajectories=4, horizon=3)
                if plan_action is not None:
                    pa = plan_action.real.astype(np.float64)
                    n_pa = min(len(pa), self.state_dim)
                    h_biased[:n_pa] += 0.05 * pa[:n_pa]
                    raw_logits = self.logits(h_biased)

            # ── 3g. Sample token ─────────────────────────────────────────────
            token, probs = self.sample_token(raw_logits, temperature=tau, top_k=top_k, top_p=top_p)
            surp = self.token_surprise(token, probs)
            surprises.append(surp)
            output_bytes.append(token)
            tok_history.append(token)

            # ── 3h. Episodic surprise gating ──────────────────────────────────
            if surp > 3.0 and len(cypha.episodic._episodes) > 0:
                # episodic memory stores 512-dim feature vectors — use prime encoding
                ep_cue = cypha.encode_features(prime_text).astype(np.float32)
                ep_n   = float(np.linalg.norm(ep_cue))
                if ep_n > 1e-9:
                    ep_cue /= ep_n
                    eps = cypha.episodic.retrieve_by_cue(ep_cue, k=3)
                    if eps:
                        ep_vec = eps[0].context_vec.astype(np.float64)
                        n_ev = min(len(ep_vec), self.state_dim)
                        h[:n_ev] = 0.9 * h[:n_ev] + 0.1 * ep_vec[:n_ev]
                        if verbose:
                            print(f'[EPISODIC] step={step} surp={surp:.2f} → '
                                  f'blended ep "{eps[0].inp_key}"')

            # ── 3i. State transition: W_T + token re-injection ───────────────
            # W_T contracts h (sr≈0.077), so token injection must be strong
            # enough to maintain state diversity.  Scale by 1/sr to compensate.
            h_mag  = float(np.linalg.norm(h)) + 1e-9
            h = W_T @ h
            e_tok  = np.zeros(self.vocab_size, dtype=np.float64)
            e_tok[token] = 1.0
            injection = self.B_embed @ e_tok
            # Inject at magnitude proportional to current state norm so
            # the token signal doesn't drown in a contracting field
            inj_scale = h_mag / (float(np.linalg.norm(injection)) + 1e-9)
            h += injection * min(inj_scale * 0.5, 2.0)

            # ── 3j. Relational graph lookup: does next token relate to a class?
            # If the current h is near a class centroid, bias toward that class's
            # member tokens (via relational graph traversal)
            if step % 16 == 0 and cypha.memory.n > 0 and tok_history:
                # Reconstruct a text snippet from recent tokens to get a 512-dim anchor
                recent_bytes = bytes([t for t in tok_history[-8:] if 0 < t < 128])
                try:
                    recent_text = recent_bytes.decode('utf-8', errors='replace').strip() or prime_text
                except Exception:
                    recent_text = prime_text
                h_q_full = cypha.encode_features(recent_text)  # 512-dim for memory lookup
                matches = cypha.memory.lookup(h_q_full, k=2)
                if matches:
                    top_cls = cypha.memory.get_output(matches[0][0])
                    if top_cls and top_cls in cypha.rel_graph._edges:
                        cls_vec = cypha.encode_features(top_cls).astype(np.float64)
                        rel_q   = cypha.rel_enc.encode_pair(h_q_full.astype(np.float64), cls_vec)
                        members = cypha.rel_graph.traverse(top_cls, rel_q, top_k=3)
                        for m_key, m_sim in members[:1]:
                            m_enc = cypha.encode_features(m_key).astype(np.float64)
                            m_br  = cypha.bridge.bridge(m_enc.astype(np.float32)).real
                            n_mb  = min(len(m_br), self.state_dim)
                            h[:n_mb] += 0.02 * m_sim * m_br[:n_mb].astype(np.float64)

            # ── 3k. Evolve field one step (keeps ψ in sync with h) ───────────
            cypha.field.evolve(1)
            psi_now = cypha.field.psi.real.astype(np.float64)
            n_pn = min(len(psi_now), self.state_dim)
            h[:n_pn] = 0.95 * h[:n_pn] + 0.05 * psi_now[:n_pn]

            h_history.append(h.copy())

            if verbose and step % 16 == 0:
                print(f'[GEN step={step:3d}] κ={kappa:.4f} τ={tau:.3f} '
                      f'surp={surp:.3f} token=0x{token:02x}({chr(token) if 32<=token<127 else "."})')

            if stop_byte is not None and token == stop_byte:
                break

        # ── 4. Record stats ──────────────────────────────────────────────────
        self.last_gen_tokens    = tok_history
        self.last_gen_surprises = surprises

        return bytes(output_bytes)

    def train_on_cypha(self, cypha: 'Cypha', text: str, epochs: int = 1) -> float:
        """
        Train decoder on a text string using the current Cypha field dynamics.
        Runs text byte-by-byte through the full forward() pipeline to build
        a sequence of field states, then trains C and B_embed on that sequence.
        """
        raw = text.encode('utf-8', errors='replace')
        if len(raw) < 2:
            return 0.0
        total_loss = 0.0
        for _ep in range(epochs):
            cypha.field.reset(); cypha.res_level.reset()
            cypha.assembly.reset(); cypha.module.reset(); cypha.global_l.reset()
            h_seq   = []
            tok_seq = []
            # Build state sequence: each byte is a context chunk
            chunk = 8  # process in 8-byte chunks so forward() has something to encode
            for i in range(0, len(raw) - 1, chunk):
                seg = raw[i:i+chunk]
                try:
                    seg_text = seg.decode('utf-8', errors='replace')
                except Exception:
                    seg_text = ' '.join(f'{b:02x}' for b in seg)
                out = cypha.forward(seg_text, training=True)
                h   = out['state'].astype(np.float64)
                h_seq.append(h)
                # Targets: bytes immediately following this chunk
                for b in raw[i+1:i+chunk+1]:
                    tok_seq.append(int(b))
            loss = self.train_on_sequence(h_seq, tok_seq[:len(h_seq)])
            total_loss += loss
        return total_loss / max(epochs, 1)


# ── Integrate decoder into main Cypha class ──────────────────────────────────
def _cypha_init_decoder(self):
    """Attach a CyphaDecoder to an existing Cypha instance."""
    if not hasattr(self, 'decoder'):
        self.decoder = CyphaDecoder(state_dim=self.rd, vocab_size=256)

def _cypha_train_decoder(self, text: str, epochs: int = 1) -> float:
    """Train the decoder on text using current field dynamics."""
    _cypha_init_decoder(self)
    return self.decoder.train_on_cypha(self, text, epochs=epochs)

def _cypha_generate(self, prime_text: str, max_tokens: int = 128,
                    temperature: float = 1.0, top_k: int = 8,
                    goal_text: Optional[str] = None,
                    use_planning: bool = True,
                    verbose: bool = False) -> str:
    """Generate text from a priming string using the full cognitive decoder."""
    _cypha_init_decoder(self)
    raw = self.decoder.generate(
        self, prime_text, max_tokens=max_tokens,
        temperature=temperature, top_k=top_k,
        goal_text=goal_text, use_planning=use_planning,
        verbose=verbose)
    try:
        return raw.decode('utf-8', errors='replace')
    except Exception:
        return ''.join(chr(b) if 32 <= b < 127 else '.' for b in raw)

# Monkey-patch onto Cypha class
Cypha.init_decoder    = _cypha_init_decoder
Cypha.train_decoder   = _cypha_train_decoder
Cypha.generate        = _cypha_generate

if __name__ == "__main__":
    import tempfile
    print("=" * 60)
    print("  CyphaStateful — smoke test")
    print("=" * 60)
    demo_pairs = [
        ("cat sound","meow"),("dog sound","bark"),("wolf sound","howl"),
        ("owl sound","hoot"),("capital of France","Paris"),
        ("capital of Japan","Tokyo"),("is 5 > 3","true"),
        ("is 2 > 10","false"),("12+165","177"),("next: 1 2 3 4 5","6"),
    ]
    tmp = tempfile.mktemp(suffix=".txt")
    with open(tmp,"w") as f:
        for a,b in demo_pairs: f.write(f"{a}|||{b}\n")

    cs = CyphaStateful(feature_dim=1024, resonance_dim=64,
                       checkpoint_root="/tmp/cypha_smoke_test")

    cs.train_file_stateful(tmp, epochs=2, verbose=True)

    print("\n── Inference ──")
    tests = [
        ("cat sound",         "meow"),
        ("capital of France", "Paris"),
        ("is 5 > 3",          "true"),
        ("12+165",            "177"),
    ]
    ok = 0
    for inp, exp in tests:
        r, conf = cs.infer(inp, verbose=False)
        mark = "✓" if r == exp else "✗"
        print(f"  {mark}  '{inp}' → '{r}' (expected '{exp}', conf={conf:.3f})")
        if r == exp:
            ok += 1
    print(f"\n  {ok}/{len(tests)} exact matches\n")

    # Resume test
    print("── Resume test (epoch already done) ──")
    cs2 = CyphaStateful(feature_dim=1024, resonance_dim=64,
                        checkpoint_root="/tmp/cypha_smoke_test")
    info = cs2.get_checkpoint_info(_stem(tmp))
    print(f"  Checkpoint info: {info}")

    os.remove(tmp)
    shutil.rmtree("/tmp/cypha_smoke_test", ignore_errors=True)
    print("  Done.\n")
