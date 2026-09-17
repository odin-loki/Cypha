"""
cypha_multimodal_profile.py
════════════════════════════
Profiles Cypha's learning dynamics across 6 data modalities:

  1. TEXT         — bag-of-words with class-specific vocabulary
  2. AUDIO (PCM)  — synthetic int16 PCM waveforms (tones, noise, sweeps)
  3. RF / IQ      — synthetic int8 IQ signals (AM, FM, BPSK, CW, USB, QPSK)
  4. IMAGE        — synthetic grayscale image feature arrays (arr: prefix)
  5. STRUCTURED   — tabular float32 feature vectors (arr: prefix)
  6. VIDEO        — temporal frame sequence aggregates (arr: prefix)

For each modality:
  — Trains a fresh Cypha instance with 6 classes × 2000 examples
  — Probes every 100 steps: probe_acc, mean_margin, margin_σ
  — Detects the phase transition (σ collapse)
  — Reports post-transition behaviour: improving / stable / degrading
  — Identifies whether a counterproductive zone exists

Run:
    python cypha_multimodal_profile.py

Output:
    cypha_multimodal_results.json   — full per-step data for all modalities
    cypha_multimodal_report.txt     — human-readable comparison report
"""

import sys, os, math, time, json, base64
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# ── Locate Cypha ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from Cypha import Cypha, AnchorMemory, ThoughtProcessor, EPSILON
    print(f"✓ Cypha loaded from {_HERE}/Cypha.py")
except ImportError as e:
    print(f"✗ Cannot import Cypha: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

N_CLASSES      = 6       # classes per modality
EXAMPLES_PC    = 2000    # training examples per class
PROBE_EVERY    = 100     # measure accuracy/margin every N steps
N_PROBE        = 150     # examples in held-out probe set
FEATURE_DIM    = 512
RESONANCE_DIM  = 256
SAMPLE_RATE    = 16000   # Hz for audio
IQ_SAMPLES     = 512     # IQ samples per RF example (1024 bytes)
AUDIO_SAMPLES  = 8000    # PCM samples per audio example (16000 bytes int16)
IMG_PIXELS     = 64      # image side length → 64×64 grayscale
VIDEO_FRAMES   = 8       # frames per video clip
VIDEO_FRAME_PX = 16      # frame side length → 16×16 per frame

rng = np.random.default_rng(42)

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. TEXT ──────────────────────────────────────────────────────────────────

_TEXT_VOCAB = [
    # 6 class-specific keyword sets
    ["quantum_entanglement", "superposition_state", "wavefunction_collapse",
     "hilbert_space", "eigenvalue_spectrum", "decoherence_rate",
     "density_matrix", "bell_inequality", "qubit_coherence", "tunneling_prob"],

    ["convolutional_layer", "backpropagation_gradient", "activation_relu",
     "batch_normalisation", "dropout_regularise", "attention_mechanism",
     "transformer_encoder", "embedding_dimension", "cross_entropy_loss", "softmax_output"],

    ["haemoglobin_saturation", "action_potential_spike", "synaptic_vesicle",
     "mitochondrial_membrane", "atp_synthesis_rate", "receptor_binding",
     "ion_channel_conductance", "neurotransmitter_release", "axon_myelination", "dendrite_arbour"],

    ["geodesic_curvature", "riemann_tensor", "spacetime_manifold",
     "christoffel_symbol", "metric_signature", "stress_energy_tensor",
     "schwarzschild_radius", "gravitational_wave", "event_horizon", "cosmological_constant"],

    ["tcp_handshake_syn", "packet_fragmentation", "routing_table_bgp",
     "ssl_certificate_chain", "dns_resolution", "arp_broadcast",
     "icmp_echo_request", "firewall_stateful", "latency_jitter", "bandwidth_throttle"],

    ["derivative_chain_rule", "fourier_transform_convolution", "laplacian_operator",
     "eigenfunction_basis", "hilbert_transform", "dirac_delta", "greens_function",
     "variational_calculus", "stochastic_differential", "measure_theoretic_integral"],
]

_TEXT_SHARED = [
    "analysis", "system", "process", "function", "signal", "data",
    "method", "result", "parameter", "component", "structure", "value",
    "matrix", "vector", "state", "level", "model", "output", "input",
]

def text_example(cls: int, noise_p: float = 0.04) -> str:
    kw  = list(rng.choice(_TEXT_VOCAB[cls], size=8, replace=False))
    sh  = list(rng.choice(_TEXT_SHARED, size=12, replace=True))
    noise = []
    if rng.random() < noise_p:
        wc = int(rng.integers(0, N_CLASSES))
        while wc == cls: wc = int(rng.integers(0, N_CLASSES))
        noise = [rng.choice(_TEXT_VOCAB[wc])]
    tokens = kw + sh + noise
    rng.shuffle(tokens)
    return " ".join(tokens)

# ── 2. AUDIO (PCM int16, pcm: prefix) ────────────────────────────────────────

_AUDIO_CLASSES = [
    # (description, generator_fn)
    # Each returns float64 signal in [-1, 1], length AUDIO_SAMPLES
    "pure_tone",
    "harmonic_chord",
    "white_noise",
    "sweep_chirp",
    "click_train",
    "am_modulated",
]

def audio_example(cls: int, noise_p: float = 0.03) -> str:
    t = np.arange(AUDIO_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    # Class-specific carrier frequencies (well separated in mel space)
    freqs = [220.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0]
    f0 = freqs[cls] * (1.0 + rng.normal(0, 0.01))  # slight jitter

    if cls == 0:   # pure tone
        sig = np.sin(2 * np.pi * f0 * t)
    elif cls == 1: # harmonic chord (multiple harmonics)
        sig = (np.sin(2*np.pi*f0*t) +
               0.5*np.sin(2*np.pi*2*f0*t) +
               0.25*np.sin(2*np.pi*3*f0*t))
        sig /= (np.abs(sig).max() + 1e-9)
    elif cls == 2: # white noise (filtered to band)
        sig = rng.normal(0, 1, AUDIO_SAMPLES)
        # Simple bandpass: keep only freq range [f0/1.5, f0*1.5]
        # Approximate via spectral masking
        F  = np.fft.rfft(sig)
        fr = np.fft.rfftfreq(AUDIO_SAMPLES, 1/SAMPLE_RATE)
        mask = (fr >= f0/1.5) & (fr <= f0*1.5)
        F_m = F * mask
        sig = np.fft.irfft(F_m, n=AUDIO_SAMPLES)
        n_ = np.abs(sig).max()
        if n_ > 1e-9: sig /= n_
    elif cls == 3: # sweep chirp (frequency rises f0 → 2*f0)
        phase = 2 * np.pi * (f0 * t + (f0 / (2 * AUDIO_SAMPLES/SAMPLE_RATE)) * t**2)
        sig = np.sin(phase)
    elif cls == 4: # click train at period corresponding to f0
        sig = np.zeros(AUDIO_SAMPLES)
        period = int(SAMPLE_RATE / f0)
        if period < 1: period = 1
        for i in range(0, AUDIO_SAMPLES, period):
            sig[i] = 1.0
            if i+1 < AUDIO_SAMPLES: sig[i+1] = -1.0
    else:          # AM modulated (carrier f0, modulator f0/10)
        carrier = np.sin(2 * np.pi * f0 * t)
        modulator = 0.5 * (1.0 + np.sin(2 * np.pi * (f0/10.0) * t))
        sig = carrier * modulator

    # Add small noise
    sig = sig + rng.normal(0, 0.02, AUDIO_SAMPLES)
    if noise_p > 0 and rng.random() < noise_p:
        # Add a component from a wrong class
        wc = int(rng.integers(0, N_CLASSES))
        while wc == cls: wc = int(rng.integers(0, N_CLASSES))
        wrong_sig_fn = lambda: audio_example(wc, noise_p=0.0)
        # Just add amplitude-scaled version
        wf  = freqs[wc]
        sig = sig + 0.1 * np.sin(2*np.pi*wf*t)

    # Clip and quantise to int16
    sig = np.clip(sig, -1.0, 1.0)
    pcm = (sig * 32700).astype(np.int16)
    return "pcm:" + pcm.tobytes().hex()

# ── 3. RF / IQ (int8, iq: prefix) ────────────────────────────────────────────

_RF_CLASSES = ["AM", "FM", "BPSK", "CW", "USB", "QPSK"]

def rf_example(cls: int, noise_p: float = 0.04) -> str:
    """Generate IQ samples for one of 6 RF modulation types."""
    N = IQ_SAMPLES
    t = np.arange(N, dtype=np.float64) / N
    fc = 0.1  # normalised carrier freq
    phi = rng.uniform(0, 2*np.pi)  # random carrier phase (irrelevant to features)

    msg_freq = 0.02  # normalised message freq
    msg = np.sin(2*np.pi*msg_freq*N*t + rng.uniform(0, 2*np.pi))

    if cls == 0:   # AM  — amplitude modulated carrier
        env = 1.0 + 0.7 * msg
        I = env * np.cos(2*np.pi*fc*N*t + phi)
        Q = env * np.sin(2*np.pi*fc*N*t + phi)
    elif cls == 1: # FM  — frequency modulated
        kf = 0.05  # frequency deviation
        phase_dev = 2*np.pi*kf * np.cumsum(msg) / N
        I = np.cos(2*np.pi*fc*N*t + phi + phase_dev)
        Q = np.sin(2*np.pi*fc*N*t + phi + phase_dev)
    elif cls == 2: # BPSK — binary phase shift keying
        n_symbols = N // 32
        symbols = rng.choice([-1.0, 1.0], size=n_symbols)
        sym_seq = np.repeat(symbols, 32)[:N]
        I = sym_seq * np.cos(2*np.pi*fc*N*t + phi)
        Q = sym_seq * np.sin(2*np.pi*fc*N*t + phi)
    elif cls == 3: # CW  — continuous wave (unmodulated)
        I = np.cos(2*np.pi*fc*N*t + phi)
        Q = np.sin(2*np.pi*fc*N*t + phi)
    elif cls == 4: # USB — upper sideband (analytic signal, one-sided spectrum)
        # USB: carrier × (1 + hilbert(msg)) — analytic version is one-sided
        analytic_msg = msg + 1j * np.imag(
            np.fft.ifft(np.fft.fft(msg) * (np.arange(N) < N//2) * 2))
        cplx = (1 + 0.5*analytic_msg) * np.exp(1j*(2*np.pi*fc*N*t + phi))
        I = np.real(cplx); Q = np.imag(cplx)
    else:          # QPSK — quadrature phase shift keying (4 phases)
        n_sym = N // 32
        phases = rng.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=n_sym)
        ph_seq = np.repeat(phases, 32)[:N]
        I = np.cos(2*np.pi*fc*N*t + phi + ph_seq)
        Q = np.sin(2*np.pi*fc*N*t + phi + ph_seq)

    # Add AWGN (SNR ≈ 15 dB)
    snr = 10**(15/10)
    pwr = np.mean(I**2 + Q**2)
    noise_std = np.sqrt(pwr / (2 * snr))
    I += rng.normal(0, noise_std, N)
    Q += rng.normal(0, noise_std, N)

    # Quantise to int8 IQ (interleaved I,Q)
    scale = 100.0 / (max(np.abs(I).max(), np.abs(Q).max()) + 1e-9)
    I_i8 = np.clip(I * scale, -127, 127).astype(np.int8)
    Q_i8 = np.clip(Q * scale, -127, 127).astype(np.int8)
    iq = np.empty(2*N, dtype=np.int8)
    iq[0::2] = I_i8; iq[1::2] = Q_i8
    return "iq:" + iq.tobytes().hex()

# ── 4. IMAGE (float32 arr, arr: prefix) ──────────────────────────────────────

_IMG_CLASSES = [
    "horizontal_stripes",
    "vertical_stripes",
    "diagonal_gradient",
    "concentric_circles",
    "gaussian_blobs",
    "checkerboard",
]

def image_example(cls: int, noise_p: float = 0.04) -> str:
    """Generate a synthetic IMG_PIXELS×IMG_PIXELS float32 grayscale image."""
    P = IMG_PIXELS
    y, x = np.mgrid[0:P, 0:P].astype(np.float64)
    cy = P / 2; cx = P / 2

    if cls == 0:   # horizontal stripes
        freq = rng.uniform(2.0, 6.0)
        img = np.sin(2*np.pi*freq * y/P)
    elif cls == 1: # vertical stripes
        freq = rng.uniform(2.0, 6.0)
        img = np.sin(2*np.pi*freq * x/P)
    elif cls == 2: # diagonal gradient
        angle = rng.uniform(0, np.pi)
        img = np.cos(angle) * x/P + np.sin(angle) * y/P
    elif cls == 3: # concentric circles
        r = np.sqrt((x-cx)**2 + (y-cy)**2)
        freq = rng.uniform(1.5, 4.0)
        img = np.sin(2*np.pi*freq * r / (P/2))
    elif cls == 4: # Gaussian blobs at random positions
        n_blobs = int(rng.integers(3, 7))
        img = np.zeros((P, P))
        for _ in range(n_blobs):
            bx = rng.uniform(P*0.1, P*0.9)
            by = rng.uniform(P*0.1, P*0.9)
            sig = rng.uniform(P*0.05, P*0.15)
            img += np.exp(-((x-bx)**2 + (y-by)**2) / (2*sig**2))
        img /= (img.max() + 1e-9)
    else:          # checkerboard
        sq = int(rng.integers(4, 10))
        img = ((x.astype(int)//sq + y.astype(int)//sq) % 2).astype(np.float64)

    # Normalise to [-1, 1]
    img = img - img.mean()
    std = img.std()
    if std > 1e-9: img /= std
    img = np.clip(img, -3, 3) / 3.0

    # Add noise
    img += rng.normal(0, 0.05, (P, P))

    # Flatten and encode as float32 arr
    flat = img.flatten().astype(np.float32)
    return "arr:" + base64.b64encode(flat.tobytes()).decode()

# ── 5. STRUCTURED / TABULAR (float32 arr, arr: prefix) ───────────────────────

# 6 cluster centres in 64-dim space, well-separated
_STRUCT_CENTRES = rng.normal(0, 3.0, (N_CLASSES, 64)).astype(np.float32)
# Orthogonalise centres for max separation
for i in range(N_CLASSES):
    for j in range(i):
        _STRUCT_CENTRES[i] -= (np.dot(_STRUCT_CENTRES[i], _STRUCT_CENTRES[j]) /
                                (np.dot(_STRUCT_CENTRES[j], _STRUCT_CENTRES[j]) + 1e-9) *
                                _STRUCT_CENTRES[j])
    n = np.linalg.norm(_STRUCT_CENTRES[i])
    if n > 1e-9: _STRUCT_CENTRES[i] /= n
    _STRUCT_CENTRES[i] *= 5.0  # scale up for clear separation

def structured_example(cls: int, noise_p: float = 0.03) -> str:
    """Structured tabular feature vector — Gaussian cluster per class."""
    centre = _STRUCT_CENTRES[cls]
    noise_std = rng.uniform(0.3, 0.8)
    vec = centre + rng.normal(0, noise_std, 64).astype(np.float32)

    # Occasional cross-class contamination
    if rng.random() < noise_p:
        wc = int(rng.integers(0, N_CLASSES))
        while wc == cls: wc = int(rng.integers(0, N_CLASSES))
        vec = 0.85 * vec + 0.15 * _STRUCT_CENTRES[wc]

    return "arr:" + base64.b64encode(vec.tobytes()).decode()

# ── 6. VIDEO (arr: prefix — aggregated frame features) ───────────────────────

_VIDEO_CLASSES = [
    "static_scene",       # frames nearly identical
    "horizontal_pan",     # feature shifts horizontally per frame
    "vertical_pan",       # feature shifts vertically
    "zoom_in",            # spatial frequency increases per frame
    "periodic_motion",    # oscillates back and forth
    "random_flash",       # high temporal variance
]

def video_example(cls: int, noise_p: float = 0.04) -> str:
    """
    Generate VIDEO_FRAMES frames of VIDEO_FRAME_PX×VIDEO_FRAME_PX,
    aggregate into a single temporal feature vector via:
      [frame_mean, frame_std, temporal_diff_mean, temporal_diff_std, ...]
    Encode as float32 arr.
    """
    P  = VIDEO_FRAME_PX
    F  = VIDEO_FRAMES
    frames = np.zeros((F, P, P), dtype=np.float64)

    y, x = np.mgrid[0:P, 0:P].astype(np.float64)

    for f_idx in range(F):
        t = f_idx / max(F-1, 1)   # 0→1 over clip
        if cls == 0:   # static
            frames[f_idx] = np.sin(2*np.pi*3*y/P) + rng.normal(0, 0.05, (P,P))
        elif cls == 1: # horizontal pan
            shift = t * P * 0.5
            frames[f_idx] = np.sin(2*np.pi*3*(y-shift)/P) + rng.normal(0, 0.05, (P,P))
        elif cls == 2: # vertical pan
            shift = t * P * 0.5
            frames[f_idx] = np.sin(2*np.pi*3*(x-shift)/P) + rng.normal(0, 0.05, (P,P))
        elif cls == 3: # zoom in (increasing frequency)
            freq = 2.0 + t * 4.0
            r = np.sqrt((x-P/2)**2 + (y-P/2)**2)
            frames[f_idx] = np.sin(2*np.pi*freq*r/(P/2)) + rng.normal(0, 0.05, (P,P))
        elif cls == 4: # periodic motion (oscillate)
            shift = np.sin(2*np.pi*t*2) * P * 0.2
            frames[f_idx] = np.sin(2*np.pi*3*(y-shift)/P) + rng.normal(0, 0.05, (P,P))
        else:          # random flash
            if rng.random() < 0.3:
                frames[f_idx] = rng.normal(0, 1.0, (P,P))
            else:
                frames[f_idx] = rng.normal(0, 0.1, (P,P))

    # Temporal aggregation: compute statistics across frames
    # mean_frame, std_frame, diff_mean, diff_std per-pixel
    frame_mean = frames.mean(axis=0)    # (P,P)
    frame_std  = frames.std(axis=0)     # (P,P)
    diff_frames = np.abs(np.diff(frames, axis=0))  # (F-1,P,P)
    diff_mean  = diff_frames.mean(axis=0)  # (P,P)
    diff_std   = diff_frames.std(axis=0)   # (P,P)

    # Stack and flatten → 4 × P × P = 4 × 256 = 1024 floats
    agg = np.stack([frame_mean, frame_std, diff_mean, diff_std], axis=0).flatten().astype(np.float32)

    # L2-normalise
    n = np.linalg.norm(agg)
    if n > 1e-9: agg /= n

    if noise_p > 0 and rng.random() < noise_p:
        wc = int(rng.integers(0, N_CLASSES))
        while wc == cls: wc = int(rng.integers(0, N_CLASSES))
        wrong = np.frombuffer(
            base64.b64decode(video_example(wc, noise_p=0.0).split("arr:")[1]),
            dtype=np.float32).copy()
        agg = 0.85*agg + 0.15*wrong
        n = np.linalg.norm(agg)
        if n > 1e-9: agg /= n

    return "arr:" + base64.b64encode(agg.tobytes()).decode()

# ══════════════════════════════════════════════════════════════════════════════
# MODALITY REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

MODALITIES = {
    "text":       {
        "fn":      text_example,
        "classes": [f"topic_{chr(65+i)}" for i in range(N_CLASSES)],
        "desc":    "Bag-of-words keyword classification (6 scientific domains)",
        "encoder": "OmegaEncoder text path",
    },
    "audio":      {
        "fn":      audio_example,
        "classes": [f"sound_{n}" for n in _AUDIO_CLASSES],
        "desc":    "Int16 PCM waveforms — mel-scale spectral encoder",
        "encoder": "pcm: → _encode_audio",
    },
    "rf_iq":      {
        "fn":      rf_example,
        "classes": _RF_CLASSES,
        "desc":    "Int8 IQ RF signals — phase-invariant spectral encoder",
        "encoder": "iq: → _encode_iq",
    },
    "image":      {
        "fn":      image_example,
        "classes": [f"img_{n}" for n in _IMG_CLASSES],
        "desc":    f"Float32 {IMG_PIXELS}×{IMG_PIXELS} grayscale images — Omega arr encoder",
        "encoder": "arr: → encode_array → float32 Omega",
    },
    "structured": {
        "fn":      structured_example,
        "classes": [f"cluster_{i}" for i in range(N_CLASSES)],
        "desc":    "64-dim float32 Gaussian clusters — tabular features",
        "encoder": "arr: → encode_array → float32 Omega",
    },
    "video":      {
        "fn":      video_example,
        "classes": [f"motion_{n}" for n in _VIDEO_CLASSES],
        "desc":    f"Temporal {VIDEO_FRAMES}×{VIDEO_FRAME_PX}² frame aggregates",
        "encoder": "arr: → encode_array → float32 Omega (temporal stats)",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def make_dataset(modality_fn, class_labels, examples_pc=EXAMPLES_PC, n_probe=N_PROBE):
    """
    Returns:
        train_pairs  — [(input_str, label), ...]  shuffled
        probe_pairs  — [(input_str, label), ...]  held-out
        wrong_pool   — {label: [other_class_inputs, ...]}  for window negatives
    """
    all_pairs = []
    for cls_idx, label in enumerate(class_labels):
        for _ in range(examples_pc):
            all_pairs.append((modality_fn(cls_idx), label))

    idxs = rng.permutation(len(all_pairs))
    all_pairs = [all_pairs[i] for i in idxs]

    # Hold out last n_probe examples as probe set
    probe  = all_pairs[-n_probe:]
    train  = all_pairs[:-n_probe]

    # Build wrong-class pool for window negatives
    pool = defaultdict(list)
    for inp, label in train:
        pool[label].append(inp)

    return train, probe, dict(pool)

def get_neg(label, pool, k=2):
    negs = []
    for lbl, examples in pool.items():
        if lbl != label and examples:
            negs.append(examples[rng.integers(0, len(examples))])
        if len(negs) >= k:
            break
    return negs

# ══════════════════════════════════════════════════════════════════════════════
# LIVE METRIC PROBE
# ══════════════════════════════════════════════════════════════════════════════

def probe_metrics(cypha: Cypha, probe_pairs: List[Tuple[str,str]]) -> Dict:
    """Measure accuracy and margin distribution on held-out probe set."""
    correct = 0; margins = []
    for inp, label in probe_pairs:
        try:
            pred, conf = cypha.infer(inp, verbose=False)
            if pred == label: correct += 1
            margins.append(float(conf))
        except Exception:
            pass
    n = len(probe_pairs)
    return {
        "probe_acc":   round(correct / max(n, 1), 4),
        "mean_margin": round(float(np.mean(margins)) if margins else 0., 4),
        "margin_std":  round(float(np.std(margins))  if margins else 0., 4),
        "n_anchors":   cypha.memory.n,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE MODALITY TRAINING RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_modality(name: str, info: Dict) -> Dict:
    """Train Cypha on one modality and return the full learning trajectory."""
    print(f"\n{'═'*68}")
    print(f"  MODALITY: {name.upper()}")
    print(f"  {info['desc']}")
    print(f"  Encoder: {info['encoder']}")
    print(f"{'─'*68}")
    print(f"  Generating {N_CLASSES} classes × {EXAMPLES_PC} examples...")

    t_gen = time.time()
    train, probe, pool = make_dataset(info['fn'], info['classes'])
    print(f"  Generated in {time.time()-t_gen:.1f}s  ({len(train)} train, {len(probe)} probe)")

    print(f"  Initialising fresh Cypha instance...")
    cypha = Cypha(feature_dim=FEATURE_DIM, resonance_dim=RESONANCE_DIM)

    trajectory = []
    step_times = []
    t_start = time.time()

    for step_i, (inp, label) in enumerate(train):
        t0 = time.perf_counter()
        negs = get_neg(label, pool, k=2)
        cypha.train_step(inp, label, negatives=negs)
        step_times.append((time.perf_counter() - t0) * 1000)

        n = step_i + 1

        if n % PROBE_EVERY == 0 or n == 1:
            metrics = probe_metrics(cypha, probe)
            metrics["n"] = n
            trajectory.append(metrics)

            elapsed = time.time() - t_start
            sps = n / elapsed
            eta_s = (len(train) - n) / max(sps, 0.01)

            print(f"  n={n:>6,}  acc={metrics['probe_acc']:.3f}  "
                  f"margin={metrics['mean_margin']:.4f}  "
                  f"σ={metrics['margin_std']:.4f}  "
                  f"anchors={metrics['n_anchors']:>4}  "
                  f"[{sps:.0f} sps  ETA {eta_s/60:.1f}m]")

    # Final probe
    final = probe_metrics(cypha, probe)
    final["n"] = len(train)
    if not trajectory or trajectory[-1]["n"] != len(train):
        trajectory.append(final)

    elapsed_total = time.time() - t_start
    t50  = float(np.percentile(step_times, 50))
    t95  = float(np.percentile(step_times, 95))

    # ── Detect phase transition ───────────────────────────────────────────────
    accs  = np.array([p["probe_acc"]  for p in trajectory])
    sigs  = np.array([p["margin_std"] for p in trajectory])
    ns    = np.array([p["n"]          for p in trajectory])

    transition_n   = None
    sig_at_trans   = None
    post_behaviour = "no_transition"

    # Find first point where acc >= 0.95 and σ drops significantly
    if accs.max() >= 0.90:
        # Find the biggest single-step drop in σ
        sig_diffs = np.diff(sigs)
        if len(sig_diffs) > 0:
            min_diff_idx = int(np.argmin(sig_diffs))
            if sig_diffs[min_diff_idx] < -0.005:   # threshold: meaningful drop
                transition_n = int(ns[min_diff_idx])
                sig_at_trans = float(sigs[min_diff_idx + 1])
                # Post-transition trend
                post_ns  = ns[min_diff_idx+1:]
                post_acc = accs[min_diff_idx+1:]
                if len(post_acc) > 2:
                    slope = np.polyfit(post_ns, post_acc, 1)[0]
                    if slope > 1e-6:
                        post_behaviour = "improving"
                    elif slope < -1e-5:
                        post_behaviour = "degrading"   # counterproductive zone
                    else:
                        post_behaviour = "stable"

    print(f"\n  ── Run complete ──")
    print(f"  Total: {elapsed_total:.1f}s  |  step: p50={t50:.1f}ms p95={t95:.1f}ms")
    print(f"  Final: acc={final['probe_acc']:.3f}  margin={final['mean_margin']:.4f}  "
          f"σ={final['margin_std']:.4f}  anchors={final['n_anchors']}")
    if transition_n:
        print(f"  Transition at n={transition_n:,}  (σ={sig_at_trans:.4f})  → {post_behaviour}")
    else:
        print(f"  No transition detected  (max_acc={accs.max():.3f})")

    return {
        "modality":      name,
        "description":   info["desc"],
        "encoder":       info["encoder"],
        "classes":       info["classes"],
        "n_train":       len(train),
        "n_probe":       len(probe),
        "trajectory":    trajectory,
        "transition_n":  transition_n,
        "sig_at_trans":  sig_at_trans,
        "post_behaviour": post_behaviour,
        "final":         final,
        "timing_ms":     {"p50": round(t50,3), "p95": round(t95,3)},
        "elapsed_s":     round(elapsed_total, 2),
    }

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(results: List[Dict], path: str):
    SEP  = "═" * 72
    SEP2 = "─" * 70
    lines = []
    W = lines.append

    W(SEP)
    W("  CYPHA MULTI-MODAL LEARNING PROFILE")
    W(f"  {N_CLASSES} classes × {EXAMPLES_PC} examples/class  |  probe every {PROBE_EVERY} steps")
    W(SEP)

    # ── Summary table ─────────────────────────────────────────────────────────
    W("")
    W(f"  {'Modality':<14}  {'FinalAcc':>9}  {'FinalMargin':>12}  {'FinalSigma':>11}  "
      f"{'TransitN':>9}  {'PostTrans':>11}  {'p50ms':>7}")
    W(f"  {SEP2}")
    for r in results:
        f   = r["final"]
        tn  = f"{r['transition_n']:,}" if r["transition_n"] else "—"
        W(f"  {r['modality']:<14}  {f['probe_acc']:>9.3f}  {f['mean_margin']:>12.4f}  "
          f"{f['margin_std']:>11.4f}  {tn:>9}  {r['post_behaviour']:>11}  "
          f"{r['timing_ms']['p50']:>7.1f}")
    W("")

    # ── Per-modality deep dive ────────────────────────────────────────────────
    for r in results:
        W(SEP2)
        W(f"  {r['modality'].upper()}")
        W(f"  {r['description']}")
        W(f"  Encoder: {r['encoder']}")
        W(SEP2)

        traj = r["trajectory"]
        ns   = [p["n"]           for p in traj]
        accs = [p["probe_acc"]   for p in traj]
        sigs = [p["margin_std"]  for p in traj]
        marg = [p["mean_margin"] for p in traj]
        anch = [p["n_anchors"]   for p in traj]

        # Show key trajectory points
        show_idx = list(range(0, len(traj), max(1, len(traj)//10))) + [len(traj)-1]
        show_idx = sorted(set(show_idx))
        W(f"  {'n':>8}  {'acc':>6}  {'margin':>8}  {'sigma':>7}  {'anchors':>8}")
        W(f"  {'─'*50}")
        for i in show_idx:
            p = traj[i]
            W(f"  {p['n']:>8,}  {p['probe_acc']:>6.3f}  "
              f"{p['mean_margin']:>8.4f}  {p['margin_std']:>7.4f}  {p['n_anchors']:>8}")

        # Transition analysis
        W("")
        if r["transition_n"]:
            W(f"  PHASE TRANSITION:")
            W(f"    n = {r['transition_n']:,}  (at {100*r['transition_n']/r['n_train']:.0f}% of training)")
            W(f"    σ at transition: {r['sig_at_trans']:.4f}")
            W(f"    Post-transition: {r['post_behaviour']}")
            # Find acc at transition
            trans_probe = [p for p in traj if p["n"] <= r["transition_n"]]
            if trans_probe:
                W(f"    Acc at transition: {trans_probe[-1]['probe_acc']:.3f}")
        else:
            W(f"  NO PHASE TRANSITION DETECTED")
            W(f"  Max acc: {max(accs):.3f}   Learning likely incomplete or hard wall.")
            if max(accs) < 0.5:
                W(f"  *** HARD WALL: Cypha cannot separate this modality's classes. ***")
            elif max(accs) < 0.80:
                W(f"  *** PARTIAL LEARNING: Classes are partially separable but noisy. ***")

        W(f"  Timing: p50={r['timing_ms']['p50']:.1f}ms  p95={r['timing_ms']['p95']:.1f}ms  "
          f"total={r['elapsed_s']:.1f}s")
        W("")

    # ── Cross-modality interpretation ─────────────────────────────────────────
    W(SEP)
    W("  INTERPRETATION — GAME THEORY PARALLEL")
    W(SEP)
    W("""
  From Phase 9 analysis: Cypha is a LVQ2.1 system with k=1 prototype per class.
  Learning proceeds as: PROTOTYPE DRIFT → PHASE TRANSITION → REFINEMENT.
  The counterproductive zone arises from LVQ2.1 OVERSHOOT in noisy/complex spaces.

  MODALITY DIFFICULTY RANKING (by transition_n / n_train ratio):
    Lower ratio  = easier (transition reached early, little training needed)
    Higher ratio = harder (prototype needs more examples to find correct position)
    None         = hard wall or insufficient training budget
    Degrading    = counterproductive zone present (game-data equivalent)

  MAPPING TO GAME THEORY:
    text       → clean, well-separated classes         = "easy domain"
    structured → Gaussian clusters, fastest learner    = "synthetic game positions"
    image      → spatial structure, medium noise       = "poker" (defined classes)
    audio      → frequency discrimination              = "poker" difficulty range
    rf_iq      → very similar signals, hard boundary   = "chess" (hard to separate)
    video      → temporal + spatial complexity         = "Go" (high-dim, noisy)
""")
    W(SEP)

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(f"\n  → Report written: {path}")
    print(text)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cypha Multi-Modal Profiler")
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES.keys()),
                        choices=list(MODALITIES.keys()),
                        help="Which modalities to run (default: all 6)")
    parser.add_argument("--examples-pc", type=int,   default=EXAMPLES_PC)
    parser.add_argument("--probe-every",  type=int,   default=PROBE_EVERY)
    parser.add_argument("--n-probe",      type=int,   default=N_PROBE)
    parser.add_argument("--n-classes",    type=int,   default=N_CLASSES)
    args = parser.parse_args()

    EXAMPLES_PC = args.examples_pc
    PROBE_EVERY  = args.probe_every
    N_PROBE      = args.n_probe

    print(f"\n{'═'*68}")
    print(f"  CYPHA MULTI-MODAL PROFILER")
    print(f"  Modalities: {', '.join(args.modalities)}")
    print(f"  {N_CLASSES} classes × {EXAMPLES_PC} examples/class  probe_every={PROBE_EVERY}")
    print(f"{'═'*68}\n")

    all_results = []
    total_t = time.time()

    for mod_name in args.modalities:
        result = run_modality(mod_name, MODALITIES[mod_name])
        all_results.append(result)

    # ── Write JSON ──────────────────────────────────────────────────────────
    json_path = "cypha_multimodal_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  → JSON written: {json_path}")

    # ── Write report ────────────────────────────────────────────────────────
    write_report(all_results, "cypha_multimodal_report.txt")

    print(f"\n  Total wall time: {(time.time()-total_t)/60:.1f} minutes")
    print(f"  Done.\n")
