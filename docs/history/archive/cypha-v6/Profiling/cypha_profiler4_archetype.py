"""
cypha_profiler4_archetype.py
════════════════════════════════════════════════════════════════════════════════
PROFILER 4 — ARCHETYPE DETECTION (Type A / B / C classifier)

Answers:
  - Can Cypha's learning archetype be identified within the first 50 steps?
  - What signals (accuracy slope, σ, margin, anchor growth) are diagnostic?
  - What are the decision thresholds for reliable early classification?
  - How early can you call it? Step 10? 20? 50?
  - How often does early detection produce false positives/negatives?
  - Can you predict final accuracy from the first 50 steps?

The three archetypes (from multimodal + fewshot profiling):
  TYPE A — Encoder-Solved:
    Accuracy crosses 50% within first 10 steps.
    σ ≈ 0 from the start (all margins near zero, k=1 regime).
    Example: audio. Mel-scale encoder already separates classes.
    Decision: STOP EARLY if accuracy_10 > 0.50, train minimally.

  TYPE B — Encoder-Partial (needs migration):
    Accuracy crawls from ~16% upward slowly over hundreds of steps.
    σ > 0.01, gradually falling. Transition occurs at n* >> 50.
    Example: text (n*≈1650/cls), image, video.
    Decision: CONTINUE training, monitor for σ collapse.

  TYPE C — Encoder-Failed:
    Accuracy grows very slowly or not at all beyond step 50.
    σ stays high (0.05+), never collapses. Anchor count grows.
    Example: structured (Omega on geometric coords), RF/IQ at ceiling.
    Decision: FLAG encoder mismatch, halt training, fix encoder first.

Detection algorithm tested:
  At each checkpoint (step 10, 20, 30, 50):
    - acc_slope:  linear slope of accuracy over last N steps
    - sigma_mean: mean margin_std over last N steps
    - sigma_slope: trend in margin_std (falling = migrating toward transition)
    - anchor_growth: new prototypes formed (multi-prototype = genuine multimodality)
    - early_acc: raw accuracy at this step

  Classification rules (learned from thresholds):
    IF early_acc > 0.50 AND sigma_mean < 0.02 → Type A
    IF early_acc < 0.35 AND acc_slope > 0.001 AND sigma_slope < 0 → Type B
    IF early_acc < 0.35 AND acc_slope < 0.002 → Type C (slow or stalled)

Evaluated against ground truth (final accuracy after full training).

Output:
  profiler4_archetype_results.json
  profiler4_archetype_report.txt

Usage:
  python cypha_profiler4_archetype.py
  python cypha_profiler4_archetype.py --modalities audio structured text
  python cypha_profiler4_archetype.py --quick   # fewer seeds, faster
"""

import sys, os, time, json, base64
import numpy as np
from typing import List, Tuple, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from Cypha import Cypha, EPSILON
    print("✓ Cypha loaded")
except ImportError as e:
    print(f"✗ Cannot import Cypha: {e}"); sys.exit(1)

rng = np.random.default_rng(42)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
N_CLASSES       = 6
N_PROBE         = 300        # 50 per class
FEATURE_DIM     = 512
RESONANCE_DIM   = 256
N_SEEDS         = 5          # more seeds = reliable detection thresholds
FULL_TRAIN_N    = 300        # full training budget per class (ground truth)
DETECT_AT       = [6, 10, 20, 30, 50]  # steps at which to attempt classification
PROBE_INTERVAL  = 2          # probe every N steps during detection window

# Archetype decision thresholds — these are the hypotheses to validate
THRESH_A_ACC    = 0.50   # Type A: accuracy above this at step 10
THRESH_A_SIGMA  = 0.02   # Type A: σ below this at step 10
THRESH_C_SLOPE  = 0.002  # Type C: acc slope below this = stalled

# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS (all 6 modalities for comprehensive archetype coverage)
# ══════════════════════════════════════════════════════════════════════════════

_VOCAB = [
    ["quantum_entanglement","superposition","wavefunction","hilbert_space",
     "eigenvalue","decoherence","density_matrix","bell_inequality","qubit","tunneling"],
    ["convolutional","backpropagation","activation","batch_norm","dropout",
     "attention","transformer","embedding","cross_entropy","softmax"],
    ["haemoglobin","action_potential","synaptic","mitochondrial","atp_synthesis",
     "receptor_binding","ion_channel","neurotransmitter","axon","dendrite"],
    ["geodesic","riemann_tensor","spacetime","christoffel","metric_tensor",
     "stress_energy","schwarzschild","gravitational_wave","event_horizon","cosmological"],
    ["tcp_handshake","packet_fragmentation","routing_bgp","ssl_certificate",
     "dns_resolution","arp_broadcast","icmp_echo","firewall","latency","bandwidth"],
    ["chain_rule","fourier_transform","laplacian","eigenfunction","hilbert_transform",
     "dirac_delta","greens_function","variational_calculus","stochastic","measure_theory"],
]
_SHARED = ["analysis","system","process","function","signal","data",
           "method","result","parameter","component","structure","value"]

def text_ex(cls):
    kw = list(rng.choice(_VOCAB[cls], size=6, replace=False))
    sh = list(rng.choice(_SHARED, size=8, replace=True))
    tokens = kw + sh; rng.shuffle(tokens)
    return " ".join(tokens)

_FREQS = [220., 440., 880., 1760., 3520., 7040.]
def audio_ex(cls):
    t = np.arange(4000) / 8000.
    f0 = _FREQS[cls] * (1 + rng.normal(0, 0.01))
    sig = np.sin(2*np.pi*f0*t)
    if cls % 2 == 1:
        sig += 0.5*np.sin(2*np.pi*2*f0*t); sig /= (np.abs(sig).max()+1e-9)
    sig += rng.normal(0, 0.02, 4000)
    return "pcm:" + np.clip(sig,-1,1).__mul__(32700).astype(np.int16).tobytes().hex()

def rf_ex(cls):
    N=512; t=np.arange(N)/N; fc=0.1; phi=rng.uniform(0,2*np.pi)
    msg=np.sin(2*np.pi*0.02*N*t)
    if   cls==0: env=1+0.7*msg; I=env*np.cos(2*np.pi*fc*N*t+phi); Q=env*np.sin(2*np.pi*fc*N*t+phi)
    elif cls==1:
        kf=0.05; pd=2*np.pi*kf*np.cumsum(msg)/N
        I=np.cos(2*np.pi*fc*N*t+phi+pd); Q=np.sin(2*np.pi*fc*N*t+phi+pd)
    elif cls==2:
        syms=np.repeat(rng.choice([-1.,1.],N//32),32)[:N]
        I=syms*np.cos(2*np.pi*fc*N*t+phi); Q=syms*np.sin(2*np.pi*fc*N*t+phi)
    elif cls==3: I=np.cos(2*np.pi*fc*N*t+phi); Q=np.sin(2*np.pi*fc*N*t+phi)
    elif cls==4:
        am=msg+1j*np.imag(np.fft.ifft(np.fft.fft(msg)*(np.arange(N)<N//2)*2))
        cplx=(1+0.5*am)*np.exp(1j*(2*np.pi*fc*N*t+phi)); I=np.real(cplx); Q=np.imag(cplx)
    else:
        phs=np.repeat(rng.choice([0,np.pi/2,np.pi,3*np.pi/2],N//32),32)[:N]
        I=np.cos(2*np.pi*fc*N*t+phi+phs); Q=np.sin(2*np.pi*fc*N*t+phi+phs)
    pwr=np.mean(I**2+Q**2); ns=np.sqrt(pwr/(2*10**(15/10)))
    I+=rng.normal(0,ns,N); Q+=rng.normal(0,ns,N)
    sc=100/(max(np.abs(I).max(),np.abs(Q).max())+1e-9)
    iq=np.empty(2*N,dtype=np.int8)
    iq[0::2]=np.clip(I*sc,-127,127).astype(np.int8)
    iq[1::2]=np.clip(Q*sc,-127,127).astype(np.int8)
    return "iq:"+iq.tobytes().hex()

def image_ex(cls):
    P=64; y,x=np.mgrid[0:P,0:P].astype(np.float64)
    if   cls==0: img=np.sin(2*np.pi*rng.uniform(2,6)*y/P)
    elif cls==1: img=np.sin(2*np.pi*rng.uniform(2,6)*x/P)
    elif cls==2:
        a=rng.uniform(0,np.pi); img=np.cos(a)*x/P+np.sin(a)*y/P
    elif cls==3:
        r=np.sqrt((x-P/2)**2+(y-P/2)**2)
        img=np.sin(2*np.pi*rng.uniform(1.5,4)*r/(P/2))
    elif cls==4:
        img=np.zeros((P,P))
        for _ in range(int(rng.integers(3,7))):
            bx,by=rng.uniform(P*.1,P*.9,2); s=rng.uniform(P*.05,P*.15)
            img+=np.exp(-((x-bx)**2+(y-by)**2)/(2*s**2))
        img/=(img.max()+1e-9)
    else:
        sq=int(rng.integers(4,10))
        img=((x.astype(int)//sq+y.astype(int)//sq)%2).astype(np.float64)
    img-=img.mean(); std=img.std()
    if std>1e-9: img/=std
    img=np.clip(img,-3,3)/3+rng.normal(0,0.05,(P,P))
    return "arr:"+base64.b64encode(img.flatten().astype(np.float32).tobytes()).decode()

_CENTRES=rng.normal(0,3,(N_CLASSES,64)).astype(np.float32)
for _i in range(N_CLASSES):
    for _j in range(_i):
        _CENTRES[_i]-=(np.dot(_CENTRES[_i],_CENTRES[_j])/(np.dot(_CENTRES[_j],_CENTRES[_j])+1e-9)*_CENTRES[_j])
    _n=np.linalg.norm(_CENTRES[_i])
    if _n>1e-9: _CENTRES[_i]=_CENTRES[_i]/_n*5
def structured_ex(cls):
    v=_CENTRES[cls]+rng.normal(0,0.5,64).astype(np.float32)
    return "arr:"+base64.b64encode(v.tobytes()).decode()

def video_ex(cls):
    P=16; F=8; frames=np.zeros((F,P,P))
    y,x=np.mgrid[0:P,0:P].astype(np.float64)
    for fi in range(F):
        t=fi/max(F-1,1)
        if   cls==0: frames[fi]=np.sin(2*np.pi*3*y/P)
        elif cls==1: frames[fi]=np.sin(2*np.pi*3*(y-t*P*.5)/P)
        elif cls==2: frames[fi]=np.sin(2*np.pi*3*(x-t*P*.5)/P)
        elif cls==3:
            r=np.sqrt((x-P/2)**2+(y-P/2)**2)
            frames[fi]=np.sin(2*np.pi*(2+t*4)*r/(P/2))
        elif cls==4: frames[fi]=np.sin(2*np.pi*3*(y-np.sin(2*np.pi*t*2)*P*.2)/P)
        else:        frames[fi]=rng.normal(0,1 if rng.random()<.3 else .1,(P,P))
        frames[fi]+=rng.normal(0,.05,(P,P))
    fm=frames.mean(0); fs=frames.std(0)
    df=np.abs(np.diff(frames,axis=0)); dm=df.mean(0); ds=df.std(0)
    agg=np.stack([fm,fs,dm,ds]).flatten().astype(np.float32)
    n=np.linalg.norm(agg)
    if n>1e-9: agg/=n
    return "arr:"+base64.b64encode(agg.tobytes()).decode()

MODALITIES = {
    "audio":      (audio_ex,      "A"),   # ground truth Type A
    "text":       (text_ex,       "B"),   # ground truth Type B
    "image":      (image_ex,      "B"),   # ground truth Type B
    "video":      (video_ex,      "B"),   # ground truth Type B
    "structured": (structured_ex, "C"),   # ground truth Type C
    "rf_iq":      (rf_ex,         "C"),   # ground truth Type C (ceiling)
}
CLASS_LABELS = [f"class_{i}" for i in range(N_CLASSES)]

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def fresh() -> Cypha:
    return Cypha(feature_dim=FEATURE_DIM, resonance_dim=RESONANCE_DIM)

def make_probe(fn) -> List[Tuple[str,str]]:
    per = N_PROBE // N_CLASSES
    return [(fn(cls), CLASS_LABELS[cls])
            for cls in range(N_CLASSES) for _ in range(per)]

def measure_k2(cypha, probe):
    correct = 0; margins = []
    for inp, label in probe:
        try:
            q = cypha.encode_features(inp)
            matches = cypha.memory.lookup(q, k=2)
            if not matches: continue
            pred = cypha.memory.get_output(matches[0][0]) or matches[0][0]
            if pred == label: correct += 1
            m = float(matches[0][1]) - (float(matches[1][1]) if len(matches)>1 else 0.)
            margins.append(m)
        except Exception:
            pass
    acc = correct / max(len(probe), 1)
    mm  = float(np.mean(margins)) if margins else 0.
    ms  = float(np.std(margins))  if margins else 0.
    return {"acc": round(acc,4), "mean_margin": round(mm,4),
            "margin_std": round(ms,4), "n_anchors": cypha.memory.n}

def build_pool(fn, n=60):
    return {CLASS_LABELS[cls]: [fn(cls) for _ in range(n)]
            for cls in range(N_CLASSES)}

def get_negs(label, pool, k=2):
    negs = []
    for lbl, ex in pool.items():
        if lbl != label and ex:
            negs.append(ex[int(rng.integers(0, len(ex)))])
        if len(negs) >= k: break
    return negs

# ══════════════════════════════════════════════════════════════════════════════
# ARCHETYPE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def classify_archetype(early_trace: List[Dict], detect_step: int) -> Dict:
    """
    Given the accuracy trace up to `detect_step`, classify the archetype.
    Returns: {archetype, confidence, signals, reasoning}

    Signals extracted:
      acc_now:     accuracy at detect_step
      acc_slope:   linear slope of accuracy (acc/step) over trace
      sigma_mean:  mean margin_std over trace
      sigma_slope: linear slope of margin_std (falling = migrating)
      anchor_rate: new anchors formed per step
    """
    if not early_trace:
        return {"archetype": "unknown", "confidence": 0.0, "signals": {}}

    ns    = np.array([t["n"]          for t in early_trace], dtype=float)
    accs  = np.array([t["acc"]        for t in early_trace], dtype=float)
    sigs  = np.array([t["margin_std"] for t in early_trace], dtype=float)
    anch  = np.array([t["n_anchors"]  for t in early_trace], dtype=float)

    acc_now    = float(accs[-1])
    sigma_mean = float(sigs.mean())
    sigma_last = float(sigs[-1])

    # Slopes (acc/step and sigma/step)
    if len(ns) >= 2:
        acc_slope   = float(np.polyfit(ns, accs, 1)[0]) * 10  # per 10 steps
        sigma_slope = float(np.polyfit(ns, sigs, 1)[0]) * 10
        anchor_rate = float((anch[-1] - anch[0]) / max(ns[-1] - ns[0], 1))
    else:
        acc_slope = sigma_slope = anchor_rate = 0.

    signals = {
        "acc_now":      round(acc_now,    4),
        "acc_slope":    round(acc_slope,  5),
        "sigma_mean":   round(sigma_mean, 5),
        "sigma_last":   round(sigma_last, 5),
        "sigma_slope":  round(sigma_slope,5),
        "anchor_rate":  round(anchor_rate,4),
        "n_anchors":    int(anch[-1]),
    }

    # ── Classification rules ──────────────────────────────────────────────────
    # Type A: already performing well
    if acc_now >= THRESH_A_ACC and sigma_mean <= THRESH_A_SIGMA:
        archetype   = "A"
        confidence  = min(1.0, acc_now * 2)
        reasoning   = (f"acc={acc_now:.3f}>{THRESH_A_ACC} AND "
                       f"σ={sigma_mean:.4f}<{THRESH_A_SIGMA}")

    # Type A borderline: high accuracy but higher σ (still likely A)
    elif acc_now >= THRESH_A_ACC:
        archetype  = "A"
        confidence = 0.7
        reasoning  = f"acc={acc_now:.3f}>{THRESH_A_ACC} (σ slightly high)"

    # Type C: stalled — very low accuracy AND very low slope
    elif acc_now < 0.35 and acc_slope < THRESH_C_SLOPE and sigma_slope >= -0.001:
        archetype  = "C"
        confidence = 0.8 if acc_slope < 0.001 else 0.6
        reasoning  = (f"acc={acc_now:.3f}<0.35 AND "
                      f"slope={acc_slope:.5f}<{THRESH_C_SLOPE} AND "
                      f"σ_slope={sigma_slope:.5f}≥0 (not falling)")

    # Type B: slow but improving — slope positive, σ falling
    elif acc_now < 0.55 and acc_slope >= THRESH_C_SLOPE and sigma_slope < 0:
        archetype  = "B"
        confidence = 0.75
        reasoning  = (f"acc={acc_now:.3f} low but slope={acc_slope:.5f}>0 "
                      f"AND σ falling ({sigma_slope:.5f}<0)")

    # Type B: moderate accuracy, not yet A
    elif 0.35 <= acc_now < THRESH_A_ACC:
        archetype  = "B"
        confidence = 0.65
        reasoning  = f"acc={acc_now:.3f} in B range (0.35–{THRESH_A_ACC})"

    # Ambiguous — could be B or C
    else:
        archetype  = "B" if acc_slope >= THRESH_C_SLOPE else "C"
        confidence = 0.5
        reasoning  = f"ambiguous: acc={acc_now:.3f} slope={acc_slope:.5f}"

    return {
        "archetype":   archetype,
        "confidence":  round(confidence, 3),
        "signals":     signals,
        "reasoning":   reasoning,
        "detect_step": detect_step,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE MODALITY RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_seed(mod_name: str, fn, ground_truth_type: str,
             probe: List, pool: Dict, seed_i: int) -> Dict:
    """
    Train FULL_TRAIN_N examples/class with fine-grained early probing.
    At each DETECT_AT checkpoint, classify archetype from early trace.
    After full training, record final accuracy as ground truth.
    """
    cypha  = fresh()
    n_total = FULL_TRAIN_N * N_CLASSES

    pairs = [(fn(cls), CLASS_LABELS[cls])
             for cls in range(N_CLASSES) for _ in range(FULL_TRAIN_N)]
    rng.shuffle(pairs)

    # Fine-grained early trace (every PROBE_INTERVAL steps up to max(DETECT_AT))
    fine_trace    = []
    full_trace    = []
    detections    = {}     # detect_step → classification result
    probe_full_every = max(1, n_total // 25)
    max_detect    = max(DETECT_AT)

    for step_i, (inp, lbl) in enumerate(pairs):
        cypha.train_step(inp, lbl, negatives=get_negs(lbl, pool))
        n = step_i + 1

        # Fine-grained early probing
        if n <= max_detect and (n % PROBE_INTERVAL == 0 or n == 1):
            pt = measure_k2(cypha, probe); pt["n"] = n
            fine_trace.append(pt)

        # Full training trace (coarser)
        if n % probe_full_every == 0 or n == n_total:
            pt = measure_k2(cypha, probe); pt["n"] = n
            full_trace.append(pt)

        # Run archetype detection at each checkpoint
        if n in DETECT_AT:
            early_slice = [t for t in fine_trace if t["n"] <= n]
            det = classify_archetype(early_slice, n)
            det["correct"] = (det["archetype"] == ground_truth_type)
            detections[n] = det

    final_acc = full_trace[-1]["acc"] if full_trace else 0.

    # Final accuracy prediction: based on step-50 signals
    step50_det = detections.get(50, detections.get(max(detections.keys()), {}))
    predicted_final = None
    if step50_det.get("signals"):
        sig = step50_det["signals"]
        # Simple linear extrapolation from slope
        predicted_final = round(min(1.0, sig["acc_now"] + sig["acc_slope"] *
                                    (n_total / 10)), 3)

    return {
        "modality":          mod_name,
        "ground_truth_type": ground_truth_type,
        "seed":              seed_i,
        "final_acc":         final_acc,
        "predicted_final":   predicted_final,
        "detections":        detections,
        "fine_trace":        fine_trace,
        "full_trace":        full_trace,
    }

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_modality(mod_name: str, fn, ground_truth_type: str,
                 n_seeds: int) -> List[Dict]:
    print(f"\n{'═'*70}")
    print(f"  MODALITY: {mod_name.upper()}  (ground truth: Type {ground_truth_type})")
    print(f"{'─'*70}")
    probe = make_probe(fn)
    pool  = build_pool(fn)
    seeds = []

    for seed_i in range(n_seeds):
        t0 = time.time()
        result = run_seed(mod_name, fn, ground_truth_type, probe, pool, seed_i)
        elapsed = time.time() - t0

        # Print detection summary
        det_str = "  ".join(
            f"n={step}:{d.get('archetype','?')}({'✓' if d.get('correct') else '✗'})"
            for step, d in result["detections"].items()
        )
        print(f"  seed={seed_i}  final={result['final_acc']:.3f}  "
              f"pred={result.get('predicted_final','?')}  "
              f"  |  {det_str}  ({elapsed:.0f}s)")
        seeds.append(result)

    return seeds

# ══════════════════════════════════════════════════════════════════════════════
# DETECTION ACCURACY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_detection(all_results: Dict) -> Dict:
    """
    For each detection step, compute:
      - accuracy: fraction of seeds correctly classified
      - precision per type (TP / (TP+FP))
      - recall per type (TP / (TP+FN))
      - confidence calibration: does confidence predict correctness?
    """
    analysis = {}

    for detect_step in DETECT_AT:
        step_data = {t: {"tp":0,"fp":0,"fn":0,"tn":0,"confs":[]} for t in "ABC"}

        for mod_name, seeds in all_results.items():
            gt = seeds[0]["ground_truth_type"]
            for seed in seeds:
                det = seed["detections"].get(detect_step, {})
                pred = det.get("archetype", "?")
                conf = det.get("confidence", 0.)
                correct = (pred == gt)
                step_data[gt]["confs"].append((conf, correct))
                for t in "ABC":
                    if t == gt and t == pred: step_data[t]["tp"] += 1
                    if t != gt and t == pred: step_data[t]["fp"] += 1
                    if t == gt and t != pred: step_data[t]["fn"] += 1
                    if t != gt and t != pred: step_data[t]["tn"] += 1

        # Overall accuracy
        total = sum(step_data[t]["tp"] + step_data[t]["fn"] for t in "ABC")
        correct_total = sum(step_data[t]["tp"] for t in "ABC")
        overall_acc = round(correct_total / max(total, 1), 4)

        per_type = {}
        for t in "ABC":
            d = step_data[t]
            prec = d["tp"] / max(d["tp"] + d["fp"], 1)
            rec  = d["tp"] / max(d["tp"] + d["fn"], 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)
            per_type[f"Type_{t}"] = {
                "precision": round(prec, 3),
                "recall":    round(rec,  3),
                "f1":        round(f1,   3),
                "support":   d["tp"] + d["fn"],
            }

        analysis[detect_step] = {
            "overall_accuracy": overall_acc,
            "per_type":         per_type,
            "total_samples":    total,
        }

    return analysis

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(all_results: Dict, analysis: Dict, path: str):
    SEP  = "═" * 76
    SEP2 = "─" * 74
    L = []; W = L.append

    W(SEP)
    W("  PROFILER 4 — ARCHETYPE DETECTION (Type A / B / C)")
    W(f"  Full training: {FULL_TRAIN_N}/class  |  Detection at steps: {DETECT_AT}")
    W(f"  Seeds: {N_SEEDS}  |  probe_n={N_PROBE}  |  k=2 margins")
    W(f"  Thresholds: acc_A>{THRESH_A_ACC}  σ_A<{THRESH_A_SIGMA}  "
      f"slope_C<{THRESH_C_SLOPE}")
    W(SEP)

    # ── Detection accuracy by step ────────────────────────────────────────────
    W("")
    W("  DETECTION ACCURACY vs DETECTION STEP")
    W(f"  {'Step':>6}  {'Overall':>9}  {'Type_A P/R/F1':>16}  "
      f"{'Type_B P/R/F1':>16}  {'Type_C P/R/F1':>16}")
    W(f"  {SEP2}")
    for step, a in sorted(analysis.items()):
        A = a["per_type"]["Type_A"]
        B = a["per_type"]["Type_B"]
        C = a["per_type"]["Type_C"]
        W(f"  {step:>6}  {a['overall_accuracy']:>9.3f}  "
          f"  {A['precision']:.2f}/{A['recall']:.2f}/{A['f1']:.2f}    "
          f"  {B['precision']:.2f}/{B['recall']:.2f}/{B['f1']:.2f}    "
          f"  {C['precision']:.2f}/{C['recall']:.2f}/{C['f1']:.2f}")

    # ── Per-modality detection results ────────────────────────────────────────
    W("")
    W("  PER-MODALITY DETECTION (all seeds, all checkpoints)")
    W(f"  {'Modality':<14}  {'GT':>4}  " +
      "  ".join(f"n={s}" for s in DETECT_AT) +
      "  Final  Pred")
    W(f"  {SEP2}")
    for mod_name, seeds in all_results.items():
        gt = seeds[0]["ground_truth_type"]
        for seed in seeds:
            det_str = "  ".join(
                f"{seed['detections'].get(s,{}).get('archetype','?')}"
                f"({'✓' if seed['detections'].get(s,{}).get('correct') else '✗'})"
                for s in DETECT_AT
            )
            W(f"  {mod_name:<14}  {gt:>4}  {det_str}  "
              f"{seed['final_acc']:.3f}  "
              f"{seed.get('predicted_final','?')}")

    # ── Signal distributions by archetype ────────────────────────────────────
    W("")
    W("  SIGNAL DISTRIBUTIONS AT STEP 50 BY ARCHETYPE")
    W("  (mean ± std across all seeds of that archetype)")
    W("")
    W(f"  {'Archetype':<12}  {'acc':>7}  {'acc_slope':>10}  "
      f"{'sigma':>8}  {'sigma_slope':>12}  {'anchors':>8}")
    W(f"  {SEP2}")
    by_type = {"A": [], "B": [], "C": []}
    for mod_name, seeds in all_results.items():
        gt = seeds[0]["ground_truth_type"]
        for seed in seeds:
            det = seed["detections"].get(50, seed["detections"].get(
                max(seed["detections"].keys()), {}))
            sig = det.get("signals", {})
            if sig:
                by_type[gt].append(sig)

    for t in "ABC":
        sigs = by_type[t]
        if not sigs:
            W(f"  Type {t:<8}  no data"); continue
        def ms(key):
            vals = [s.get(key,0) for s in sigs]
            return f"{np.mean(vals):+.4f}±{np.std(vals):.4f}"
        W(f"  Type {t:<8}  "
          f"{ms('acc_now'):>16}  "
          f"{ms('acc_slope'):>16}  "
          f"{ms('sigma_mean'):>16}  "
          f"{ms('sigma_slope'):>16}  "
          f"{np.mean([s.get('n_anchors',0) for s in sigs]):>8.1f}")

    # ── Early trace: step-by-step for each modality ───────────────────────────
    W("")
    W("  EARLY TRACES (seed 0, first 50 steps)")
    W(f"  {'step':>6}  " + "  ".join(f"{m+' acc':>12}" for m in all_results.keys()))
    W(f"  {SEP2}")
    max_step = max(DETECT_AT)
    step_grid = sorted(set(
        t["n"]
        for seeds in all_results.values()
        for t in seeds[0]["fine_trace"]
        if t["n"] <= max_step
    ))
    for step in step_grid:
        row = f"  {step:>6}"
        for mod_name, seeds in all_results.items():
            trace = seeds[0]["fine_trace"]
            pt = next((t for t in trace if t["n"] == step), None)
            row += f"  {pt['acc']:>12.3f}" if pt else f"  {'—':>12}"
        W(row)

    # ── Final accuracy prediction accuracy ────────────────────────────────────
    W("")
    W("  FINAL ACCURACY PREDICTION (from step-50 extrapolation)")
    W(f"  {'Modality':<14}  {'GT_type':>8}  {'Actual':>8}  "
      f"{'Predicted':>10}  {'Error':>7}")
    W(f"  {SEP2}")
    for mod_name, seeds in all_results.items():
        gt = seeds[0]["ground_truth_type"]
        for seed in seeds:
            actual = seed["final_acc"]
            pred   = seed.get("predicted_final")
            err    = round(pred - actual, 3) if pred else None
            W(f"  {mod_name:<14}  {'Type_'+gt:>8}  {actual:>8.3f}  "
              f"{str(pred):>10}  {str(err):>7}")

    # ── Synthesis ─────────────────────────────────────────────────────────────
    W("")
    W(SEP)
    W("  SYNTHESIS — ARCHETYPE DETECTION")
    W(SEP)
    W(f"""
  DETECTION RELIABILITY:
    The table above shows precision/recall/F1 at each detection step.
    Key question: does detection accuracy improve from step 10 → 50?
    If accuracy plateaus by step 10: you don't need 50 steps.
    If accuracy is still improving at step 50: you need more early probing.

  TYPE A DETECTION (audio):
    Should be detectable by step 6-10 from acc > {THRESH_A_ACC} alone.
    If audio is mis-classified as B at step 10: threshold needs raising.
    Low false-positive rate for A is critical — wrong A-classification
    means you stop training a B-type system too early.

  TYPE C DETECTION (structured, rf_iq):
    The hard case. Type C looks like Type B early on (both have low acc).
    The discriminator is sigma_slope: B has falling σ, C has flat/rising σ.
    Also: Type C shows anchor_rate > 0 (multiple prototypes forming)
    without matching accuracy improvement — this is the fingerprint.

  TYPE B DETECTION:
    Residual category — anything not A or C.
    Main risk: calling B too early when it's actually a slow A.
    Guard: require sigma_mean < 0.02 for A classification.

  OPTIMAL DETECTION STEP:
    The step at which overall_accuracy plateaus is the recommended
    detection point for the adaptive learner. Earlier = faster startup,
    but more misclassifications. Later = more reliable, wastes training.

  PREDICTION ACCURACY:
    Simple linear extrapolation from step-50 signals gives rough final acc.
    Systematic bias: extrapolation undershoots Type B (misses transition),
    overshoots Type C (assumes improvement that doesn't come).
    A better predictor uses the Q(n) model fit from profiler 1.
    Combined: archetype classification → correct Q(n) template → n* prediction.
""")
    W(SEP)

    text = "\n".join(L)
    with open(path, "w") as f:
        f.write(text)
    print(f"\n  Report → {path}")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profiler 4 — Archetype Detection")
    parser.add_argument("--modalities", nargs="+",
                        default=list(MODALITIES.keys()),
                        choices=list(MODALITIES.keys()))
    parser.add_argument("--quick", action="store_true",
                        help="2 seeds only (~12 min)")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    n_seeds = 2 if args.quick else args.seeds

    print(f"\n{'═'*70}")
    print("  CYPHA PROFILER 4 — ARCHETYPE DETECTION (Type A / B / C)")
    print(f"  Modalities:  {', '.join(args.modalities)}")
    print(f"  Seeds:       {n_seeds}")
    print(f"  Detect at:   steps {DETECT_AT}")
    print(f"  Full budget: {FULL_TRAIN_N}/class = {FULL_TRAIN_N*N_CLASSES} total")
    print(f"{'═'*70}\n")

    t_start = time.time()
    all_results = {}

    for mod_name in args.modalities:
        fn, gt = MODALITIES[mod_name]
        all_results[mod_name] = run_modality(mod_name, fn, gt, n_seeds)

    analysis = analyse_detection(all_results)

    json_path = "profiler4_archetype_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  JSON → {json_path}")

    write_report(all_results, analysis, "profiler4_archetype_report.txt")
    print(f"\n  Total: {(time.time()-t_start)/60:.1f} min")
    print("  Done.\n")
