"""
cypha_profiler1_transition.py
════════════════════════════════════════════════════════════════════════════════
PROFILER 1 — PHASE TRANSITION SCALING (k=2 probe fix)

Answers:
  - At what step does the phase transition occur for each modality?
  - Does transition_n / n_total (the ratio) stay constant within a modality?
  - Is the ratio predictable from the training budget alone?
  - Does the transition exist at small n/class budgets, or only above a threshold?
  - What is the minimum n/class required for a transition to occur?

Fix over cypha_fewshot_online.py EXP5:
  - Probe function uses k=2 lookup to get real margin differences
  - σ collapse detector now has real signal to work with
  - Runs all 6 modalities (text, audio, rf_iq, image, structured, video)
  - Wider n/class sweep: [5, 10, 25, 50, 100, 200, 500, 1000]
  - Finer probe interval (every 2% of n_total, minimum 1 step)
  - Reports: transition_n, sigma_before, sigma_after, ratio, acc_at_trans,
             post_slope (improving/stable/degrading), Q(n) fit parameters

Output:
  profiler1_transition_results.json
  profiler1_transition_report.txt

Usage:
  python cypha_profiler1_transition.py
  python cypha_profiler1_transition.py --modalities text audio
  python cypha_profiler1_transition.py --max-n-class 200   # quick run
"""

import sys, os, math, time, json, base64
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
N_CLASSES      = 6
N_PROBE        = 300          # probe set size (50 per class)
FEATURE_DIM    = 512
RESONANCE_DIM  = 256
N_CLASS_SWEEP  = [5, 10, 25, 50, 100, 200, 500, 1000]
N_SEEDS        = 2            # repeat each condition for stability
PROBE_POINTS   = 40           # number of probe measurements per run
MIN_SIGMA_DROP = 0.003        # minimum σ drop to count as transition

# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS (all 6 modalities)
# ══════════════════════════════════════════════════════════════════════════════

# ── Text ──────────────────────────────────────────────────────────────────────
_TEXT_VOCAB = [
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
    kw = list(rng.choice(_TEXT_VOCAB[cls], size=6, replace=False))
    sh = list(rng.choice(_SHARED, size=8, replace=True))
    tokens = kw + sh; rng.shuffle(tokens)
    return " ".join(tokens)

# ── Audio ─────────────────────────────────────────────────────────────────────
_FREQS = [220., 440., 880., 1760., 3520., 7040.]
def audio_ex(cls):
    t = np.arange(4000) / 8000.0
    f0 = _FREQS[cls] * (1 + rng.normal(0, 0.01))
    sig = np.sin(2*np.pi*f0*t)
    if cls % 2 == 1:
        sig += 0.5*np.sin(2*np.pi*2*f0*t); sig /= (np.abs(sig).max()+1e-9)
    sig += rng.normal(0, 0.02, 4000)
    return "pcm:" + np.clip(sig,-1,1).astype(np.float64).__mul__(32700).astype(np.int16).tobytes().hex()

# ── RF/IQ ─────────────────────────────────────────────────────────────────────
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

# ── Image ─────────────────────────────────────────────────────────────────────
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

# ── Structured ────────────────────────────────────────────────────────────────
_CENTRES=rng.normal(0,3,(N_CLASSES,64)).astype(np.float32)
for _i in range(N_CLASSES):
    for _j in range(_i):
        _CENTRES[_i]-=(np.dot(_CENTRES[_i],_CENTRES[_j])/(np.dot(_CENTRES[_j],_CENTRES[_j])+1e-9)*_CENTRES[_j])
    _n=np.linalg.norm(_CENTRES[_i])
    if _n>1e-9: _CENTRES[_i]=_CENTRES[_i]/_n*5
def structured_ex(cls):
    v=_CENTRES[cls]+rng.normal(0,0.5,64).astype(np.float32)
    return "arr:"+base64.b64encode(v.tobytes()).decode()

# ── Video ─────────────────────────────────────────────────────────────────────
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
    n=np.linalg.norm(agg);
    if n>1e-9: agg/=n
    return "arr:"+base64.b64encode(agg.tobytes()).decode()

MODALITIES = {
    "text":       (text_ex,       "OmegaEncoder text path"),
    "audio":      (audio_ex,      "pcm: → _encode_audio"),
    "rf_iq":      (rf_ex,         "iq: → _encode_iq"),
    "image":      (image_ex,      "arr: → float32 Omega (spatial)"),
    "structured": (structured_ex, "arr: → float32 Omega (geometric)"),
    "video":      (video_ex,      "arr: → float32 Omega (temporal)"),
}
CLASS_LABELS = [f"class_{i}" for i in range(N_CLASSES)]

# ══════════════════════════════════════════════════════════════════════════════
# CORE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def fresh() -> Cypha:
    return Cypha(feature_dim=FEATURE_DIM, resonance_dim=RESONANCE_DIM)

def make_probe(fn) -> List[Tuple[str,str]]:
    probe = []
    per = N_PROBE // N_CLASSES
    for cls in range(N_CLASSES):
        for _ in range(per):
            probe.append((fn(cls), CLASS_LABELS[cls]))
    return probe

def measure_k2(cypha: Cypha, probe: List[Tuple[str,str]]) -> Dict:
    """
    Probe with k=2 lookup to get real margins.
    Patches memory.lookup to return k=2, restores after.
    """
    correct = 0; margins = []

    for inp, label in probe:
        try:
            # Encode the query
            q_vec = cypha.encode_features(inp)
            # k=2 lookup directly on memory
            matches = cypha.memory.lookup(q_vec, k=2)
            if not matches:
                continue
            # Get prediction
            pred_key, pred_sim = matches[0]
            pred_label = cypha.memory.get_output(pred_key) or pred_key
            if pred_label == label:
                correct += 1
            # Real margin
            if len(matches) >= 2:
                margin = float(matches[0][1]) - float(matches[1][1])
            else:
                margin = 0.5
            margins.append(margin)
        except Exception:
            pass

    n = len(probe)
    acc = correct / max(n, 1)
    mm  = float(np.mean(margins)) if margins else 0.0
    ms  = float(np.std(margins))  if margins else 0.0
    return {
        "acc":        round(acc, 4),
        "mean_margin": round(mm, 4),
        "margin_std":  round(ms, 4),
        "n_anchors":   cypha.memory.n,
    }

def build_pool(fn, n=50) -> Dict[str,List[str]]:
    pool = {}
    for cls in range(N_CLASSES):
        pool[CLASS_LABELS[cls]] = [fn(cls) for _ in range(n)]
    return pool

def get_negs(label, pool, k=2):
    negs = []
    for lbl, examples in pool.items():
        if lbl != label and examples:
            negs.append(examples[int(rng.integers(0,len(examples)))])
        if len(negs) >= k: break
    return negs

# ══════════════════════════════════════════════════════════════════════════════
# Q(n) MODEL FIT
# ══════════════════════════════════════════════════════════════════════════════

def fit_qn(trace: List[Dict]) -> Dict:
    if len(trace) < 5:
        return {"fit_failed": True, "reason": "too_few_points"}
    ns   = np.array([t["n"]   for t in trace], dtype=float)
    accs = np.array([t["acc"] for t in trace], dtype=float)
    if accs.max() < 0.25 or (accs.max()-accs.min()) < 0.05:
        return {"fit_failed": True, "reason": f"insufficient_range ({accs.max():.3f})"}
    y = (accs - accs.min()) / (accs.max() - accs.min())
    n_max = ns.max(); best_r2=-np.inf; best_p=None
    for n_half in np.linspace(n_max*.05, n_max*.95, 50):
        w = max(n_half*.3, 1.)
        rec = 1/(1+np.exp(-(ns-n_half)/w))
        for lam in np.logspace(-6, -2, 50):
            plas = np.exp(-lam*ns)
            q = rec*plas; qm=q.max()
            if qm<1e-9: continue
            q/=qm
            r2 = 1 - np.sum((y-q)**2)/max(np.sum((y-y.mean())**2),1e-9)
            if r2>best_r2: best_r2=r2; best_p=(n_half,w,lam)
    if best_p is None:
        return {"fit_failed": True, "reason": "grid_search_failed"}
    n_half,w,lam = best_p
    ng = np.linspace(0, n_max*2, 20000)
    qg = (1/(1+np.exp(-(ng-n_half)/w)))*np.exp(-lam*ng)
    n_star = float(ng[np.argmax(qg)])
    return {
        "n_half":    round(n_half,2),
        "width":     round(w,2),
        "lambda":    round(float(lam),8),
        "n_star":    round(n_star,1),
        "r_squared": round(float(best_r2),4),
    }

# ══════════════════════════════════════════════════════════════════════════════
# TRANSITION DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def detect_transition(trace: List[Dict]) -> Dict:
    """
    Find the phase transition from real k=2 margin_std data.
    Looks for largest single-step σ drop.
    Also reports:
      - sigma_pre:  mean σ before transition
      - sigma_post: mean σ after transition
      - acc_at_trans
      - post_slope: acc trend after transition (per 1000 steps)
      - behaviour: improving / stable / degrading
    """
    if len(trace) < 4:
        return {"found": False, "reason": "too_few_points"}

    sigs = np.array([t["margin_std"] for t in trace])
    ns   = np.array([t["n"]          for t in trace])
    accs = np.array([t["acc"]         for t in trace])

    # Need meaningful σ variance
    if sigs.max() - sigs.min() < MIN_SIGMA_DROP:
        return {
            "found":       False,
            "reason":      f"sigma_range_too_small ({sigs.max()-sigs.min():.4f})",
            "max_acc":     float(accs.max()),
            "sigma_mean":  float(sigs.mean()),
        }

    diffs = np.diff(sigs)
    best_i = int(np.argmin(diffs))

    if diffs[best_i] >= -MIN_SIGMA_DROP:
        return {
            "found":   False,
            "reason":  f"no_significant_drop (max_drop={diffs[best_i]:.4f})",
            "max_acc": float(accs.max()),
        }

    trans_n   = int(ns[best_i])
    trans_acc = float(accs[best_i])
    sig_drop  = float(diffs[best_i])

    pre_mask  = ns <= trans_n
    post_mask = ns >  trans_n
    sig_pre   = float(sigs[pre_mask].mean()) if pre_mask.sum() > 0 else 0.
    sig_post  = float(sigs[post_mask].mean()) if post_mask.sum() > 0 else 0.

    # Post-transition accuracy trend
    post_accs = accs[post_mask]; post_ns = ns[post_mask]
    behaviour = "no_post_data"
    post_slope = None
    if len(post_accs) > 2:
        slope = float(np.polyfit(post_ns, post_accs, 1)[0]) * 1000
        post_slope = round(slope, 6)
        if   slope >  0.5e-3: behaviour = "improving"
        elif slope < -0.5e-3: behaviour = "degrading"
        else:                 behaviour = "stable"

    return {
        "found":        True,
        "transition_n": trans_n,
        "transition_acc": trans_acc,
        "sigma_drop":   round(sig_drop, 5),
        "sigma_pre":    round(sig_pre, 5),
        "sigma_post":   round(sig_post, 5),
        "sigma_ratio":  round(sig_pre / max(sig_post, 1e-9), 2),
        "post_slope":   post_slope,
        "behaviour":    behaviour,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_one(mod_name: str, fn, n_pc: int, probe: List, pool: Dict,
            seed_i: int) -> Dict:
    """Train one Cypha instance on n_pc examples/class, probe throughout."""
    cypha = fresh()
    pairs = []
    for cls in range(N_CLASSES):
        for _ in range(n_pc):
            pairs.append((fn(cls), CLASS_LABELS[cls]))
    rng.shuffle(pairs)

    n_total      = len(pairs)
    probe_every  = max(1, n_total // PROBE_POINTS)
    trace        = []

    for step_i, (inp, lbl) in enumerate(pairs):
        cypha.train_step(inp, lbl, negatives=get_negs(lbl, pool))
        n = step_i + 1
        if n % probe_every == 0 or n == 1 or n == n_total:
            m = measure_k2(cypha, probe)
            m["n"] = n
            trace.append(m)

    transition = detect_transition(trace)
    qn_fit     = fit_qn(trace)

    ratio = None
    if transition.get("found") and transition.get("transition_n"):
        ratio = round(transition["transition_n"] / n_total, 3)

    return {
        "modality":    mod_name,
        "n_per_class": n_pc,
        "n_total":     n_total,
        "seed":        seed_i,
        "transition":  transition,
        "trans_ratio": ratio,
        "qn_fit":      qn_fit,
        "final_acc":   float(trace[-1]["acc"]) if trace else 0.,
        "trace":       trace,
    }

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_modality(mod_name: str, fn, n_class_sweep: List[int]) -> List[Dict]:
    print(f"\n{'═'*70}")
    print(f"  MODALITY: {mod_name.upper()}")
    print(f"{'─'*70}")
    probe = make_probe(fn)
    pool  = build_pool(fn)
    all_runs = []

    for n_pc in n_class_sweep:
        seed_runs = []
        for seed_i in range(N_SEEDS):
            t0 = time.time()
            result = run_one(mod_name, fn, n_pc, probe, pool, seed_i)
            elapsed = time.time() - t0
            tr = result["transition"]
            tn = f"n={tr['transition_n']:,}" if tr.get("found") else "—"
            rat = f"{result['trans_ratio']:.3f}" if result["trans_ratio"] else "—"
            beh = tr.get("behaviour", tr.get("reason",""))
            print(f"  n/cls={n_pc:>5}  seed={seed_i}  "
                  f"trans={tn:<12}  ratio={rat}  "
                  f"final={result['final_acc']:.3f}  "
                  f"beh={beh}  ({elapsed:.0f}s)")
            seed_runs.append(result)

        # Aggregate seeds
        found_runs = [r for r in seed_runs if r["transition"].get("found")]
        agg = {
            "modality":    mod_name,
            "n_per_class": n_pc,
            "n_total":     n_pc * N_CLASSES,
            "n_seeds":     N_SEEDS,
            "seeds":       seed_runs,
            # Aggregated transition
            "trans_found_count": len(found_runs),
            "trans_n_mean":      round(float(np.mean([r["transition"]["transition_n"] for r in found_runs])),1) if found_runs else None,
            "trans_n_std":       round(float(np.std( [r["transition"]["transition_n"] for r in found_runs])),1) if found_runs else None,
            "trans_ratio_mean":  round(float(np.mean([r["trans_ratio"] for r in found_runs])),3) if found_runs else None,
            "trans_ratio_std":   round(float(np.std( [r["trans_ratio"] for r in found_runs])),3) if found_runs else None,
            "final_acc_mean":    round(float(np.mean([r["final_acc"] for r in seed_runs])),4),
            "final_acc_std":     round(float(np.std( [r["final_acc"] for r in seed_runs])),4),
            # Q(n) from first seed with valid fit
            "qn_fit": next((r["qn_fit"] for r in seed_runs if not r["qn_fit"].get("fit_failed")), {"fit_failed": True}),
        }
        all_runs.append(agg)

    return all_runs

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(results: Dict, path: str):
    SEP  = "═" * 76
    SEP2 = "─" * 74
    L = []; W = L.append

    W(SEP)
    W("  PROFILER 1 — PHASE TRANSITION SCALING  (k=2 probe, real σ data)")
    W(f"  {N_CLASSES} classes  |  probe_n={N_PROBE}  |  seeds={N_SEEDS}  |  probe_pts={PROBE_POINTS}")
    W(SEP)

    # ── Summary table ─────────────────────────────────────────────────────────
    W("")
    W("  TRANSITION DETECTION SUMMARY")
    W(f"  {'Modality':<14}  {'n/cls':>6}  {'n_total':>8}  {'trans_n':>10}  "
      f"{'ratio':>7}  {'σ_pre':>7}  {'σ_post':>7}  {'σ_ratio':>8}  "
      f"{'final_acc':>10}  {'behaviour'}")
    W(f"  {SEP2}")

    for mod_name, mod_data in results.items():
        for agg in mod_data:
            if agg["trans_found_count"] > 0:
                tn  = f"{agg['trans_n_mean']:.0f}±{agg['trans_n_std']:.0f}"
                rat = f"{agg['trans_ratio_mean']:.3f}"
                # Get σ stats from first found seed
                found = next(r for r in agg["seeds"] if r["transition"].get("found"))
                tr = found["transition"]
                sp  = f"{tr['sigma_pre']:.4f}"
                spo = f"{tr['sigma_post']:.4f}"
                srat = f"{tr['sigma_ratio']:.1f}x"
                beh = tr.get("behaviour","—")
            else:
                tn=rat=sp=spo=srat="—"; beh="no_transition"
            W(f"  {mod_name:<14}  {agg['n_per_class']:>6}  {agg['n_total']:>8}  "
              f"{tn:>10}  {rat:>7}  {sp:>7}  {spo:>7}  {srat:>8}  "
              f"{agg['final_acc_mean']:>10.3f}  {beh}")

    # ── Ratio consistency ─────────────────────────────────────────────────────
    W("")
    W("  RATIO CONSISTENCY (transition_n / n_total)")
    W("  A consistent ratio means the transition is a fixed fraction of training,")
    W("  predictable from the budget alone — enabling adaptive stopping.")
    W("")
    W(f"  {'Modality':<14}  " +
      "  ".join(f"n/cls={n:>4}" for n in N_CLASS_SWEEP))
    W(f"  {SEP2}")
    for mod_name, mod_data in results.items():
        row = f"  {mod_name:<14}  "
        for agg in mod_data:
            rat = f"{agg['trans_ratio_mean']:.3f}" if agg["trans_ratio_mean"] else "  —  "
            row += f"  {rat:>9}"
        W(row)

    # ── Q(n) fits ─────────────────────────────────────────────────────────────
    W("")
    W("  Q(n) = recall(n) × plasticity(n) MODEL FITS")
    W("  Q(n) = sigmoid((n−n½)/w) × exp(−λ·n)")
    W("")
    W(f"  {'Modality':<14}  {'n/cls':>6}  {'n½':>8}  {'λ':>10}  "
      f"{'n*':>8}  {'R²':>6}  {'n*/n_total':>11}")
    W(f"  {SEP2}")
    for mod_name, mod_data in results.items():
        for agg in mod_data:
            fit = agg["qn_fit"]
            if fit.get("fit_failed"):
                W(f"  {mod_name:<14}  {agg['n_per_class']:>6}  "
                  f"fit failed: {fit.get('reason','')}")
            else:
                n_total = agg["n_total"]
                nstar_ratio = round(fit["n_star"]/n_total,3) if n_total else "—"
                W(f"  {mod_name:<14}  {agg['n_per_class']:>6}  "
                  f"{fit['n_half']:>8.1f}  {fit['lambda']:>10.2e}  "
                  f"{fit['n_star']:>8.0f}  {fit['r_squared']:>6.3f}  "
                  f"{nstar_ratio:>11}")

    # ── Per-modality analysis ──────────────────────────────────────────────────
    W("")
    W("  PER-MODALITY TRACE SUMMARIES")
    for mod_name, mod_data in results.items():
        W(f"\n  {mod_name.upper()}")
        for agg in mod_data:
            W(f"  n/cls={agg['n_per_class']}  (seed 0 trace):")
            seed0_trace = agg["seeds"][0]["trace"]
            show_every = max(1, len(seed0_trace)//8)
            W(f"    {'n':>8}  {'acc':>6}  {'margin':>8}  {'σ':>7}  {'anchors':>8}")
            for i, pt in enumerate(seed0_trace):
                if i % show_every == 0 or i == len(seed0_trace)-1:
                    W(f"    {pt['n']:>8,}  {pt['acc']:>6.3f}  "
                      f"{pt['mean_margin']:>8.4f}  {pt['margin_std']:>7.4f}  "
                      f"{pt['n_anchors']:>8}")

    # ── Synthesis ─────────────────────────────────────────────────────────────
    W("")
    W(SEP)
    W("  SYNTHESIS")
    W(SEP)
    W("""
  RATIO CONSISTENCY:
    If trans_ratio is stable across n/class values for a modality:
      → Transition is a constant fraction of training. Predictable.
      → Adaptive learner can stop at: n_stop = ratio × n_budget
    If trans_ratio varies with n/class:
      → Transition depends on absolute example count, not fraction.
      → Adaptive learner must monitor σ in real time.
    If no transition detected at any budget:
      → Modality is Type C (encoder-limited). No stopping criterion applies.
      → Flag as "encoder mismatch" and halt training early.

  Q(n) UNIVERSALITY:
    n* / n_total ratio should be close to trans_ratio if the model is correct.
    Systematic deviation indicates the plasticity decay (λ) is misestimated.
    Compare n* column to trans_n column — should agree within 20%.

  MINIMUM VIABLE BUDGET:
    The smallest n/class where a transition is detected gives the practical
    minimum training requirement for that modality.
    Below this threshold, training does not converge — it wastes budget.
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
    parser = argparse.ArgumentParser(description="Profiler 1 — Transition Scaling")
    parser.add_argument("--modalities", nargs="+",
                        default=list(MODALITIES.keys()),
                        choices=list(MODALITIES.keys()))
    parser.add_argument("--max-n-class", type=int, default=1000,
                        help="Cap the n/class sweep (e.g. 200 for quick run)")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    N_SEEDS = args.seeds
    sweep = [n for n in N_CLASS_SWEEP if n <= args.max_n_class]

    print(f"\n{'═'*70}")
    print("  CYPHA PROFILER 1 — PHASE TRANSITION SCALING")
    print(f"  Modalities: {', '.join(args.modalities)}")
    print(f"  n/class sweep: {sweep}")
    print(f"  Seeds: {N_SEEDS}  |  Probe points: {PROBE_POINTS}  |  k=2 margins")
    print(f"{'═'*70}\n")

    t_start = time.time()
    all_results = {}

    for mod_name in args.modalities:
        fn, _ = MODALITIES[mod_name]
        all_results[mod_name] = run_modality(mod_name, fn, sweep)

    # Save JSON
    json_path = "profiler1_transition_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  JSON → {json_path}")

    write_report(all_results, "profiler1_transition_report.txt")

    print(f"\n  Total: {(time.time()-t_start)/60:.1f} min")
    print("  Done.\n")
