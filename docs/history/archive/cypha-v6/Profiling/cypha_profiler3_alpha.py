"""
cypha_profiler3_alpha.py
════════════════════════════════════════════════════════════════════════════════
PROFILER 3 — EMA-ALPHA DYNAMICS & SWITCHING HYPOTHESIS

Answers:
  - Is alpha=0.05 (fixed-low) truly better than adaptive pre-transition?
  - Is adaptive better than fixed-low post-transition?
  - Does switching from fixed-low to adaptive AT the transition beat both?
  - What is the optimal fixed alpha per modality?
  - Does the optimal alpha depend on n_per_class (training budget)?
  - Is the transition detectable in real time well enough to switch on?

Hypothesis from EXP4 analysis:
  - Pre-transition: fixed low alpha (0.05) outperforms adaptive
    because adaptive keeps update rate permanently elevated in uncertain regime
  - Post-transition: adaptive is better because it selectively reinforces
    hard examples without disturbing well-converged centroids
  - Switching at detected transition beats both fixed strategies

Strategies tested:
  1. fixed_005     — alpha=0.05 throughout
  2. fixed_015     — alpha=0.15 throughout (Cypha default)
  3. fixed_030     — alpha=0.30 throughout
  4. fixed_040     — alpha=0.40 throughout
  5. adaptive      — ThoughtProcessor uncertainty-weighted (Cypha default)
  6. switch_005_ad — alpha=0.05 pre-transition, adaptive post-transition
  7. switch_015_ad — alpha=0.15 pre-transition, adaptive post-transition
  8. switch_ad_005 — adaptive pre-transition, 0.05 post-transition (reverse)
  9. anneal        — linearly decays from 0.40 → 0.05 over n_total steps

Each strategy is tested across:
  - 3 modalities: text, audio, structured
  - 3 n/class budgets: 50, 200, 500
  - 2 seeds per condition

Transition detection uses real-time σ monitoring (k=2 probe every 5% of budget).

Output:
  profiler3_alpha_results.json
  profiler3_alpha_report.txt

Usage:
  python cypha_profiler3_alpha.py
  python cypha_profiler3_alpha.py --modalities audio text
  python cypha_profiler3_alpha.py --quick   # n/class=[50] only, 1 seed
"""

import sys, os, time, json, base64
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable

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
N_CLASSES     = 6
N_PROBE       = 300       # 50 per class
FEATURE_DIM   = 512
RESONANCE_DIM = 256
N_CLASS_SWEEP = [50, 200, 500]
N_SEEDS       = 2
PROBE_POINTS  = 30        # measurement points per run
SIGMA_DROP_THRESH = 0.003 # σ drop to detect transition in real time

# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS
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

_CENTRES = rng.normal(0, 3, (N_CLASSES, 64)).astype(np.float32)
for _i in range(N_CLASSES):
    for _j in range(_i):
        _CENTRES[_i] -= (np.dot(_CENTRES[_i],_CENTRES[_j]) /
                         (np.dot(_CENTRES[_j],_CENTRES[_j])+1e-9) * _CENTRES[_j])
    _n = np.linalg.norm(_CENTRES[_i])
    if _n > 1e-9: _CENTRES[_i] = _CENTRES[_i] / _n * 5.0

def structured_ex(cls):
    v = _CENTRES[cls] + rng.normal(0, 0.5, 64).astype(np.float32)
    return "arr:" + base64.b64encode(v.tobytes()).decode()

MODALITIES = {
    "text":       text_ex,
    "audio":      audio_ex,
    "structured": structured_ex,
}
CLASS_LABELS = [f"class_{i}" for i in range(N_CLASSES)]

# ══════════════════════════════════════════════════════════════════════════════
# ALPHA STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def make_strategy(name: str, n_total: int) -> Dict:
    """
    Returns a strategy dict with:
      get_alpha(step, sigma_history, transition_detected) → float
      label: str
    """
    def fixed(a):
        def _f(step, sig_hist, trans_det): return a
        return _f

    def adaptive_fn(step, sig_hist, trans_det):
        return None   # None = use ThoughtProcessor's suggested_alpha

    def anneal_fn(step, sig_hist, trans_det):
        # Linear decay 0.40 → 0.05 over n_total
        frac = step / max(n_total, 1)
        return round(0.40 - frac * (0.40 - 0.05), 4)

    def switch_low_then_adaptive(pre_alpha):
        def _f(step, sig_hist, trans_det):
            if trans_det: return None   # adaptive post-transition
            return pre_alpha            # fixed pre-transition
        return _f

    def switch_adaptive_then_low(post_alpha):
        def _f(step, sig_hist, trans_det):
            if trans_det: return post_alpha  # fixed post-transition
            return None                      # adaptive pre-transition
        return _f

    strategies = {
        "fixed_005":     {"fn": fixed(0.05),                    "label": "Fixed α=0.05"},
        "fixed_015":     {"fn": fixed(0.15),                    "label": "Fixed α=0.15"},
        "fixed_030":     {"fn": fixed(0.30),                    "label": "Fixed α=0.30"},
        "fixed_040":     {"fn": fixed(0.40),                    "label": "Fixed α=0.40"},
        "adaptive":      {"fn": adaptive_fn,                    "label": "Adaptive (TP)"},
        "switch_005_ad": {"fn": switch_low_then_adaptive(0.05), "label": "Switch 0.05→adaptive"},
        "switch_015_ad": {"fn": switch_low_then_adaptive(0.15), "label": "Switch 0.15→adaptive"},
        "switch_ad_005": {"fn": switch_adaptive_then_low(0.05), "label": "Switch adaptive→0.05"},
        "anneal":        {"fn": anneal_fn,                      "label": "Anneal 0.40→0.05"},
    }
    return strategies[name]

STRATEGY_NAMES = [
    "fixed_005", "fixed_015", "fixed_030", "fixed_040",
    "adaptive",
    "switch_005_ad", "switch_015_ad", "switch_ad_005",
    "anneal",
]

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def fresh() -> Cypha:
    return Cypha(feature_dim=FEATURE_DIM, resonance_dim=RESONANCE_DIM)

def make_probe(fn) -> List[Tuple[str,str]]:
    per = N_PROBE // N_CLASSES
    return [(fn(cls), CLASS_LABELS[cls])
            for cls in range(N_CLASSES) for _ in range(per)]

def measure_k2(cypha: Cypha, probe: List[Tuple[str,str]]) -> Dict:
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
    return {"acc": round(acc,4), "mean_margin": round(mm,4), "margin_std": round(ms,4),
            "n_anchors": cypha.memory.n}

def build_pool(fn, n=60) -> Dict[str,List[str]]:
    return {CLASS_LABELS[cls]: [fn(cls) for _ in range(n)]
            for cls in range(N_CLASSES)}

def get_negs(label, pool, k=2):
    negs = []
    for lbl, ex in pool.items():
        if lbl != label and ex:
            negs.append(ex[int(rng.integers(0, len(ex)))])
        if len(negs) >= k: break
    return negs

def detect_transition_realtime(sigma_history: List[float]) -> bool:
    """Returns True if the last σ value is a significant drop from previous."""
    if len(sigma_history) < 2: return False
    drop = sigma_history[-1] - sigma_history[-2]
    return drop < -SIGMA_DROP_THRESH

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE RUN WITH ALPHA STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

def run_strategy(mod_name: str, fn, n_pc: int, strategy_name: str,
                 probe: List, pool: Dict, seed_i: int) -> Dict:
    """
    Train with a given alpha strategy. The strategy function is called at each
    step to get the current alpha. None = use ThoughtProcessor's suggested_alpha.

    Tracks:
      - Full accuracy/margin trace
      - Alpha used at each probe point
      - Transition detection moment
      - Per-phase accuracy (pre/post transition)
    """
    cypha    = fresh()
    n_total  = n_pc * N_CLASSES
    strat    = make_strategy(strategy_name, n_total)
    alpha_fn = strat["fn"]

    pairs = [(fn(cls), CLASS_LABELS[cls])
             for cls in range(N_CLASSES) for _ in range(n_pc)]
    rng.shuffle(pairs)

    probe_every       = max(1, n_total // PROBE_POINTS)
    trace             = []
    sigma_history     = []
    transition_detected = False
    transition_n        = None
    alphas_used         = []   # alpha at each step

    # Patch memory.store to inject fixed alpha when strategy says so
    _orig_store = cypha.memory.store

    for step_i, (inp, lbl) in enumerate(pairs):
        current_alpha = alpha_fn(step_i, sigma_history, transition_detected)

        # Patch store for this step
        if current_alpha is not None:
            _a = current_alpha
            def _patched(key, state, output, dedup_threshold=None,
                         ema_alpha=0.15, __a=_a, __orig=_orig_store):
                return __orig(key, state, output, dedup_threshold, __a)
            cypha.memory.store = _patched
        else:
            cypha.memory.store = _orig_store

        cypha.train_step(inp, lbl, negatives=get_negs(lbl, pool))
        alphas_used.append(current_alpha if current_alpha is not None else -1)  # -1 = adaptive

        n = step_i + 1
        if n % probe_every == 0 or n == 1 or n == n_total:
            pt = measure_k2(cypha, probe)
            pt["n"]     = n
            pt["alpha"] = round(current_alpha, 4) if current_alpha is not None else -1
            pt["trans"] = transition_detected
            trace.append(pt)
            sigma_history.append(pt["margin_std"])

            # Check for transition
            if not transition_detected and detect_transition_realtime(sigma_history):
                transition_detected = True
                transition_n = n
                pt["trans_detected_here"] = True

    # Restore original store
    cypha.memory.store = _orig_store

    # Compute per-phase accuracy
    pre_trans_accs  = [t["acc"] for t in trace if not t.get("trans") and t["n"] != 1]
    post_trans_accs = [t["acc"] for t in trace if t.get("trans")]
    final_acc       = trace[-1]["acc"] if trace else 0.

    # Compute mean alpha used (treating -1 as adaptive → substitute 0.279 = median of 0.15+0.25*0.516)
    alpha_arr = [a if a >= 0 else 0.279 for a in alphas_used]
    mean_alpha = round(float(np.mean(alpha_arr)), 4)

    return {
        "modality":         mod_name,
        "n_per_class":      n_pc,
        "strategy":         strategy_name,
        "strategy_label":   strat["label"],
        "seed":             seed_i,
        "final_acc":        final_acc,
        "transition_n":     transition_n,
        "trans_ratio":      round(transition_n / n_total, 3) if transition_n else None,
        "acc_pre_trans":    round(float(np.mean(pre_trans_accs)),  4) if pre_trans_accs  else None,
        "acc_post_trans":   round(float(np.mean(post_trans_accs)), 4) if post_trans_accs else None,
        "mean_alpha":       mean_alpha,
        "trace":            trace,
    }

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_modality(mod_name: str, fn, n_class_sweep: List[int],
                 n_seeds: int) -> Dict:
    print(f"\n{'═'*72}")
    print(f"  MODALITY: {mod_name.upper()}")
    print(f"{'─'*72}")
    probe = make_probe(fn)
    pool  = build_pool(fn)

    mod_results = {}

    for n_pc in n_class_sweep:
        print(f"\n  n/class = {n_pc}  (n_total={n_pc*N_CLASSES})")
        npc_results = {}

        for strat_name in STRATEGY_NAMES:
            seed_runs = []
            for seed_i in range(n_seeds):
                r = run_strategy(mod_name, fn, n_pc, strat_name,
                                 probe, pool, seed_i)
                seed_runs.append(r)

            # Aggregate across seeds
            final_accs  = [r["final_acc"] for r in seed_runs]
            trans_ns    = [r["transition_n"] for r in seed_runs if r["transition_n"]]
            label       = seed_runs[0]["strategy_label"]

            agg = {
                "strategy":        strat_name,
                "label":           label,
                "final_acc_mean":  round(float(np.mean(final_accs)), 4),
                "final_acc_std":   round(float(np.std(final_accs)),  4),
                "trans_n_mean":    round(float(np.mean(trans_ns)),   1) if trans_ns else None,
                "seeds":           seed_runs,
            }
            npc_results[strat_name] = agg

            # Print summary row
            trans_str = f"trans@{int(agg['trans_n_mean']):,}" if agg["trans_n_mean"] else "no_trans"
            print(f"    {label:<28}  "
                  f"acc={agg['final_acc_mean']:.3f}±{agg['final_acc_std']:.3f}  "
                  f"{trans_str}")

        # Best strategy for this n/class
        best = max(npc_results.values(), key=lambda x: x["final_acc_mean"])
        print(f"  → BEST: {best['label']}  acc={best['final_acc_mean']:.3f}")
        mod_results[n_pc] = npc_results

    return mod_results

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(results: Dict, path: str):
    SEP  = "═" * 76
    SEP2 = "─" * 74
    L = []; W = L.append

    W(SEP)
    W("  PROFILER 3 — EMA-ALPHA DYNAMICS & SWITCHING HYPOTHESIS")
    W(f"  {N_CLASSES} classes  |  probe_n={N_PROBE}  |  k=2 margins  |  seeds={N_SEEDS}")
    W(f"  n/class sweep: {N_CLASS_SWEEP}")
    W(SEP)

    # ── Ranking table per modality ─────────────────────────────────────────────
    for mod_name, mod_data in results.items():
        W(f"\n  MODALITY: {mod_name.upper()}")
        W("")

        for n_pc, npc_data in mod_data.items():
            W(f"  n/class={n_pc}  (n_total={n_pc*N_CLASSES})")
            W(f"  {'Rank':>5}  {'Strategy':<28}  {'FinalAcc':>9}  {'Std':>5}  {'TransN':>8}")
            W(f"  {SEP2}")

            ranked = sorted(npc_data.values(),
                            key=lambda x: x["final_acc_mean"], reverse=True)
            for rank_i, r in enumerate(ranked, 1):
                tn = f"{int(r['trans_n_mean']):,}" if r["trans_n_mean"] else "—"
                W(f"  {rank_i:>5}  {r['label']:<28}  "
                  f"{r['final_acc_mean']:>9.3f}  {r['final_acc_std']:>5.3f}  "
                  f"{tn:>8}")
            W("")

    # ── Switch strategies: did they beat both parent strategies? ──────────────
    W("")
    W("  SWITCHING STRATEGY ANALYSIS")
    W("  Does switching at transition beat both parent strategies?")
    W("")
    W(f"  {'Modality':<14}  {'n/cls':>6}  "
      f"{'fixed_005':>10}  {'adaptive':>10}  {'switch_005_ad':>14}  "
      f"{'switch_015_ad':>14}  {'switch_ad_005':>14}  {'winner'}")
    W(f"  {SEP2}")

    for mod_name, mod_data in results.items():
        for n_pc, npc_data in mod_data.items():
            f005  = npc_data["fixed_005"]["final_acc_mean"]
            fadap = npc_data["adaptive"]["final_acc_mean"]
            sw1   = npc_data["switch_005_ad"]["final_acc_mean"]
            sw2   = npc_data["switch_015_ad"]["final_acc_mean"]
            sw3   = npc_data["switch_ad_005"]["final_acc_mean"]
            best_sw = max(sw1, sw2, sw3)
            # Winner: does best switch beat both parents?
            if best_sw > f005 and best_sw > fadap:
                winner = "SWITCH ✓"
            elif f005 >= fadap:
                winner = "fixed_005"
            else:
                winner = "adaptive"
            W(f"  {mod_name:<14}  {n_pc:>6}  "
              f"{f005:>10.3f}  {fadap:>10.3f}  {sw1:>14.3f}  "
              f"{sw2:>14.3f}  {sw3:>14.3f}  {winner}")

    # ── Annealing comparison ──────────────────────────────────────────────────
    W("")
    W("  ANNEALING vs FIXED-LOW  (does smooth decay beat step function?)")
    W("")
    W(f"  {'Modality':<14}  {'n/cls':>6}  {'fixed_005':>10}  "
      f"{'anneal':>8}  {'diff':>6}  {'winner'}")
    W(f"  {SEP2}")
    for mod_name, mod_data in results.items():
        for n_pc, npc_data in mod_data.items():
            f005 = npc_data["fixed_005"]["final_acc_mean"]
            ann  = npc_data["anneal"]["final_acc_mean"]
            diff = ann - f005
            W(f"  {mod_name:<14}  {n_pc:>6}  {f005:>10.3f}  "
              f"{ann:>8.3f}  {diff:>+6.3f}  "
              f"{'anneal' if diff > 0.005 else 'fixed_005' if diff < -0.005 else 'tie'}")

    # ── Alpha effect on convergence speed ─────────────────────────────────────
    W("")
    W("  CONVERGENCE SPEED: n to reach 0.80 accuracy")
    W("  (from cold-start traces, n/class=200)")
    W("")
    W(f"  {'Modality':<14}  {'Strategy':<28}  {'n@0.80':>8}")
    W(f"  {SEP2}")
    for mod_name, mod_data in results.items():
        n_pc = 200
        if n_pc not in mod_data: continue
        for strat_name in ["fixed_005", "fixed_015", "adaptive", "switch_005_ad", "anneal"]:
            if strat_name not in mod_data[n_pc]: continue
            r = mod_data[n_pc][strat_name]
            # Find first step where acc >= 0.80
            traces = r["seeds"][0]["trace"]
            n80 = next((t["n"] for t in traces if t["acc"] >= 0.80), None)
            n80_str = f"{n80:,}" if n80 else f">{n_pc*N_CLASSES:,}"
            W(f"  {mod_name:<14}  {r['label']:<28}  {n80_str:>8}")

    # ── Traces for best and worst strategy ────────────────────────────────────
    W("")
    W("  ACCURACY TRACES — best vs worst strategy per modality (n/class=200, seed 0)")
    for mod_name, mod_data in results.items():
        n_pc = 200
        if n_pc not in mod_data: continue
        W(f"\n  {mod_name}  n/class=200:")
        npc = mod_data[n_pc]
        ranked = sorted(npc.values(), key=lambda x: x["final_acc_mean"], reverse=True)
        best_s  = ranked[0]
        worst_s = ranked[-1]
        best_trace  = best_s["seeds"][0]["trace"]
        worst_trace = worst_s["seeds"][0]["trace"]
        W(f"  {'n':>8}  {'best_acc':>9} ({best_s['label']})  "
          f"{'worst_acc':>10} ({worst_s['label']})")
        n_show = min(len(best_trace), len(worst_trace))
        show_e = max(1, n_show // 10)
        for i in range(0, n_show, show_e):
            bt = best_trace[i]; wt = worst_trace[i]
            W(f"  {bt['n']:>8,}  {bt['acc']:>9.3f}  {wt['acc']:>10.3f}")

    # ── Synthesis ─────────────────────────────────────────────────────────────
    W("")
    W(SEP)
    W("  SYNTHESIS — ALPHA DYNAMICS")
    W(SEP)
    W("""
  HYPOTHESIS VERDICT:
    "Pre-transition: fixed low beats adaptive"
    → CONFIRMED if fixed_005 > adaptive at same probe points pre-transition
    → REJECTED if adaptive matches or exceeds fixed_005 throughout

    "Post-transition: adaptive beats fixed"
    → CONFIRMED if acc_post_trans(adaptive) > acc_post_trans(fixed_005)
    → Look at trace values after transition_n

    "Switching at transition beats both"
    → CONFIRMED if switch_005_ad or switch_015_ad > max(fixed_005, adaptive)
    → Even if marginally — shows the mechanism is real

  ANNEALING:
    Annealing mimics switching without needing transition detection.
    If anneal beats switch: transition detection is unreliable/unnecessary.
    If switch beats anneal: real-time detection adds value.

  OPTIMAL ALPHA BY MODALITY:
    The best fixed alpha per modality reveals the regime:
    - Best α < 0.15 → pre-transition dominated run (centroid still migrating)
    - Best α ≈ 0.15 → balanced run (transition midway)  
    - Best α > 0.15 → post-transition dominated (fast adaptation needed)

  ADAPTIVE LEARNER IMPLICATION:
    If switch beats both parents → use switch strategy in adaptive learner:
      1. Train with fixed_low (0.05) until transition detected
      2. Switch to adaptive post-transition
      3. Transition detection: real-time σ drop (k=2 probe required)
    If anneal ≈ switch → simpler: just anneal α from 0.40 to 0.05 over budget.
    If fixed_low always wins → adaptive mechanism should be disabled entirely.
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
    parser = argparse.ArgumentParser(description="Profiler 3 — Alpha Dynamics")
    parser.add_argument("--modalities", nargs="+",
                        default=list(MODALITIES.keys()),
                        choices=list(MODALITIES.keys()))
    parser.add_argument("--quick", action="store_true",
                        help="n/class=[50] only, 1 seed (~8 min)")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    n_class_sweep = [50] if args.quick else N_CLASS_SWEEP
    n_seeds       = 1   if args.quick else args.seeds

    print(f"\n{'═'*72}")
    print("  CYPHA PROFILER 3 — EMA-ALPHA DYNAMICS & SWITCHING HYPOTHESIS")
    print(f"  Modalities:    {', '.join(args.modalities)}")
    print(f"  n/class sweep: {n_class_sweep}")
    print(f"  Strategies:    {len(STRATEGY_NAMES)}")
    print(f"  Seeds:         {n_seeds}")
    print(f"{'═'*72}\n")

    t_start = time.time()
    all_results = {}

    for mod_name in args.modalities:
        fn = MODALITIES[mod_name]
        all_results[mod_name] = run_modality(mod_name, fn, n_class_sweep, n_seeds)

    json_path = "profiler3_alpha_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  JSON → {json_path}")

    write_report(all_results, "profiler3_alpha_report.txt")
    print(f"\n  Total: {(time.time()-t_start)/60:.1f} min")
    print("  Done.\n")
