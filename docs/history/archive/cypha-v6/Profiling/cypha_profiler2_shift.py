"""
cypha_profiler2_shift.py
════════════════════════════════════════════════════════════════════════════════
PROFILER 2 — DISTRIBUTION SHIFT & ADAPTATION

Answers:
  - How much does accuracy drop when the data distribution shifts?
  - Does pseudo-label adaptation recover the lost accuracy?
  - Is recovery proportional to shift magnitude?
  - Does adaptation cause catastrophic forgetting of the original distribution?
  - Does a replay buffer (interleaving original examples) prevent forgetting?
  - What confidence threshold actually triggers writes during adaptation?
  - Is there a shift magnitude beyond which adaptation fails entirely?

Fixes over cypha_fewshot_online.py EXP3:
  - 2000 adaptation steps (vs 200 — 10x more)
  - Multiple shift magnitudes: 0.1, 0.2, 0.3, 0.5, 0.7 (vs just 0.3)
  - Confidence threshold sweep: 0.0 (always write), 1.1, 1.5, 2.0
  - Replay buffer test: 50% original A-examples interleaved with B-adaptation
  - Probes acc(A) AND acc(B) throughout adaptation, not just at end
  - Reports: drop, recovery, forgetting, write_count, write_accuracy per step

Output:
  profiler2_shift_results.json
  profiler2_shift_report.txt

Usage:
  python cypha_profiler2_shift.py
  python cypha_profiler2_shift.py --modalities text audio
  python cypha_profiler2_shift.py --quick   # fewer shift magnitudes, shorter adapt
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
N_CLASSES        = 6
N_PROBE          = 300           # probe set per distribution (50/class)
FEATURE_DIM      = 512
RESONANCE_DIM    = 256
TRAIN_N_CLASS    = 200           # labelled training examples/class on dist A
ADAPT_STEPS      = 2000          # unlabelled adaptation steps on dist B
PROBE_EVERY      = 100           # probe interval during adaptation
SHIFT_MAGNITUDES = [0.1, 0.2, 0.3, 0.5, 0.7]   # vocabulary/frequency blend ratio
CONF_THRESHOLDS  = [0.0, 1.1, 1.5, 2.0]         # exp(sim/temp) gates for writes
REPLAY_RATIO     = 0.5           # fraction of adaptation steps using A-replay

# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS (3 most interpretable modalities for shift analysis)
# ══════════════════════════════════════════════════════════════════════════════

# ── Text ──────────────────────────────────────────────────────────────────────
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

def text_ex(cls: int, shift: float = 0.0) -> str:
    kw = list(rng.choice(_VOCAB[cls], size=6, replace=False))
    if shift > 0:
        adj = (cls + 1) % N_CLASSES
        n_shift = max(1, int(round(shift * 6)))
        kw[-n_shift:] = list(rng.choice(_VOCAB[adj], size=n_shift, replace=False))
    sh = list(rng.choice(_SHARED, size=8, replace=True))
    tokens = kw + sh; rng.shuffle(tokens)
    return " ".join(tokens)

# ── Audio ─────────────────────────────────────────────────────────────────────
_FREQS = [220., 440., 880., 1760., 3520., 7040.]

def audio_ex(cls: int, shift: float = 0.0) -> str:
    t = np.arange(4000) / 8000.
    f0 = _FREQS[cls] * (1 + rng.normal(0, 0.01))
    if shift > 0:
        f1 = _FREQS[(cls + 1) % N_CLASSES]
        f0 = (1 - shift) * f0 + shift * f1
    sig = np.sin(2 * np.pi * f0 * t)
    if cls % 2 == 1:
        sig += 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
        sig /= (np.abs(sig).max() + 1e-9)
    sig += rng.normal(0, 0.02 + shift * 0.05, 4000)
    sig = np.clip(sig, -1, 1)
    return "pcm:" + (sig * 32700).astype(np.int16).tobytes().hex()

# ── Video (temporal aggregates) ────────────────────────────────────────────────
def video_ex(cls: int, shift: float = 0.0) -> str:
    P = 16; F = 8
    frames = np.zeros((F, P, P))
    y, x = np.mgrid[0:P, 0:P].astype(np.float64)
    target_cls = cls
    if shift > 0 and rng.random() < shift:
        target_cls = (cls + 1) % N_CLASSES   # probabilistic class blend
    for fi in range(F):
        t = fi / max(F - 1, 1)
        if   target_cls == 0: frames[fi] = np.sin(2*np.pi*3*y/P)
        elif target_cls == 1: frames[fi] = np.sin(2*np.pi*3*(y - t*P*.5)/P)
        elif target_cls == 2: frames[fi] = np.sin(2*np.pi*3*(x - t*P*.5)/P)
        elif target_cls == 3:
            r = np.sqrt((x-P/2)**2 + (y-P/2)**2)
            frames[fi] = np.sin(2*np.pi*(2+t*4)*r/(P/2))
        elif target_cls == 4: frames[fi] = np.sin(2*np.pi*3*(y - np.sin(2*np.pi*t*2)*P*.2)/P)
        else: frames[fi] = rng.normal(0, 1 if rng.random() < .3 else .1, (P,P))
        frames[fi] += rng.normal(0, .05 + shift * .1, (P,P))
    fm = frames.mean(0); fs = frames.std(0)
    df = np.abs(np.diff(frames, axis=0)); dm = df.mean(0); ds = df.std(0)
    agg = np.stack([fm, fs, dm, ds]).flatten().astype(np.float32)
    n = np.linalg.norm(agg)
    if n > 1e-9: agg /= n
    return "arr:" + base64.b64encode(agg.tobytes()).decode()

MODALITIES = {
    "text":  (text_ex,  "Text — vocabulary blend shift"),
    "audio": (audio_ex, "Audio — carrier frequency blend shift"),
    "video": (video_ex, "Video — motion pattern blend shift"),
}
CLASS_LABELS = [f"class_{i}" for i in range(N_CLASSES)]

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def fresh() -> Cypha:
    return Cypha(feature_dim=FEATURE_DIM, resonance_dim=RESONANCE_DIM)

def make_probe(fn, shift: float = 0.0) -> List[Tuple[str,str]]:
    per = N_PROBE // N_CLASSES
    return [(fn(cls, shift=shift), CLASS_LABELS[cls])
            for cls in range(N_CLASSES) for _ in range(per)]

def measure(cypha: Cypha, probe: List[Tuple[str,str]]) -> Tuple[float, float]:
    """Returns (accuracy, mean_confidence). Uses k=2 for real margins."""
    correct = 0; confs = []
    for inp, label in probe:
        try:
            q = cypha.encode_features(inp)
            matches = cypha.memory.lookup(q, k=2)
            if not matches: continue
            pred_label = cypha.memory.get_output(matches[0][0]) or matches[0][0]
            if pred_label == label: correct += 1
            margin = float(matches[0][1]) - (float(matches[1][1]) if len(matches)>1 else 0.)
            confs.append(float(np.exp(margin / max(cypha.temperature, 0.1))))
        except Exception:
            pass
    acc  = correct / max(len(probe), 1)
    conf = float(np.mean(confs)) if confs else 0.
    return round(acc, 4), round(conf, 4)

def build_pool(fn, shift: float = 0.0, n: int = 100) -> Dict[str, List[str]]:
    return {
        CLASS_LABELS[cls]: [fn(cls, shift=shift) for _ in range(n)]
        for cls in range(N_CLASSES)
    }

def get_negs(label: str, pool: Dict, k: int = 2) -> List[str]:
    negs = []
    for lbl, examples in pool.items():
        if lbl != label and examples:
            negs.append(examples[int(rng.integers(0, len(examples)))])
        if len(negs) >= k: break
    return negs

# Cypha contains unpicklable queues — we retrain from cached pairs instead.
_SAVED_PAIRS_A = []

def retrain_fresh(fn, n_pc: int, pool_A: Dict) -> Cypha:
    """Retrain a fresh Cypha on the cached dist-A pairs (same seed as base model)."""
    cypha = fresh()
    for inp, lbl in _SAVED_PAIRS_A:
        cypha.train_step(inp, lbl, negatives=get_negs(lbl, pool_A))
    return cypha

# ══════════════════════════════════════════════════════════════════════════════
# ADAPTATION EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def adapt_experiment(fn, n_pc: int, shift: float,
                     conf_threshold: float, use_replay: bool,
                     pool_A: Dict, pool_B: Dict,
                     probe_A: List, probe_B: List) -> Dict:
    """
    Run ADAPT_STEPS pseudo-label adaptation on dist B.
    conf_threshold: minimum exp(margin/temp) required to write. 0.0 = always write.
    use_replay: if True, interleave original A examples (REPLAY_RATIO of steps).
    Returns full trace + summary stats.
    """
    cypha = retrain_fresh(fn, n_pc, pool_A)

    # Build unlabelled B pool (true labels known for write_acc evaluation)
    unlabelled = []
    per = ADAPT_STEPS // N_CLASSES + 10
    for cls in range(N_CLASSES):
        for _ in range(per):
            unlabelled.append((fn(cls, shift=shift), CLASS_LABELS[cls]))
    rng.shuffle(unlabelled)

    # Build A replay pool
    replay_pool = []
    if use_replay:
        for cls in range(N_CLASSES):
            for _ in range(ADAPT_STEPS // N_CLASSES):
                replay_pool.append((fn(cls, shift=0.0), CLASS_LABELS[cls]))
        rng.shuffle(replay_pool)

    trace          = []
    writes         = 0
    correct_writes = 0
    triggered      = 0   # times conf threshold checked

    acc_A0, conf_A0 = measure(cypha, probe_A)
    acc_B0, conf_B0 = measure(cypha, probe_B)
    trace.append({"step": 0, "acc_A": acc_A0, "acc_B": acc_B0,
                  "conf_A": conf_A0, "conf_B": conf_B0,
                  "writes": 0, "write_acc": 0.})

    replay_idx = 0
    for step_i, (inp, true_label) in enumerate(unlabelled[:ADAPT_STEPS]):

        # Pseudo-label on B
        q    = cypha.encode_features(inp)
        mats = cypha.memory.lookup(q, k=2)
        if mats:
            pred_label = cypha.memory.get_output(mats[0][0]) or mats[0][0]
            margin = float(mats[0][1]) - (float(mats[1][1]) if len(mats)>1 else 0.)
            conf   = float(np.exp(margin / max(cypha.temperature, 0.1)))
            triggered += 1

            if conf_threshold == 0.0 or conf >= conf_threshold:
                cypha.train_step(inp, pred_label, negatives=get_negs(pred_label, pool_B))
                writes += 1
                if pred_label == true_label: correct_writes += 1

        # Replay: interleave A example
        if use_replay and replay_idx < len(replay_pool):
            r_inp, r_lbl = replay_pool[replay_idx]
            cypha.train_step(r_inp, r_lbl, negatives=get_negs(r_lbl, pool_A))
            replay_idx += 1

        if (step_i + 1) % PROBE_EVERY == 0:
            acc_A, conf_A = measure(cypha, probe_A)
            acc_B, conf_B = measure(cypha, probe_B)
            write_acc = correct_writes / max(writes, 1)
            trace.append({
                "step":      step_i + 1,
                "acc_A":     acc_A,
                "acc_B":     acc_B,
                "conf_A":    conf_A,
                "conf_B":    conf_B,
                "writes":    writes,
                "write_acc": round(write_acc, 4),
            })

    final_acc_A, _ = measure(cypha, probe_A)
    final_acc_B, _ = measure(cypha, probe_B)
    write_acc_final = correct_writes / max(writes, 1)

    return {
        "conf_threshold":      conf_threshold,
        "use_replay":          use_replay,
        "writes":              writes,
        "write_accuracy":      round(write_acc_final, 4),
        "triggered":           triggered,
        "write_rate":          round(writes / max(triggered, 1), 4),
        "acc_A_before":        acc_A0,
        "acc_B_before":        acc_B0,
        "acc_A_final":         final_acc_A,
        "acc_B_final":         final_acc_B,
        "recovery":            round(final_acc_B - acc_B0, 4),
        "forgetting":          round(acc_A0 - final_acc_A, 4),
        "net_gain":            round(final_acc_B - acc_B0 - (acc_A0 - final_acc_A), 4),
        "trace":               trace,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SHIFT SWEEP FOR ONE MODALITY
# ══════════════════════════════════════════════════════════════════════════════

def run_modality(mod_name: str, fn, shift_mags=None, adapt_steps=None, conf_thrs=None) -> Dict:
    if shift_mags  is None: shift_mags  = SHIFT_MAGNITUDES
    if adapt_steps is None: adapt_steps = ADAPT_STEPS
    if conf_thrs   is None: conf_thrs   = CONF_THRESHOLDS
    print(f"\n{'═'*70}")
    print(f"  MODALITY: {mod_name.upper()}")
    print(f"{'─'*70}")

    pool_A  = build_pool(fn, shift=0.0)
    probe_A = make_probe(fn, shift=0.0)

    # Pre-build and cache A training pairs (reused for each adapt_experiment call)
    global _SAVED_PAIRS_A
    _SAVED_PAIRS_A = []
    for cls in range(N_CLASSES):
        for _ in range(TRAIN_N_CLASS):
            _SAVED_PAIRS_A.append((fn(cls, shift=0.0), CLASS_LABELS[cls]))
    rng.shuffle(_SAVED_PAIRS_A)

    # Train once on dist A to measure baseline
    print("  Training base model on dist A...")
    t0 = time.time()
    cypha_base = fresh()
    for inp, lbl in _SAVED_PAIRS_A:
        cypha_base.train_step(inp, lbl, negatives=get_negs(lbl, pool_A))
    acc_A_train, _ = measure(cypha_base, probe_A)
    print(f"  Base model acc(A) = {acc_A_train:.3f}  ({time.time()-t0:.0f}s)")

    shift_results = []

    for shift in shift_mags:
        print(f"\n  shift={shift:.1f}")
        probe_B = make_probe(fn, shift=shift)
        pool_B  = build_pool(fn, shift=shift)

        # Measure baseline drop before any adaptation
        acc_B_base, _ = measure(cypha_base, probe_B)
        drop = round(acc_A_train - acc_B_base, 4)
        print(f"    Baseline: acc(A)={acc_A_train:.3f}  acc(B)={acc_B_base:.3f}  drop={drop:+.3f}")

        cond_results = []

        # Run all (conf_threshold × replay) combinations
        for conf_thr in conf_thrs:
            for use_replay in [False, True]:
                label = f"conf={conf_thr}_replay={int(use_replay)}"
                result = adapt_experiment(
                    fn, TRAIN_N_CLASS, shift, conf_thr, use_replay,
                    pool_A, pool_B, probe_A, probe_B
                )
                cond_results.append(result)
                print(f"    {label:<28}  "
                      f"writes={result['writes']:>5}  "
                      f"write_acc={result['write_accuracy']:.3f}  "
                      f"recovery={result['recovery']:+.3f}  "
                      f"forget={result['forgetting']:+.3f}  "
                      f"net={result['net_gain']:+.3f}")

        shift_results.append({
            "shift":         shift,
            "acc_A_base":    acc_A_train,
            "acc_B_before":  acc_B_base,
            "drop":          drop,
            "conditions":    cond_results,
        })

    return {
        "modality":        mod_name,
        "train_n_class":   TRAIN_N_CLASS,
        "adapt_steps":     adapt_steps,
        "acc_A_base":      acc_A_train,
        "shifts":          shift_results,
    }

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(results: Dict, path: str):
    SEP  = "═" * 78
    SEP2 = "─" * 76
    L = []; W = L.append

    W(SEP)
    W("  PROFILER 2 — DISTRIBUTION SHIFT & ADAPTATION")
    W(f"  Train={TRAIN_N_CLASS}/class on A  |  Adapt={ADAPT_STEPS} steps on B  |  "
      f"probe_n={N_PROBE}")
    W(f"  Shifts={SHIFT_MAGNITUDES}  |  Conf thresholds={CONF_THRESHOLDS}")
    W(f"  Replay ratio={REPLAY_RATIO} (when enabled)")
    W(SEP)

    for mod_name, mod_data in results.items():
        W(f"\n  {'─'*74}")
        W(f"  MODALITY: {mod_name.upper()}  "
          f"(base acc(A)={mod_data['acc_A_base']:.3f})")
        W(f"  {'─'*74}")

        # ── Drop table ────────────────────────────────────────────────────────
        W("")
        W("  ACCURACY DROP BY SHIFT MAGNITUDE:")
        W(f"  {'shift':>6}  {'acc(A)':>7}  {'acc(B)pre':>10}  {'drop':>6}  "
          f"{'drop_pct':>9}")
        for s in mod_data["shifts"]:
            drop_pct = s["drop"] / max(s["acc_A_base"], 0.01) * 100
            W(f"  {s['shift']:>6.1f}  {s['acc_A_base']:>7.3f}  "
              f"{s['acc_B_before']:>10.3f}  {s['drop']:>+6.3f}  "
              f"{drop_pct:>8.1f}%")

        # ── Adaptation results ────────────────────────────────────────────────
        W("")
        W("  ADAPTATION RESULTS (best conf_threshold, no replay):")
        W(f"  {'shift':>6}  {'conf_thr':>9}  {'writes':>7}  {'write_acc':>10}  "
          f"{'recovery':>9}  {'forgetting':>11}  {'net_gain':>9}")
        for s in mod_data["shifts"]:
            # Find best net_gain without replay
            no_replay = [c for c in s["conditions"] if not c["use_replay"]]
            best = max(no_replay, key=lambda c: c["net_gain"])
            W(f"  {s['shift']:>6.1f}  {best['conf_threshold']:>9.1f}  "
              f"{best['writes']:>7}  {best['write_accuracy']:>10.3f}  "
              f"{best['recovery']:>+9.3f}  {best['forgetting']:>+11.3f}  "
              f"{best['net_gain']:>+9.3f}")

        W("")
        W("  REPLAY BUFFER EFFECT (shift=0.3, all conf thresholds):")
        s3 = next((s for s in mod_data["shifts"] if abs(s["shift"]-0.3)<0.01), None)
        if s3:
            W(f"  {'conf_thr':>9}  {'replay':>7}  {'writes':>7}  "
              f"{'write_acc':>10}  {'recovery':>9}  {'forget':>8}  {'net':>7}")
            for c in s3["conditions"]:
                W(f"  {c['conf_threshold']:>9.1f}  "
                  f"{'yes' if c['use_replay'] else 'no':>7}  "
                  f"{c['writes']:>7}  {c['write_accuracy']:>10.3f}  "
                  f"{c['recovery']:>+9.3f}  {c['forgetting']:>+8.3f}  "
                  f"{c['net_gain']:>+7.3f}")

        W("")
        W("  WRITE_RATE vs CONF_THRESHOLD (shift=0.3, no replay):")
        W("  (write_rate = fraction of inference steps that triggered a write)")
        if s3:
            no_replay = [c for c in s3["conditions"] if not c["use_replay"]]
            W(f"  {'conf_thr':>9}  {'write_rate':>11}  {'writes':>7}  "
              f"{'write_acc':>10}")
            for c in no_replay:
                W(f"  {c['conf_threshold']:>9.1f}  "
                  f"{c['write_rate']:>11.4f}  "
                  f"{c['writes']:>7}  {c['write_accuracy']:>10.3f}")

        # ── Adaptation traces (shift=0.3, best conf, both replay conditions) ──
        W("")
        W("  ADAPTATION TRACE (shift=0.3, best conf threshold):")
        if s3:
            no_replay = [c for c in s3["conditions"] if not c["use_replay"]]
            best_conf = max(no_replay, key=lambda c: c["net_gain"])
            replay_eq = next(c for c in s3["conditions"]
                             if c["use_replay"] and
                             c["conf_threshold"] == best_conf["conf_threshold"])
            W(f"  conf_threshold={best_conf['conf_threshold']}")
            W(f"  {'step':>6}  {'acc_A(no-rpl)':>14}  {'acc_B(no-rpl)':>14}  "
              f"{'acc_A(replay)':>14}  {'acc_B(replay)':>14}")
            n_rows = min(len(best_conf["trace"]), len(replay_eq["trace"]))
            for i in range(n_rows):
                r0 = best_conf["trace"][i]
                r1 = replay_eq["trace"][i]
                W(f"  {r0['step']:>6}  {r0['acc_A']:>14.3f}  {r0['acc_B']:>14.3f}  "
                  f"{r1['acc_A']:>14.3f}  {r1['acc_B']:>14.3f}")

    # ── Cross-modality synthesis ───────────────────────────────────────────────
    W("")
    W(SEP)
    W("  SYNTHESIS — DISTRIBUTION SHIFT")
    W(SEP)
    W(f"""
  DROP vs SHIFT:
    Expected: monotonically increasing drop with shift magnitude.
    If drop is non-linear (slow then fast), there is a 'tolerance zone'
    where the centroid is robust to small perturbations.
    If drop is linear, every increase in shift costs proportional accuracy.

  RECOVERY vs WRITE_ACCURACY:
    Pseudo-labelling only helps when write_accuracy > ~70-85% (from EXP2).
    Recovery = f(write_accuracy × n_writes) — both matter.
    High write_acc + few writes (high conf_threshold) = slow but clean recovery.
    Low  write_acc + many writes (low conf_threshold) = fast but polluted.
    The optimal conf_threshold maximises net_gain = recovery - forgetting.

  REPLAY BUFFER:
    Replay prevents forgetting by keeping A-centroid anchored.
    Net_gain with replay > net_gain without replay when:
      forgetting_no_replay > recovery_improvement_from_replay
    If forgetting is already ~0, replay adds no benefit.

  CATASTROPHIC FORGETTING CONDITION:
    Forgetting is large when:
      - The shift is large (B is far from A)
      - Write accuracy is low (wrong labels pull centroid toward B noise)
      - No replay (A-centroid has no anchor)
    Forgetting approaches zero when:
      - The shift is small (B overlaps A)
      - Write accuracy is high (B-pseudo-labels are correct)
      - Replay is active

  SHIFT TOLERANCE THRESHOLD:
    The shift magnitude where acc(B) first drops below 0.80 defines the
    practical deployment tolerance. Beyond this, adaptation is required.
    The shift magnitude where adaptation fails (recovery < 0.05) defines
    the recovery limit — beyond which retraining from scratch is needed.
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
    parser = argparse.ArgumentParser(description="Profiler 2 — Distribution Shift")
    parser.add_argument("--modalities", nargs="+",
                        default=list(MODALITIES.keys()),
                        choices=list(MODALITIES.keys()))
    parser.add_argument("--quick", action="store_true",
                        help="Fewer shifts, shorter adaptation (5 min/modality)")
    args = parser.parse_args()

    # Quick mode: override module-level config
    sm  = [0.2, 0.5]   if args.quick else SHIFT_MAGNITUDES
    ast = 500           if args.quick else ADAPT_STEPS
    ct  = [0.0, 1.5]   if args.quick else CONF_THRESHOLDS

    print(f"\n{'═'*70}")
    print("  CYPHA PROFILER 2 — DISTRIBUTION SHIFT & ADAPTATION")
    print(f"  Modalities:  {', '.join(args.modalities)}")
    print(f"  Shifts:      {sm}")
    print(f"  Conf gates:  {ct}")
    print(f"  Adapt steps: {ast}  |  Train/class: {TRAIN_N_CLASS}")
    print(f"{'═'*70}\n")

    t_start = time.time()
    all_results = {}

    for mod_name in args.modalities:
        fn, _ = MODALITIES[mod_name]
        all_results[mod_name] = run_modality(mod_name, fn, sm, ast, ct)

    json_path = "profiler2_shift_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  JSON → {json_path}")

    write_report(all_results, "profiler2_shift_report.txt")
    print(f"\n  Total: {(time.time()-t_start)/60:.1f} min")
    print("  Done.\n")
