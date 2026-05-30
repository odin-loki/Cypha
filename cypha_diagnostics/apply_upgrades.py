#!/usr/bin/env python3
"""
Apply concrete upgrades based on diagnostic findings. Tests each one immediately.
Upgrades:
  U-A: Fix deliberation band (most impactful — kills accuracy on binary tasks)
  U-B: Empirical Bayes LR warm-start (best LR from phase3/4 sweep)
  U-C: RFF encoder with auto_gamma (replace VectorEncoder in bench)
  U-D: Multi-pass training (2 passes over training data)
  U-E: Quasi-random RFF (Halton sequences instead of random ω)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.datasets import make_blobs, make_classification, load_digits, load_iris, load_wine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Cypha import CyphaDIF, VectorEncoder, RFFEncoder

RESULTS = ROOT / "cypha_diagnostics" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 13, 99, 17]

def scale_split(X, y, seed=42, test=0.25):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test, random_state=seed,
                                               stratify=y if len(np.unique(y)) > 1 else None)
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), y_tr, y_te

def eval_clf(clf, X_te, y_te):
    preds = [clf.infer(x)[0] for x in X_te]
    true_s = [str(c) for c in y_te]
    return float(accuracy_score(true_s, preds))

def train_online(clf, X_tr, y_tr, n_passes=1):
    idx = np.arange(len(X_tr))
    for _ in range(n_passes):
        np.random.default_rng(42).shuffle(idx)
        for i in idx:
            clf.train_step(X_tr[i], str(y_tr[i]))


# Load diagnostic summary if available
try:
    with open(RESULTS / "DIAGNOSTIC_SUMMARY.json") as f:
        diag = json.load(f)
    ph3 = diag.get("phase3", {})
    ph4 = diag.get("phase4", {})
    ph1 = diag.get("phase1", {})
    ph2 = diag.get("phase2", {})
    print("[UP] Loaded diagnostic results", flush=True)
except FileNotFoundError:
    diag = {}; ph1 = {}; ph2 = {}; ph3 = {}; ph4 = {}
    print("[UP] No diagnostic results found — using defaults", flush=True)


# Benchmark suite
BENCHMARKS = {
    "S1_2class": make_classification(n_samples=3000, n_features=10, n_informative=5,
                                     n_redundant=2, n_classes=2, random_state=42),
    "S2_10class": make_blobs(n_samples=5000, centers=10, n_features=30, cluster_std=1.5, random_state=42),
    "R1_iris": (load_iris().data, load_iris().target),
    "R2_wine": (load_wine().data, load_wine().target),
    "R3_digits": (load_digits().data, load_digits().target),
}

# SVM ceilings
print("[UP] Computing SVM ceilings...", flush=True)
ceilings = {}
for name, (X, y) in BENCHMARKS.items():
    X_tr, X_te, y_tr, y_te = scale_split(X, y, seed=42)
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    svm.fit(X_tr, y_tr)
    ceilings[name] = float(accuracy_score([str(c) for c in y_te], [str(c) for c in svm.predict(X_te)]))
    print(f"  SVM ceiling [{name}]: {ceilings[name]:.4f}", flush=True)


def run_config(config_name, build_fn, datasets=None, n_passes=1):
    """Run config on all benchmarks, return {ds: mean_acc}."""
    results = {}
    ds_to_test = datasets or BENCHMARKS
    for name, (X, y) in ds_to_test.items():
        accs = []
        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = scale_split(X, y, seed=seed)
            clf = build_fn(X_tr, seed)
            train_online(clf, X_tr, y_tr, n_passes=n_passes)
            a = eval_clf(clf, X_te, y_te)
            accs.append(a)
        results[name] = round(float(np.mean(accs)), 4)
    print(f"  [{config_name}] {results}", flush=True)
    return results


print("\n" + "="*65, flush=True)
print("UPGRADE EVALUATION", flush=True)
print("="*65, flush=True)

upgrade_results = {}

# ─── BASELINE (no upgrades) ───────────────────────────────────────────────
print("\n[BASE] Default config, no-deliberation:", flush=True)
def build_base(X_tr, seed):
    d = X_tr.shape[1]
    clf = CyphaDIF(encoder=VectorEncoder(d), field_dim=max(32, d*2),
                   delta_lr=0.05, world_lr=0.008, enc_lr=0.002,
                   replay_ratio=0.30, rng=np.random.default_rng(seed))
    clf.temperature = 1.0
    clf.deliberation_lo = 1.0; clf.deliberation_hi = 0.0
    return clf
base_results = run_config("BASE", build_base)
upgrade_results["BASE"] = base_results


# ─── U-A: Fix deliberation (disable it) ───────────────────────────────────
# The existing profile has [0.4, 0.6] which was shown to suppress predictions.
# U-A compares profile-with-deliberation vs profile-without for binary tasks.
print("\n[U-A] Profile WITH deliberation [0.4,0.6] vs WITHOUT:", flush=True)
from cypha_bench.adapters.bench_models import BenchClassifier

def build_profile_nodeli(X_tr, seed):
    clf = BenchClassifier(X_tr.shape[1], seed=seed)
    clf.dif.deliberation_lo = 1.0
    clf.dif.deliberation_hi = 0.0
    return clf.dif

def build_profile_deli(X_tr, seed):
    clf = BenchClassifier(X_tr.shape[1], seed=seed)
    # Uses profile defaults: lo=0.4, hi=0.6
    return clf.dif

ua_nodeli = run_config("UA_nodeli", build_profile_nodeli)
ua_deli = run_config("UA_deli", build_profile_deli)
upgrade_results["UA_nodeli"] = ua_nodeli
upgrade_results["UA_deli"] = ua_deli
deli_penalty = {k: round(ua_nodeli.get(k, 0) - ua_deli.get(k, 0), 4) for k in ua_nodeli}
print(f"  Deliberation penalty (nodeli - deli): {deli_penalty}", flush=True)


# ─── U-B: Empirical Bayes best LR ─────────────────────────────────────────
# From Phase 3/4 diagnostic, the best LR config was identified per benchmark.
# Here we test a few specific better LR combos vs default.
print("\n[U-B] Better learning rates (empirical Bayes warm-start proxy):", flush=True)

# Extract best LR from phase3 if available
best_lrs = {}
for ds_k in ph3:
    cfg = ph3[ds_k].get("best_lr_config", "d0.05_w0.008")
    parts = cfg.split("_")
    try:
        best_lrs[ds_k] = {"delta": float(parts[0][1:]), "world": float(parts[1][1:])}
    except:
        best_lrs[ds_k] = {"delta": 0.05, "world": 0.008}

# Apply best LR from diagnostics (or tested candidates)
for lr_label, dlr, wlr in [
    ("LR_default", 0.05, 0.008),
    ("LR_high_delta", 0.10, 0.008),
    ("LR_balanced", 0.08, 0.015),
    ("LR_slow", 0.03, 0.005),
]:
    def build_lr(X_tr, seed, _dlr=dlr, _wlr=wlr):
        d = X_tr.shape[1]
        clf = CyphaDIF(encoder=VectorEncoder(d), field_dim=max(32, d*2),
                       delta_lr=_dlr, world_lr=_wlr,
                       rng=np.random.default_rng(seed))
        clf.deliberation_lo = 1.0; clf.deliberation_hi = 0.0
        return clf
    res = run_config(lr_label, build_lr)
    upgrade_results[lr_label] = res


# ─── U-C: RFF encoder (replace VectorEncoder) ─────────────────────────────
print("\n[U-C] RFF encoder (auto_gamma) at D=128, 256, 512:", flush=True)
for D in [64, 128, 256, 512]:
    def build_rff(X_tr, seed, _D=D):
        d = X_tr.shape[1]
        enc = RFFEncoder(d, D=_D, gamma=1.0, seed=seed)
        enc.auto_gamma(X_tr[:min(500, len(X_tr))])
        clf = CyphaDIF(encoder=enc, field_dim=_D,
                       delta_lr=0.05, world_lr=0.008,
                       rng=np.random.default_rng(seed))
        clf.deliberation_lo = 1.0; clf.deliberation_hi = 0.0
        return clf
    res = run_config(f"RFF_D{D}", build_rff)
    upgrade_results[f"RFF_D{D}"] = res


# ─── U-D: Multi-pass training (2-5 passes) ────────────────────────────────
print("\n[U-D] Multi-pass training:", flush=True)
for n_p in [1, 2, 3]:
    def build_mp(X_tr, seed):
        d = X_tr.shape[1]
        clf = CyphaDIF(encoder=VectorEncoder(d), field_dim=max(32, d*2),
                       delta_lr=0.05, world_lr=0.008,
                       rng=np.random.default_rng(seed))
        clf.deliberation_lo = 1.0; clf.deliberation_hi = 0.0
        return clf
    res = run_config(f"MULTIPASS_{n_p}", build_mp, n_passes=n_p)
    upgrade_results[f"MULTIPASS_{n_p}"] = res


# ─── U-E: Quasi-random RFF (Halton sequences) ─────────────────────────────
print("\n[U-E] Quasi-random RFF (Halton vs random):", flush=True)

def halton(n, dim, base_start=2):
    """Low-discrepancy Halton sequence for n samples, dim dimensions."""
    def halton_1d(n, base):
        seq = np.zeros(n)
        f, r = 1.0, 0
        for i in range(n):
            f /= base
            r += f
            if r >= 1.0:
                r -= 1.0
            seq[i] = r
        # Proper implementation
        result = np.zeros(n)
        for i in range(1, n + 1):
            f, r = 1, 0
            ii = i
            while ii > 0:
                f /= base
                r += f * (ii % base)
                ii //= base
            result[i - 1] = r
        return result
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    cols = []
    for d in range(dim):
        p = primes[d % len(primes)]
        seq = np.zeros(n)
        for i in range(1, n + 1):
            f, r = 1, 0
            ii = i
            while ii > 0:
                f /= p
                r += f * (ii % p)
                ii //= p
            seq[i - 1] = r
        cols.append(seq)
    H = np.stack(cols, axis=1)
    return H


class HaltonRFFEncoder(RFFEncoder):
    """RFF with quasi-random Halton frequency sampling instead of Gaussian random."""
    def _init_weights(self) -> None:
        import math
        D, d = self._D, self._input_dim
        # Use Halton for uniform [0,1]^D×d, then inverse-Gaussian transform
        H = halton(D, d)
        # Box-Muller transform to get N(0,1) from uniform pairs
        # Stack two Halton draws
        H1 = halton(D, d)
        H2 = halton(D, d)
        H1 = np.clip(H1, 1e-10, 1 - 1e-10)
        H2 = np.clip(H2, 1e-10, 1 - 1e-10)
        W_normal = np.sqrt(-2 * np.log(H1)) * np.cos(2 * np.pi * H2)
        self.W = (W_normal * self._gamma).astype(np.float64)
        rng = np.random.default_rng(self._seed)
        self.b = rng.uniform(0, 2.0 * math.pi, self._D).astype(np.float64)
        self._scale = math.sqrt(2.0 / self._D)


for D in [128, 256]:
    def build_halton(X_tr, seed, _D=D):
        d = X_tr.shape[1]
        enc = HaltonRFFEncoder(d, D=_D, gamma=1.0, seed=seed)
        enc.auto_gamma(X_tr[:min(500, len(X_tr))])
        clf = CyphaDIF(encoder=enc, field_dim=_D,
                       delta_lr=0.05, world_lr=0.008,
                       rng=np.random.default_rng(seed))
        clf.deliberation_lo = 1.0; clf.deliberation_hi = 0.0
        return clf
    res = run_config(f"HALTON_D{D}", build_halton)
    upgrade_results[f"HALTON_D{D}"] = res


# ─── U-F: RFF + Best LR + Multi-pass (combined best) ──────────────────────
print("\n[U-F] Combined: RFF256 + best LR + 2 passes:", flush=True)
def build_combined(X_tr, seed):
    d = X_tr.shape[1]
    enc = RFFEncoder(d, D=256, gamma=1.0, seed=seed)
    enc.auto_gamma(X_tr[:min(500, len(X_tr))])
    clf = CyphaDIF(encoder=enc, field_dim=256,
                   delta_lr=0.08, world_lr=0.015, enc_lr=0.002,
                   replay_ratio=0.35,
                   rng=np.random.default_rng(seed))
    clf.deliberation_lo = 1.0; clf.deliberation_hi = 0.0
    return clf
uf_res = run_config("COMBINED_BEST", build_combined, n_passes=2)
upgrade_results["COMBINED_BEST"] = uf_res


# ─── Summary ──────────────────────────────────────────────────────────────
print("\n" + "="*65, flush=True)
print("UPGRADE COMPARISON vs SVM CEILING", flush=True)
print("="*65, flush=True)

for name, res in sorted(upgrade_results.items()):
    gaps = {ds: round(ceilings.get(ds, 1.0) - res.get(ds, 0.0), 3) for ds in res}
    avg_acc = round(float(np.mean(list(res.values()))), 4)
    avg_gap = round(float(np.mean(list(gaps.values()))), 4)
    vs_base = round(avg_acc - float(np.mean(list(base_results.values()))), 4)
    print(f"  {name:22s} avg={avg_acc:.4f}  gap_to_SVM={avg_gap:.4f}  vs_base={vs_base:+.4f}  {res}", flush=True)

# Keep-or-revert decisions
print("\n" + "="*65, flush=True)
print("KEEP / REVERT DECISIONS", flush=True)
print("="*65, flush=True)
base_avg = float(np.mean(list(base_results.values())))
threshold = base_avg - 0.005  # must not regress by more than 0.5%

keep = {}
for name, res in upgrade_results.items():
    if name == "BASE":
        continue
    avg = float(np.mean(list(res.values())))
    decision = "KEEP" if avg >= threshold else "REVERT"
    keep[name] = {"avg": round(avg, 4), "delta": round(avg - base_avg, 4), "decision": decision}
    print(f"  {name:22s} avg={avg:.4f} ({avg-base_avg:+.4f})  → {decision}", flush=True)

with open(RESULTS / "upgrade_evaluation.json", "w") as f:
    json.dump({"ceilings": ceilings, "base": base_results, "upgrades": upgrade_results, "decisions": keep}, f, indent=2)
print(f"\nSaved {RESULTS / 'upgrade_evaluation.json'}", flush=True)
print("[UPGRADE EVAL DONE]", flush=True)
