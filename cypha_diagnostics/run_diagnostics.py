#!/usr/bin/env python3
"""
Cypha Full Diagnostic Suite — Phase 1-4 per cypha_diagnostic_plan.md
Covers: baseline establishment, encoder quality, NIG calibration, online learning dynamics.
"""
from __future__ import annotations
import json, os, sys, time, warnings
from pathlib import Path
import numpy as np
from sklearn.datasets import (
    make_classification, make_blobs, load_iris, load_wine,
    load_breast_cancer, load_digits, make_regression
)
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score, silhouette_score
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, SVR
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from scipy.stats import shapiro
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Cypha import CyphaDIF, VectorEncoder, RFFEncoder  # noqa: E402

warnings.filterwarnings("ignore")

OUT = ROOT / "cypha_diagnostics" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 13, 99, 17]
print(f"[DIAG] Running diagnostics. Output: {OUT}", flush=True)
print(f"[DIAG] Git commit: {ROOT}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def _make_cypha(X_tr, seed, delta_lr=0.05, world_lr=0.008, enc_lr=0.002,
                temp=1.0, replay=0.30, no_deliberation=True):
    d = X_tr.shape[1]
    rng = np.random.default_rng(seed)
    clf = CyphaDIF(encoder=VectorEncoder(d), field_dim=max(32, d*2),
                   delta_lr=delta_lr, world_lr=world_lr, enc_lr=enc_lr,
                   replay_ratio=replay, rng=rng)
    clf.temperature = temp
    if no_deliberation:
        clf.deliberation_lo = 1.0
        clf.deliberation_hi = 0.0
    return clf


def _make_cypha_rff(X_tr, seed, D=256, no_deliberation=True):
    d = X_tr.shape[1]
    rng = np.random.default_rng(seed)
    enc = RFFEncoder(d, D=D, gamma=1.0, seed=seed)
    enc.auto_gamma(X_tr[:min(500, len(X_tr))])
    clf = CyphaDIF(encoder=enc, field_dim=D, rng=rng)
    if no_deliberation:
        clf.deliberation_lo = 1.0
        clf.deliberation_hi = 0.0
    return clf


def train_online(clf, X_tr, y_tr, n_passes=1):
    idx = np.arange(len(X_tr))
    for _ in range(n_passes):
        for i in idx:
            clf.train_step(X_tr[i], str(y_tr[i]))


def eval_clf(clf, X_te, y_te):
    labels = sorted(set(str(c) for c in y_te))
    preds = [clf.infer(x)[0] for x in X_te]
    true_s = [str(c) for c in y_te]
    acc = float(accuracy_score(true_s, preds))
    f1 = float(f1_score(true_s, preds, average="macro", zero_division=0))
    return acc, f1


def extract_h(clf, X):
    """Extract encoded latent vectors h for all X."""
    hs = []
    for x in X:
        f = clf.encoder_fn(x).astype(np.float64)
        h = clf.encoder.project(f)
        hs.append(h)
    return np.stack(hs)


def scale_split(X, y, seed=42, test=0.25):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test, random_state=seed,
                                               stratify=y if len(np.unique(y)) > 1 else None)
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), y_tr, y_te


def fdr_score(H, y):
    """Mean Fisher Discriminant Ratio across dimensions."""
    labels = np.unique(y)
    if len(labels) < 2:
        return 0.0
    mu_all = H.mean(0)
    s_between = np.zeros(H.shape[1])
    s_within = np.zeros(H.shape[1])
    for c in labels:
        Hc = H[y == c]
        n = len(Hc)
        mu_c = Hc.mean(0)
        s_between += n * (mu_c - mu_all) ** 2
        s_within += Hc.var(0) * n
    s_within = np.maximum(s_within, 1e-12)
    return float(np.mean(s_between / s_within))


def calibration_ece(clf, X_te, y_te, n_bins=10):
    """Expected Calibration Error (binary bucket)."""
    true_s = [str(c) for c in y_te]
    confs, corrects = [], []
    for x, t in zip(X_te, true_s):
        pred, conf = clf.infer(x)
        confs.append(float(conf))
        corrects.append(1.0 if pred == t else 0.0)
    confs, corrects = np.array(confs), np.array(corrects)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        acc_bin = corrects[mask].mean()
        conf_bin = confs[mask].mean()
        ece += mask.sum() / len(confs) * abs(acc_bin - conf_bin)
    return float(ece)


def steps_to_accuracy(clf_fn, X_tr, y_tr, X_te, y_te, target=0.80):
    """How many samples until clf reaches target accuracy? Returns -1 if never."""
    clf = clf_fn()
    idx = np.arange(len(X_tr))
    np.random.default_rng(42).shuffle(idx)
    labels_te = [str(c) for c in y_te]
    window = 50
    for step, i in enumerate(idx):
        clf.train_step(X_tr[i], str(y_tr[i]))
        if (step + 1) % window == 0:
            preds = [clf.infer(x)[0] for x in X_te[:200]]
            acc = accuracy_score(labels_te[:200], preds)
            if acc >= target:
                return step + 1
    return -1


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Baseline Establishment
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PHASE 1 — Baseline Establishment", flush=True)
print("="*70, flush=True)

DATASETS = {
    "S1_2class_linear": make_classification(n_samples=3000, n_features=10, n_informative=5,
                                            n_redundant=2, n_classes=2, random_state=42),
    "S2_5class_gauss": make_blobs(n_samples=3000, centers=5, n_features=20, cluster_std=1.5, random_state=42),
    "S2_10class_gauss": make_blobs(n_samples=5000, centers=10, n_features=30, cluster_std=1.5, random_state=42),
    "S3_nonlinear_xor": None,  # Built below
    "S4_drift": None,           # Built below
    "R1_iris": (load_iris().data, load_iris().target),
    "R2_wine": (load_wine().data, load_wine().target),
    "R3_digits": (load_digits().data, load_digits().target),
    "R4_breast_cancer": (load_breast_cancer().data, load_breast_cancer().target),
}

# S3: XOR-style nonlinear
rng0 = np.random.default_rng(42)
X_xor = rng0.standard_normal((4000, 20))
y_xor = ((X_xor[:, 0] > 0) ^ (X_xor[:, 1] > 0)).astype(int)
DATASETS["S3_nonlinear_xor"] = (X_xor, y_xor)

# S4: Multimodal per class (3 Gaussians per class)
X_multi, y_multi = [], []
for c in range(6):
    for cluster in range(3):
        center = rng0.standard_normal(15) * 2 + c * 1.5
        Xi = rng0.standard_normal((166, 15)) + center
        X_multi.append(Xi); y_multi.append(np.full(166, c))
X_multi = np.vstack(X_multi); y_multi = np.concatenate(y_multi)
perm = rng0.permutation(len(X_multi))
X_multi, y_multi = X_multi[perm], y_multi[perm]
DATASETS["S4_multimodal"] = (X_multi, y_multi)

phase1 = {}

for ds_name, data in DATASETS.items():
    if data is None:
        continue
    X, y = data
    print(f"\n  [{ds_name}] n={len(X)}, d={X.shape[1]}, classes={len(np.unique(y))}", flush=True)
    accs_cypha_nodeli, accs_cypha_deli, accs_sgd, accs_pa, accs_gnb, accs_rf, accs_svm = [],[],[],[],[],[],[]
    eces, convergence_steps = [], []

    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = scale_split(X, y, seed=seed)
        labels_str = [str(c) for c in sorted(np.unique(y))]
        true_te = [str(c) for c in y_te]

        # Cypha WITHOUT deliberation
        clf_nd = _make_cypha(X_tr, seed, no_deliberation=True)
        train_online(clf_nd, X_tr, y_tr)
        acc_nd, _ = eval_clf(clf_nd, X_te, y_te)
        accs_cypha_nodeli.append(acc_nd)
        eces.append(calibration_ece(clf_nd, X_te[:100], y_te[:100]))

        # Cypha WITH deliberation [0.4, 0.6] (matching current profile)
        clf_d = _make_cypha(X_tr, seed, no_deliberation=False)
        clf_d.deliberation_lo = 0.4
        clf_d.deliberation_hi = 0.6
        train_online(clf_d, X_tr, y_tr)
        acc_d, _ = eval_clf(clf_d, X_te, y_te)
        accs_cypha_deli.append(acc_d)

        # SGD online
        sgd = SGDClassifier(max_iter=1, warm_start=True, random_state=seed, loss="log_loss")
        for xi, yi in zip(X_tr, y_tr):
            sgd.partial_fit([xi], [yi], classes=np.unique(y))
        acc_sgd = float(accuracy_score(true_te, [str(p) for p in sgd.predict(X_te)]))
        accs_sgd.append(acc_sgd)

        # Passive-Aggressive
        pa = PassiveAggressiveClassifier(max_iter=1, warm_start=True, random_state=seed)
        for xi, yi in zip(X_tr, y_tr):
            pa.partial_fit([xi], [yi], classes=np.unique(y))
        acc_pa = float(accuracy_score(true_te, [str(p) for p in pa.predict(X_te)]))
        accs_pa.append(acc_pa)

        # Gaussian NB online
        gnb = GaussianNB()
        for xi, yi in zip(X_tr, y_tr):
            gnb.partial_fit([xi], [yi], classes=np.unique(y))
        acc_gnb = float(accuracy_score(true_te, [str(p) for p in gnb.predict(X_te)]))
        accs_gnb.append(acc_gnb)

        # Batch RF ceiling
        rf = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf.fit(X_tr, y_tr)
        acc_rf = float(accuracy_score(true_te, [str(p) for p in rf.predict(X_te)]))
        accs_rf.append(acc_rf)

        # Batch SVM ceiling
        svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)
        svm.fit(X_tr, y_tr)
        acc_svm = float(accuracy_score(true_te, [str(p) for p in svm.predict(X_te)]))
        accs_svm.append(acc_svm)

    ceiling = float(np.mean(accs_svm))
    cypha_mean = float(np.mean(accs_cypha_nodeli))
    headroom = ceiling - cypha_mean
    online_gap = float(np.mean(accs_sgd)) - cypha_mean
    saturation = "SATURATED" if headroom < 0.02 else ("TIGHT" if headroom < 0.10 else "OPEN")

    res = {
        "cypha_nodeli_mean": round(cypha_mean, 4),
        "cypha_nodeli_std": round(float(np.std(accs_cypha_nodeli)), 4),
        "cypha_deli_mean": round(float(np.mean(accs_cypha_deli)), 4),
        "deliberation_penalty": round(cypha_mean - float(np.mean(accs_cypha_deli)), 4),
        "sgd_online_mean": round(float(np.mean(accs_sgd)), 4),
        "pa_online_mean": round(float(np.mean(accs_pa)), 4),
        "gnb_online_mean": round(float(np.mean(accs_gnb)), 4),
        "rf_batch_ceiling": round(float(np.mean(accs_rf)), 4),
        "svm_batch_ceiling": round(ceiling, 4),
        "headroom": round(headroom, 4),
        "online_gap_vs_sgd": round(online_gap, 4),
        "ece_mean": round(float(np.mean(eces)), 4),
        "saturation": saturation,
    }
    phase1[ds_name] = res
    print(f"    Cypha(no-deli)={cypha_mean:.3f}  Cypha(deli)={float(np.mean(accs_cypha_deli)):.3f}  "
          f"SGD={float(np.mean(accs_sgd)):.3f}  RF={float(np.mean(accs_rf)):.3f}  "
          f"SVM={ceiling:.3f}  headroom={headroom:.3f}  [{saturation}]  "
          f"deli_penalty={cypha_mean - float(np.mean(accs_cypha_deli)):.3f}", flush=True)

with open(OUT / "phase1_baseline.json", "w") as f:
    json.dump(phase1, f, indent=2)
print(f"\n[PHASE 1 DONE] Saved {OUT / 'phase1_baseline.json'}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Encoder Quality Analysis
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PHASE 2 — Encoder Quality Analysis", flush=True)
print("="*70, flush=True)

OPEN_DS = {k: v for k, v in DATASETS.items()
           if phase1.get(k, {}).get("saturation") == "OPEN"}
if not OPEN_DS:
    OPEN_DS = {k: v for k, v in DATASETS.items() if k in ("S1_2class_linear", "R3_digits", "S2_10class_gauss")}

phase2 = {}
for ds_name, data in list(OPEN_DS.items())[:4]:  # top 4 open
    X, y = data
    X_tr, X_te, y_tr, y_te = scale_split(X, y, seed=42)
    labels = np.array([str(c) for c in y_te])
    print(f"\n  [{ds_name}]", flush=True)

    # 2.1 FDR and silhouette in encoded h space
    clf = _make_cypha(X_tr, 42, no_deliberation=True)
    train_online(clf, X_tr, y_tr)
    H_te = extract_h(clf, X_te)
    fdr = fdr_score(H_te, y_te)
    try:
        sil = float(silhouette_score(H_te, y_te, sample_size=min(500, len(y_te))))
    except:
        sil = 0.0

    # 2.2 Linearity test: linear(h) vs kernel(h) vs kernel(X)
    y_te_int = np.array(y_te, dtype=int) if y_te.dtype != object else np.array([int(c) for c in y_te])
    lin_h, kern_h, kern_x = 0.0, 0.0, 0.0
    try:
        lr_h = LogisticRegression(max_iter=500, random_state=42)
        lr_h.fit(extract_h(clf, X_tr), y_tr)
        lin_h = float(accuracy_score([str(c) for c in y_te], [str(c) for c in lr_h.predict(H_te)]))
    except:
        pass
    try:
        svm_h = SVC(kernel="rbf", C=5, gamma="scale", random_state=42)
        svm_h.fit(extract_h(clf, X_tr), y_tr)
        kern_h = float(accuracy_score([str(c) for c in y_te], [str(c) for c in svm_h.predict(H_te)]))
    except:
        pass
    try:
        svm_x = SVC(kernel="rbf", C=5, gamma="scale", random_state=42)
        svm_x.fit(X_tr, y_tr)
        kern_x = float(accuracy_score([str(c) for c in y_te], [str(c) for c in svm_x.predict(X_te)]))
    except:
        pass

    # 2.3 D_rff sweep
    drff_results = {}
    for D in [16, 32, 64, 128, 256, 512]:
        accs_d = []
        for seed in SEEDS[:3]:
            X_tr2, X_te2, y_tr2, y_te2 = scale_split(X, y, seed=seed)
            clf_r = _make_cypha_rff(X_tr2, seed, D=D, no_deliberation=True)
            train_online(clf_r, X_tr2, y_tr2)
            a, _ = eval_clf(clf_r, X_te2, y_te2)
            accs_d.append(a)
        drff_results[str(D)] = round(float(np.mean(accs_d)), 4)

    # 2.4 Normality tests on h (Shapiro-Wilk per dim per class)
    H_tr = extract_h(clf, X_tr)
    sw_passes = {}
    for c in np.unique(y_tr):
        Hc = H_tr[np.array(y_tr) == c]
        if len(Hc) < 8:
            continue
        passes = 0
        for dim in range(min(H_tr.shape[1], 20)):
            try:
                _, pval = shapiro(Hc[:min(200, len(Hc)), dim])
                if pval > 0.05:
                    passes += 1
            except:
                pass
        sw_passes[str(c)] = passes / min(H_tr.shape[1], 20)
    mean_gaussian_frac = float(np.mean(list(sw_passes.values()))) if sw_passes else 0.0

    res = {
        "fdr": round(fdr, 4),
        "silhouette": round(sil, 4),
        "linear_h_acc": round(lin_h, 4),
        "kernel_h_acc": round(kern_h, 4),
        "kernel_x_acc": round(kern_x, 4),
        "nonlinearity_gap": round(kern_h - lin_h, 4),  # > 0.05 -> LLR linearity ceiling
        "drff_sweep": drff_results,
        "drff_best": max(drff_results, key=drff_results.get),
        "mean_gaussian_fraction_sw": round(mean_gaussian_frac, 4),
        "bottleneck_diagnosis": (
            "ENCODER_QUALITY" if fdr < 0.5 else
            "LLR_LINEARITY" if kern_h - lin_h > 0.05 else
            "NIG_MISSPEC" if mean_gaussian_frac < 0.5 else
            "NONE_OBVIOUS"
        ),
    }
    phase2[ds_name] = res
    print(f"    FDR={fdr:.3f} Sil={sil:.3f} Linear(h)={lin_h:.3f} Kernel(h)={kern_h:.3f} "
          f"Kernel(X)={kern_x:.3f} nonlin_gap={kern_h-lin_h:.3f}", flush=True)
    print(f"    D_rff sweep: {drff_results}  best_D={res['drff_best']}", flush=True)
    print(f"    Gaussian frac (SW): {mean_gaussian_frac:.3f}  Bottleneck: {res['bottleneck_diagnosis']}", flush=True)

with open(OUT / "phase2_encoder.json", "w") as f:
    json.dump(phase2, f, indent=2)
print(f"\n[PHASE 2 DONE] Saved {OUT / 'phase2_encoder.json'}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: NIG Calibration
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PHASE 3 — NIG Calibration & ECE", flush=True)
print("="*70, flush=True)

phase3 = {}
# Use R3_digits and S2_10class_gauss (high-headroom targets)
for ds_name in ["R3_digits", "S2_10class_gauss", "S1_2class_linear"]:
    data = DATASETS.get(ds_name)
    if data is None:
        continue
    X, y = data
    X_tr, X_te, y_tr, y_te = scale_split(X, y, seed=42)
    true_te = [str(c) for c in y_te]
    print(f"\n  [{ds_name}]", flush=True)

    # 3.1 ECE with different learning rate configs
    lr_results = {}
    for delta_lr in [0.01, 0.03, 0.05, 0.10, 0.20]:
        for world_lr in [0.003, 0.008, 0.02]:
            cfg = f"d{delta_lr}_w{world_lr}"
            eces_lr = []
            accs_lr = []
            for seed in SEEDS[:3]:
                X_tr2, X_te2, y_tr2, y_te2 = scale_split(X, y, seed=seed)
                clf = _make_cypha(X_tr2, seed, delta_lr=delta_lr, world_lr=world_lr,
                                  no_deliberation=True)
                train_online(clf, X_tr2, y_tr2)
                acc, _ = eval_clf(clf, X_te2, y_te2)
                eces_lr.append(calibration_ece(clf, X_te2[:150], y_te2[:150]))
                accs_lr.append(acc)
            lr_results[cfg] = {
                "acc": round(float(np.mean(accs_lr)), 4),
                "ece": round(float(np.mean(eces_lr)), 4),
            }
    # Best config by accuracy
    best_cfg = max(lr_results, key=lambda k: lr_results[k]["acc"])
    best_acc = lr_results[best_cfg]["acc"]
    best_ece = lr_results[best_cfg]["ece"]

    # 3.2 ECE baseline (default config)
    clf_def = _make_cypha(X_tr, 42, no_deliberation=True)
    train_online(clf_def, X_tr, y_tr)
    ece_default = calibration_ece(clf_def, X_te[:200], y_te[:200])
    acc_default, _ = eval_clf(clf_def, X_te, y_te)

    # Find best LR combo per acc
    sorted_configs = sorted(lr_results.items(), key=lambda x: -x[1]["acc"])[:5]
    print(f"    Default: acc={acc_default:.3f} ECE={ece_default:.3f}", flush=True)
    print(f"    Best LR config: {best_cfg} acc={best_acc:.3f} ECE={best_ece:.3f}", flush=True)
    print(f"    Top 5 LR configs: {[(c, v['acc']) for c, v in sorted_configs]}", flush=True)

    phase3[ds_name] = {
        "default_acc": round(acc_default, 4),
        "default_ece": round(ece_default, 4),
        "best_lr_config": best_cfg,
        "best_lr_acc": round(best_acc, 4),
        "best_lr_ece": round(best_ece, 4),
        "acc_gain_from_lr": round(best_acc - acc_default, 4),
        "lr_sweep": {k: v for k, v in sorted_configs},
        "calibration_verdict": "WELL_CALIBRATED" if ece_default < 0.05 else
                               ("MODERATE" if ece_default < 0.15 else "POORLY_CALIBRATED"),
    }

with open(OUT / "phase3_calibration.json", "w") as f:
    json.dump(phase3, f, indent=2)
print(f"\n[PHASE 3 DONE] Saved {OUT / 'phase3_calibration.json'}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: Online Learning Dynamics
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PHASE 4 — Online Learning Dynamics", flush=True)
print("="*70, flush=True)

phase4 = {}

# 4.1 Catastrophic forgetting test (most critical)
print("\n  [4.1 CATASTROPHIC FORGETTING]", flush=True)
X_clf, y_clf = make_blobs(n_samples=10000, centers=10, n_features=20,
                          cluster_std=1.5, random_state=42)
sc = StandardScaler(); X_clf = sc.fit_transform(X_clf)

def run_forgetting_test(seed):
    rng = np.random.default_rng(seed)
    # Phase 1: train on classes 0-4
    mask_p1 = y_clf < 5
    X_p1, y_p1 = X_clf[mask_p1], y_clf[mask_p1]
    idx_p1 = rng.permutation(len(X_p1))[:3000]
    X_p1s, y_p1s = X_p1[idx_p1], y_p1[idx_p1]

    # Hold-out test set for classes 0-4
    X_te_p1, y_te_p1 = X_p1[:300], y_p1[:300]

    clf = _make_cypha(X_clf, seed, no_deliberation=True)
    for xi, yi in zip(X_p1s, y_p1s):
        clf.train_step(xi, str(yi))

    # Accuracy after phase 1
    acc_before = eval_clf(clf, X_te_p1, y_te_p1)[0]

    # Phase 2: train on classes 5-9 ONLY
    mask_p2 = y_clf >= 5
    X_p2, y_p2 = X_clf[mask_p2], y_clf[mask_p2]
    idx_p2 = rng.permutation(len(X_p2))[:3000]
    for xi, yi in zip(X_p2[idx_p2], y_p2[idx_p2]):
        clf.train_step(xi, str(yi))

    # Re-test classes 0-4
    acc_after = eval_clf(clf, X_te_p1, y_te_p1)[0]
    forgetting = (acc_before - acc_after) / max(acc_before, 1e-9)
    return {"acc_before": acc_before, "acc_after": acc_after, "forgetting_ratio": forgetting}

forget_results = [run_forgetting_test(s) for s in SEEDS]
mean_forget = float(np.mean([r["forgetting_ratio"] for r in forget_results]))
verdict_forget = "EXCELLENT" if mean_forget < 0.05 else ("MODERATE" if mean_forget < 0.2 else "SEVERE")
phase4["catastrophic_forgetting"] = {
    "mean_forgetting_ratio": round(mean_forget, 4),
    "per_seed": [{"acc_before": round(r["acc_before"], 3), "acc_after": round(r["acc_after"], 3),
                  "ratio": round(r["forgetting_ratio"], 3)} for r in forget_results],
    "verdict": verdict_forget,
}
print(f"    Forgetting ratio: {mean_forget:.4f}  [{verdict_forget}]", flush=True)
for r in forget_results:
    print(f"      acc_before={r['acc_before']:.3f} -> acc_after={r['acc_after']:.3f} (ratio={r['forgetting_ratio']:.3f})", flush=True)


# 4.2 Multi-pass training: does a second pass help?
print("\n  [4.2 MULTI-PASS TRAINING]", flush=True)
X_mp, y_mp = make_blobs(n_samples=4000, centers=8, n_features=25, cluster_std=1.5, random_state=42)
sc_mp = StandardScaler(); X_mp = sc_mp.fit_transform(X_mp)
X_tr_mp, X_te_mp, y_tr_mp, y_te_mp = train_test_split(X_mp, y_mp, test_size=0.25, random_state=42)

mp_results = {}
for n_passes in [1, 2, 3, 5]:
    accs = []
    for seed in SEEDS[:3]:
        clf = _make_cypha(X_tr_mp, seed, no_deliberation=True)
        train_online(clf, X_tr_mp, y_tr_mp, n_passes=n_passes)
        a, _ = eval_clf(clf, X_te_mp, y_te_mp)
        accs.append(a)
    mp_results[str(n_passes)] = round(float(np.mean(accs)), 4)
phase4["multi_pass"] = mp_results
print(f"    Multi-pass results: {mp_results}", flush=True)


# 4.3 Learning rate sweep on primary benchmark
print("\n  [4.3 LEARNING RATE IMPACT — S2_10class_gauss]", flush=True)
X_lrs, y_lrs = DATASETS["S2_10class_gauss"]
X_tr_lr, X_te_lr, y_tr_lr, y_te_lr = scale_split(X_lrs, y_lrs, seed=42)

lr_acc = {}
for delta_lr in [0.01, 0.03, 0.05, 0.10, 0.20, 0.50]:
    for world_lr in [0.002, 0.008, 0.03]:
        key = f"d{delta_lr}_w{world_lr}"
        accs = []
        for seed in SEEDS[:3]:
            X_tr2, X_te2, y_tr2, y_te2 = scale_split(X_lrs, y_lrs, seed=seed)
            clf = _make_cypha(X_tr2, seed, delta_lr=delta_lr, world_lr=world_lr, no_deliberation=True)
            train_online(clf, X_tr2, y_tr2)
            a, _ = eval_clf(clf, X_te2, y_te2)
            accs.append(a)
        lr_acc[key] = round(float(np.mean(accs)), 4)

best_lr = max(lr_acc, key=lr_acc.get)
sorted_lr = sorted(lr_acc.items(), key=lambda x: -x[1])[:8]
phase4["lr_sweep"] = {"all": lr_acc, "best": best_lr, "best_acc": lr_acc[best_lr],
                      "top8": sorted_lr}
print(f"    Best LR: {best_lr} -> acc={lr_acc[best_lr]:.4f}", flush=True)
print(f"    Top LR configs: {sorted_lr[:5]}", flush=True)


# 4.4 Label noise robustness
print("\n  [4.4 LABEL NOISE ROBUSTNESS]", flush=True)
X_noise, y_noise = make_blobs(n_samples=5000, centers=6, n_features=20, cluster_std=1.2, random_state=42)
sc_n = StandardScaler(); X_noise = sc_n.fit_transform(X_noise)
X_tr_n, X_te_n, y_tr_n, y_te_n = train_test_split(X_noise, y_noise, test_size=0.25, random_state=42)

noise_results = {}
for noise_rate in [0.0, 0.05, 0.10, 0.20, 0.30]:
    accs = []
    for seed in SEEDS[:3]:
        rng_n = np.random.default_rng(seed)
        y_noisy = y_tr_n.copy()
        n_corrupt = int(noise_rate * len(y_noisy))
        flip_idx = rng_n.choice(len(y_noisy), n_corrupt, replace=False)
        y_noisy[flip_idx] = rng_n.integers(0, 6, n_corrupt)

        clf = _make_cypha(X_tr_n, seed, no_deliberation=True)
        train_online(clf, X_tr_n, y_noisy)
        a, _ = eval_clf(clf, X_te_n, y_te_n)
        accs.append(a)
    noise_results[str(noise_rate)] = round(float(np.mean(accs)), 4)
phase4["noise_robustness"] = noise_results
print(f"    Noise robustness: {noise_results}", flush=True)


# 4.5 Convergence speed
print("\n  [4.5 CONVERGENCE SPEED]", flush=True)
X_conv, y_conv = make_blobs(n_samples=8000, centers=5, n_features=20, cluster_std=1.3, random_state=42)
sc_cv = StandardScaler(); X_conv = sc_cv.fit_transform(X_conv)
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_conv, y_conv, test_size=0.25, random_state=42)

def acc_at_step(clf_fn, X_tr, y_tr, X_te, y_te, checkpoints):
    clf = clf_fn()
    idx = list(range(len(X_tr)))
    acc_curve = {}
    step = 0
    for i in idx:
        clf.train_step(X_tr[i], str(y_tr[i]))
        step += 1
        if step in checkpoints:
            a, _ = eval_clf(clf, X_te[:200], y_te[:200])
            acc_curve[step] = round(a, 4)
    return acc_curve

checkpoints = [50, 100, 200, 500, 1000, 2000, 5000]
cypha_curve = acc_at_step(lambda: _make_cypha(X_tr_c, 42, no_deliberation=True),
                          X_tr_c, y_tr_c, X_te_c, y_te_c, checkpoints)

# Compare with SGD
def sgd_curve(X_tr, y_tr, X_te, y_te, checkpoints):
    sgd = SGDClassifier(loss="log_loss", random_state=42)
    curve = {}
    for step, (xi, yi) in enumerate(zip(X_tr, y_tr), 1):
        sgd.partial_fit([xi], [yi], classes=np.unique(y_tr))
        if step in checkpoints:
            a = accuracy_score([str(c) for c in y_te[:200]], [str(c) for c in sgd.predict(X_te[:200])])
            curve[step] = round(float(a), 4)
    return curve

sgd_c = sgd_curve(X_tr_c, y_tr_c, X_te_c, y_te_c, checkpoints)
phase4["convergence"] = {"cypha": cypha_curve, "sgd": sgd_c}
print(f"    Cypha curve: {cypha_curve}", flush=True)
print(f"    SGD curve:   {sgd_c}", flush=True)

with open(OUT / "phase4_online_dynamics.json", "w") as f:
    json.dump(phase4, f, indent=2)
print(f"\n[PHASE 4 DONE] Saved {OUT / 'phase4_online_dynamics.json'}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5: Targeted Best-Config Validation
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("PHASE 5 — Best Config Validation on D01 Problem Cases", flush=True)
print("="*70, flush=True)

phase5 = {}

# Test the two biggest problem areas from baseline
for ds_name in ["S1_2class_linear", "R3_digits"]:
    data = DATASETS.get(ds_name)
    if data is None:
        continue
    X, y = data
    print(f"\n  [{ds_name}] — testing best configs from Phase 3+4:", flush=True)

    # Extract best LR from phase3
    p3 = phase3.get(ds_name, phase3.get("R3_digits", {}))
    best_cfg_str = p3.get("best_lr_config", "d0.05_w0.008")
    parts = best_cfg_str.split("_")
    best_d = float(parts[0][1:]) if parts else 0.05
    best_w = float(parts[1][1:]) if len(parts) > 1 else 0.008

    results = {}
    for label, kwargs in [
        ("default", {}),
        ("best_lr", {"delta_lr": best_d, "world_lr": best_w}),
        ("nodeli_best_lr", {"delta_lr": best_d, "world_lr": best_w}),
        ("multipass2", {"delta_lr": best_d, "world_lr": best_w}),
        ("rff128", {}),
        ("rff256", {}),
    ]:
        accs = []
        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = scale_split(X, y, seed=seed)
            if label == "rff128":
                clf = _make_cypha_rff(X_tr, seed, D=128, no_deliberation=True)
                train_online(clf, X_tr, y_tr)
            elif label == "rff256":
                clf = _make_cypha_rff(X_tr, seed, D=256, no_deliberation=True)
                train_online(clf, X_tr, y_tr)
            elif label == "multipass2":
                clf = _make_cypha(X_tr, seed, no_deliberation=True, **kwargs)
                train_online(clf, X_tr, y_tr, n_passes=2)
            else:
                clf = _make_cypha(X_tr, seed, no_deliberation=True, **kwargs)
                train_online(clf, X_tr, y_tr)
            a, _ = eval_clf(clf, X_te, y_te)
            accs.append(a)
        results[label] = round(float(np.mean(accs)), 4)

    best_label = max(results, key=results.get)
    phase5[ds_name] = results
    print(f"    Results: {results}", flush=True)
    print(f"    Best config: {best_label} -> {results[best_label]:.4f}", flush=True)

with open(OUT / "phase5_best_configs.json", "w") as f:
    json.dump(phase5, f, indent=2)
print(f"\n[PHASE 5 DONE] Saved {OUT / 'phase5_best_configs.json'}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70, flush=True)
print("DIAGNOSTIC SUMMARY", flush=True)
print("="*70, flush=True)

summary = {
    "phase1_open_benchmarks": [k for k, v in phase1.items() if v.get("saturation") == "OPEN"],
    "phase1_saturated": [k for k, v in phase1.items() if v.get("saturation") == "SATURATED"],
    "deliberation_penalties": {k: v.get("deliberation_penalty", 0) for k, v in phase1.items()},
    "worst_online_gap": max(phase1.values(), key=lambda v: v.get("online_gap_vs_sgd", 0)).get("online_gap_vs_sgd"),
    "phase2_bottlenecks": {k: v.get("bottleneck_diagnosis") for k, v in phase2.items()},
    "phase2_best_drff": {k: v.get("drff_best") for k, v in phase2.items()},
    "phase3_best_lr_gain": {k: v.get("acc_gain_from_lr", 0) for k, v in phase3.items()},
    "phase4_forgetting_ratio": phase4.get("catastrophic_forgetting", {}).get("mean_forgetting_ratio"),
    "phase4_forgetting_verdict": phase4.get("catastrophic_forgetting", {}).get("verdict"),
    "phase4_multipass_gain": None,
    "phase5_best_improvements": {k: max(v.values()) - v.get("default", 0) for k, v in phase5.items()},
}
if phase4.get("multi_pass"):
    mp = phase4["multi_pass"]
    summary["phase4_multipass_gain"] = round(mp.get("2", 0) - mp.get("1", 0), 4)

with open(OUT / "DIAGNOSTIC_SUMMARY.json", "w") as f:
    json.dump({"phase1": phase1, "phase2": phase2, "phase3": phase3,
               "phase4": phase4, "phase5": phase5, "summary": summary}, f, indent=2)

print(f"\nOpen benchmarks (headroom > 10%): {summary['phase1_open_benchmarks']}", flush=True)
print(f"Deliberation penalties: {summary['deliberation_penalties']}", flush=True)
print(f"Bottleneck diagnoses: {summary['phase2_bottlenecks']}", flush=True)
print(f"Best D_rff: {summary['phase2_best_drff']}", flush=True)
print(f"LR sweep gains: {summary['phase3_best_lr_gain']}", flush=True)
print(f"Forgetting ratio: {summary['phase4_forgetting_ratio']} [{summary['phase4_forgetting_verdict']}]", flush=True)
print(f"Multi-pass gain: {summary['phase4_multipass_gain']}", flush=True)
print(f"\nAll results saved to {OUT}", flush=True)
print("\n[DIAGNOSTICS COMPLETE]", flush=True)
