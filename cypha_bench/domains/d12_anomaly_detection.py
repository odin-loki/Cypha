"""Domain 12 — anomaly detection (NSL-KDD or synthetic)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from bench_common import (
    DEFAULT_SEED,
    clf_epistemic,
    finalize_domain,
    make_classifier,
    safe_auroc,
    train_classifier_online,
    data_path,
    rng,
)

NSL_KDD_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level",
]


def _synthetic_nsl_kdd(n_train: int = 5000, n_test: int = 2000, n_features: int = 41):
    g = rng(DEFAULT_SEED)
    X_train = g.standard_normal((n_train, n_features)).astype(np.float32)
    y_train = (g.random(n_train) < 0.2).astype(int)
    X_test = g.standard_normal((n_test, n_features)).astype(np.float32)
    attack_shift = g.standard_normal(n_features) * 0.5 + 1.5
    attack_mask = g.random(n_test) < 0.3
    X_test[attack_mask] += attack_shift
    y_test = attack_mask.astype(int)
    attack_types = np.where(y_test, "dos", "normal")
    return X_train, y_train, X_test, y_test, attack_types


def load_nsl_kdd():
    train_p = data_path("nsl_kdd", "KDDTrain+.txt")
    test_p = data_path("nsl_kdd", "KDDTest+.txt")
    if not (train_p.exists() and test_p.exists()):
        return _synthetic_nsl_kdd()

    train = pd.read_csv(train_p, header=None, names=NSL_KDD_COLS)
    test = pd.read_csv(test_p, header=None, names=NSL_KDD_COLS)
    for col in ["protocol_type", "service", "flag"]:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    train["binary_label"] = (train["label"] != "normal").astype(int)
    test["binary_label"] = (test["label"] != "normal").astype(int)
    feature_cols = [c for c in NSL_KDD_COLS if c not in ("label", "difficulty_level")]
    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train["binary_label"].values
    X_test = test[feature_cols].values.astype(np.float32)
    y_test = test["binary_label"].values
    attack_types = test["label"].values
    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train),
        y_train,
        scaler.transform(X_test),
        y_test,
        attack_types,
    )


def experiment_12a_binary_intrusion():
    X_train, y_train, X_test, y_test, _ = load_nsl_kdd()
    normal_mask = y_train == 0
    X_norm = X_train[normal_mask]
    clf = make_classifier(X_norm.shape[1])
    g = rng(DEFAULT_SEED)
    for _pass in range(3):
        order = g.permutation(len(X_norm))
        for i in order:
            clf.train_step(X_norm[i], "normal")

    scores = np.array([clf_epistemic(clf, x) for x in X_test])
    auroc = safe_auroc(y_test, scores)

    iso = IsolationForest(random_state=DEFAULT_SEED, contamination=0.2)
    iso.fit(X_norm)
    iso_scores = -iso.decision_function(X_test)
    iso_auroc = safe_auroc(y_test, iso_scores)

    return {
        "cypha_ood_auroc": auroc,
        "isolation_forest_auroc": iso_auroc,
        "data_source": "nsl_kdd" if data_path("nsl_kdd", "KDDTrain+.txt").exists() else "synthetic",
    }


def experiment_12b_attack_types():
    X_train, y_train, X_test, y_test, attack_types = load_nsl_kdd()
    main_types = {"normal": 0, "dos": 1, "probe": 2, "r2l": 3, "u2r": 4}

    def _map_label(lbl: str) -> int | None:
        lbl = str(lbl).lower()
        for key in main_types:
            if key in lbl:
                return main_types[key]
        return None

    tr_labels, tr_X = [], []
    for x, raw in zip(X_train, y_train):
        if raw == 0:
            tr_labels.append("normal")
            tr_X.append(x)
        else:
            tr_labels.append("attack")
            tr_X.append(x)
    tr_X = np.stack(tr_X)
    clf = make_classifier(tr_X.shape[1], seed=DEFAULT_SEED + 1)
    train_classifier_online(clf, tr_X, tr_labels, passes=3)

    scores = []
    for x in X_test[: min(2000, len(X_test))]:
        scores.append(clf_epistemic(clf, x))
    return {
        "mean_epistemic_attack": float(np.mean(scores)),
        "n_test": len(scores),
    }


def experiment_12c_online_latency():
    X_train, y_train, X_test, y_test, _ = load_nsl_kdd()
    clf = make_classifier(X_test.shape[1], seed=DEFAULT_SEED + 2)
    normal_idx = np.where(y_train == 0)[0][:500]
    for idx in normal_idx:
        clf.train_step(X_train[idx], "normal")

    # Stream attacks; count steps until accuracy > 0.8 on attack subset
    attack_idx = np.where(y_test == 1)[0][:200]
    if len(attack_idx) == 0:
        return {"detection_latency_steps": None}

    correct, latency = 0, None
    for step, idx in enumerate(attack_idx, start=1):
        x = X_test[idx]
        clf.train_step(x, "attack")
        pred, _ = clf.infer(x)
        if pred == "attack":
            correct += 1
        acc = correct / step
        if acc >= 0.8 and latency is None:
            latency = step
    return {"detection_latency_steps": latency, "final_attack_acc": float(correct / max(len(attack_idx), 1))}


def run() -> dict:
    experiments = {
        "12A_binary_intrusion": experiment_12a_binary_intrusion(),
        "12B_attack_types": experiment_12b_attack_types(),
        "12C_online_detection": experiment_12c_online_latency(),
    }
    return finalize_domain("d12", experiments)


if __name__ == "__main__":
    print(run())
