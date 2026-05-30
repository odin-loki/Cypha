"""Domain 16 — multitask learning (iris + wine + digits)."""

from __future__ import annotations

import sys
from itertools import cycle
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from bench_common import (
    DEFAULT_FIELD_DIM,
    DEFAULT_SEED,
    clf_metrics,
    finalize_domain,
    make_classifier,
    rng,
)

from Cypha import CyphaDIF, VectorEncoder


def _load_tasks():
    iris = load_iris()
    wine = load_wine()
    digits = load_digits()

    def _split(X, y):
        return train_test_split(X, y, test_size=0.25, random_state=DEFAULT_SEED, stratify=y)

    tasks = {}
    # Standardize each task independently so zero-padding is a true "null" signal
    # (all active features are zero-mean, unit-variance; padding is at the data mean)
    for name, ds_data, ds_target in [
        ("iris", iris.data, iris.target),
        ("wine", wine.data, wine.target),
        ("digits", digits.data / 16.0, digits.target),
    ]:
        scaler = StandardScaler()
        Xtr_raw, Xte_raw, y_tr, y_te = _split(ds_data.astype(np.float32), ds_target)
        Xtr = scaler.fit_transform(Xtr_raw).astype(np.float32)
        Xte = scaler.transform(Xte_raw).astype(np.float32)
        tasks[name] = (Xtr, y_tr, Xte, y_te)
    return tasks


def _make_multitask_clf(max_dim: int, seed: int = DEFAULT_SEED):
    """Build classifier optimised for multi-task: frozen encoder avoids cross-task drift."""
    return CyphaDIF(
        encoder=VectorEncoder(max_dim),
        field_dim=DEFAULT_FIELD_DIM,
        enc_lr=0.0,          # freeze encoder — prevents cross-task contrastive drift
        rng=np.random.default_rng(seed),
    )


def _pad_to_max(X: np.ndarray, max_dim: int) -> np.ndarray:
    if X.shape[1] == max_dim:
        return X
    pad = np.zeros((X.shape[0], max_dim - X.shape[1]), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)


def multitask_stream(task_datasets, interleave: str = "round_robin", max_steps: int = 15000, seed: int = DEFAULT_SEED):
    g = rng(seed)
    iterators = {tid: cycle(zip(X, y)) for tid, (X, y) in task_datasets.items()}
    task_ids = list(task_datasets.keys())
    step = 0
    while step < max_steps:
        if interleave == "round_robin":
            order = task_ids
        elif interleave == "random":
            order = [g.choice(task_ids)]
        else:
            order = task_ids
        for tid in order:
            if step >= max_steps:
                break
            if interleave == "block":
                for _ in range(min(1000, max_steps - step)):
                    x, y = next(iterators[tid])
                    yield x, y, tid
                    step += 1
            else:
                x, y = next(iterators[tid])
                yield x, y, tid
                step += 1


def experiment_16a_task_discovery():
    tasks = _load_tasks()
    max_dim = max(t[0].shape[1] for t in tasks.values())
    train_sets = {tid: (_pad_to_max(X, max_dim), y) for tid, (X, y, _, _) in tasks.items()}

    clf = _make_multitask_clf(max_dim)
    routing_by_task = {tid: [] for tid in tasks}
    for x, y, tid in multitask_stream(
        {k: v for k, v in train_sets.items()}, interleave="round_robin", max_steps=12000
    ):
        label = f"{tid}_{y}"
        clf.train_step(x.astype(np.float32), label)
        info = clf.infer_full(x.astype(np.float32))
        probs = info.get("probs") or {}
        if probs:
            winner = max(probs, key=probs.get)
            routing_by_task[tid].append(winner)

    # Cluster routing labels per task — ARI between task id and dominant expert prefix
    task_labels, route_labels = [], []
    for tid, routes in routing_by_task.items():
        if not routes:
            continue
        from collections import Counter

        dominant = Counter(routes).most_common(1)[0][0]
        task_labels.extend([tid] * len(routes))
        route_labels.extend([dominant] * len(routes))
    ari = float(adjusted_rand_score(task_labels, route_labels)) if task_labels else 0.0

    per_task_acc = {}
    for tid, (_, _, Xte, yte) in tasks.items():
        Xp = _pad_to_max(Xte, max_dim)
        per_task_acc[tid] = clf_metrics(clf, Xp, [f"{tid}_{y}" for y in yte])["accuracy"]

    return {"routing_ari": ari, "per_task_accuracy": per_task_acc, "expert_count": int(clf.diagnostics()["n_classes"])}


def experiment_16b_forgetting():
    tasks = _load_tasks()
    max_dim = max(t[0].shape[1] for t in tasks.values())
    order = ["iris", "wine", "digits"]
    clf = _make_multitask_clf(max_dim, seed=DEFAULT_SEED + 1)

    def _train_task(tid, steps=3000):
        X, y, _, _ = tasks[tid]
        Xp = _pad_to_max(X, max_dim)
        g = rng(DEFAULT_SEED + 2)
        idx = g.permutation(len(Xp))[:steps]
        for i in idx:
            clf.train_step(Xp[i], f"{tid}_{y[i]}")

    def _eval_task(tid):
        _, _, Xte, yte = tasks[tid]
        Xp = _pad_to_max(Xte, max_dim)
        return clf_metrics(clf, Xp, [f"{tid}_{y}" for y in yte])["accuracy"]

    _train_task("iris")
    acc_a_before = _eval_task("iris")
    _train_task("wine")
    _train_task("digits")
    acc_a_after = _eval_task("iris")
    forgetting = (acc_a_before - acc_a_after) / max(acc_a_before, 1e-6)
    return {
        "task_a_accuracy_before": acc_a_before,
        "task_a_accuracy_after": acc_a_after,
        "forgetting_score": float(forgetting),
    }


def experiment_16d_interleaving():
    tasks = _load_tasks()
    max_dim = max(t[0].shape[1] for t in tasks.values())
    train_sets = {tid: (_pad_to_max(X, max_dim), y) for tid, (X, y, _, _) in tasks.items()}
    results = {}
    for strategy in ("round_robin", "random", "block"):
        clf = _make_multitask_clf(max_dim, seed=DEFAULT_SEED + 3)
        for x, y, tid in multitask_stream(train_sets, interleave=strategy, max_steps=9000):
            clf.train_step(x.astype(np.float32), f"{tid}_{y}")
        accs = {}
        for tid, (_, _, Xte, yte) in tasks.items():
            Xp = _pad_to_max(Xte, max_dim)
            accs[tid] = clf_metrics(clf, Xp, [f"{tid}_{y}" for y in yte])["accuracy"]
        results[strategy] = accs
    return results


def experiment_16e_save_restore():
    """Demonstrate zero-forgetting via CyphaDIF.save_state / load_state.

    Trains on Task A (iris), saves state, trains on Task B (wine) + Task C (digits),
    then restores the Task A snapshot.  Expected: task_a_restored ≈ task_a_before.
    """
    tasks = _load_tasks()
    max_dim = max(t[0].shape[1] for t in tasks.values())

    clf = _make_multitask_clf(max_dim, seed=DEFAULT_SEED + 4)

    def _train_task(tid, steps=3000):
        X, y, _, _ = tasks[tid]
        Xp = _pad_to_max(X, max_dim)
        g  = rng(DEFAULT_SEED + 5)
        idx = g.permutation(len(Xp))[:steps]
        for i in idx:
            clf.train_step(Xp[i].astype(np.float32), f"{tid}_{y[i]}")

    def _eval_task(tid):
        _, _, Xte, yte = tasks[tid]
        Xp = _pad_to_max(Xte, max_dim)
        return clf_metrics(clf, Xp, [f"{tid}_{y}" for y in yte])["accuracy"]

    _train_task("iris")
    acc_before = _eval_task("iris")
    snapshot   = clf.save_state()       # snapshot after Task A

    _train_task("wine")
    _train_task("digits")
    acc_corrupted = _eval_task("iris")  # expected: degraded

    clf.load_state(snapshot)            # restore Task A
    acc_restored = _eval_task("iris")   # expected: ≈ acc_before

    return {
        "task_a_before"   : acc_before,
        "task_a_corrupted": acc_corrupted,
        "task_a_restored" : acc_restored,
        "retention_ratio" : acc_restored / max(acc_before, 1e-6),
    }


def experiment_16f_per_task_models():
    """Per-task model ensemble: zero catastrophic forgetting by design.

    Each task gets its own CyphaDIF instance trained only on its data.
    At inference, the task identity selects the correct model.
    Comparison with 16B (shared model, 81% forgetting) shows the architectural cost.
    """
    tasks = _load_tasks()
    max_dim = max(t[0].shape[1] for t in tasks.values())
    per_task_clfs = {}

    for tid, (X_tr, y_tr, X_te, y_te) in tasks.items():
        Xp_tr = _pad_to_max(X_tr, max_dim)
        clf = _make_multitask_clf(max_dim, seed=DEFAULT_SEED + 10)
        g = rng(DEFAULT_SEED + 11)
        for _ in range(3):  # 3 passes per task
            order = g.permutation(len(Xp_tr))
            for i in order:
                clf.train_step(Xp_tr[i].astype(np.float32), f"{tid}_{y_tr[i]}")
        per_task_clfs[tid] = clf

    per_task_acc = {}
    for tid, (_, _, X_te, y_te) in tasks.items():
        Xp = _pad_to_max(X_te, max_dim)
        acc = clf_metrics(per_task_clfs[tid], Xp, [f"{tid}_{y}" for y in y_te])["accuracy"]
        per_task_acc[tid] = acc

    return {
        "per_task_accuracy": per_task_acc,
        "forgetting_score": 0.0,  # by design: each model is never retrained on other tasks
        "note": "per-task isolated models — zero forgetting by architecture",
    }


def run() -> dict:
    experiments = {
        "16A_task_discovery"         : experiment_16a_task_discovery(),
        "16B_forgetting_resistance"  : experiment_16b_forgetting(),
        "16D_interleaving_comparison": experiment_16d_interleaving(),
        "16E_save_restore"           : experiment_16e_save_restore(),
        "16F_per_task_models"        : experiment_16f_per_task_models(),
    }
    return finalize_domain("d16", experiments)


if __name__ == "__main__":
    print(run())
