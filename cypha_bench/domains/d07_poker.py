"""Domain 07 — poker decision classification."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

import os

from cypha_bench.adapters.bench_models import BenchClassifier
from cypha_bench.common.baselines import offline_classification_baselines, sgd_online_classifier
from cypha_bench.common.metrics import cypha_metrics, evaluate_classification, online_train_classifier, save_figure, save_table, standardize_train_test
from cypha_bench.common.paths import scale
from cypha_bench.config.load_profile import classification_params, load_profile
from cypha_bench.encoders import PokerEncoder


def generate_poker_dataset(n_hands: int = 12000, rng_seed: int = 42):
    enc = PokerEncoder()
    rng = np.random.default_rng(rng_seed)
    x_rows, labels = [], []
    for _ in range(n_hands):
        vec, label = enc.generate_random_situation(rng)
        x_rows.append(vec)
        labels.append(label)
    return np.asarray(x_rows, dtype=np.float64), np.asarray(labels)


def run() -> dict:
    x, y = generate_poker_dataset(n_hands=scale(12000, 3000))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train, x_test = standardize_train_test(x_train, x_test)

    model = BenchClassifier(x_train.shape[1])
    n_passes = 1
    if os.environ.get("CYPHA_BENCH_USE_PROFILE", "1").strip().lower() not in ("0", "false", "no"):
        n_passes = max(1, int(classification_params(load_profile()).get("n_epochs", 1)))
    online_train_classifier(model, x_train, y_train, label_fn=str, passes=n_passes)
    scores = evaluate_classification(model, x_test, y_test, label_fn=str)
    cypha = cypha_metrics(model, x_test, y_test, task="classification")
    baselines = offline_classification_baselines(x_train, y_train, x_test, y_test)
    sgd = sgd_online_classifier(x_train, y_train, x_test, y_test, classes=np.unique(y_train))

    hand_strength = x_test[:, 0]
    epistemic = []
    for xi in x_test:
        _, _, ep = model.predict(xi)
        epistemic.append(ep)
    epistemic = np.asarray(epistemic)
    boundary_dist = np.abs(hand_strength - 0.35)
    rho, _ = spearmanr(epistemic, boundary_dist)
    boundary_corr = float(rho) if np.isfinite(rho) else 0.0

    metrics = {
        "domain": "d07_poker",
        "n_hands": int(len(x)),
        "cypha_scores": scores,
        "cypha_metrics": cypha,
        "baselines": baselines,
        "sgd_online": sgd,
        "boundary_uncertainty_spearman": boundary_corr,
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        ["CyphaDIF", "SGD", "RF"],
        [scores["accuracy"], sgd["accuracy"], baselines["random_forest"]["accuracy"]],
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Poker decision accuracy")
    axes[1].scatter(boundary_dist, epistemic, s=8, alpha=0.4)
    axes[1].set_xlabel("|hand_strength - 0.35|")
    axes[1].set_ylabel("Epistemic proxy")
    axes[1].set_title("Uncertainty vs fold/call boundary")
    plt.tight_layout()

    save_table("d07_poker", metrics)
    save_figure(fig, "fig07_poker_decision_accuracy")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(run())
