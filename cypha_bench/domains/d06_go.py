"""Domain 06 — Go territory estimation."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from cypha_bench.adapters.bench_models import BenchClassifier, BenchRegressor
from cypha_bench.common.baselines import offline_classification_baselines, offline_regression_baselines, sgd_online_classifier, sgd_online_regressor
from cypha_bench.common.metrics import cypha_metrics, evaluate_classification, evaluate_regression, online_train_classifier, online_train_regressor, save_figure, save_table, standardize_train_test
from cypha_bench.common.paths import scale
from cypha_bench.encoders import GoEncoder


def generate_synthetic_go_position(rng: np.random.Generator, n_stones: int = 20):
    board = np.zeros((9, 9), dtype=np.float32)
    positions = rng.choice(81, size=n_stones, replace=False)
    for i, pos in enumerate(positions):
        r, c = divmod(int(pos), 9)
        board[r, c] = 1.0 if i < n_stones // 2 else -1.0
    territory = float(board.sum())
    return board, territory


def generate_dataset(n_samples: int = 8000, rng_seed: int = 42):
    rng = np.random.default_rng(rng_seed)
    enc = GoEncoder()
    x_rows, y_reg, y_cls = [], [], []
    for _ in range(n_samples):
        n_stones = int(rng.integers(5, 40))
        board, territory = generate_synthetic_go_position(rng, n_stones)
        features = enc.encode(board)
        x_rows.append(features)
        y_reg.append(territory)
        y_cls.append("black" if territory >= 0 else "white")
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_reg, dtype=np.float64), y_cls


def run() -> dict:
    x, y_reg, y_cls = generate_dataset(n_samples=scale(8000, 2000))
    x_train, x_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        x, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    x_train, x_test = standardize_train_test(x_train, x_test)

    reg = BenchRegressor(x_train.shape[1])
    online_train_regressor(reg, x_train, y_train_reg)
    reg_scores = evaluate_regression(reg, x_test, y_test_reg)
    reg_cypha = cypha_metrics(reg, x_test, y_test_reg, task="regression")
    reg_baselines = offline_regression_baselines(x_train, y_train_reg, x_test, y_test_reg)
    reg_sgd = sgd_online_regressor(x_train, y_train_reg, x_test, y_test_reg)

    clf = BenchClassifier(x_train.shape[1])
    online_train_classifier(clf, x_train, y_train_cls, label_fn=str)
    cls_scores = evaluate_classification(clf, x_test, y_test_cls, label_fn=str)
    cls_cypha = cypha_metrics(clf, x_test, y_test_cls, task="classification")
    cls_baselines = offline_classification_baselines(x_train, y_train_cls, x_test, y_test_cls)
    cls_sgd = sgd_online_classifier(x_train, y_train_cls, x_test, y_test_cls, classes=np.unique(y_train_cls))

    metrics = {
        "domain": "d06_go",
        "n_samples": int(len(x)),
        "regression": {
            "cypha_scores": reg_scores,
            "cypha_metrics": reg_cypha,
            "baselines": reg_baselines,
            "sgd_online": reg_sgd,
        },
        "classification": {
            "cypha_scores": cls_scores,
            "cypha_metrics": cls_cypha,
            "baselines": cls_baselines,
            "sgd_online": cls_sgd,
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        ["CyphaDIF", "SGD", "RF"],
        [reg_scores["rmse"], reg_sgd["rmse"], reg_baselines["random_forest"]["rmse"]],
    )
    axes[0].set_title("Territory regression RMSE")
    axes[1].bar(
        ["CyphaDIF", "SGD", "LogReg"],
        [
            cls_scores["accuracy"],
            cls_sgd["accuracy"],
            cls_baselines["logistic_regression"]["accuracy"],
        ],
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Outcome classification accuracy")
    plt.tight_layout()

    save_table("d06_go", metrics)
    save_figure(fig, "fig06_go_territory_regression")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(run())
