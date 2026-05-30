"""Domain 02 — regression benchmarks."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes, make_regression
from sklearn.model_selection import train_test_split

from cypha_bench.adapters.bench_models import BenchRegressor
from cypha_bench.config.load_profile import load_profile, regression_params
from cypha_bench.common.baselines import offline_regression_baselines, sgd_online_regressor
from cypha_bench.common.metrics import cypha_metrics, evaluate_regression, online_train_regressor, save_figure, save_table, standardize_train_test


def _load_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, loader in (
        ("diabetes", load_diabetes),
        ("california_housing", fetch_california_housing),
    ):
        try:
            x, y = loader(return_X_y=True)
            datasets[name] = (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        except Exception:
            continue

    if len(datasets) < 2:
        rng = np.random.default_rng(42)
        x, y = make_regression(n_samples=1500, n_features=12, noise=0.5, random_state=42)
        datasets["synthetic_regression"] = (x, y)
        x2, y2 = make_regression(n_samples=1200, n_features=8, noise=1.0, random_state=7)
        datasets["synthetic_regression_alt"] = (x2, y2)
    return datasets


def _run_dataset(name: str, x: np.ndarray, y: np.ndarray) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_test = standardize_train_test(x_train, x_test)

    model = BenchRegressor(x_train.shape[1])
    passes = 1
    if os.environ.get("CYPHA_BENCH_USE_PROFILE", "1").strip().lower() not in ("0", "false", "no"):
        passes = int(regression_params(load_profile()).get("n_epochs", 1))
    online_train_regressor(model, x_train, y_train, passes=passes)
    cypha_scores = evaluate_regression(model, x_test, y_test)
    cypha = cypha_metrics(model, x_test, y_test, task="regression")
    baselines = offline_regression_baselines(x_train, y_train, x_test, y_test)
    sgd = sgd_online_regressor(x_train, y_train, x_test, y_test)

    return {
        "dataset": name,
        "cypha_scores": cypha_scores,
        "cypha_metrics": cypha,
        "baselines": baselines,
        "sgd_online": sgd,
    }


def run() -> dict:
    results = [_run_dataset(name, x, y) for name, (x, y) in _load_datasets().items()]
    metrics = {"domain": "d02_regression", "datasets": results}

    names = [r["dataset"] for r in results]
    cypha_rmse = [r["cypha_scores"]["rmse"] for r in results]
    sgd_rmse = [r["sgd_online"]["rmse"] for r in results]
    rf_rmse = [r["baselines"]["random_forest"]["rmse"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.25, cypha_rmse, 0.25, label="CyphaDIF")
    ax.bar(xpos, sgd_rmse, 0.25, label="SGD online")
    ax.bar(xpos + 0.25, rf_rmse, 0.25, label="Random Forest")
    ax.set_xticks(xpos, names)
    ax.set_ylabel("RMSE")
    ax.set_title("Regression RMSE comparison")
    ax.legend()
    plt.tight_layout()

    save_table("d02_regression", metrics)
    save_figure(fig, "fig02_regression_rmse_comparison")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(run())
