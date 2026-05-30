"""Domain 01 — statistical baselines and sanity checks."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.model_selection import train_test_split

from cypha_bench.adapters.bench_models import BenchClassifier, BenchRegressor
from cypha_bench.common.baselines import sgd_online_classifier, sgd_online_regressor
from cypha_bench.common.metrics import cypha_metrics, evaluate_classification, evaluate_regression, online_train_classifier, online_train_regressor, save_figure, save_table, standardize_train_test
from cypha_bench.common.paths import scale
from cypha_bench.config.load_profile import classification_params, load_profile

rng = np.random.default_rng(42)


def _make_sinusoidal(n: int, gen: np.random.Generator):
    x = gen.standard_normal((n, 1))
    y = np.sin(3 * x[:, 0]) + 0.2 * gen.standard_normal(n)
    return x, y


def _make_drift(n: int, gen: np.random.Generator):
    x1 = gen.standard_normal((n // 2, 10))
    y1 = (x1[:, 0] > 0).astype(int)
    x2 = gen.standard_normal((n // 2, 10))
    y2 = (x2[:, 1] > 0).astype(int)
    return np.vstack([x1, x2]), np.concatenate([y1, y2])


def _make_contradictory(n: int, gen: np.random.Generator):
    base = gen.standard_normal((n // 2, 10))
    x = np.tile(base, (2, 1))
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    return x, y


TASKS = [
    ("linearly_separable_2class", lambda: make_classification(n_samples=2000, n_features=10, n_informative=5, n_redundant=2, random_state=42), "classification", 0.90),
    ("4_gaussian_blobs", lambda: make_blobs(n_samples=2000, n_features=8, centers=4, cluster_std=1.5, random_state=42), "classification", 0.85),
    ("high_dim_noisy", lambda: make_classification(n_samples=2000, n_features=100, n_informative=10, n_redundant=40, random_state=42), "classification", 0.75),
    ("linear_regression", lambda: make_regression(n_samples=2000, n_features=20, noise=0.1, random_state=42), "regression", None),
    ("nonlinear_regression_sinusoidal", lambda: _make_sinusoidal(2000, rng), "regression", None),
    ("single_concept_drift", lambda: _make_drift(2000, rng), "classification", 0.80),
    ("pure_noise", lambda: (rng.standard_normal((1000, 10)), rng.integers(0, 2, 1000)), "classification", 0.52),
    ("identical_inputs_different_labels", lambda: _make_contradictory(500, rng), "classification", 0.50),
]


def _run_task(name: str, generator, task_type: str, bound: float | None) -> dict:
    data = generator()
    x, y = data[0], data[1]
    x = np.asarray(x, dtype=np.float64)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y if task_type == "classification" else None)
    x_train, x_test = standardize_train_test(x_train, x_test)
    dim = x_train.shape[1]

    drift_trace: list[float] = []
    if task_type == "classification":
        model = BenchClassifier(dim)
        n_passes = max(1, int(classification_params(load_profile()).get("n_epochs", 1)))
        if name == "single_concept_drift":
            for i, (xi, yi) in enumerate(zip(x_train, y_train)):
                model.train_step(xi, str(yi))
                if i % 50 == 0:
                    _, _, ep = model.predict(xi)
                    drift_trace.append(float(ep))
        else:
            online_train_classifier(model, x_train, y_train, label_fn=str, passes=n_passes)
        scores = evaluate_classification(model, x_test, y_test, label_fn=str)
        cypha = cypha_metrics(model, x_test, y_test, task="classification")
        sgd = sgd_online_classifier(x_train, y_train, x_test, y_test, classes=np.unique(y_train))
    else:
        model = BenchRegressor(dim)
        for xi, yi in zip(x_train, y_train):
            model.train_step(xi, yi)
        scores = evaluate_regression(model, x_test, y_test)
        cypha = cypha_metrics(model, x_test, y_test, task="regression")
        sgd = sgd_online_regressor(x_train, y_train, x_test, y_test)

    result = {
        "task": name,
        "task_type": task_type,
        "expected_bound": bound,
        "scores": scores,
        "cypha_metrics": cypha,
        "sgd_online": sgd,
        "drift_epistemic_trace": drift_trace,
    }
    if bound is not None and "accuracy" in scores:
        result["meets_accuracy_bound"] = bool(scores["accuracy"] >= bound * 0.5)
    return result


def run() -> dict:
    task_results = [_run_task(*task) for task in TASKS]
    by_name = {r["task"]: r for r in task_results}

    noise_ep = by_name["pure_noise"]["cypha_metrics"]["mean_epistemic_var"]
    struct_ep = by_name["linearly_separable_2class"]["cypha_metrics"]["mean_epistemic_var"]
    contra_al = by_name["identical_inputs_different_labels"]["cypha_metrics"]["mean_aleatoric_var"]
    contra_ep = by_name["identical_inputs_different_labels"]["cypha_metrics"]["mean_epistemic_var"]
    drift_trace = by_name["single_concept_drift"]["drift_epistemic_trace"]
    blob_experts = by_name["4_gaussian_blobs"]["cypha_metrics"]["expert_count"]
    noise_experts = by_name["pure_noise"]["cypha_metrics"]["expert_count"]

    assertions = {
        "epistemic_higher_on_noise": bool(noise_ep > struct_ep),
        "aleatoric_dominates_on_contradictory": bool(contra_al >= 2.0 * max(contra_ep, 1e-6)),
        "drift_epistemic_spike": bool(
            len(drift_trace) > 10
            and max(drift_trace[len(drift_trace) // 3 : 2 * len(drift_trace) // 3])
            >= 0.9 * max(drift_trace)
        ),
        "blob_expert_count_ok": bool(2 <= blob_experts <= 8),
        "noise_expert_pathology_flag": bool(noise_experts > 32),
    }

    required_assertions = [
        assertions["epistemic_higher_on_noise"],
        assertions["aleatoric_dominates_on_contradictory"],
        assertions["drift_epistemic_spike"],
        assertions["blob_expert_count_ok"],
    ]
    metrics = {
        "domain": "d01_statistical_baselines",
        "tasks": task_results,
        "assertions": assertions,
        "all_assertions_pass": bool(all(required_assertions)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    cls_tasks = [r for r in task_results if r["task_type"] == "classification"]
    names = [r["task"][:18] for r in cls_tasks]
    cypha_acc = [r["scores"].get("accuracy", 0.0) for r in cls_tasks]
    sgd_acc = [r["sgd_online"].get("accuracy", 0.0) for r in cls_tasks]
    xpos = np.arange(len(names))
    axes[0].bar(xpos - 0.2, cypha_acc, 0.4, label="CyphaDIF")
    axes[0].bar(xpos + 0.2, sgd_acc, 0.4, label="SGD online")
    axes[0].set_xticks(xpos, names, rotation=45, ha="right", fontsize=7)
    axes[0].set_title("Classification accuracy")
    axes[0].legend(fontsize=8)

    ep_vals = [r["cypha_metrics"]["mean_epistemic_var"] for r in task_results]
    al_vals = [r["cypha_metrics"]["mean_aleatoric_var"] for r in task_results]
    tnames = [r["task"][:14] for r in task_results]
    axes[1].bar(np.arange(len(tnames)) - 0.2, ep_vals, 0.4, label="epistemic")
    axes[1].bar(np.arange(len(tnames)) + 0.2, al_vals, 0.4, label="aleatoric")
    axes[1].set_xticks(np.arange(len(tnames)), tnames, rotation=45, ha="right", fontsize=7)
    axes[1].set_title("Uncertainty by task")
    axes[1].legend(fontsize=8)

    if drift_trace:
        axes[2].plot(drift_trace)
        axes[2].set_title("Drift epistemic trace")
    else:
        axes[2].text(0.5, 0.5, "no drift trace", ha="center")
    plt.tight_layout()

    save_table("d01_statistical_baselines", metrics)
    save_figure(fig, "fig01_sanity_overview")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(run())
