"""Domain 03 — classification benchmarks."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups, load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split

from cypha_bench.adapters.bench_models import BenchClassifier
from cypha_bench.config.load_profile import classification_params, load_profile
from cypha_bench.common.baselines import offline_classification_baselines, sgd_online_classifier
from cypha_bench.common.metrics import cypha_metrics, evaluate_classification, online_train_classifier, save_figure, save_table, standardize_train_test
from cypha_bench.common.paths import scale
from cypha_bench.encoders import TextEncoder


def _load_20news_subset(max_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import TruncatedSVD
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    idx = np.arange(len(data.data))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    idx = idx[:max_samples]
    texts = [data.data[i] for i in idx]
    targets = np.asarray(data.target)[idx]
    enc = TextEncoder(max_features=1000, ngram_range=(1, 2))
    enc.fit(texts)
    x = enc.encode_batch(texts)
    svd = TruncatedSVD(n_components=100, random_state=42)
    x = svd.fit_transform(x)
    return x.astype(np.float64), targets


def _load_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    datasets = {
        "iris": load_iris(return_X_y=True),
        "wine": load_wine(return_X_y=True),
        "breast_cancer": load_breast_cancer(return_X_y=True),
        "digits": load_digits(return_X_y=True),
    }
    out = {k: (np.asarray(v[0], dtype=np.float64), v[1]) for k, v in datasets.items()}
    out["20newsgroups_subset"] = _load_20news_subset(scale(2000, 800))
    return out


def _run_dataset(name: str, x: np.ndarray, y) -> dict:
    stratify = y if len(np.unique(y)) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify
    )
    x_train, x_test = standardize_train_test(x_train, x_test)

    model = BenchClassifier(x_train.shape[1])
    passes = 1
    if os.environ.get("CYPHA_BENCH_USE_PROFILE", "1").strip().lower() not in ("0", "false", "no"):
        passes = int(classification_params(load_profile()).get("n_epochs", 1))
    online_train_classifier(model, x_train, y_train, label_fn=str, passes=passes)
    scores = evaluate_classification(model, x_test, y_test, label_fn=str)
    cypha = cypha_metrics(model, x_test, y_test, task="classification")
    baselines = offline_classification_baselines(x_train, y_train, x_test, y_test)
    sgd = sgd_online_classifier(x_train, y_train, x_test, y_test, classes=np.unique(y_train))

    return {
        "dataset": name,
        "cypha_scores": scores,
        "cypha_metrics": cypha,
        "baselines": baselines,
        "sgd_online": sgd,
    }


def run() -> dict:
    results = [_run_dataset(name, x, y) for name, (x, y) in _load_datasets().items()]
    metrics = {"domain": "d03_classification", "datasets": results}

    names = [r["dataset"] for r in results]
    cypha_acc = [r["cypha_scores"]["accuracy"] for r in results]
    sgd_acc = [r["sgd_online"]["accuracy"] for r in results]
    lr_acc = [r["baselines"]["logistic_regression"]["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 4))
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.25, cypha_acc, 0.25, label="CyphaDIF")
    ax.bar(xpos, sgd_acc, 0.25, label="SGD online")
    ax.bar(xpos + 0.25, lr_acc, 0.25, label="Logistic Regression")
    ax.set_xticks(xpos, names, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification accuracy by dataset")
    ax.legend()
    plt.tight_layout()

    save_table("d03_classification", metrics)
    save_figure(fig, "fig03_classification_accuracy")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(run())
