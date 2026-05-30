"""Domain 08 — computer vision (MNIST or digits fallback)."""

from __future__ import annotations

import gzip
import os
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from cypha_bench.adapters.bench_models import BenchClassifier
from cypha_bench.config.load_profile import classification_params, load_profile
from cypha_bench.common.baselines import offline_classification_baselines, sgd_online_classifier
from cypha_bench.common.metrics import cypha_metrics, evaluate_classification, online_train_classifier, save_figure, save_table
from cypha_bench.common.paths import DATA_DIR, scale
from cypha_bench.encoders import ImageEncoder


def _read_images(path):
    with open(path, "rb") as handle:
        _magic, n, rows, cols = struct.unpack(">IIII", handle.read(16))
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(n, rows, cols)


def _read_labels(path):
    with open(path, "rb") as handle:
        _magic, n = struct.unpack(">II", handle.read(8))
        return np.frombuffer(handle.read(), dtype=np.uint8)


def _load_mnist():
    root = DATA_DIR / "mnist"
    train_x = root / "train-images-idx3-ubyte"
    if not train_x.exists():
        gz = root / "train-images-idx3-ubyte.gz"
        if gz.exists():
            with gzip.open(gz, "rb") as handle:
                _magic, n, rows, cols = struct.unpack(">IIII", handle.read(16))
                x_train = np.frombuffer(handle.read(), dtype=np.uint8).reshape(n, rows, cols)
            with gzip.open(root / "train-labels-idx1-ubyte.gz", "rb") as handle:
                handle.read(8)
                y_train = np.frombuffer(handle.read(), dtype=np.uint8)
            with gzip.open(root / "t10k-images-idx3-ubyte.gz", "rb") as handle:
                handle.read(16)
                x_test = np.frombuffer(handle.read(), dtype=np.uint8).reshape(-1, rows, cols)
            with gzip.open(root / "t10k-labels-idx1-ubyte.gz", "rb") as handle:
                handle.read(8)
                y_test = np.frombuffer(handle.read(), dtype=np.uint8)
            return "mnist_gz", x_train, y_train, x_test, y_test
        return None
    x_train = _read_images(train_x)
    y_train = _read_labels(root / "train-labels-idx1-ubyte")
    x_test = _read_images(root / "t10k-images-idx3-ubyte")
    y_test = _read_labels(root / "t10k-labels-idx1-ubyte")
    return "mnist", x_train, y_train, x_test, y_test


def _load_data():
    mnist = _load_mnist()
    if mnist is not None:
        return mnist
    x, y = load_digits(return_X_y=True)
    images = x.reshape(-1, 8, 8)
    images = np.repeat(np.repeat(images, 4, axis=1), 4, axis=2)
    return "sklearn_digits", images, y, images, y


def _encode_images(images: np.ndarray, mode: str) -> np.ndarray:
    enc = ImageEncoder()
    rows = []
    for img in images:
        if mode == "raw":
            rows.append(enc.raw_pixels(img))
        else:
            rows.append(enc.hog_features(img))
    return np.asarray(rows, dtype=np.float64)


def _run_encoding(source: str, x_train, y_train, x_test, y_test, mode: str, max_train: int, max_test: int) -> dict:
    if len(x_train) > max_train:
        idx = np.random.default_rng(42).choice(len(x_train), max_train, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]
    if len(x_test) > max_test:
        idx = np.random.default_rng(7).choice(len(x_test), max_test, replace=False)
        x_test, y_test = x_test[idx], y_test[idx]

    x_train = _encode_images(x_train, mode)
    x_test = _encode_images(x_test, mode)

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
        "encoding": mode,
        "cypha_scores": scores,
        "cypha_metrics": cypha,
        "baselines": baselines,
        "sgd_online": sgd,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }


def run() -> dict:
    source, x_train, y_train, x_test, y_test = _load_data()
    max_train = scale(10000, 2000)
    max_test = scale(2000, 500)
    raw = _run_encoding(source, x_train, y_train, x_test, y_test, "raw", max_train, max_test)
    hog = _run_encoding(source, x_train, y_train, x_test, y_test, "hog", max_train, max_test)

    metrics = {
        "domain": "d08_computer_vision",
        "data_source": source,
        "experiments": [raw, hog],
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Raw CyphaDIF", "Raw SGD", "HOG CyphaDIF", "HOG SGD"]
    acc = [
        raw["cypha_scores"]["accuracy"],
        raw["sgd_online"]["accuracy"],
        hog["cypha_scores"]["accuracy"],
        hog["sgd_online"]["accuracy"],
    ]
    ax.bar(labels, acc)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Vision classification ({source})")
    plt.tight_layout()

    save_table("d08_computer_vision", metrics)
    save_figure(fig, "fig08_mnist_accuracy_by_encoding")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(run())
