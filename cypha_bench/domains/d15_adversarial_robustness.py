"""Domain 15 — adversarial robustness (digits/MNIST noise tests)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from bench_common import (
    DEFAULT_SEED,
    clf_epistemic,
    clf_metrics,
    finalize_domain,
    make_classifier,
    rng,
    train_classifier_online,
)


def _hog_like_features(images: np.ndarray, cell: int = 4) -> np.ndarray:
    """Lightweight gradient histogram features for 8x8 digits."""
    imgs = images.reshape(-1, 8, 8).astype(np.float64) / 16.0
    feats = []
    for img in imgs:
        gx = np.gradient(img, axis=1)
        gy = np.gradient(img, axis=0)
        mag = np.sqrt(gx**2 + gy**2)
        angles = (np.arctan2(gy, gx) % np.pi) / np.pi
        n_bins = 9
        hist = np.zeros((2, 2, n_bins))
        for i in range(2):
            for j in range(2):
                patch_mag = mag[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4]
                patch_ang = angles[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4]
                for b in range(n_bins):
                    mask = (patch_ang >= b / n_bins) & (patch_ang < (b + 1) / n_bins)
                    hist[i, j, b] = patch_mag[mask].sum()
        feats.append(hist.ravel())
    return np.stack(feats).astype(np.float32)


def load_digit_features():
    data = load_digits()
    X = _hog_like_features(data.images)
    y = data.target
    return train_test_split(X, y, test_size=0.25, random_state=DEFAULT_SEED, stratify=y)


def add_gaussian_noise(X: np.ndarray, std: float, g: np.random.Generator) -> np.ndarray:
    return X + g.standard_normal(X.shape).astype(np.float32) * std


def adversarial_fgsm_proxy(clf, x: np.ndarray, y_true: int, epsilon: float = 0.1) -> np.ndarray:
    x_adv = x.copy().astype(np.float64)
    pred_orig, _ = clf.infer(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += 1e-4
        x_minus[i] -= 1e-4
        p_plus, _ = clf.infer(x_plus)
        p_minus, _ = clf.infer(x_minus)
        if str(p_plus) != str(y_true):
            x_adv[i] += epsilon
        elif str(p_minus) != str(y_true):
            x_adv[i] -= epsilon
    return np.clip(x_adv, 0.0, None).astype(np.float32)


def experiment_15a_noise_robustness():
    X_train, X_test, y_train, y_test = load_digit_features()
    clf = make_classifier(X_train.shape[1])
    train_classifier_online(clf, X_train, y_train, passes=4)
    g = rng(DEFAULT_SEED)
    levels = [0.0, 0.1, 0.2, 0.5, 1.0]
    curve = {}
    for std in levels:
        Xn = add_gaussian_noise(X_test, std, g)
        m = clf_metrics(clf, Xn, y_test)
        curve[str(std)] = {
            "accuracy": m["accuracy"],
            "mean_epistemic_var": m["mean_epistemic_var"],
        }
    return curve


def experiment_15b_feature_dropout():
    X_train, X_test, y_train, y_test = load_digit_features()
    clf = make_classifier(X_train.shape[1], seed=DEFAULT_SEED + 1)
    train_classifier_online(clf, X_train, y_train, passes=4)
    g = rng(DEFAULT_SEED + 1)
    rates = [0.1, 0.25, 0.5, 0.75]
    curve = {}
    for rate in rates:
        Xd = X_test.copy()
        mask = g.random(Xd.shape) < rate
        Xd[mask] = 0.0
        m = clf_metrics(clf, Xd, y_test)
        curve[str(rate)] = {
            "accuracy": m["accuracy"],
            "mean_epistemic_var": m["mean_epistemic_var"],
        }
    return curve


def experiment_15c_adversarial():
    X_train, X_test, y_train, y_test = load_digit_features()
    clf = make_classifier(X_train.shape[1], seed=DEFAULT_SEED + 2)
    train_classifier_online(clf, X_train, y_train, passes=4)
    subset = X_test[: min(500, len(X_test))]
    ysub = y_test[: len(subset)]
    natural_epi, adv_epi = [], []
    acc_nat, acc_adv = 0, 0
    for x, y in zip(subset, ysub):
        pred, _ = clf.infer(x)
        if str(pred) == str(y):
            acc_nat += 1
        natural_epi.append(clf_epistemic(clf, x))
        x_adv = adversarial_fgsm_proxy(clf, x, int(y))
        pred_a, _ = clf.infer(x_adv)
        if str(pred_a) == str(y):
            acc_adv += 1
        adv_epi.append(clf_epistemic(clf, x_adv))
    n = len(subset)
    return {
        "accuracy_natural": acc_nat / n,
        "accuracy_adversarial": acc_adv / n,
        "mean_epistemic_natural": float(np.mean(natural_epi)),
        "mean_epistemic_adversarial": float(np.mean(adv_epi)),
    }


def run() -> dict:
    experiments = {
        "15A_gaussian_noise": experiment_15a_noise_robustness(),
        "15B_feature_dropout": experiment_15b_feature_dropout(),
        "15C_adversarial_fgsm_proxy": experiment_15c_adversarial(),
    }
    return finalize_domain("d15", experiments)


if __name__ == "__main__":
    print(run())
