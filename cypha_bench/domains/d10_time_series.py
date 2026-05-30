"""Domain 10 — time series (ECG5000 / synthetic, yfinance financial)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from bench_common import (
    DEFAULT_FINANCIAL_PERIOD,
    DEFAULT_FINANCIAL_TICKERS,
    DEFAULT_FINANCIAL_WINDOW,
    DEFAULT_SEED,
    clf_epistemic,
    clf_metrics,
    finalize_domain,
    make_classifier,
    make_regressor,
    reg_metrics,
    safe_auroc,
    train_classifier_online,
    train_regressor_online,
    data_path,
    rng,
)
from cypha_bench.encoders.timeseries_encoder import TimeSeriesEncoder


def _synthetic_ecg5000(n_train: int = 500, n_test: int = 450, length: int = 140, n_classes: int = 5):
    g = rng(DEFAULT_SEED)
    def _make(n):
        X, y = [], []
        for i in range(n):
            cls = i % n_classes
            t = np.linspace(0, 4 * np.pi, length)
            wave = np.sin(t + cls) + 0.3 * np.sin(3 * t + cls * 0.5)
            wave += g.standard_normal(length) * 0.05
            X.append(wave.astype(np.float32))
            y.append(cls + 1)
        return np.stack(X), np.array(y, dtype=int)
    return _make(n_train), _make(n_test)


def load_ecg5000():
    train_p = data_path("ecg5000", "ECG5000_TRAIN.txt")
    test_p = data_path("ecg5000", "ECG5000_TEST.txt")
    if train_p.exists() and test_p.exists():
        train = np.loadtxt(train_p)
        test = np.loadtxt(test_p)
        X_train, y_train = train[:, 1:], train[:, 0].astype(int)
        X_test, y_test = test[:, 1:], test[:, 0].astype(int)
        return X_train, y_train, X_test, y_test
    (X_train, y_train), (X_test, y_test) = _synthetic_ecg5000()
    return X_train, y_train, X_test, y_test


def _encode_ecg_rows(rows: np.ndarray, enc: TimeSeriesEncoder) -> np.ndarray:
    return np.stack([enc.encode_series(row) for row in rows], axis=0)


def experiment_10a_ecg():
    X_train, y_train, X_test, y_test = load_ecg5000()
    enc = TimeSeriesEncoder(window_size=min(50, X_train.shape[1]), n_fft_coeffs=10)
    Xtr = _encode_ecg_rows(X_train, enc)
    Xte = _encode_ecg_rows(X_test, enc)
    clf = make_classifier(Xtr.shape[1])
    train_classifier_online(clf, Xtr, y_train, passes=4)
    m = clf_metrics(clf, Xte, y_test)
    m["data_source"] = "ecg5000" if data_path("ecg5000", "ECG5000_TRAIN.txt").exists() else "synthetic"
    return m


def experiment_10b_sliding():
    X_train, y_train, _, _ = load_ecg5000()
    enc = TimeSeriesEncoder(window_size=10, n_fft_coeffs=6)
    X_rows, y_rows = [], []
    for series, label in zip(X_train, y_train):
        windows, _ = enc.sliding_windows(series, step=5)
        if len(windows) == 0:
            continue
        X_rows.append(windows)
        y_rows.extend([label] * len(windows))
    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=int)
    split = int(0.8 * len(X))
    clf = make_classifier(X.shape[1], seed=DEFAULT_SEED + 1)
    train_classifier_online(clf, X[:split], y[:split], passes=4)
    return clf_metrics(clf, X[split:], y[split:])


def experiment_10c_ood():
    X_train, y_train, X_test, y_test = load_ecg5000()
    normal_cls = int(np.min(y_train))
    tr_mask = y_train == normal_cls
    enc = TimeSeriesEncoder(window_size=min(50, X_train.shape[1]))
    Xtr = _encode_ecg_rows(X_train[tr_mask], enc)
    Xte = _encode_ecg_rows(X_test, enc)
    clf = make_classifier(Xtr.shape[1], seed=DEFAULT_SEED + 2)
    train_classifier_online(clf, Xtr, y_train[tr_mask], passes=3)
    scores, labels = [], []
    for x, yt in zip(Xte, y_test):
        scores.append(clf_epistemic(clf, x))
        labels.append(0 if yt == normal_cls else 1)
    return {"ood_auroc": safe_auroc(np.array(labels), np.array(scores))}


def _load_financial_returns():
    try:
        import yfinance as yf
    except ImportError:
        yf = None

    enc = TimeSeriesEncoder(window_size=DEFAULT_FINANCIAL_WINDOW)
    all_X, all_y, ticker_ids = [], [], []
    if yf is not None:
        for ti, ticker in enumerate(DEFAULT_FINANCIAL_TICKERS):
            try:
                df = yf.download(ticker, period=DEFAULT_FINANCIAL_PERIOD, interval="1d", progress=False)
                if df is None or df.empty:
                    continue
                close = df["Close"].values.astype(np.float64).ravel()
                if len(close) < DEFAULT_FINANCIAL_WINDOW + 2:
                    continue
                log_returns = np.diff(np.log(close + 1e-12))
                X, indices = enc.sliding_windows(log_returns, step=1)
                y = (log_returns[indices] > 0).astype(int)
                all_X.append(X)
                all_y.append(y)
                ticker_ids.extend([ticker] * len(y))
            except Exception:
                continue

    if not all_X:
        g = rng(DEFAULT_SEED + 3)
        for ti, ticker in enumerate(DEFAULT_FINANCIAL_TICKERS):
            series = np.cumsum(g.standard_normal(800) * 0.01).astype(np.float64)
            log_returns = np.diff(series)
            X, indices = enc.sliding_windows(log_returns, step=1)
            y = (log_returns[indices] > 0).astype(int)
            all_X.append(X)
            all_y.append(y)
            ticker_ids.extend([ticker] * len(y))

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    return X, y, ticker_ids


def experiment_10d_financial():
    X, y, _ = _load_financial_returns()
    split = int(0.8 * len(X))
    clf = make_classifier(X.shape[1], seed=DEFAULT_SEED + 4)
    train_classifier_online(clf, X[:split], y[:split], passes=3)
    m = clf_metrics(clf, X[split:], y[split:])
    m["note"] = "near_chance_expected"
    return m


def run() -> dict:
    experiments = {
        "10A_ecg_classification": experiment_10a_ecg(),
        "10B_ecg_sliding_window": experiment_10b_sliding(),
        "10C_ecg_ood_detection": experiment_10c_ood(),
        "10D_financial_return_sign": experiment_10d_financial(),
    }
    return finalize_domain("d10", experiments)


if __name__ == "__main__":
    print(run())
