"""Sliding-window time series feature encoder."""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft


class TimeSeriesEncoder:
    """
    Sliding window encoder for time series data.
    Extracts statistical + frequency features from a window of samples.
    """

    def __init__(self, window_size: int = 50, n_fft_coeffs: int = 10):
        self.window_size = window_size
        self.n_fft_coeffs = n_fft_coeffs
        self.feature_dim = 8 + n_fft_coeffs

    def encode_window(self, window: np.ndarray) -> np.ndarray:
        assert len(window) == self.window_size
        features: list[float] = []

        features.append(float(np.mean(window)))
        features.append(float(np.std(window)))
        features.append(float(np.min(window)))
        features.append(float(np.max(window)))
        features.append(float(np.percentile(window, 25)))
        features.append(float(np.percentile(window, 75)))
        features.append(float(np.mean(np.abs(np.diff(window)))))
        features.append(float(np.sum(np.sign(np.diff(window)) != 0)))

        fft_coeffs = np.abs(rfft(window))
        features.extend(fft_coeffs[: self.n_fft_coeffs].tolist())

        return np.array(features, dtype=np.float32)

    def sliding_windows(
        self, series: np.ndarray, step: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(series)
        indices = list(range(self.window_size, n, step))
        X = np.array(
            [self.encode_window(series[i - self.window_size : i]) for i in indices]
        )
        return X, np.array(indices)

    def encode_series(self, series: np.ndarray) -> np.ndarray:
        series = np.asarray(series, dtype=np.float64).ravel()
        if len(series) == self.window_size:
            return self.encode_window(series)
        if len(series) > self.window_size:
            X, _ = self.sliding_windows(series, step=max(1, len(series) // self.window_size))
            if len(X) == 0:
                padded = np.pad(series, (0, self.window_size - len(series)), mode="edge")
                return self.encode_window(padded)
            return X.mean(axis=0).astype(np.float32)
        padded = np.pad(series, (0, self.window_size - len(series)), mode="edge")
        return self.encode_window(padded)
