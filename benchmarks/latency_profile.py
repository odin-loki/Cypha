#!/usr/bin/env python3
"""Latency benchmark — O(1) per-token inference for CyphaLM."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cypha_lm.config import CyphaLMConfig
from cypha_lm.model.cypha_lm import CyphaLM
from experiments._common import scale


def bench_predict(model: CyphaLM, n_tokens: int, warmup: int = 50) -> dict:
    rng = np.random.default_rng(0)
    tokens = [int(rng.integers(0, model.config.vocab_size)) for _ in range(warmup + n_tokens)]

    model.reset_context()
    for t in tokens[:warmup]:
        model.predict_next(t)

    latencies: list[float] = []
    for t in tokens[warmup:]:
        t0 = time.perf_counter()
        model.predict_next(t)
        latencies.append(time.perf_counter() - t0)

    arr = np.asarray(latencies)
    return {
        "n_tokens": n_tokens,
        "mean_ms": float(arr.mean() * 1000),
        "p50_ms": float(np.percentile(arr, 50) * 1000),
        "p95_ms": float(np.percentile(arr, 95) * 1000),
        "tokens_per_sec": float(1.0 / max(arr.mean(), 1e-9)),
    }


def bench_train_step(model: CyphaLM, n_steps: int) -> dict:
    rng = np.random.default_rng(1)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        a = int(rng.integers(0, model.config.vocab_size))
        b = int(rng.integers(0, model.config.vocab_size))
        model.train_step(a, b)
    elapsed = time.perf_counter() - t0
    return {
        "n_steps": n_steps,
        "total_seconds": elapsed,
        "steps_per_sec": n_steps / max(elapsed, 1e-9),
    }


def main() -> dict:
    n_predict = scale(2000, 200)
    n_train = scale(1000, 100)
    cfg = CyphaLMConfig(vocab_size=128, d_embed=64, d_state=128)
    model = CyphaLM(cfg)

    predict_stats = bench_predict(model, n_predict)
    train_stats = bench_train_step(model, n_train)

    result = {"predict": predict_stats, "train_step": train_stats}
    print(
        f"Predict: {predict_stats['mean_ms']:.3f} ms/token "
        f"({predict_stats['tokens_per_sec']:.0f} tok/s)"
    )
    print(f"Train: {train_stats['steps_per_sec']:.0f} steps/s")
    return result


if __name__ == "__main__":
    main()
