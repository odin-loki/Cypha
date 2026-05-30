#!/usr/bin/env python3
"""Perplexity evaluation benchmark for CyphaLM."""

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
from experiments._common import perplexity_from_losses, scale


def synthetic_tokens(n: int, vocab: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, vocab)) for _ in range(n)]


def evaluate_perplexity(model: CyphaLM, tokens: list[int]) -> float:
    losses = []
    model.reset_context()
    for i in range(len(tokens) - 1):
        pred = model.predict_next(tokens[i])
        losses.append(float(-pred["log_probs"][tokens[i + 1]]))
    return perplexity_from_losses(np.asarray(losses))


def main() -> dict:
    vocab = 128
    train_steps = scale(5000, 500)
    eval_len = scale(1000, 200)

    cfg = CyphaLMConfig(vocab_size=vocab, d_embed=64, seed=42)
    model = CyphaLM(cfg)
    train = synthetic_tokens(train_steps + 1, vocab, 42)

    t0 = time.perf_counter()
    for i in range(len(train) - 1):
        model.train_step(train[i], train[i + 1])
    train_time = time.perf_counter() - t0

    test = synthetic_tokens(eval_len + 1, vocab, 99)
    ppl = evaluate_perplexity(model, test)

    result = {
        "train_steps": train_steps,
        "eval_tokens": eval_len,
        "perplexity": ppl,
        "train_seconds": train_time,
        "tokens_per_second": train_steps / max(train_time, 1e-6),
    }
    print(f"Perplexity: {ppl:.4f}  ({train_steps} train steps in {train_time:.2f}s)")
    return result


if __name__ == "__main__":
    main()
