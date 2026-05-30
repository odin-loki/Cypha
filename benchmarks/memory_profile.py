#!/usr/bin/env python3
"""Memory footprint benchmark for CyphaLM components."""

from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cypha_lm.config import CyphaLMConfig
from cypha_lm.model.cypha_lm import CyphaLM
from experiments._common import scale


def _array_bytes(*arrays: np.ndarray) -> int:
    return sum(a.nbytes for a in arrays)


def profile_model(cfg: CyphaLMConfig, n_steps: int) -> dict:
    tracemalloc.start()
    model = CyphaLM(cfg)
    snap_create = tracemalloc.get_traced_memory()[1]

    rng = np.random.default_rng(42)
    tokens = [int(rng.integers(0, cfg.vocab_size)) for _ in range(n_steps + 1)]
    for i in range(n_steps):
        model.train_step(tokens[i], tokens[i + 1])
    snap_train = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    static_bytes = (
        model.embed._table.nbytes
        + sum(W.nbytes for W in model.ssm.W_fast + model.ssm.W_slow)
        + model.gria.W.nbytes
        + model.gria.alpha.nbytes
        + model.gria.bias.nbytes
        + model._proj_ssm.nbytes
        + model._proj_dif.nbytes
        + model._proj_embed.nbytes
    )

    return {
        "peak_bytes_tracemalloc": snap_train,
        "after_init_bytes": snap_create,
        "static_array_bytes": static_bytes,
        "n_experts": model.dif.expert_count(),
        "n_steps": n_steps,
    }


def main() -> dict:
    n_steps = scale(5000, 500)
    cfg = CyphaLMConfig(vocab_size=256, d_embed=64, d_state=128, max_experts=128)
    result = profile_model(cfg, n_steps)
    mb = result["peak_bytes_tracemalloc"] / (1024 * 1024)
    print(f"Peak memory: {mb:.2f} MB ({result['n_experts']} experts after {n_steps} steps)")
    return result


if __name__ == "__main__":
    main()
