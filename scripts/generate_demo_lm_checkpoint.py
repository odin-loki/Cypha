#!/usr/bin/env python3
"""Train a tiny CyphaLM and save a demo checkpoint for examples/ and REST smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from cypha_bench.adapters.cyphalm_bench import load_cyphalm_config
from cypha_lm.config import CyphaLMConfig
from cypha_lm.model.cypha_lm import CyphaLM

OUT = _REPO / "examples" / "demo_cyphalm" / "demo"


def main() -> None:
    text = (
        "Call me Ishmael. Some years ago—never mind how long precisely—"
        "having little or no money in my purse, and nothing particular to interest me on shore, "
        "I thought I would sail about a little and see the watery part of the world."
    ) * 5
    chars = sorted(set(text))
    char2id = {c: i + 1 for i, c in enumerate(chars[:127])}
    ids = [char2id.get(c, 0) for c in text]

    base = load_cyphalm_config()
    base.update(
        {
            "vocab_size": 128,
            "d_embed": 32,
            "d_state": 64,
            "field_dim": 96,
            "max_experts": 32,
            "n_experts": 2,
            "gria_lr": 0.08,
            "train_ssm": False,
        }
    )
    model = CyphaLM(CyphaLMConfig(**base))
    model.train_sequence(ids[: min(2000, len(ids) - 1)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT))
    print(f"Saved demo checkpoint to {OUT}.json / {OUT}.npz")
    print(f"Device: {model.device}")
    print("Load with:")
    print(f"  CYPHA_LM_CHECKPOINT={OUT}")
    print(f"  curl -X POST .../lm/load -d '{{\"checkpoint_path\": \"{OUT.as_posix()}\"}}'")


if __name__ == "__main__":
    main()
