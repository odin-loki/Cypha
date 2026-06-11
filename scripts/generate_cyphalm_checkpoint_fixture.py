#!/usr/bin/env python3
"""Export Python char_lstm checkpoint for native load parity (cyphalm_checkpoint_parity)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from cypha_lm.config import CyphaLMConfig
from cypha_lm.model.cypha_lm import CyphaLM

_OUT = _ROOT / "parity_fixtures" / "cyphalm_checkpoint" / "char_lstm"


def _synthetic_ids(n: int, vocab: int, seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(1, vocab, size=n)]


def _eval_bpc(model: CyphaLM, ids: list[int], n_eval: int) -> float:
    model.reset_context()
    n = min(n_eval, len(ids) - 1)
    bits: list[float] = []
    for i in range(n):
        pred = model.predict_next(int(ids[i]))
        nxt = int(ids[i + 1])
        bits.append(float(-pred["log_probs"][nxt] / np.log(2)))
    return float(np.mean(bits)) if bits else float("nan")


def main() -> None:
    cfg = CyphaLMConfig(
        vocab_size=32,
        d_embed=8,
        lstm_hidden=16,
        context_mode="char_lstm",
        seed=42,
        lstm_lr=0.05,
        train_epochs=1,
    )
    train_ids = _synthetic_ids(400, cfg.vocab_size)
    eval_ids = _synthetic_ids(80, cfg.vocab_size, seed=99)
    model = CyphaLM(cfg)
    for ep in range(cfg.train_epochs):
        model.reset_context()
        for i in range(len(train_ids) - 1):
            model.train_step(int(train_ids[i]), int(train_ids[i + 1]))
    bpc = _eval_bpc(model, eval_ids, n_eval=len(eval_ids) - 1)

    _OUT.mkdir(parents=True, exist_ok=True)
    ckpt_base = _OUT / "checkpoint"
    model.save(str(ckpt_base))

    sidecar = {
        "fixture_schema": 1,
        "name": "cyphalm_checkpoint_char_lstm",
        "checkpoint_json": "checkpoint.json",
        "eval_ids": eval_ids,
        "expected_bpc": bpc,
        "atol_bpc": 0.08,
    }
    (_OUT / "sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {_OUT / 'sidecar.json'} bpc={bpc:.4f}")


if __name__ == "__main__":
    main()
