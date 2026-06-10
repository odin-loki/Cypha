#!/usr/bin/env python3
"""One-time emitter: parity_fixtures/cyphalm_*/sidecar.json from Python CyphaLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from cypha_lm.config import CyphaLMConfig
from cypha_lm.model.cypha_lm import CyphaLM

_OUT_CHAR = _ROOT / "parity_fixtures" / "cyphalm_char_lstm"
_OUT_SSM = _ROOT / "parity_fixtures" / "cyphalm_ssm"


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


def emit_char_lstm() -> None:
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
    lstm = model.lstm_head
    sidecar = {
        "fixture_schema": 1,
        "name": "cyphalm_char_lstm",
        "mode": "char_lstm",
        "config": {
            "vocab_size": cfg.vocab_size,
            "d_embed": cfg.d_embed,
            "lstm_hidden": cfg.lstm_hidden,
            "context_mode": "char_lstm",
            "seed": cfg.seed,
            "lstm_lr": cfg.lstm_lr,
            "train_epochs": cfg.train_epochs,
        },
        "vocab_size": cfg.vocab_size,
        "field_dim": cfg.field_dim,
        "hidden": cfg.lstm_hidden,
        "train_ids": train_ids[:200],
        "eval_ids": eval_ids,
        "n_train": 199,
        "n_eval": len(eval_ids) - 1,
        "expected_bpc": bpc,
        "atol_bpc": 0.25,
        "char_lstm": {
            "E": np.asarray(lstm.E, dtype=np.float64).reshape(-1).tolist(),
            "Wx": np.asarray(lstm.Wx, dtype=np.float64).reshape(-1).tolist(),
            "Wh": np.asarray(lstm.Wh, dtype=np.float64).reshape(-1).tolist(),
            "b": np.asarray(lstm.b, dtype=np.float64).reshape(-1).tolist(),
            "Wy": np.asarray(lstm.Wy, dtype=np.float64).reshape(-1).tolist(),
            "by": np.asarray(lstm.by, dtype=np.float64).reshape(-1).tolist(),
        },
    }
    _OUT_CHAR.mkdir(parents=True, exist_ok=True)
    (_OUT_CHAR / "sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {_OUT_CHAR / 'sidecar.json'} bpc={bpc:.4f}")


def emit_ssm_bpc() -> None:
    cfg = CyphaLMConfig(
        vocab_size=32,
        d_embed=8,
        d_state=16,
        ssm_layers=1,
        field_dim=16,
        context_mode="ssm_only",
        seed=42,
        gria_lr=0.05,
        train_epochs=1,
        train_ssm=False,
    )
    train_ids = _synthetic_ids(300, cfg.vocab_size)
    eval_ids = _synthetic_ids(60, cfg.vocab_size, seed=77)
    model = CyphaLM(cfg)
    for i in range(min(250, len(train_ids) - 1)):
        model.train_step(int(train_ids[i]), int(train_ids[i + 1]))
    bpc = _eval_bpc(model, eval_ids, n_eval=len(eval_ids) - 1)
    sidecar = {
        "fixture_schema": 1,
        "name": "cyphalm_ssm",
        "mode": "ssm",
        "config": {
            "vocab_size": cfg.vocab_size,
            "d_embed": cfg.d_embed,
            "d_state": cfg.d_state,
            "ssm_layers": cfg.ssm_layers,
            "field_dim": cfg.field_dim,
            "context_mode": "ssm_only",
            "seed": cfg.seed,
            "gria_lr": cfg.gria_lr,
        },
        "train_ids": train_ids[:180],
        "eval_ids": eval_ids,
        "n_train": 179,
        "n_eval": len(eval_ids) - 1,
        "expected_bpc": bpc,
        "atol_bpc": 0.5,
    }
    _OUT_SSM.mkdir(parents=True, exist_ok=True)
    (_OUT_SSM / "sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {_OUT_SSM / 'sidecar.json'} bpc={bpc:.4f}")


def main() -> None:
    emit_char_lstm()
    emit_ssm_bpc()


if __name__ == "__main__":
    main()
