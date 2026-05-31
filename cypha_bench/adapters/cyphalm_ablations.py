"""CyphaLM context-mode ablation runner for D04/D17."""

from __future__ import annotations

from typing import Any

from cypha_bench.adapters.char_lstm_baseline import char_lstm_baseline_bpc
from cypha_bench.adapters.cyphalm_bench import (
    bigram_baseline_bpc,
    eval_held_out_bpc,
    fivegram_baseline_bpc,
    fourgram_baseline_bpc,
    make_cyphalm,
    train_with_learning_curve,
    trigram_baseline_bpc,
)
from cypha_bench.common.paths import is_fast


def run_lm_ablations(
    corpus,
    limits: dict[str, int],
    modes: list[str],
    *,
    profile: str = "d17",
) -> dict[str, Any]:
    """
    Train/eval CyphaLM under each context_mode and score n-gram / LSTM baselines.

    Returns held-out BPC per ablation mode plus baseline BPC values on the same split.
    """
    train_ids = corpus.train_ids
    test_ids = corpus.eval_ids
    vocab_size = corpus.vocab_size
    n_train = min(limits["n_train"], len(train_ids) - 1)
    train_slice = train_ids[:n_train]

    result: dict[str, Any] = {
        "modes_run": list(modes),
        "n_train": n_train,
        "n_eval": limits["n_eval"],
        "bigram_bpc": bigram_baseline_bpc(train_slice, test_ids, vocab_size),
        "trigram_bpc": trigram_baseline_bpc(train_slice, test_ids, vocab_size),
    }

    if not is_fast():
        result["fourgram_bpc"] = fourgram_baseline_bpc(train_slice, test_ids, vocab_size)
        result["fivegram_bpc"] = fivegram_baseline_bpc(train_slice, test_ids, vocab_size)
        result["char_lstm_bpc"] = char_lstm_baseline_bpc(
            train_slice,
            test_ids,
            vocab_size,
            n_train_steps=n_train,
        )

    ablation_bpc: dict[str, float] = {}
    for mode in modes:
        model, _ = make_cyphalm({"context_mode": mode}, profile=profile)
        train_with_learning_curve(
            model,
            train_ids,
            test_ids,
            n_train=n_train,
            snapshot_every=limits["snapshot_every"],
            n_eval_snapshot=min(200, limits["n_eval"]),
        )
        ablation_bpc[mode] = eval_held_out_bpc(model, test_ids, n_eval=limits["n_eval"])

    result["ablation_bpc"] = ablation_bpc
    return result
