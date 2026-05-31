"""Domain 17 — CyphaLM integration (WikiText / Gutenberg / synthetic)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parents[1]
_REPO = _BENCH.parent
for path in (_REPO, _BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bench_common import finalize_domain, rng

from cypha_bench.adapters.cyphalm_bench import (
    DEFAULT_CYPHALM_CONFIG,
    bigram_baseline_bpc,
    build_char_vocab,
    cyphalm_bench_limits,
    cyphalm_dif_metrics,
    encode_text,
    eval_held_out_bpc,
    load_char_corpus,
    make_cyphalm,
)


def _load_text_corpus(max_chars: int | None = None) -> tuple[str, str]:
    """Prefer WikiText for D17; fall back to Gutenberg / synthetic."""
    text, source = load_char_corpus(max_chars=max_chars, prefer_gutenberg=False)
    return text, source


def experiment_17a_bpc():
    limits = cyphalm_bench_limits()
    text, source = _load_text_corpus(limits["max_corpus_chars"])
    char2id, _ = build_char_vocab(text, DEFAULT_CYPHALM_CONFIG["vocab_size"] - 1)
    ids = encode_text(text, char2id)
    if len(ids) < 512:
        return {"bpc": float("nan"), "source": source, "skipped": True}

    model, _ = make_cyphalm()
    train_ids = ids[: int(0.8 * len(ids))]
    test_ids = ids[int(0.8 * len(ids)) :]

    n_train = min(limits["n_train"], len(train_ids) - 1)
    stats = model.train_sequence(train_ids[:n_train])
    online_losses = stats["loss"]
    online_bpc = float(np.mean(online_losses) / np.log(2))
    tail = max(1, len(online_losses) // 10)
    final_train_bpc = float(np.mean(online_losses[-tail:]) / np.log(2))

    held_out_bpc = eval_held_out_bpc(model, test_ids, n_eval=limits["n_eval"])
    bigram_bpc = bigram_baseline_bpc(
        train_ids[:n_train], test_ids, DEFAULT_CYPHALM_CONFIG["vocab_size"]
    )

    return {
        "cyphalm_bpc": held_out_bpc,
        "online_train_bpc": online_bpc,
        "final_train_bpc": final_train_bpc,
        "bigram_bpc": bigram_bpc,
        "source": source,
        "train_tokens": n_train,
        "cypha_dif": cyphalm_dif_metrics(model),
    }


def experiment_17b_alpha_spectrum():
    text, _ = _load_text_corpus(max_chars=min(30_000, cyphalm_bench_limits()["max_corpus_chars"]))
    char2id, _ = build_char_vocab(text)
    ids = encode_text(text, char2id)
    model, _ = make_cyphalm()
    model.train_sequence(ids[: min(3000, len(ids) - 1)])
    profile = model.compression_profile()
    expert_alpha = np.asarray(profile.get("expert_alpha_spectrum", []), dtype=np.float64)
    if expert_alpha.size == 0:
        frac_edge = float("nan")
    else:
        frac_edge = float(np.mean(np.abs(expert_alpha - 0.5) < 0.1))
    return {
        "mean_alpha": float(profile.get("mean_alpha", float("nan"))),
        "fraction_edge_of_chaos": frac_edge,
        "n_experts": int(profile.get("n_experts", 0)),
    }


def experiment_17d_online_adaptation():
    from cypha_bench.common.paths import DATA_DIR

    text_a, _ = _load_text_corpus(max_chars=20_000)
    holmes = DATA_DIR / "gutenberg" / "sherlock_holmes.txt"
    text_b = (
        holmes.read_text(encoding="utf-8", errors="ignore")[:20_000]
        if holmes.exists()
        else text_a[::-1]
    )
    char2id, _ = build_char_vocab(text_a + text_b)
    ids_a = encode_text(text_a, char2id)
    ids_b = encode_text(text_b, char2id)

    model, _ = make_cyphalm()
    model.train_sequence(ids_a[:2000])
    pre = model.train_sequence(ids_b[:500])
    bpc_before = float(np.sum(pre["loss"]) / max(len(pre["loss"]), 1) / np.log(2))
    model.train_sequence(ids_b[500:3500])
    post = model.train_sequence(ids_b[3500:4000])
    bpc_after = float(np.sum(post["loss"]) / max(len(post["loss"]), 1) / np.log(2))
    return {
        "bpc_ood_before_adapt": bpc_before,
        "bpc_ood_after_adapt": bpc_after,
        "bpc_improvement": bpc_before - bpc_after,
    }


def run() -> dict:
    try:
        experiments = {
            "17A_bits_per_character": experiment_17a_bpc(),
            "17B_alpha_spectrum": experiment_17b_alpha_spectrum(),
            "17D_online_adaptation": experiment_17d_online_adaptation(),
        }
    except ImportError as exc:
        experiments = {
            "skipped": True,
            "reason": f"cypha_lm not available: {exc}",
        }
    return finalize_domain("d17", experiments)


if __name__ == "__main__":
    print(run())
