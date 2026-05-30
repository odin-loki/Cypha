"""Domain 17 — CyphaLM integration (WikiText / Gutenberg / synthetic)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from bench_common import DEFAULT_SEED, data_path, finalize_domain, rng

DEFAULT_CYPHALM_CONFIG = {
    "vocab_size": 128,
    "d_embed": 64,
    "d_state": 128,
    "tau_fast": 1.0,
    "tau_slow": 20.0,
    "ssm_layers": 2,
    "field_dim": 160,
    "max_experts": 128,
    "alpha_init": 0.5,
    "context_length": 256,
    "seed": DEFAULT_SEED,
}


def _load_text_corpus(max_chars: int = 80000) -> tuple[str, str]:
    wt = data_path("wikitext2", "wikitext-2", "wiki.train.tokens")
    if wt.exists():
        text = wt.read_text(encoding="utf-8")[:max_chars]
        return text, "wikitext2"

    guten = data_path("gutenberg", "moby_dick.txt")
    if guten.exists():
        return guten.read_text(encoding="utf-8", errors="ignore")[:max_chars], "gutenberg"

    g = rng(DEFAULT_SEED)
    vocab = "abcdefghijklmnopqrstuvwxyz "
    synthetic = "".join(g.choice(list(vocab), size=max_chars))
    synthetic = " ".join(synthetic[i : i + 5] for i in range(0, len(synthetic), 5))
    return synthetic, "synthetic"


def _char_vocab(text: str, vocab_size: int = 128) -> dict[str, int]:
    chars = sorted(set(text))[:vocab_size - 1]
    return {c: i + 1 for i, c in enumerate(chars)}


def _encode_text(text: str, char2id: dict[str, int]) -> list[int]:
    unk = 0
    return [char2id.get(c, unk) for c in text if c in char2id or c == " "]


def _make_cyphalm():
    ensure_repo = str(_BENCH.parent)
    if ensure_repo not in sys.path:
        sys.path.insert(0, ensure_repo)
    from cypha_lm.config import CyphaLMConfig
    from cypha_lm.model.cypha_lm import CyphaLM

    cfg = CyphaLMConfig(**DEFAULT_CYPHALM_CONFIG)
    return CyphaLM(cfg), cfg


def experiment_17a_bpc():
    text, source = _load_text_corpus()
    char2id = _char_vocab(text, DEFAULT_CYPHALM_CONFIG["vocab_size"] - 1)
    ids = _encode_text(text, char2id)
    if len(ids) < 512:
        return {"bpc": float("nan"), "source": source, "skipped": True}

    model, _ = _make_cyphalm()
    train_ids = ids[: int(0.8 * len(ids))]
    test_ids = ids[int(0.8 * len(ids)) :]

    # Train on more tokens (up to 40000) for better convergence
    n_train = min(40000, len(train_ids) - 1)
    stats = model.train_sequence(train_ids[:n_train])
    online_losses = stats["loss"]
    online_bpc = float(np.mean(online_losses) / np.log(2))
    # Final-phase BPC: last 10% of training (model has mostly converged)
    tail = max(1, len(online_losses) // 10)
    final_train_bpc = float(np.mean(online_losses[-tail:]) / np.log(2))

    # Held-out evaluation: predict without updating weights
    eval_losses = []
    n_eval = min(2000, len(test_ids) - 1)
    for i in range(n_eval):
        pred = model.predict_next(test_ids[i])
        lp = pred["log_probs"]
        nxt = test_ids[i + 1]
        eval_losses.append(float(-lp[nxt]))
    held_out_bpc = float(np.mean(eval_losses) / np.log(2)) if eval_losses else float("nan")

    # Bigram baseline
    counts = np.zeros((DEFAULT_CYPHALM_CONFIG["vocab_size"],) * 2)
    for a, b in zip(train_ids, train_ids[1:]):
        counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True) + 1e-12
    probs = counts / row_sums
    bi_bits = []
    for a, b in zip(test_ids, test_ids[1:]):
        p = max(probs[a, b], 1e-12)
        bi_bits.append(-np.log2(p))
    bigram_bpc = float(np.mean(bi_bits)) if bi_bits else float("nan")

    return {
        "cyphalm_bpc": held_out_bpc,        # primary metric: held-out test BPC
        "online_train_bpc": online_bpc,      # reference: average online training loss
        "final_train_bpc": final_train_bpc,  # reference: converged training BPC
        "bigram_bpc": bigram_bpc,
        "source": source,
        "train_tokens": n_train,
    }


def experiment_17b_alpha_spectrum():
    text, _ = _load_text_corpus(max_chars=30000)
    char2id = _char_vocab(text)
    ids = _encode_text(text, char2id)
    model, _ = _make_cyphalm()
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
    text_a, _ = _load_text_corpus(max_chars=20000)
    holmes = data_path("gutenberg", "sherlock_holmes.txt")
    text_b = (
        holmes.read_text(encoding="utf-8", errors="ignore")[:20000]
        if holmes.exists()
        else text_a[::-1]
    )
    char2id = _char_vocab(text_a + text_b)
    ids_a = _encode_text(text_a, char2id)
    ids_b = _encode_text(text_b, char2id)

    model, _ = _make_cyphalm()
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
