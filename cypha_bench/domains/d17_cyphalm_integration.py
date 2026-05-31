"""Domain 17 — CyphaLM integration (WikiText / Gutenberg)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BENCH = Path(__file__).resolve().parents[1]
_REPO = _BENCH.parent
for path in (_REPO, _BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bench_common import finalize_domain

from cypha_bench.adapters.cyphalm_bench import (
    bigram_baseline_bpc,
    cyphalm_bench_limits,
    cyphalm_dif_metrics,
    eval_held_out_bpc,
    load_cyphalm_config,
    make_cyphalm,
    prepare_lm_corpus,
    require_real_corpus,
    train_with_learning_curve,
    trigram_baseline_bpc,
)
from cypha_bench.common.metrics import save_figure


def _load_corpus():
    limits = cyphalm_bench_limits()
    corpus = prepare_lm_corpus(prefer_wikitext=True, max_train_chars=limits["max_corpus_chars"])
    require_real_corpus(corpus.source, domain="D17")
    return corpus


def experiment_17a_bpc():
    limits = cyphalm_bench_limits()
    corpus = _load_corpus()
    if len(corpus.train_ids) < 512:
        return {"skipped": True, "source": corpus.source}

    model, cfg = make_cyphalm(profile="d17")
    n_train = min(limits["n_train"], len(corpus.train_ids) - 1)
    curve = train_with_learning_curve(
        model,
        corpus.train_ids,
        corpus.eval_ids,
        n_train=n_train,
        snapshot_every=limits["snapshot_every"],
        n_eval_snapshot=min(200, limits["n_eval"]),
    )

    held_out_bpc = eval_held_out_bpc(model, corpus.eval_ids, n_eval=limits["n_eval"])
    train_slice = corpus.train_ids[:n_train]
    bigram_bpc = bigram_baseline_bpc(train_slice, corpus.eval_ids, corpus.vocab_size)
    trigram_bpc = trigram_baseline_bpc(train_slice, corpus.eval_ids, corpus.vocab_size)

    return {
        "cyphalm_bpc": held_out_bpc,
        "online_train_bpc": curve["online_train_bpc"],
        "final_train_bpc": curve["final_train_bpc"],
        "bigram_bpc": bigram_bpc,
        "trigram_bpc": trigram_bpc,
        "delta_vs_bigram": held_out_bpc - bigram_bpc,
        "delta_vs_trigram": held_out_bpc - trigram_bpc,
        "source": corpus.source,
        "split": corpus.split,
        "train_tokens": n_train,
        "learning_curve": curve,
        "device": model.device,
        "profile": load_cyphalm_config(profile="d17"),
        "cypha_dif": cyphalm_dif_metrics(model),
    }


def experiment_17b_alpha_spectrum():
    limits = cyphalm_bench_limits()
    corpus = _load_corpus()
    model, _ = make_cyphalm(profile="d17")
    n = min(limits["n_train"], len(corpus.train_ids) - 1)
    model.train_sequence(corpus.train_ids[:n])
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
    from cypha_bench.adapters.cyphalm_bench import encode_text
    from cypha_bench.common.paths import DATA_DIR

    corpus = _load_corpus()
    holmes = DATA_DIR / "gutenberg" / "sherlock_holmes.txt"
    if holmes.exists():
        text_b = holmes.read_text(encoding="utf-8", errors="ignore")[:20_000]
        ids_b = encode_text(text_b, corpus.char2id)
    else:
        ids_b = list(reversed(corpus.train_ids[:4000]))

    model, _ = make_cyphalm(profile="d17")
    warm = min(2000, len(corpus.train_ids) - 1)
    model.train_sequence(corpus.train_ids[:warm])
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


def _save_learning_figure(result: dict) -> None:
    curve = result.get("learning_curve") or {}
    steps = curve.get("steps") or []
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, curve["held_out_bpc"], marker="o", label="CyphaLM held-out")
    bb = result.get("bigram_bpc")
    tb = result.get("trigram_bpc")
    if bb is not None:
        ax.axhline(bb, color="gray", linestyle="--", label=f"Bigram ({bb:.2f})")
    if tb is not None:
        ax.axhline(tb, color="orange", linestyle=":", label=f"Trigram ({tb:.2f})")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Bits per character")
    ax.set_title(f"D17 CyphaLM ({result.get('source', '?')})")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, "fig17_learning_curve")
    plt.close(fig)


def run() -> dict:
    try:
        exp17a = experiment_17a_bpc()
        if not exp17a.get("skipped"):
            _save_learning_figure(exp17a)
        experiments = {
            "17A_bits_per_character": exp17a,
            "17B_alpha_spectrum": experiment_17b_alpha_spectrum(),
            "17D_online_adaptation": experiment_17d_online_adaptation(),
        }
    except ImportError as exc:
        experiments = {
            "skipped": True,
            "reason": f"cypha_lm not available: {exc}",
        }
    except RuntimeError as exc:
        experiments = {
            "skipped": True,
            "reason": str(exc),
        }
    return finalize_domain("d17", experiments)


if __name__ == "__main__":
    print(run())
