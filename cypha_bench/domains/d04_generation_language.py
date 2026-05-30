"""Domain 04 — character-level language modelling."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier

from cypha_bench.adapters.bench_models import BenchClassifier
from cypha_bench.common.metrics import save_figure, save_table
from cypha_bench.common.paths import DATA_DIR, scale
from cypha_bench.encoders import CharNgramEncoder


def _synthetic_corpus(length: int = 20000) -> str:
    rng = np.random.default_rng(42)
    alphabet = "abcdefghijklmnopqrstuvwxyz .,\n"
    return "".join(rng.choice(list(alphabet)) for _ in range(length))


def _load_corpus() -> tuple[str, str]:
    candidates = [
        DATA_DIR / "gutenberg" / "moby_dick.txt",
        DATA_DIR / "gutenberg" / "alice.txt",
        DATA_DIR / "gutenberg" / "sherlock_holmes.txt",
    ]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8", errors="replace") as handle:
                return path.name, handle.read()
    return "synthetic", _synthetic_corpus()


def _run_char_lm(text: str, n_steps: int = 5000, context_length: int = 50) -> dict:
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2label = {i: str(i) for i in range(len(chars))}

    enc = CharNgramEncoder(n=5, vocab_size=200)
    enc.build_vocab(text[: min(len(text), 50000)])

    clf = BenchClassifier(enc.dim)
    sgd = SGDClassifier(loss="log_loss", random_state=42)
    classes = np.arange(len(chars))

    window = text[:context_length]
    steps: list[int] = []
    bpc_curve: list[float] = []
    ep_curve: list[float] = []
    expert_curve: list[int] = []
    sgd_bpc: list[float] = []
    n_trained = 0
    limit = min(len(text) - 1, n_steps + context_length)

    for i in range(context_length, limit):
        x = enc.encode(window)
        next_char = text[i]
        if next_char not in char2idx:
            window = window[1:] + next_char
            continue
        next_idx = char2idx[next_char]
        label = str(next_idx)
        clf.train_step(x, label)
        sgd.partial_fit(x.reshape(1, -1), [next_idx], classes=classes)

        if n_trained % 500 == 0:
            _, probs, ep_var = clf.predict(x)
            true_prob = float(probs[next_idx]) if next_idx < len(probs) else 1e-10
            bpc = float(-np.log2(max(true_prob, 1e-10)))
            sgd_probs = sgd.predict_proba(x.reshape(1, -1))[0]
            sgd_true = float(sgd_probs[next_idx]) if next_idx < len(sgd_probs) else 1e-10
            steps.append(n_trained)
            bpc_curve.append(bpc)
            sgd_bpc.append(float(-np.log2(max(sgd_true, 1e-10))))
            ep_curve.append(float(ep_var))
            expert_curve.append(clf.expert_count())

        window = window[1:] + next_char
        n_trained += 1

    return {
        "steps": steps,
        "bits_per_char": bpc_curve,
        "sgd_bits_per_char": sgd_bpc,
        "epistemic_var": ep_curve,
        "expert_count": expert_curve,
        "vocab_size": len(chars),
        "trained_steps": n_trained,
    }


def run() -> dict:
    source, text = _load_corpus()
    lm_metrics = _run_char_lm(text, n_steps=scale(5000, 1500))
    metrics = {
        "domain": "d04_generation_language",
        "corpus_source": source,
        "char_lm": lm_metrics,
        "final_bpc": lm_metrics["bits_per_char"][-1] if lm_metrics["bits_per_char"] else None,
        "final_sgd_bpc": lm_metrics["sgd_bits_per_char"][-1] if lm_metrics["sgd_bits_per_char"] else None,
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if lm_metrics["steps"]:
        axes[0].plot(lm_metrics["steps"], lm_metrics["bits_per_char"], label="CyphaDIF")
        axes[0].plot(lm_metrics["steps"], lm_metrics["sgd_bits_per_char"], label="SGD online")
        axes[0].set_xlabel("Training step")
        axes[0].set_ylabel("Bits per character")
        axes[0].set_title(f"Char LM ({source})")
        axes[0].legend()
        axes[1].plot(lm_metrics["steps"], lm_metrics["expert_count"])
        axes[1].set_xlabel("Training step")
        axes[1].set_ylabel("Expert count")
        axes[1].set_title("Expert growth")
    plt.tight_layout()

    save_table("d04_generation_language", metrics)
    save_figure(fig, "fig04_char_lm_training")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(run())
