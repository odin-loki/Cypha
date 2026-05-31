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


def _eval_held_out_bpc(
    clf: BenchClassifier,
    sgd: "SGDClassifier",
    enc: "CharNgramEncoder",
    text: str,
    char2idx: dict,
    context_length: int,
) -> tuple[float, float]:
    """Evaluate mean bits-per-character on a held-out suffix of text.

    Returns (cypha_mean_bpc, sgd_mean_bpc).
    Both metrics are averaged over all held-out positions where the character
    is in the training vocabulary, giving a fair comparison.
    """
    bpcs: list[float] = []
    sgd_bpcs: list[float] = []
    window = text[:context_length]
    # Snapshot label order once for consistency across the evaluation loop.
    with clf.dif.memory._lock:
        ordered_labels = list(clf.dif.memory._classes.keys())
    label_set = set(ordered_labels)

    for i in range(context_length, len(text) - 1):
        next_char = text[i]
        if next_char not in char2idx:
            window = window[1:] + next_char
            continue
        next_idx = char2idx[next_char]
        label_str = str(next_idx)
        x = enc.encode(window)

        # CyphaDIF: probability for the correct next character.
        _, probs, _ = clf.predict(x)
        if label_str in label_set:
            slot = ordered_labels.index(label_str)
            true_prob = float(probs[slot]) if 0 <= slot < len(probs) else 1e-10
        else:
            # Character not in training vocab — the model cannot assign non-floor prob.
            true_prob = 1e-10
        bpcs.append(float(-np.log2(max(true_prob, 1e-10))))

        # SGD baseline.
        sgd_probs = sgd.predict_proba(x.reshape(1, -1))[0]
        sgd_true = float(sgd_probs[next_idx]) if next_idx < len(sgd_probs) else 1e-10
        sgd_bpcs.append(float(-np.log2(max(sgd_true, 1e-10))))

        window = window[1:] + next_char

    mean_bpc = float(np.mean(bpcs)) if bpcs else float("nan")
    mean_sgd_bpc = float(np.mean(sgd_bpcs)) if sgd_bpcs else float("nan")
    return mean_bpc, mean_sgd_bpc


def _run_char_lm(text: str, n_steps: int = 5000, context_length: int = 50) -> dict:
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}

    enc = CharNgramEncoder(n=5, vocab_size=200)
    enc.build_vocab(text[: min(len(text), 50000)])

    # 80/20 train/test split — evaluate on held-out suffix only.
    split = int(len(text) * 0.8)
    train_text = text[:split]
    test_text = text[split:]

    clf = BenchClassifier(enc.dim)
    sgd = SGDClassifier(loss="log_loss", random_state=42)
    classes = np.arange(len(chars))

    window = train_text[:context_length]
    steps: list[int] = []
    bpc_curve: list[float] = []
    ep_curve: list[float] = []
    expert_curve: list[int] = []
    sgd_bpc: list[float] = []
    n_trained = 0
    limit = min(len(train_text) - 1, n_steps + context_length)

    for i in range(context_length, limit):
        x = enc.encode(window)
        next_char = train_text[i]
        if next_char not in char2idx:
            window = window[1:] + next_char
            continue
        next_idx = char2idx[next_char]
        label = str(next_idx)
        clf.train_step(x, label)
        sgd.partial_fit(x.reshape(1, -1), [next_idx], classes=classes)

        if n_trained % 500 == 0 and n_trained > 0:
            # Snapshot: evaluate on a short window of test text (fast approximation).
            test_slice = test_text[:min(200, len(test_text))]
            snap_bpc, snap_sgd = _eval_held_out_bpc(
                clf, sgd, enc, test_slice, char2idx, context_length
            )
            with clf.dif.memory._lock:
                n_experts = len(clf.dif.memory._classes)
            ep_var = float(clf.dif.infer_full(x).get("entropy", 0.0))
            steps.append(n_trained)
            bpc_curve.append(snap_bpc)
            sgd_bpc.append(snap_sgd)
            ep_curve.append(ep_var)
            expert_curve.append(n_experts)

        window = window[1:] + next_char
        n_trained += 1

    # Final held-out BPC over the full test set.
    final_bpc, final_sgd_bpc = _eval_held_out_bpc(
        clf, sgd, enc, test_text, char2idx, context_length
    )

    return {
        "steps": steps,
        "bits_per_char": bpc_curve,
        "sgd_bits_per_char": sgd_bpc,
        "epistemic_var": ep_curve,
        "expert_count": expert_curve,
        "vocab_size": len(chars),
        "trained_steps": n_trained,
        "final_bpc": final_bpc,
        "final_sgd_bpc": final_sgd_bpc,
        "eval_method": "held_out_20pct_suffix",
    }


def run() -> dict:
    source, text = _load_corpus()
    lm_metrics = _run_char_lm(text, n_steps=scale(5000, 1500))
    metrics = {
        "domain": "d04_generation_language",
        "corpus_source": source,
        "char_lm": lm_metrics,
        # final_bpc is already computed as held-out mean BPC inside _run_char_lm.
        "final_bpc": lm_metrics.get("final_bpc"),
        "final_sgd_bpc": lm_metrics.get("final_sgd_bpc"),
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
