"""Branch A document classification — CyphaDIF on frozen text embeddings."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from cypha_bench.adapters.bench_models import BenchClassifier
from cypha_bench.adapters.frozen_text_embeddings import embed_texts
from cypha_bench.common.metrics import evaluate_classification, online_train_classifier, standardize_train_test
from cypha_bench.common.paths import DATA_DIR, scale
from cypha_bench.config.load_profile import classification_params, load_profile
from cypha_bench.domains.d09_documents import _gutenberg_segments, _load_20news


def _gutenberg_segment_texts() -> list[str]:
    _, segments, _ = _gutenberg_segments()
    return list(segments)


def run_branch_a_documents(
    *,
    n_samples: int | None = None,
    backend: str = "auto",
    freeze_encoder_proj: bool = True,
) -> dict[str, Any]:
    """
    Train CyphaDIF on frozen embeddings of 20 Newsgroups; OOD epistemic on Gutenberg.

    Returns accuracy, baselines, and Mann-Whitney epistemic separation vs Gutenberg.
    """
    n_samples = n_samples or scale(2000, 800)
    _, _, _, y_all, train_texts = _load_20news(n_samples)
    y_all = np.asarray(y_all)

    x_all, embed_meta = embed_texts(train_texts, backend=backend)
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    x_train, x_test = standardize_train_test(x_train, x_test)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0

    passes = max(1, int(classification_params(load_profile()).get("n_epochs", 1)))
    t0 = time.perf_counter()
    model = BenchClassifier(x_train.shape[1], seed=42)
    if freeze_encoder_proj:
        model.dif.encoder._frozen = True
    online_train_classifier(model, x_train, y_train, label_fn=str, passes=passes)
    train_s = time.perf_counter() - t0

    scores = evaluate_classification(model, x_test, y_test, label_fn=str)
    ep_in = [float(model.predict(xi)[2]) for xi in x_test[: min(200, len(x_test))]]

    gutenberg_texts = _gutenberg_segment_texts()
    x_ood_raw, _ = embed_texts(gutenberg_texts, backend=backend)
    x_ood = (x_ood_raw - mean) / std
    ep_ood = [float(model.predict(xi)[2]) for xi in x_ood]

    u_stat, p_val = mannwhitneyu(ep_ood, ep_in, alternative="greater")

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(x_train, y_train)
    lr_acc = float(lr.score(x_test, y_test))

    return {
        "n_samples": int(n_samples),
        "n_classes": int(len(np.unique(y_all))),
        "embedding": embed_meta,
        "freeze_encoder_proj": bool(freeze_encoder_proj),
        "cypha_scores": scores,
        "cypha_accuracy": float(scores["accuracy"]),
        "logreg_accuracy": lr_acc,
        "train_seconds": train_s,
        "gutenberg_ood": {
            "n_segments": int(len(gutenberg_texts)),
            "mean_epistemic_in": float(np.mean(ep_in)),
            "mean_epistemic_ood": float(np.mean(ep_ood)),
            "mannwhitney_u": float(u_stat),
            "p_value": float(p_val),
            "ood_epistemic_higher": bool(np.mean(ep_ood) > np.mean(ep_in)),
        },
    }
