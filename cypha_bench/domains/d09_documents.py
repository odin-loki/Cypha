"""Domain 09 — document understanding (20news + Gutenberg)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

import os

from cypha_bench.adapters.bench_models import BenchClassifier
from cypha_bench.common.baselines import offline_classification_baselines, sgd_online_classifier
from cypha_bench.common.metrics import cypha_metrics, evaluate_classification, online_train_classifier, save_figure, save_table, standardize_train_test
from cypha_bench.common.paths import DATA_DIR, scale
from cypha_bench.config.load_profile import classification_params, load_profile
from cypha_bench.encoders import DocumentEncoder


def _load_20news(max_samples: int = 2000):
    from sklearn.decomposition import TruncatedSVD
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    idx = np.arange(len(data.data))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    idx = idx[:max_samples]
    texts = [data.data[i] for i in idx]
    targets = np.asarray(data.target)[idx]
    enc = DocumentEncoder(max_features=2000, ngram_range=(1, 2))
    enc.fit(texts)
    x = enc.encode_batch(texts).astype(np.float64)
    svd = TruncatedSVD(n_components=min(100, x.shape[1] - 1), random_state=42)
    x = svd.fit_transform(x)
    return enc, svd, x, targets, texts


def _gutenberg_segments():
    enc = DocumentEncoder(max_features=2000, ngram_range=(1, 2))
    books = {
        "alice": DATA_DIR / "gutenberg" / "alice.txt",
        "sherlock": DATA_DIR / "gutenberg" / "sherlock_holmes.txt",
        "moby": DATA_DIR / "gutenberg" / "moby_dick.txt",
    }
    segments, labels = [], []
    for name, path in books.items():
        if not path.exists():
            text = f"Synthetic passage for {name}. " * 200
            segments.extend([text[i : i + 500] for i in range(0, 2000, 500)])
            labels.extend([name] * 4)
            continue
        segs = enc.segment_book(str(path), segment_chars=500)
        take = min(len(segs), scale(400, 80))
        segments.extend(segs[:take])
        labels.extend([name] * take)
    if not segments:
        segments = ["Synthetic document segment."] * 30
        labels = ["synthetic"] * 30
    return enc, segments, labels


def _apply_svd(raw: np.ndarray, svd) -> np.ndarray:
    """Pad/truncate raw TF-IDF features to match SVD input dim, then transform."""
    n_svd_feat = svd.components_.shape[1]
    if raw.shape[1] < n_svd_feat:
        raw = np.pad(raw, ((0, 0), (0, n_svd_feat - raw.shape[1])))
    else:
        raw = raw[:, :n_svd_feat]
    return svd.transform(raw)


def run() -> dict:
    news_enc, news_svd, x_news, y_news, news_texts = _load_20news(scale(2000, 800))
    x_train, x_test, y_train, y_test = train_test_split(
        x_news, y_news, test_size=0.2, random_state=42, stratify=y_news
    )
    x_train, x_test = standardize_train_test(x_train, x_test)

    model = BenchClassifier(x_train.shape[1])
    n_passes = 1
    if os.environ.get("CYPHA_BENCH_USE_PROFILE", "1").strip().lower() not in ("0", "false", "no"):
        n_passes = max(1, int(classification_params(load_profile()).get("n_epochs", 1)))
    online_train_classifier(model, x_train, y_train, label_fn=str, passes=n_passes)
    news_scores = evaluate_classification(model, x_test, y_test, label_fn=str)
    news_cypha = cypha_metrics(model, x_test, y_test, task="classification")
    news_baselines = offline_classification_baselines(x_train, y_train, x_test, y_test)
    news_sgd = sgd_online_classifier(x_train, y_train, x_test, y_test, classes=np.unique(y_train))

    doc_enc, gutenberg_segments, book_labels = _gutenberg_segments()
    doc_enc.fit(news_texts + gutenberg_segments)
    gutenberg_x = doc_enc.encode_batch(gutenberg_segments).astype(np.float64)
    gutenberg_x = _apply_svd(gutenberg_x, news_svd)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    pad = x_train.shape[1]
    if gutenberg_x.shape[1] < pad:
        gutenberg_x = np.pad(gutenberg_x, ((0, 0), (0, pad - gutenberg_x.shape[1])))
    elif gutenberg_x.shape[1] > pad:
        gutenberg_x = gutenberg_x[:, :pad]
    gutenberg_x = (gutenberg_x - mean) / std

    ep_in, ep_ood = [], []
    for xi in x_test[: min(200, len(x_test))]:
        _, _, ep = model.predict(xi)
        ep_in.append(ep)
    for xi in gutenberg_x:
        _, _, ep = model.predict(xi)
        ep_ood.append(ep)
    u_stat, p_val = mannwhitneyu(ep_ood, ep_in, alternative="greater")

    book_x = doc_enc.encode_batch(gutenberg_segments).astype(np.float64)
    book_x = _apply_svd(book_x, news_svd)
    if book_x.shape[1] < pad:
        book_x = np.pad(book_x, ((0, 0), (0, pad - book_x.shape[1])))
    else:
        book_x = book_x[:, :pad]
    book_x = (book_x - mean) / std
    book_train, book_test, lbl_train, lbl_test = train_test_split(
        book_x, book_labels, test_size=0.2, random_state=42, stratify=book_labels
    )
    book_model = BenchClassifier(book_train.shape[1])
    online_train_classifier(book_model, book_train, lbl_train, label_fn=str)
    book_scores = evaluate_classification(book_model, book_test, lbl_test, label_fn=str)
    book_sgd = sgd_online_classifier(book_train, lbl_train, book_test, lbl_test, classes=np.unique(lbl_train))

    metrics = {
        "domain": "d09_documents",
        "20news": {
            "n_samples": int(len(x_news)),
            "cypha_scores": news_scores,
            "cypha_metrics": news_cypha,
            "baselines": news_baselines,
            "sgd_online": news_sgd,
        },
        "gutenberg_ood": {
            "mean_epistemic_in": float(np.mean(ep_in)),
            "mean_epistemic_ood": float(np.mean(ep_ood)),
            "mannwhitney_u": float(u_stat),
            "p_value": float(p_val),
        },
        "gutenberg_book_classification": {
            "n_segments": int(len(book_x)),
            "cypha_scores": book_scores,
            "sgd_online": book_sgd,
        },
    }

    if os.environ.get("CYPHA_BENCH_BRANCH_A", "0") == "1":
        from cypha_bench.adapters.branch_a_documents import run_branch_a_documents

        backend = os.environ.get("CYPHA_BRANCH_A_BACKEND", "auto")
        metrics["branch_a_frozen_embed"] = run_branch_a_documents(backend=backend)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        ["CyphaDIF", "SGD", "LogReg"],
        [
            news_scores["accuracy"],
            news_sgd["accuracy"],
            news_baselines["logistic_regression"]["accuracy"],
        ],
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("20 Newsgroups subset")
    axes[1].boxplot([ep_in, ep_ood], tick_labels=["in-domain", "Gutenberg OOD"])
    axes[1].set_title("Epistemic variance OOD test")
    plt.tight_layout()

    save_table("d09_documents", metrics)
    save_figure(fig, "fig09_documents_overview")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(run())
