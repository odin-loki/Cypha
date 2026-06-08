"""Frozen text embeddings for Branch A (CyphaDIF on semantic vectors)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer


def embed_texts_hashing(
    texts: list[str],
    *,
    n_features: int = 512,
    n_components: int = 128,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic hashing embedder (no train-time fit; offline fallback)."""
    hv = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
    )
    raw = hv.transform(texts).astype(np.float64)
    n_comp = min(n_components, max(2, raw.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    dense = svd.fit_transform(raw)
    meta = {
        "backend": "hashing_svd",
        "n_features": n_features,
        "n_components": n_comp,
        "explained_variance_ratio": float(np.sum(svd.explained_variance_ratio_)),
    }
    return dense.astype(np.float64), meta


def embed_texts_sentence_transformer(
    texts: list[str],
    *,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Encode with sentence-transformers if installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    vecs = np.asarray(vecs, dtype=np.float64)
    meta = {
        "backend": "sentence_transformers",
        "model_name": model_name,
        "dim": int(vecs.shape[1]),
    }
    return vecs, meta


def embed_texts(
    texts: list[str],
    *,
    backend: str = "auto",
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Return frozen embeddings for a text list.

    backend: auto | sentence_transformers | hashing
    """
    if backend in ("auto", "sentence_transformers"):
        st = embed_texts_sentence_transformer(texts, model_name=model_name)
        if st is not None:
            return st
        if backend == "sentence_transformers":
            raise RuntimeError(
                "sentence-transformers not installed. "
                "pip install sentence-transformers"
            )
    return embed_texts_hashing(texts)
