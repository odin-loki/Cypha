from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer


class TextEncoder:
    """TF-IDF encoder for document batches."""

    def __init__(self, max_features: int = 1000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer="word",
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        self.fitted = False

    def fit(self, documents: list[str]) -> None:
        self.vectorizer.fit(documents)
        self.fitted = True

    def encode(self, text: str) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before encode().")
        vec = self.vectorizer.transform([text])
        if issparse(vec):
            vec = vec.toarray()
        return vec.flatten().astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before encode_batch().")
        vecs = self.vectorizer.transform(texts)
        if issparse(vecs):
            vecs = vecs.toarray()
        return vecs.astype(np.float32)


class CharNgramEncoder:
    """Character n-gram bag encoder for streaming language modelling."""

    def __init__(self, n: int = 5, vocab_size: int = 200) -> None:
        self.n = n
        self.vocab_size = vocab_size
        self._vocab: dict[str, int] = {}

    @property
    def dim(self) -> int:
        return self.vocab_size

    def build_vocab(self, text: str) -> None:
        ngrams = [text[i : i + self.n] for i in range(max(len(text) - self.n, 0))]
        counts = Counter(ngrams)
        self._vocab = {ng: i for i, (ng, _) in enumerate(counts.most_common(self.vocab_size))}

    def encode(self, window: str) -> np.ndarray:
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        for i in range(max(len(window) - self.n, 0)):
            ng = window[i : i + self.n]
            idx = self._vocab.get(ng)
            if idx is not None:
                vec[idx] += 1.0
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec
