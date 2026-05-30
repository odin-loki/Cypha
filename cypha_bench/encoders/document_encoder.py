from __future__ import annotations

import numpy as np

from cypha_bench.encoders.text_encoder import TextEncoder


class DocumentEncoder(TextEncoder):
    """Alias for document TF-IDF encoding with a larger default vocabulary."""

    def __init__(self, max_features: int = 2000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        super().__init__(max_features=max_features, ngram_range=ngram_range)

    def segment_book(self, path: str, segment_chars: int = 500) -> list[str]:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        return [
            text[i : i + segment_chars]
            for i in range(0, max(len(text) - segment_chars, 0), segment_chars)
        ]
