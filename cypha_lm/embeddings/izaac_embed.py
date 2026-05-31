"""GF(2^n) permutation polynomial token embeddings (Izaac)."""

from __future__ import annotations

import math
from typing import Any

import galois
import numpy as np

from cypha_lm.array_backend import asnumpy


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _valid_permutation_exponent(k: int, n: int) -> bool:
    modulus = (1 << n) - 1
    return _gcd(2**k + 1, modulus) == 1


def _find_valid_k(poly_degree_exp: int, n: int) -> int:
    k = poly_degree_exp
    while k < n + 10:
        if _valid_permutation_exponent(k, n):
            return k
        k += 1
    raise ValueError(f"No valid permutation exponent found for GF(2^{n})")


class IzaacEmbedding:
    """
    GF(2^n) permutation polynomial token embedding.

    p(x) = a * x^(2^k + 1) + b over GF(2^n) is bijective when a != 0 and
    gcd(2^k + 1, 2^n - 1) = 1. Token vectors are n-bit coefficient blocks
    tiled to ``d_embed`` dimensions.
    """

    def __init__(
        self,
        vocab_size: int,
        d_embed: int,
        poly_degree_exp: int = 1,
        seed: int = 42,
        xp: Any = None,
    ) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if d_embed < 1:
            raise ValueError("d_embed must be >= 1")

        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.poly_degree_exp = poly_degree_exp
        self.seed = seed
        self._xp = xp if xp is not None else np

        self.n = math.ceil(math.log2(vocab_size))
        if self.n < 1:
            self.n = 1

        self.n_blocks = int(math.ceil(d_embed / self.n))
        self.k = _find_valid_k(poly_degree_exp, self.n)
        self._exp = 2**self.k + 1

        self._gf = galois.GF(2**self.n)
        rng = np.random.default_rng(seed)
        nonzero = [int(v) for v in self._gf.elements if int(v) != 0]
        self._a = self._gf(int(rng.choice(nonzero)))
        self._b = self._gf(int(rng.integers(0, 2**self.n)))

        self._table = self._xp.asarray(self._build_table_cpu())

    def _poly(self, token_id: int):
        x = self._gf(int(token_id))
        return self._a * x**self._exp + self._b

    def _field_to_bits(self, element) -> np.ndarray:
        bits = np.array([int(v) for v in element.vector()], dtype=np.float64)
        return bits

    def _build_table_cpu(self) -> np.ndarray:
        table = np.zeros((self.vocab_size, self.d_embed), dtype=np.float64)
        for t in range(self.vocab_size):
            block = self._field_to_bits(self._poly(t))
            for b in range(self.n_blocks):
                start = b * self.n
                end = min(start + self.n, self.d_embed)
                table[t, start:end] = block[: end - start]
        return table

    def embed(self, token_id: int) -> Any:
        if not 0 <= token_id < self.vocab_size:
            raise IndexError(
                f"token_id {token_id} out of range [0, {self.vocab_size})"
            )
        return self._table[token_id].copy()

    def embed_batch(self, token_ids: np.ndarray) -> Any:
        token_ids = np.asarray(token_ids, dtype=np.int64).ravel()
        if np.any((token_ids < 0) | (token_ids >= self.vocab_size)):
            raise IndexError("token_ids contain out-of-range values")
        return self._table[token_ids].copy()

    def distance(self, id_a: int, id_b: int) -> int:
        """Hamming weight of the difference p(a) - p(b) in GF(2^n)."""
        diff = self._poly(id_a) - self._poly(id_b)
        return int(np.sum(self._field_to_bits(diff)))

    def nearest_token(self, vec: np.ndarray) -> int:
        vec = np.asarray(vec, dtype=np.float64).ravel()
        if vec.shape[0] != self.d_embed:
            raise ValueError(f"Expected vector of shape ({self.d_embed},)")
        table = asnumpy(self._table)
        dists = np.sum((table - vec) ** 2, axis=1)
        return int(np.argmin(dists))
