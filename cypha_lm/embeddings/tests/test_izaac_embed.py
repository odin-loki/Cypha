"""Tests for Izaac GF(2^n) permutation polynomial embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from cypha_lm.embeddings.izaac_embed import IzaacEmbedding
from cypha_lm.tests.conftest import assert_no_torch_on_import


@pytest.fixture
def embed() -> IzaacEmbedding:
    return IzaacEmbedding(vocab_size=64, d_embed=64, seed=42)


def test_bijectivity(embed: IzaacEmbedding) -> None:
    table = np.stack([embed.embed(t) for t in range(embed.vocab_size)], axis=0)
    unique_rows = np.unique(table, axis=0)
    assert unique_rows.shape[0] == embed.vocab_size


def test_decode_roundtrip(embed: IzaacEmbedding) -> None:
    for t in range(embed.vocab_size):
        vec = embed.embed(t)
        recovered = embed.nearest_token(vec)
        assert recovered == t


def test_gf_distance_symmetry(embed: IzaacEmbedding) -> None:
    rng = np.random.default_rng(0)
    for _ in range(1000):
        a, b = rng.integers(0, embed.vocab_size, size=2)
        assert embed.distance(int(a), int(b)) == embed.distance(int(b), int(a))


def test_dimension_shape(embed: IzaacEmbedding) -> None:
    ids = np.array([0, 1, 5, 10, 63], dtype=np.int64)
    batch = embed.embed_batch(ids)
    assert batch.shape == (len(ids), embed.d_embed)


def test_seed_reproducibility() -> None:
    e1 = IzaacEmbedding(vocab_size=64, d_embed=64, seed=7)
    e2 = IzaacEmbedding(vocab_size=64, d_embed=64, seed=7)
    e3 = IzaacEmbedding(vocab_size=64, d_embed=64, seed=99)
    assert np.allclose(e1.embed(10), e2.embed(10))
    assert not np.allclose(e1.embed(10), e3.embed(10))


def test_zero_token(embed: IzaacEmbedding) -> None:
    vec = embed.embed(0)
    assert vec.shape == (embed.d_embed,)
    assert np.all(np.isfinite(vec))
    assert not np.any(np.isnan(vec))


def test_vocab_boundary(embed: IzaacEmbedding) -> None:
    last = embed.vocab_size - 1
    vec = embed.embed(last)
    assert vec.shape == (embed.d_embed,)
    assert embed.nearest_token(vec) == last


def test_algebraic_structure(embed: IzaacEmbedding) -> None:
    neighbour_dists = []
    random_dists = []
    rng = np.random.default_rng(1)
    for t in range(1, embed.vocab_size - 1):
        neighbour_dists.append(embed.distance(t, t + 1))
    for _ in range(500):
        a, b = rng.integers(0, embed.vocab_size, size=2)
        if abs(int(a) - int(b)) > 1:
            random_dists.append(embed.distance(int(a), int(b)))
    assert np.mean(neighbour_dists) <= np.mean(random_dists) + 2.0


def test_no_torch_dependency() -> None:
    assert_no_torch_on_import("from cypha_lm.embeddings.izaac_embed import IzaacEmbedding")
