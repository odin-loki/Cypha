"""Unit tests for cypha_views.transforms."""

from __future__ import annotations

from collections import Counter

from cypha_views.transforms import (
    apply_transform,
    block_shuffle,
    identity,
    rotate_start,
)


def test_identity_unchanged() -> None:
    ids = [10, 20, 30, 40, 50]
    assert identity(ids) == ids
    assert apply_transform("forward", ids) == ids


def test_rotate_start_same_length() -> None:
    ids = list(range(12))
    for offset in (0, 3, 7, 11, 100):
        out = rotate_start(ids, offset=offset)
        assert len(out) == len(ids)
        assert sorted(out) == sorted(ids)


def test_block_shuffle_preserves_multiset() -> None:
    ids = list(range(100))
    for seed in (0, 1, 42):
        out = block_shuffle(ids, block_size=16, seed=seed)
        assert Counter(out) == Counter(ids)
        assert apply_transform("block_shuffle", ids, block_size=16, seed=seed) == out
