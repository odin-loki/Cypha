"""Tests for view-slot embeddings."""

from __future__ import annotations

import numpy as np

from cypha_lm.embeddings.view_embed import CANONICAL_VIEW_SLOTS, ViewEmbedding


def test_canonical_slots() -> None:
    emb = ViewEmbedding(16, 8, seed=0, learnable=True)
    assert emb.slot_for_view("forward") == CANONICAL_VIEW_SLOTS["forward"]
    assert emb.slot_for_view("block_shuffle") == 1


def test_learnable_update() -> None:
    emb = ViewEmbedding(4, 8, seed=0, learnable=True)
    before = emb.forward(0).copy()
    grad = np.ones(8, dtype=np.float64)
    emb.update(0, grad, lr=0.1)
    after = emb.forward(0)
    assert not np.allclose(before, after)
    assert np.allclose(after, before - 0.1)


def test_fixed_no_update() -> None:
    emb = ViewEmbedding(4, 8, seed=0, learnable=False)
    before = emb.forward(0).copy()
    emb.update(0, np.ones(8), lr=0.5)
    assert np.allclose(before, emb.forward(0))
