"""score_matrix use_field modes and shape invariants."""
from __future__ import annotations

import numpy as np
import pytest

from Cypha import CyphaDIF, VectorEncoder


def _clf(seed: int = 3):
    rng = np.random.default_rng(seed)
    clf = CyphaDIF(VectorEncoder(6), field_dim=40, rng=np.random.default_rng(seed))
    for t in range(50):
        lbl = str(t % 3)
        o = np.zeros(6)
        o[int(lbl) * 2] = 1.5
        clf.train_step(rng.normal(0, 0.3, 6) + o, lbl)
    return clf


def test_score_matrix_use_field_both_finite():
    clf = _clf()
    rng = np.random.default_rng(11)
    H = rng.normal(0, 0.5, (12, clf.feat_dim))
    L0, lab0 = clf.score_matrix(H, use_field=False)
    L1, lab1 = clf.score_matrix(H, use_field=True)
    assert lab0 == lab1
    assert L0.shape == L1.shape
    assert np.all(np.isfinite(L0))
    assert np.all(np.isfinite(L1))


def test_score_matrix_empty_classes_returns_empty():
    clf = CyphaDIF(VectorEncoder(3), field_dim=16, rng=np.random.default_rng(0))
    H = np.zeros((2, clf.feat_dim), dtype=np.float64)
    LLR, labels = clf.score_matrix(H)
    assert labels == []
    assert LLR.shape == (2, 0)


def test_batch_infer_serial_parity_small_batch():
    clf = _clf()
    rng = np.random.default_rng(22)
    xs = [rng.normal(0, 0.4, 6) for _ in range(8)]
    batch = clf.batch_infer(xs, use_field=True)
    for i, x in enumerate(xs):
        p, c = clf.infer(x)
        assert batch[i][0] == p
        assert abs(batch[i][1] - c) < 1e-9
