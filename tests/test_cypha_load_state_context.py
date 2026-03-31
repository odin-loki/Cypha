"""``CyphaDIF.load_state`` must restore Tier-1 context so ``score_matrix`` matches (PORT / registry)."""
from __future__ import annotations

import numpy as np


def test_cypha_dif_save_load_preserves_score_matrix_for_same_encode():
    from Cypha import CyphaDIF, VectorEncoder

    rng = np.random.default_rng(101)
    clf = CyphaDIF(VectorEncoder(4), field_dim=40, rng=rng)
    for i in range(40):
        clf.train_step(rng.standard_normal(4), str(i % 3))
    xq = np.asarray([0.15, -0.2, 0.05, 0.1], dtype=np.float64)
    _, h = clf._encode(xq)
    LLR0, labels = clf.score_matrix(h.reshape(1, -1), use_field=False)

    st = clf.save_state()
    clf2 = CyphaDIF(VectorEncoder(4), field_dim=40, rng=np.random.default_rng(999))
    clf2.load_state(st)
    _, h2 = clf2._encode(xq)
    LLR1, labels2 = clf2.score_matrix(h2.reshape(1, -1), use_field=False)

    assert labels == labels2
    np.testing.assert_allclose(LLR0, LLR1, rtol=0, atol=1e-12)
