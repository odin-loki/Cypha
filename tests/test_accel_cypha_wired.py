"""
Parity: `score_matrix` with production `fused_score_llr` vs pure NumPy reference.
"""

from __future__ import annotations

from unittest import mock

import numpy as np

import Cypha as cy


def _np_fused_llr(H, mu0, inv_v, D, D_sq, u_k=None, ctx_arr=None):
    H = np.ascontiguousarray(H, dtype=np.float64)
    mu0 = np.ascontiguousarray(mu0, dtype=np.float64).ravel()
    inv_v = np.ascontiguousarray(inv_v, dtype=np.float64).ravel()
    D = np.ascontiguousarray(D, dtype=np.float64)
    D_sq = np.ascontiguousarray(D_sq, dtype=np.float64).ravel()
    K = D.shape[0]
    if u_k is None:
        u_k = np.zeros(K, dtype=np.float64)
    if ctx_arr is None:
        ctx_arr = np.zeros(K, dtype=np.float64)
    R = (H - mu0) * inv_v
    return R @ D.T - 0.5 * D_sq - u_k + ctx_arr


def test_score_matrix_auto_gemm_matches_numpy_reference():
    rng = np.random.default_rng(0)
    d, K = 8, 3
    H = rng.standard_normal((12, d)).astype(np.float64)
    enc = cy.VectorEncoder(d)
    clf = cy.CyphaDIF(enc, rng=rng)
    labels = [str(k) for k in range(K)]
    for _ in range(4):
        for k in range(K):
            clf.train_step(rng.standard_normal(d), labels[k])

    with mock.patch.object(cy, "fused_score_llr", side_effect=_np_fused_llr):
        LLR_ref, labs_ref = clf.score_matrix(H, use_field=True)
    LLR_act, labs_act = clf.score_matrix(H, use_field=True)

    assert labs_ref == labs_act
    np.testing.assert_allclose(LLR_act, LLR_ref, rtol=0.0, atol=1e-9)
