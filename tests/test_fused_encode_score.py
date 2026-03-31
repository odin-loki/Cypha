"""fused_features_to_latent_and_llr matches project_features + fused_score_llr."""

from __future__ import annotations

import numpy as np

from cypha_accel.score_batch import fused_features_to_latent_and_llr, fused_score_llr, project_features


def test_fused_features_matches_two_step():
    rng = np.random.default_rng(2)
    N, d, K = 17, 11, 4
    F = rng.standard_normal((N, d)).astype(np.float64)
    W = rng.standard_normal((d, d)).astype(np.float64)
    mu0 = rng.standard_normal(d).astype(np.float64)
    inv_v = np.abs(rng.standard_normal(d)).astype(np.float64) + 0.1
    D = rng.standard_normal((K, d)).astype(np.float64)
    D_sq = (D * D) @ inv_v
    u_k = rng.standard_normal(K).astype(np.float64)
    ctx_arr = rng.standard_normal(K).astype(np.float64)

    H1, LLR1 = fused_features_to_latent_and_llr(F, W, mu0, inv_v, D, D_sq, u_k, ctx_arr)
    H2 = project_features(F, W)
    LLR2 = fused_score_llr(H2, mu0, inv_v, D, D_sq, u_k, ctx_arr)

    np.testing.assert_allclose(H1, H2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(LLR1, LLR2, rtol=0, atol=1e-12)
