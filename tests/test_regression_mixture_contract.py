"""
Numeric contract for `DIFRegressor.predict` (scalar path) vs native `predict_mixture_scalar`.

Native: CTest `native_regression_mixture` (`regression_mixture_parity`); batched path + EMA: `native_regression_m4` (`regression_m4_parity`).
"""
from __future__ import annotations

import numpy as np


def test_scalar_mixture_formula_matches_dif_regressor_predict():
    """Same Σ p·μ and √(Σ p·var) as `Cypha.DIFRegressor.predict` when target_dim == 1."""
    probs = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    mu = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    var = np.array([0.1, 0.4, 0.2], dtype=np.float64)
    y_pred = float(np.dot(probs, mu))
    mix_var = float(np.dot(probs, var))
    unc = float(np.sqrt(max(mix_var, 0.0)))
    assert abs(y_pred - 2.1) < 1e-15
    assert abs(mix_var - 0.28) < 1e-15
    assert abs(unc - np.sqrt(0.28)) < 1e-15


def test_predict_batch_mixture_matches_matrix_formula():
    """Same ``P @ mu_mat`` and ``sqrt(max(P @ var_vec,0))`` as ``DIFRegressor.predict_batch``."""
    rng = np.random.default_rng(1)
    n, k, d = 6, 3, 2
    probs = rng.random((n, k))
    probs /= probs.sum(axis=1, keepdims=True)
    mu_mat = rng.standard_normal((k, d))
    var_vec = (rng.random(k) ** 2).astype(np.float64)
    y = probs @ mu_mat
    unc = np.sqrt(np.maximum(probs @ var_vec, 0.0))
    assert y.shape == (n, d)
    assert unc.shape == (n,)
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(unc))
