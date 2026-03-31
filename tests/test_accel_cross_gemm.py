"""cypha_accel.cross_gemm — matches NumPy R @ D.T (CPU or GPU backend)."""
from __future__ import annotations

import numpy as np
import pytest

from cypha_accel.cross_gemm import cross_r_dT


def test_cross_matches_numpy():
    rng = np.random.default_rng(0)
    R = rng.standard_normal((32, 12))
    D = rng.standard_normal((5, 12))
    ref = R @ D.T
    got = cross_r_dT(R, D)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_cross_bad_shape_raises():
    R = np.zeros((2, 3))
    D = np.zeros((4, 4))
    with pytest.raises(ValueError):
        cross_r_dT(R, D)
