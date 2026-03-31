"""GH gate: vectorized ``cypha_accel.nig_gh`` vs Cypha scalar GH helpers."""
from __future__ import annotations

import numpy as np

import Cypha as cy
from cypha_accel.nig_gh import gig_e_inv_v_vec, nig_r_eff_vec


def test_gig_e_inv_v_vec_matches_scalar():
    rng = np.random.default_rng(42)
    chi = np.abs(rng.standard_normal(500)) * 2.0 + 1e-5
    psi = 1.3
    lam = -1.0
    ref = np.array([cy._gig_E_inv_V(lam, float(c), psi) for c in chi])
    got = gig_e_inv_v_vec(lam, chi, psi)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-10)


def test_nig_r_eff_vec_matches_scalar():
    rng = np.random.default_rng(43)
    mp = np.abs(rng.standard_normal(300)) * 3.0
    R_base = 0.85
    chi, psi = 1.0, 1.0
    ref = np.array([cy._nig_R_eff(float(x), R_base, chi, psi) for x in mp])
    got = nig_r_eff_vec(mp, R_base, chi, psi)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-10)
