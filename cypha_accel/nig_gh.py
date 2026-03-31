"""
Vectorised GH–NIG effective noise (R_eff) for batch gates.

Kept in cypha_accel so fused CUDA paths can import without pulling in Cypha.py
(and avoid circular imports with score_batch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_EPS = 1e-8

try:
    from scipy.special import kv as _kv_init

    _BESSEL_X = np.linspace(1e-6, 120.0, 16384)
    _K0v = _kv_init(0.0, _BESSEL_X)
    _K1v = _kv_init(1.0, _BESSEL_X)
    _K2v = _kv_init(2.0, _BESSEL_X)
    _K2_K1_TABLE = _K2v / np.maximum(_K1v, 1e-300)
    _BESSEL_TABLES_OK = True
    del _kv_init, _K0v, _K1v, _K2v
except Exception:
    _npz = Path(__file__).resolve().parents[1] / "bessel_ratios.npz"
    try:
        if _npz.is_file():
            _z = np.load(_npz)
            _BESSEL_X = np.asarray(_z["x"], dtype=np.float64).reshape(-1)
            _K2_K1_TABLE = np.asarray(_z["k2_k1"], dtype=np.float64).reshape(-1)
            if _BESSEL_X.shape[0] == 16384 and _K2_K1_TABLE.shape[0] == 16384:
                _BESSEL_TABLES_OK = True
            else:
                raise ValueError("bessel_ratios.npz wrong length")
        else:
            raise FileNotFoundError(str(_npz))
    except Exception:
        _BESSEL_TABLES_OK = False
        _BESSEL_X = None
        _K2_K1_TABLE = None


def gig_e_inv_v_vec(lam: float, chi: np.ndarray, psi: float) -> np.ndarray:
    """E[1/V] for V ~ GIG(lambda, chi_i, psi); matches Cypha scalar _gig_E_inv_V element-wise."""
    chi0 = np.asarray(chi, dtype=np.float64)
    psi = float(psi)
    out = np.empty_like(chi0)
    mask_bad = (chi0 < _EPS) | (psi < _EPS)
    if np.any(mask_bad):
        out[mask_bad] = psi / np.maximum(chi0[mask_bad], _EPS)
    if not np.any(~mask_bad):
        return out

    chi_g = np.maximum(chi0[~mask_bad], _EPS)
    x = np.sqrt(chi_g * psi)
    sub = np.empty_like(chi_g)
    sm = x < 1e-6
    if np.any(sm):
        sub[sm] = psi / chi_g[sm]
    if not np.any(~sm):
        out[~mask_bad] = sub
        return out

    chi_b = chi_g[~sm]
    x_b = x[~sm]
    out_b = np.empty_like(x_b)
    if _BESSEL_TABLES_OK and abs(lam - (-1.0)) < 1e-9:
        use_tab = x_b <= 120.0
        if np.any(use_tab):
            xt = np.clip(x_b[use_tab], _BESSEL_X[0], _BESSEL_X[-1])
            ratio = np.interp(xt, _BESSEL_X, _K2_K1_TABLE)
            out_b[use_tab] = np.sqrt(psi / chi_b[use_tab]) * ratio
        if np.any(~use_tab):
            from scipy.special import kv as _kv

            xi = x_b[~use_tab]
            ci = chi_b[~use_tab]
            k_lam = _kv(abs(lam), xi)
            k_lam1 = _kv(abs(lam - 1), xi)
            ratio = k_lam1 / np.maximum(k_lam, 1e-300)
            out_b[~use_tab] = np.sqrt(psi / ci) * ratio
    else:
        try:
            from scipy.special import kv as _kv

            k_lam = _kv(abs(lam), x_b)
            k_lam1 = _kv(abs(lam - 1), x_b)
            ratio = k_lam1 / np.maximum(k_lam, 1e-300)
            out_b = np.sqrt(psi / chi_b) * ratio
        except Exception:
            out_b = psi / chi_b
    sub[~sm] = out_b
    out[~mask_bad] = sub
    return out


def nig_r_eff_vec(mp: np.ndarray, R: float, chi: float, psi: float) -> np.ndarray:
    mp = np.maximum(np.asarray(mp, dtype=np.float64), 0.0)
    chi_post = chi + mp / max(R, _EPS)
    e_inv = gig_e_inv_v_vec(-1.0, chi_post, psi)
    return R / np.maximum(e_inv, _EPS)
