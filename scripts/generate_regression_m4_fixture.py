#!/usr/bin/env python3
"""Write ``parity_fixtures/regression_m4/sidecar.json`` for ``native_regression_m4`` CTest."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "parity_fixtures" / "regression_m4" / "sidecar.json"


def _rff_rls_train_step(
    phi: np.ndarray,
    w: np.ndarray,
    b: float,
    P: np.ndarray,
    y_raw: float,
    y_mean: float,
    y_std: float,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Mirror ``RFFRegressor.train_step`` (Cypha.py)."""
    phi = np.asarray(phi, dtype=np.float64).ravel()
    D = int(phi.shape[0])
    phi_b = np.append(phi, 1.0)
    yn = (float(y_raw) - y_mean) / y_std
    pred = float(phi @ w) + b
    err = yn - pred
    Pp = P @ phi_b
    denom = 1.0 + float(phi_b @ Pp)
    Pn = P - np.outer(Pp, Pp) / denom
    delta = (Pp / denom) * err
    wn = w + delta[:D]
    bn = float(b + delta[D])
    loss = err * err * (y_std**2)
    return wn, bn, Pn, float(loss)


def _mke_expert_rls_scalar_step(
    phi: np.ndarray,
    w: np.ndarray,
    P: np.ndarray,
    pi: float,
    gh_scale: float,
    err: float,
    forgetting_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One expert inner loop from ``MKERegressor.train_step`` (scalar target)."""
    phi = np.asarray(phi, dtype=np.float64).ravel()
    P = np.asarray(P, dtype=np.float64).copy()
    w = np.asarray(w, dtype=np.float64).copy()
    if pi < 0.02:
        return w, P
    if forgetting_factor < 1.0:
        P /= forgetting_factor
    Pphi = P @ phi
    denom = 1.0 + pi * float(phi @ Pphi)
    K_g = pi * Pphi / denom * gh_scale
    w = w + K_g * err
    P = P - pi * np.outer(K_g, phi) @ P
    return w, P


def _softmax_row(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Match ``Cypha._softmax`` (K≤8 list path vs K>8 vector path)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    k = int(x.shape[0])
    if k <= 8:
        mx = float(x.max())
        e = np.array([math.exp(float(v) - mx) for v in x], dtype=np.float64)
        s = float(e.sum()) + eps
        return e / s
    x2 = x - x.max()
    e = np.exp(x2)
    return e / (e.sum() + eps)


def _two_stage_predict(
    llr: np.ndarray,
    x: np.ndarray,
    w1: np.ndarray,
    b1: float,
    phi2: np.ndarray,
    w2: np.ndarray,
    b2: float,
    y_mean: float,
    y_std: float,
) -> float:
    """Mirror ``TwoStageDIFRegressor.predict`` (normalized combine then scale)."""
    llr = np.asarray(llr, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    w1 = np.asarray(w1, dtype=np.float64).ravel()
    phi2 = np.asarray(phi2, dtype=np.float64).ravel()
    w2 = np.asarray(w2, dtype=np.float64).ravel()
    y_s1 = float(np.dot(np.concatenate([llr, x]), w1) + b1)
    y_s2 = float(np.dot(phi2, w2) + b2)
    return float((y_s1 + y_s2) * y_std + y_mean)


def main() -> None:
    rng = np.random.default_rng(42)
    n, k, d = 5, 4, 3
    probs = rng.random((n, k))
    probs /= probs.sum(axis=1, keepdims=True)
    mu_mat = rng.standard_normal((k, d))
    var_vec = (rng.random(k) ** 2).astype(np.float64)
    y_exp = (probs @ mu_mat).ravel()
    unc_exp = np.sqrt(np.maximum(probs @ var_vec, 0.0))

    lr = 0.06
    mu_b = np.array([1.0, -0.5, 2.0], dtype=np.float64)
    var_b = 0.08
    n_b = 7
    y_obs = np.array([0.25, 0.0, 1.75], dtype=np.float64)
    delta = y_obs - mu_b
    mu_a = mu_b + lr * delta
    var_a = (1.0 - lr) * var_b + lr * float(delta @ delta)

    y_init = np.array([3.0, -1.0], dtype=np.float64)

    rng2 = np.random.default_rng(7)
    D_rff = 12
    phi_r = rng2.standard_normal(D_rff)
    w_r = rng2.standard_normal(D_rff)
    b_r = 0.13
    lam0 = 0.5
    P_r = np.eye(D_rff + 1, dtype=np.float64) / lam0
    y_mean_r = 2.0
    y_std_r = 1.5
    y_raw_r = 0.25
    w_after, b_after, P_after, loss_exp = _rff_rls_train_step(
        phi_r, w_r, b_r, P_r, y_raw_r, y_mean_r, y_std_r
    )

    D_mke = 8
    phi_m = rng2.standard_normal(D_mke)
    w_m = rng2.standard_normal(D_mke)
    P0 = 10.0
    P_m = np.eye(D_mke, dtype=np.float64) * P0
    pi_m = 0.35
    gh_m = 0.85
    err_m = -0.42
    ff_m = 1.0
    w_mke_a, P_mke_a = _mke_expert_rls_scalar_step(phi_m, w_m, P_m, pi_m, gh_m, err_m, ff_m)

    ff2 = 0.97
    w_m2 = rng2.standard_normal(D_mke)
    P_m2 = np.eye(D_mke, dtype=np.float64) * (P0 * 0.5)
    w_mke_ff, P_mke_ff = _mke_expert_rls_scalar_step(phi_m, w_m2, P_m2, pi_m, 1.0, err_m, ff2)

    K_ts, d_ts, D2 = 3, 4, 16
    llr_ts = rng2.standard_normal(K_ts)
    x_ts = rng2.standard_normal(d_ts)
    w1_ts = rng2.standard_normal(K_ts + d_ts)
    b1_ts = -0.05
    phi2_ts = rng2.standard_normal(D2)
    w2_ts = rng2.standard_normal(D2)
    b2_ts = 0.11
    ym_ts, ys_ts = 0.5, 2.25
    y_ts_exp = _two_stage_predict(llr_ts, x_ts, w1_ts, b1_ts, phi2_ts, w2_ts, b2_ts, ym_ts, ys_ts)

    D_skip = 5
    phi_s = rng2.standard_normal(D_skip)
    w_s = rng2.standard_normal(D_skip)
    P_s = np.eye(D_skip, dtype=np.float64) * 3.0

    rng3 = np.random.default_rng(11)
    eps_rt = 1e-8
    K_mr = 6
    llr_mr = rng3.standard_normal(K_mr)
    T_mr = 1.25
    z_mr = llr_mr / (T_mr + eps_rt)
    p_mr = _softmax_row(z_mr, eps_rt)
    mu_mr = rng3.standard_normal(K_mr)
    yhat_mr = float(np.dot(p_mr, mu_mr))
    entr_mr = float(-np.dot(p_mr, np.log(p_mr + eps_rt)))

    K_10 = 10
    llr_10 = rng3.standard_normal(K_10)
    z_10 = llr_10 / (T_mr + eps_rt)
    p_10 = _softmax_row(z_10, eps_rt)
    mu_10 = rng3.standard_normal(K_10)
    yhat_10 = float(np.dot(p_10, mu_10))
    entr_10 = float(-np.dot(p_10, np.log(p_10 + eps_rt)))

    payload = {
        "batch": {
            "n": int(n),
            "k": int(k),
            "d": int(d),
            "probs": probs.astype(np.float64).ravel().tolist(),
            "mu_mat": mu_mat.astype(np.float64).ravel(order="C").tolist(),
            "var_vec": var_vec.tolist(),
            "expected_y": y_exp.tolist(),
            "expected_unc": unc_exp.astype(np.float64).tolist(),
        },
        "ema": {
            "d": 3,
            "lr": lr,
            "mu_before": mu_b.tolist(),
            "var_before": float(var_b),
            "n_before": n_b,
            "y": y_obs.tolist(),
            "mu_after": mu_a.tolist(),
            "var_after": float(var_a),
            "n_after": n_b + 1,
        },
        "ema_init": {
            "d": 2,
            "lr": lr,
            "y": y_init.tolist(),
            "mu_after": y_init.tolist(),
            "var_after": 0.0,
            "n_after": 1,
        },
        "rff_rls": {
            "D": D_rff,
            "phi": phi_r.astype(np.float64).tolist(),
            "w_before": w_r.tolist(),
            "b_before": float(b_r),
            "P_before": P_r.astype(np.float64).ravel(order="C").tolist(),
            "y_raw": float(y_raw_r),
            "y_mean": float(y_mean_r),
            "y_std": float(y_std_r),
            "expected_loss": float(loss_exp),
            "w_after": w_after.tolist(),
            "b_after": float(b_after),
            "P_after": P_after.astype(np.float64).ravel(order="C").tolist(),
        },
        "mke_rls": {
            "D": D_mke,
            "phi": phi_m.tolist(),
            "pi": float(pi_m),
            "gh_scale": float(gh_m),
            "err": float(err_m),
            "forgetting_factor": float(ff_m),
            "w_before": w_m.tolist(),
            "P_before": P_m.astype(np.float64).ravel(order="C").tolist(),
            "w_after": w_mke_a.tolist(),
            "P_after": P_mke_a.astype(np.float64).ravel(order="C").tolist(),
            "forgetting_case": {
                "forgetting_factor": float(ff2),
                "w_before": w_m2.tolist(),
                "P_before": P_m2.astype(np.float64).ravel(order="C").tolist(),
                "w_after": w_mke_ff.tolist(),
                "P_after": P_mke_ff.astype(np.float64).ravel(order="C").tolist(),
            },
            "low_pi_noop": {
                "D": D_skip,
                "phi": phi_s.tolist(),
                "pi": 0.01,
                "gh_scale": 1.0,
                "err": 1.0,
                "forgetting_factor": 1.0,
                "w_before": w_s.tolist(),
                "P_before": P_s.astype(np.float64).ravel(order="C").tolist(),
            },
        },
        "two_stage": {
            "K": K_ts,
            "d_in": d_ts,
            "D2": D2,
            "llr": llr_ts.tolist(),
            "x": x_ts.tolist(),
            "w1": w1_ts.tolist(),
            "b1": float(b1_ts),
            "phi2": phi2_ts.tolist(),
            "w2": w2_ts.tolist(),
            "b2": float(b2_ts),
            "y_mean": float(ym_ts),
            "y_std": float(ys_ts),
            "expected_y": float(y_ts_exp),
        },
        "mke_route": {
            "K": K_mr,
            "llr": llr_mr.tolist(),
            "temperature": float(T_mr),
            "eps": float(eps_rt),
            "expert_mu": mu_mr.tolist(),
            "expected_probs": p_mr.tolist(),
            "expected_y_hat": float(yhat_mr),
            "expected_entropy": float(entr_mr),
            "k_gt_8": {
                "K": K_10,
                "llr": llr_10.tolist(),
                "expected_probs": p_10.tolist(),
                "expected_y_hat": float(yhat_10),
                "expected_entropy": float(entr_10),
                "expert_mu": mu_10.tolist(),
            },
        },
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}")


if __name__ == "__main__":
    main()
