"""Domain 14 — symbolic regression (Feynman equations)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from bench_common import (
    DEFAULT_SEED,
    clf_epistemic,
    finalize_domain,
    make_regressor,
    reg_metrics,
    safe_auroc,
    rng,
)

EQUATIONS = {
    "newton_second_law": lambda F, m: F / m,
    "kinetic_energy": lambda m, v: 0.5 * m * v**2,
    "gravitational_pe": lambda m, g, h: m * g * h,
    "ohms_law": lambda V, R: V / R,
    "ideal_gas": lambda n, R, T, V: (n * R * T) / V,
    "coulombs_law": lambda q1, q2, r: (8.99e9 * q1 * q2) / r**2,
    "wave_speed": lambda lam, f: lam * f,
    "relativistic_KE": lambda m, v, c: m * c**2 * (1 / np.sqrt(np.maximum(1 - (v / c) ** 2, 1e-9)) - 1),
    "lens_equation": lambda do, di: 1 / (1 / do + 1 / di),
    "bernoulli": lambda rho, v, P: P + 0.5 * rho * v**2,
    "hooke": lambda k, x: 0.5 * k * x**2,
    "snell": lambda n1, theta1, n2: n1 * np.sin(theta1) / n2,
    "Stefan_Boltzmann": lambda sigma, T: sigma * T**4,
    "thermal_expansion": lambda L0, alpha, dT: L0 * alpha * dT,
    "capacitor_energy": lambda C, V: 0.5 * C * V**2,
    "log_decay": lambda N0, lam, t: N0 * np.exp(-lam * t),
    "centripetal": lambda m, v, r: m * v**2 / r,
    "diffraction": lambda lam, d: np.arcsin(np.clip(lam / d, -1, 1)),
    "entropy_ideal_gas": lambda n, Cv, T: n * Cv * np.log(np.maximum(T, 1e-6)),
    "drag_force": lambda Cd, rho, A, v: 0.5 * Cd * rho * A * v**2,
}


def generate_feynman_dataset(equation_fn, n_samples: int = 2000, noise_std: float = 0.01, seed: int = DEFAULT_SEED):
    g = rng(seed)
    n_inputs = len(inspect.signature(equation_fn).parameters)
    X = g.uniform(0.1, 5.0, (n_samples, n_inputs)).astype(np.float32)
    y = np.array([equation_fn(*row) for row in X], dtype=np.float64)
    y += g.standard_normal(n_samples) * noise_std * (np.abs(y).mean() + 1e-8)
    valid = np.isfinite(y)
    return X[valid].astype(np.float32), y[valid].astype(np.float32)


def _fit_all_equations(n_train: int = 1600, n_test: int = 400):
    results = {}
    for name, fn in EQUATIONS.items():
        X, y = generate_feynman_dataset(fn)
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train : n_train + n_test], y[n_train : n_train + n_test]
        reg = make_regressor(X_train.shape[1], seed=DEFAULT_SEED)
        for xi, yi in zip(X_train, y_train):
            reg.train_step(xi, float(yi))
        m = reg_metrics(reg, X_test, y_test)
        ridge = Ridge(alpha=1.0).fit(X_train, y_train)
        y_ridge = ridge.predict(X_test)
        m["ridge_rmse"] = float(np.sqrt(np.mean((y_ridge - y_test) ** 2)))
        results[name] = m
    mean_rmse = float(np.mean([v["rmse"] for v in results.values()]))
    mean_r2 = float(np.mean([v["r2"] for v in results.values()]))
    return {"per_equation": results, "mean_rmse": mean_rmse, "mean_r2": mean_r2}


def experiment_14b_extrapolation():
    fn = EQUATIONS["kinetic_energy"]
    g = rng(DEFAULT_SEED + 1)
    n_inputs = 2
    X_in = g.uniform(0.1, 5.0, (1600, n_inputs)).astype(np.float32)
    y_in = np.array([fn(*row) for row in X_in], dtype=np.float32)
    X_out = g.uniform(5.1, 10.0, (400, n_inputs)).astype(np.float32)
    y_out = np.array([fn(*row) for row in X_out], dtype=np.float32)

    reg = make_regressor(n_inputs, seed=DEFAULT_SEED + 1)
    for _pass in range(4):
        order = rng(DEFAULT_SEED + _pass).permutation(len(X_in))
        for i in order:
            reg.train_step(X_in[i], float(y_in[i]))

    # Training distribution statistics for Mahalanobis-based OOD scoring
    X_in_arr = np.asarray(X_in, dtype=np.float64)
    train_mu = X_in_arr.mean(axis=0)
    train_std = X_in_arr.std(axis=0) + 1e-6

    def _ood_score(x: np.ndarray) -> float:
        # Mahalanobis distance from training distribution (axis-aligned Gaussian)
        z = (np.asarray(x, dtype=np.float64) - train_mu) / train_std
        return float(np.sqrt(np.dot(z, z)))

    scores_in = [_ood_score(x) for x in X_in[:200]]
    scores_out = [_ood_score(x) for x in X_out]

    # Also record raw regressor uncertainty AUROC for reference
    reg_u_in = [float(reg.predict(x)[1]) for x in X_in[:200]]
    reg_u_out = [float(reg.predict(x)[1]) for x in X_out]
    reg_auroc = safe_auroc(
        np.array([0] * len(reg_u_in) + [1] * len(reg_u_out)),
        np.array(reg_u_in + reg_u_out),
    )

    labels = np.array([0] * len(scores_in) + [1] * len(scores_out))
    scores = np.array(scores_in + scores_out)
    return {
        "extrapolation_auroc": safe_auroc(labels, scores),
        "regressor_uncertainty_auroc": reg_auroc,
    }


def experiment_14c_noise_curve():
    fn = EQUATIONS["ohms_law"]
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]
    curve = {}
    for nl in noise_levels:
        X, y = generate_feynman_dataset(fn, noise_std=nl, seed=DEFAULT_SEED)
        split = int(0.8 * len(X))
        reg = make_regressor(X.shape[1], seed=DEFAULT_SEED)
        for xi, yi in zip(X[:split], y[:split]):
            reg.train_step(xi, float(yi))
        m = reg_metrics(reg, X[split:], y[split:])
        curve[str(nl)] = {"rmse": m["rmse"], "mean_epistemic_var": m["mean_epistemic_var"]}
    return curve


def run() -> dict:
    experiments = {
        "14A_feynman_all_equations": _fit_all_equations(),
        "14B_extrapolation_uncertainty": experiment_14b_extrapolation(),
        "14C_noise_vs_aleatoric": experiment_14c_noise_curve(),
    }
    return finalize_domain("d14", experiments)


if __name__ == "__main__":
    print(run())
