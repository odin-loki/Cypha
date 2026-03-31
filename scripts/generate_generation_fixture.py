#!/usr/bin/env python3
"""
Emit ``parity_fixtures/generation/sidecar.json`` for ``generation_parity``.

Each case stores pre-drawn random variates (z / u) alongside expected outputs so
the native tool can replay the exact same arithmetic without needing a matching
RNG implementation.

Re-run whenever Cypha.py generation math changes:
  python scripts/generate_generation_fixture.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary  # noqa: E402

_FIX = _ROOT / "parity_fixtures"
_OUT = _FIX / "generation" / "sidecar.json"

_MIN_VAR = 1e-4
_EPS = 1e-8

# Tolerance written into the fixture so the native tool uses the same check.
_ATOL = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_params(dif: CyphaDIF, label: str):
    """(mu_k, inv_v, delta_mu_k, v0)."""
    with dif.memory._lock:
        mu0 = dif.memory.world.mu.copy()
        v0 = dif.memory.world.v.copy()
        inv_v = dif.memory.world.inv_v.copy()
        delta_k = dif.memory._classes[label].delta_mu.copy()
    mu_k = mu0 + delta_k
    return mu_k, inv_v, delta_k, v0


def _d_buf(dif: CyphaDIF):
    with dif.memory._lock:
        K = len(dif.memory._label_order)
        D = dif.memory._D_buf[:K].copy()
        order = list(dif.memory._label_order)
        idx_map = dict(dif.memory._label_idx)
    return D, order, idx_map


def _fused_llr(H, mu0, inv_v, D):
    """LLR[i,k] without MDL penalty or context (generation path)."""
    R = (H - mu0) * inv_v
    D_sq = (D * D) @ inv_v  # (K,)
    return R @ D.T - 0.5 * D_sq  # (N, K)


def _batch_logpdf(H, mu, v):
    v_safe = np.maximum(v, _MIN_VAR)
    log_norm = 0.5 * np.sum(np.log(v_safe))
    maha = 0.5 * ((H - mu) ** 2 / v_safe).sum(axis=1)
    return -log_norm - maha


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / (e.sum() + _EPS)


def _to_field_dim(dif: CyphaDIF, h: np.ndarray) -> np.ndarray:
    h_norm = h / (float(h @ h) ** 0.5 + _EPS)
    return h_norm if dif._W_inject is None else dif._W_inject @ h_norm


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------

def _case_gaussian_no_rejection(dif: CyphaDIF, label: str, n: int, temperature: float,
                                 rng: np.random.Generator):
    mu_k, inv_v, _, v0 = _class_params(dif, label)
    d = len(mu_k)
    z = rng.standard_normal((n, d))
    std = np.sqrt(np.maximum(v0, _MIN_VAR)) * temperature
    expected_h = [mu_k + z[i] * std for i in range(n)]
    return {
        "label": label, "n": n, "temperature": temperature,
        "z": z.tolist(),
        "expected_h": [h.tolist() for h in expected_h],
    }


def _case_gaussian_rejection(dif: CyphaDIF, label: str, n: int, temperature: float,
                               max_candidates: int, rng: np.random.Generator):
    mu_k, inv_v, _, v0 = _class_params(dif, label)
    with dif.memory._lock:
        mu0 = dif.memory.world.mu.copy()
        K = len(dif.memory._label_order)
        D = dif.memory._D_buf[:K].copy()
        k_idx = dif.memory._label_idx.get(label, -1)
    d = len(mu_k)
    std = np.sqrt(np.maximum(v0, _MIN_VAR)) * temperature
    # Draw all candidates at once: (n, max_candidates, d)
    z_all = rng.standard_normal((n, max_candidates, d))
    expected_h = []
    for i in range(n):
        cands = mu_k + z_all[i] * std  # (C, d)
        llr = _fused_llr(cands, mu0, inv_v, D)  # (C, K)
        best_i = int(llr[:, k_idx].argmax()) if k_idx >= 0 else 0
        expected_h.append(cands[best_i].tolist())
    return {
        "label": label, "n": n, "temperature": temperature,
        "max_candidates": max_candidates, "k_idx": k_idx,
        "z_candidates": z_all.reshape(n * max_candidates, d).tolist(),
        "expected_h": expected_h,
    }


def _case_conditioned(dif: CyphaDIF, label: str, n: int, temperature: float,
                       rng: np.random.Generator):
    h_field = dif.field.h
    mu0_cond, v0 = dif.memory.world.condition_on_field(h_field)
    with dif.memory._lock:
        delta_k = dif.memory._classes[label].delta_mu.copy()
    mu_k = mu0_cond + delta_k
    d = len(mu_k)
    z = rng.standard_normal((n, d))
    std = np.sqrt(np.maximum(v0, _MIN_VAR)) * temperature
    expected_h = [mu_k + z[i] * std for i in range(n)]
    return {
        "label": label, "n": n, "temperature": temperature,
        "z": z.tolist(),
        "expected_h": [h.tolist() for h in expected_h],
    }


def _case_langevin(dif: CyphaDIF, label: str, n: int, n_steps: int,
                    step_size: float, temperature: float, rng: np.random.Generator):
    mu_k, inv_v, delta_k, v0 = _class_params(dif, label)
    d = len(mu_k)
    grad_llr_k = inv_v * delta_k
    v_prior = v0 * max(temperature ** 2, 0.1)
    sqrt_2step_T = math.sqrt(2.0 * step_size * max(temperature ** 2, _MIN_VAR))
    z_init = rng.standard_normal((n, d))
    z_noise = rng.standard_normal((n, n_steps, d))
    expected_h = []
    for i in range(n):
        h = mu_k + z_init[i] * np.sqrt(v0) * 0.5
        for s in range(n_steps):
            grad = grad_llr_k - (h - mu_k) / (v_prior + _EPS)
            h = h + step_size * grad + sqrt_2step_T * z_noise[i, s]
        expected_h.append(h.tolist())
    return {
        "label": label, "n": n, "n_steps": n_steps,
        "step_size": step_size, "temperature": temperature,
        "z_init": z_init.tolist(),
        "z_noise": z_noise.tolist(),
        "expected_h": expected_h,
    }


def _case_boundary(dif: CyphaDIF, label_a: str, label_b: str,
                    n: int, alpha: float, temperature: float,
                    rng: np.random.Generator):
    mu_a, inv_v, dm_a, v0 = _class_params(dif, label_a)
    mu_b, _, dm_b, _ = _class_params(dif, label_b)
    with dif.memory._lock:
        mu0 = dif.memory.world.mu.copy()
    normal = (dm_a - dm_b) * inv_v
    n_sq = float(normal @ normal) + _EPS
    target_dot = 0.5 * (float(dm_a @ (dm_a * inv_v)) - float(dm_b @ (dm_b * inv_v)))
    mu_interp = (1.0 - alpha) * mu_a + alpha * mu_b
    d = len(mu_interp)
    std = np.sqrt(np.maximum(v0, _MIN_VAR)) * max(temperature, _EPS)
    z = rng.standard_normal((n, d))
    expected_h = []
    for i in range(n):
        h = mu_interp + z[i] * std
        curr = float((h - mu0) @ normal)
        t = (curr - target_dot) / n_sq
        h = h - t * normal
        expected_h.append(h.tolist())
    return {
        "label_a": label_a, "label_b": label_b,
        "n": n, "alpha": alpha, "temperature": temperature,
        "z": z.tolist(),
        "expected_h": expected_h,
    }


def _case_ood(dif: CyphaDIF, n: int, n_candidates: int, rng: np.random.Generator):
    with dif.memory._lock:
        mu0 = dif.memory.world.mu.copy()
        v0 = dif.memory.world.v.copy()
        class_items = [(k, dif.memory._classes[k]) for k in dif.memory._classes]
    d = len(mu0)
    std = np.sqrt(np.maximum(v0, _MIN_VAR)) * 2.0
    z = rng.standard_normal((n_candidates, d))
    H = mu0 + z * std
    ll_world = _batch_logpdf(H, mu0, v0)
    max_llr = np.full(n_candidates, -np.inf)
    for lbl, cd in class_items:
        mu_k = cd.mu(mu0)
        u_k = float(np.mean(v0)) / (cd.n_obs + 1)
        ll_k = _batch_logpdf(H, mu_k, v0)
        llr_k = ll_k - ll_world - u_k
        np.maximum(max_llr, llr_k, out=max_llr)
    order = np.argsort(max_llr)
    expected_h = [H[i].tolist() for i in order[:n]]
    return {
        "n": n, "n_candidates": n_candidates,
        "z_candidates": z.tolist(),
        "expected_h": expected_h,
    }


def _case_mdl_ball(dif: CyphaDIF, label: str, n: int, radius: float,
                    rng: np.random.Generator):
    mu_k, _, _, v0 = _class_params(dif, label)
    std = np.sqrt(np.maximum(v0, _MIN_VAR))
    d = len(mu_k)
    z_dir = rng.standard_normal((n, d))
    u_mag = rng.uniform(0, 1, size=n)
    expected_h = []
    for i in range(n):
        raw_fr = z_dir[i] / std
        fr_norm = float(np.linalg.norm(raw_fr)) + _EPS
        dir_fr = raw_fr / fr_norm
        r = radius * float(u_mag[i] ** (1.0 / d))
        delta = dir_fr * r * std
        expected_h.append((mu_k + delta).tolist())
    return {
        "label": label, "n": n, "radius": radius,
        "z_dir": z_dir.tolist(),
        "u_mag": u_mag.tolist(),
        "expected_h": expected_h,
    }


def _case_ancestral(dif: CyphaDIF, n: int, temperature: float,
                     rng: np.random.Generator):
    with dif.memory._lock:
        classes = list(dif.memory._classes.keys())
        n_obs_arr = np.array([dif.memory._classes[k].n_obs for k in classes],
                             dtype=np.float64)
        mu0 = dif.memory.world.mu.copy()
        v0 = dif.memory.world.v.copy()
        deltas = [dif.memory._classes[k].delta_mu.copy() for k in classes]
    d = len(mu0)
    freq = np.maximum(n_obs_arr, 1.0)
    freq /= freq.sum()
    probs = freq ** (1.0 / (temperature + _EPS))
    probs /= probs.sum()
    # Store cumulative for categorical inverse-CDF
    u_class = rng.uniform(size=n)
    z = rng.standard_normal((n, d))
    expected_labels = []
    expected_h = []
    for i in range(n):
        # inverse CDF
        cum = 0.0
        idx = len(classes) - 1
        for ki, p in enumerate(probs):
            cum += p
            if u_class[i] < cum:
                idx = ki
                break
        label = classes[idx]
        mu_k = mu0 + deltas[idx]
        std = np.sqrt(np.maximum(v0, _MIN_VAR))
        expected_labels.append(label)
        expected_h.append((mu_k + z[i] * std).tolist())
    return {
        "n": n, "temperature": temperature,
        "class_probs": probs.tolist(),
        "u_class": u_class.tolist(),
        "z": z.tolist(),
        "expected_labels": expected_labels,
        "expected_h": expected_h,
    }


def _case_predict_next(dif: CyphaDIF, last_label: str):
    with dif.memory._lock:
        classes = list(dif.memory._classes.keys())
    # Set context last label without modifying rest of state
    orig = dif.context._last_label
    dif.context._last_label = last_label
    dist = dif.predict_next(last_label)
    dif.context._last_label = orig
    # Return probs in label order
    probs = [dist.get(k, 0.0) for k in classes]
    return {
        "last_label": last_label,
        "classes": classes,
        "expected_probs": probs,
    }


def _case_rollout(dif: CyphaDIF, seed_label: str, n_steps: int, temperature: float,
                   exploration: float, rng: np.random.Generator):
    """Run rollout with pre-drawn randoms; record z and u for native replay."""
    with dif.memory._lock:
        classes = list(dif.memory._classes.keys())
        n_obs_arr = np.array([dif.memory._classes[k].n_obs for k in classes],
                              dtype=np.float64)
        v0 = dif.memory.world.v.copy()
        mu0 = dif.memory.world.mu.copy()
        inv_v = dif.memory.world.inv_v.copy()
        D = dif.memory._D_buf[:len(classes)].copy()
    d = dif.memory.world.mu.shape[0]
    K = len(classes)
    uniform = np.ones(K) / K

    z_generate = rng.standard_normal((n_steps, d))
    u_transition = rng.uniform(size=n_steps)

    current = seed_label
    expected_labels = []
    expected_h = []

    for step in range(n_steps):
        # Generate from current class (no rejection — temperature ≤ 1)
        ki = classes.index(current)
        delta_k = dif.memory._classes[current].delta_mu.copy()
        mu_k = mu0 + delta_k
        std = np.sqrt(np.maximum(v0, _MIN_VAR)) * temperature
        h = mu_k + z_generate[step] * std
        expected_labels.append(current)
        expected_h.append(h.tolist())

        # Update context
        dif.context.record(current, correct=True)
        dif.field.inject(_to_field_dim(dif, h), strength=0.05)
        dif.field.evolve(dif.field.h, update_state=True)

        # Next distribution
        next_dist = dif.predict_next(current)
        raw_probs = np.array([next_dist.get(k, 0.0) for k in classes])
        raw_probs = raw_probs / (raw_probs.sum() + _EPS)
        mixed = (1.0 - exploration) * raw_probs + exploration * uniform
        mixed = mixed / mixed.sum()

        # Categorical from pre-drawn uniform
        cum = 0.0
        chosen = K - 1
        for ki2, p in enumerate(mixed):
            cum += p
            if u_transition[step] < cum:
                chosen = ki2
                break
        current = classes[chosen]

    return {
        "seed_label": seed_label,
        "n_steps": n_steps,
        "temperature": temperature,
        "exploration": exploration,
        "z_generate": z_generate.tolist(),
        "u_transition": u_transition.tolist(),
        "expected_labels": expected_labels,
        "expected_h": expected_h,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cypha_path = _FIX / "reference.cypha"
    if not cypha_path.is_file():
        raise SystemExit(f"missing {cypha_path} — run scripts/generate_parity_fixtures.py first")

    state = cypha_load_binary(str(cypha_path))
    dif = CyphaDIF(encoder=VectorEncoder(8), field_dim=24,
                   rng=np.random.default_rng(0))
    dif.load_state(state)

    with dif.memory._lock:
        labels = list(dif.memory._label_order)
        d = int(dif.memory.world.mu.shape[0])
    field_dim = int(dif.field.h.shape[0])

    rng = np.random.default_rng(424242)
    target = labels[0]
    label_b = labels[1]

    doc = {
        "fixture_schema": 1,
        "generator": "scripts/generate_generation_fixture.py",
        "seed": 424242,
        "d_latent": d,
        "field_dim": field_dim,
        "labels": labels,
        "atol": _ATOL,
        "cases": {
            "generate_gaussian_no_rejection": _case_gaussian_no_rejection(
                dif, target, n=4, temperature=1.2, rng=rng),
            "generate_gaussian_rejection": _case_gaussian_rejection(
                dif, target, n=3, temperature=2.0, max_candidates=8, rng=rng),
            "generate_conditioned": _case_conditioned(
                dif, target, n=3, temperature=1.0, rng=rng),
            "generate_langevin": _case_langevin(
                dif, target, n=2, n_steps=20, step_size=0.05, temperature=1.0, rng=rng),
            "generate_boundary": _case_boundary(
                dif, target, label_b, n=3, alpha=0.5, temperature=0.3, rng=rng),
            "generate_ood": _case_ood(
                dif, n=3, n_candidates=32, rng=rng),
            "generate_mdl_ball": _case_mdl_ball(
                dif, label_b, n=3, radius=1.5, rng=rng),
            "generate_ancestral": _case_ancestral(
                dif, n=5, temperature=1.0, rng=rng),
            "predict_next": _case_predict_next(dif, last_label=target),
            "rollout": _case_rollout(
                dif, seed_label=target, n_steps=6, temperature=0.8,
                exploration=0.15, rng=rng),
        },
    }

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}  (d={d}, field_dim={field_dim}, labels={labels})")


if __name__ == "__main__":
    main()
