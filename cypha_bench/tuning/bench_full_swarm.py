#!/usr/bin/env python3
"""
Full testbench-aligned hyperparameter swarm.

Scores hundreds of profile combinations on cached bench micro-tasks
(d01/d02/d03/d08-style workloads) and writes the best to everyday_profile.json.

Usage:
  python cypha_bench/tuning/bench_full_swarm.py --combos 400 --jobs 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_classification,
    make_regression,
)
from sklearn.model_selection import train_test_split

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_accel.cuda_util import cuda_gemm_usable, warmup_cuda  # noqa: E402
from cypha_accel.gpu_stress import cupy_gemm_burn  # noqa: E402
from cypha_bench.adapters.bench_models import BenchClassifier, BenchRegressor  # noqa: E402
from cypha_bench.common.metrics import evaluate_classification, evaluate_regression, standardize_train_test  # noqa: E402
from cypha_bench.domains.d08_computer_vision import _encode_images, _load_data  # noqa: E402

try:
    from joblib import Parallel, delayed  # noqa: WPS433
except ImportError:
    Parallel = None  # type: ignore
    delayed = None  # type: ignore


OUT_DIR = _REPO / "cypha_bench" / "artifacts" / "tuning"
PROFILE_PATH = _REPO / "cypha_bench" / "config" / "everyday_profile.json"


@dataclass(frozen=True)
class ParamRanges:
    world_lr: tuple[float, ...] = (0.006, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03)
    delta_lr: tuple[float, ...] = (0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14)
    enc_lr: tuple[float, ...] = (0.001, 0.002, 0.003)
    mdl_lambda: tuple[float, ...] = (0.0005, 0.001, 0.002, 0.004)
    temperature: tuple[float, ...] = (0.85, 0.95, 1.0, 1.05, 1.15, 1.2)
    field_dim: tuple[int, ...] = (64, 96, 128, 160)
    context_win: tuple[int, ...] = (12, 16, 24, 32, 48)
    target_lr: tuple[float, ...] = (0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.10)
    n_experts: tuple[int, ...] = (4, 6, 8, 10, 12)
    replay_ratio: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30)
    ood_sigma: tuple[float, ...] = (8.0, 10.0, 12.0, 15.0, 18.0, 22.0)


FLOOR_CONSTRAINTS: dict[str, float] = {
    "mnist_hog_acc": 0.84,
    "cls_digits_acc": 0.88,
    "reg_california_r2": 0.0,
    "cls_wine_acc": 0.90,
}

OBJECTIVES = ("minmax", "composite", "constrained")
REGIMES = ("vision", "tabular", "regression")

_CACHED: dict[str, Any] = {}


def _split_xy(x, y, seed=42):
    strat = y if len(np.unique(y)) > 1 else None
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=strat)
    return standardize_train_test(x_tr, x_te), y_tr, y_te


def _cache_datasets(mnist_train: int, mnist_test: int, cal_train: int) -> None:
    if _CACHED:
        return

    cls_tasks: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, loader in (
        ("iris", load_iris),
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
        ("digits", load_digits),
    ):
        x, y = loader(return_X_y=True)
        (x_tr, x_te), y_tr, y_te = _split_xy(np.asarray(x, dtype=np.float64), y)
        cls_tasks.append((name, (x_tr, y_tr, x_te, y_te)))

    x, y = make_classification(n_samples=1200, n_features=12, n_informative=6, random_state=42)
    (x_tr, x_te), y_tr, y_te = _split_xy(np.asarray(x, dtype=np.float64), y)
    cls_tasks.append(("linear_sep", (x_tr, y_tr, x_te, y_te)))

    x, y = make_blobs(n_samples=1200, n_features=8, centers=4, cluster_std=1.2, random_state=42)
    (x_tr, x_te), y_tr, y_te = _split_xy(np.asarray(x, dtype=np.float64), y)
    cls_tasks.append(("gaussian_blobs", (x_tr, y_tr, x_te, y_te)))

    reg_tasks: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for name, loader in (("diabetes", load_diabetes),):
        x, y = loader(return_X_y=True)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)
        x_tr, x_te = standardize_train_test(x_tr, x_te)
        reg_tasks.append((name, x_tr, y_tr, x_te, y_te))

    x, y = fetch_california_housing(return_X_y=True)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) > cal_train:
        idx = np.random.default_rng(42).choice(len(x), cal_train, replace=False)
        x, y = x[idx], y[idx]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)
    x_tr, x_te = standardize_train_test(x_tr, x_te)
    reg_tasks.append(("california", x_tr, y_tr, x_te, y_te))

    x, y = make_regression(n_samples=1000, n_features=16, noise=0.15, random_state=42)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)
    x_tr, x_te = standardize_train_test(x_tr, x_te)
    reg_tasks.append(("linear_reg", x_tr, y_tr, x_te, y_te))

    source, xt, yt, xe, ye = _load_data()
    if len(xt) > mnist_train:
        idx = np.random.default_rng(42).choice(len(xt), mnist_train, replace=False)
        xt, yt = xt[idx], yt[idx]
    if len(xe) > mnist_test:
        idx = np.random.default_rng(7).choice(len(xe), mnist_test, replace=False)
        xe, ye = xe[idx], ye[idx]
    hog_tr = _encode_images(xt, "hog")
    hog_te = _encode_images(xe, "hog")
    hog_tr, hog_te = standardize_train_test(hog_tr, hog_te)

    _CACHED["cls"] = cls_tasks
    _CACHED["reg"] = reg_tasks
    _CACHED["mnist_hog"] = (hog_tr, yt, hog_te, ye, source)


def _sample_profile(rng: random.Random, ranges: ParamRanges) -> dict[str, Any]:
    cls = {
        "world_lr": rng.choice(ranges.world_lr),
        "delta_lr": rng.choice(ranges.delta_lr),
        "enc_lr": rng.choice(ranges.enc_lr),
        "mdl_lambda": rng.choice(ranges.mdl_lambda),
        "temperature": rng.choice(ranges.temperature),
        "field_dim": rng.choice(ranges.field_dim),
        "context_win": rng.choice(ranges.context_win),
        "n_epochs": rng.choice([2, 3, 4]),
        "train_passes_cap": rng.choice([300, 400, 600, 800]),
    }
    reg = {
        "world_lr": rng.choice(ranges.world_lr),
        "delta_lr": rng.choice(ranges.delta_lr),
        "enc_lr": rng.choice(ranges.enc_lr),
        "mdl_lambda": rng.choice(ranges.mdl_lambda),
        "temperature": rng.choice(ranges.temperature),
        "field_dim": rng.choice(ranges.field_dim),
        "context_win": rng.choice(ranges.context_win),
        "target_lr": rng.choice(ranges.target_lr),
        "n_experts": rng.choice(ranges.n_experts),
        "n_epochs": rng.choice([1, 2, 3]),
        "train_passes_cap": rng.choice([500, 700, 900, 1200]),
    }
    arch = {
        "replay_ratio": rng.choice(ranges.replay_ratio),
        "ood_sigma": rng.choice(ranges.ood_sigma),
    }
    return {
        "classification_cyphadif": cls,
        "regression_difregressor": reg,
        "architecture": arch,
    }


def _near_choice(
    center_val: Any,
    choices: tuple[Any, ...],
    rng: random.Random,
) -> Any:
    if center_val in choices:
        idx = choices.index(center_val)
        lo = max(0, idx - 1)
        hi = min(len(choices) - 1, idx + 1)
        return rng.choice(choices[lo : hi + 1])
    return rng.choice(choices)


def _flat_center_for_regime(profile: dict[str, Any], regime: str) -> dict[str, Any]:
    """Normalize regime-style or flat profiles into sampling blocks."""
    if "regimes" not in profile:
        return profile

    regimes = profile["regimes"]
    arch = dict(profile.get("architecture", {}))
    tabular = dict(regimes.get("tabular") or profile.get("classification_cyphadif", {}))
    regression = dict(regimes.get("regression") or profile.get("regression_difregressor", {}))

    if regime == "vision":
        vision = dict(regimes.get("vision") or tabular)
        cls = {k: v for k, v in vision.items() if k not in ("replay_ratio", "ood_sigma")}
        for key in ("replay_ratio", "ood_sigma"):
            if key in vision:
                arch[key] = vision[key]
        return {
            "classification_cyphadif": cls,
            "regression_difregressor": regression,
            "architecture": arch,
        }

    return {
        "classification_cyphadif": tabular,
        "regression_difregressor": regression,
        "architecture": arch,
    }


def _sample_cls_block(
    ranges: ParamRanges,
    center_cls: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    return {
        "world_lr": _near_choice(center_cls.get("world_lr"), ranges.world_lr, rng),
        "delta_lr": _near_choice(center_cls.get("delta_lr"), ranges.delta_lr, rng),
        "enc_lr": _near_choice(center_cls.get("enc_lr"), ranges.enc_lr, rng),
        "mdl_lambda": _near_choice(center_cls.get("mdl_lambda"), ranges.mdl_lambda, rng),
        "temperature": _near_choice(center_cls.get("temperature"), ranges.temperature, rng),
        "field_dim": _near_choice(center_cls.get("field_dim"), ranges.field_dim, rng),
        "context_win": _near_choice(center_cls.get("context_win"), ranges.context_win, rng),
        "n_epochs": _near_choice(center_cls.get("n_epochs"), (2, 3, 4), rng),
        "train_passes_cap": _near_choice(center_cls.get("train_passes_cap"), (300, 400, 600, 800), rng),
    }


def _sample_reg_block(
    ranges: ParamRanges,
    center_reg: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    return {
        "world_lr": _near_choice(center_reg.get("world_lr"), ranges.world_lr, rng),
        "delta_lr": _near_choice(center_reg.get("delta_lr"), ranges.delta_lr, rng),
        "enc_lr": _near_choice(center_reg.get("enc_lr"), ranges.enc_lr, rng),
        "mdl_lambda": _near_choice(center_reg.get("mdl_lambda"), ranges.mdl_lambda, rng),
        "temperature": _near_choice(center_reg.get("temperature"), ranges.temperature, rng),
        "field_dim": _near_choice(center_reg.get("field_dim"), ranges.field_dim, rng),
        "context_win": _near_choice(center_reg.get("context_win"), ranges.context_win, rng),
        "target_lr": _near_choice(center_reg.get("target_lr"), ranges.target_lr, rng),
        "n_experts": _near_choice(center_reg.get("n_experts"), ranges.n_experts, rng),
        "n_epochs": _near_choice(center_reg.get("n_epochs"), (1, 2, 3), rng),
        "train_passes_cap": _near_choice(
            center_reg.get("train_passes_cap"),
            (500, 700, 900, 1200),
            rng,
        ),
    }


def _sample_arch_block(
    ranges: ParamRanges,
    center_arch: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    return {
        "replay_ratio": _near_choice(center_arch.get("replay_ratio"), ranges.replay_ratio, rng),
        "ood_sigma": _near_choice(center_arch.get("ood_sigma"), ranges.ood_sigma, rng),
    }


def sample_profile_narrow(
    ranges: ParamRanges,
    center_profile: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Sample a profile near center_profile for round-2 local search."""
    flat = _flat_center_for_regime(center_profile, "tabular")
    center_cls = flat.get("classification_cyphadif", {})
    center_reg = flat.get("regression_difregressor", {})
    center_arch = flat.get("architecture", {})
    return {
        "classification_cyphadif": _sample_cls_block(ranges, center_cls, rng),
        "regression_difregressor": _sample_reg_block(ranges, center_reg, rng),
        "architecture": _sample_arch_block(ranges, center_arch, rng),
    }


def sample_profile_narrow_regime(
    ranges: ParamRanges,
    center: dict[str, Any],
    regime: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Sample near center, perturbing only blocks relevant to regime."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime: {regime!r}; expected one of {REGIMES}")

    flat = _flat_center_for_regime(center, regime)
    center_cls = dict(flat.get("classification_cyphadif", {}))
    center_reg = dict(flat.get("regression_difregressor", {}))
    center_arch = dict(flat.get("architecture", {}))

    cls = center_cls
    reg = center_reg
    arch = center_arch

    if regime in ("vision", "tabular"):
        cls = _sample_cls_block(ranges, center_cls, rng)
    if regime == "vision":
        arch = _sample_arch_block(ranges, center_arch, rng)
    if regime == "regression":
        reg = _sample_reg_block(ranges, center_reg, rng)

    return {
        "classification_cyphadif": cls,
        "regression_difregressor": reg,
        "architecture": arch,
    }


def _passes_floors(scores: dict[str, float]) -> bool:
    return all(scores.get(key, -1e9) >= floor for key, floor in FLOOR_CONSTRAINTS.items())


def _regime_score(regime: str, scores: dict[str, float]) -> float:
    if regime == "vision":
        mnist = float(scores.get("mnist_hog_acc", 0.0))
        cls_keys = [k for k in scores if k.startswith("cls_") and k.endswith("_acc")]
        other_mean = sum(float(scores[k]) for k in cls_keys) / max(len(cls_keys), 1)
        return 0.75 * mnist + 0.25 * other_mean
    if regime == "tabular":
        cls_keys = [k for k in scores if k.startswith("cls_") and k.endswith("_acc")]
        if not cls_keys:
            return -1e9
        return sum(float(scores[k]) for k in cls_keys) / len(cls_keys)
    if regime == "regression":
        r2 = float(scores.get("reg_california_r2", -1.0))
        rmse = float(scores.get("reg_california_rmse", 999.0))
        return 0.5 * r2 + 0.5 * (1.0 / (1.0 + rmse))
    raise ValueError(f"unknown regime: {regime!r}; expected one of {REGIMES}")


def _regime_passes_floors(regime: str, scores: dict[str, float]) -> bool:
    if regime == "vision":
        return scores.get("mnist_hog_acc", -1e9) >= FLOOR_CONSTRAINTS["mnist_hog_acc"]
    return True


def _apply_regime_objective(rows: list[dict[str, Any]], regime: str) -> None:
    for row in rows:
        if row.get("error"):
            row["composite"] = -1e9
            continue
        scores = row.get("scores", {})
        if not _regime_passes_floors(regime, scores):
            row["composite"] = -1e9
        else:
            row["composite"] = _regime_score(regime, scores)


def _train_classifier_online(
    model, x_tr, y_tr, profile: dict[str, Any], input_dim: int
) -> None:
    from cypha_bench.config.load_profile import (
        classification_params,
        select_classification_regime,
        uses_regimes,
    )

    regime = select_classification_regime(input_dim)
    if uses_regimes(profile) or profile.get("regimes"):
        p = classification_params(profile, regime=regime)
    else:
        p = classification_params(profile)
    passes = max(1, int(p.get("n_epochs", 1)))
    for _ in range(passes):
        for xi, yi in zip(x_tr, y_tr):
            model.train_step(xi, str(int(yi)))


def _train_regressor_online(model, x_tr, y_tr, profile: dict[str, Any]) -> None:
    from cypha_bench.config.load_profile import load_profile, regression_params, uses_regimes

    prof = profile if uses_regimes(profile) or "regression_difregressor" in profile else load_profile()
    p = regression_params(prof)
    passes = max(1, int(p.get("n_epochs", 1)))
    for _ in range(passes):
        for xi, yi in zip(x_tr, y_tr):
            model.train_step(xi, float(yi))


def _eval_profile(profile: dict[str, Any], seed: int = 42) -> dict[str, float]:
    scores: dict[str, float] = {}
    t0 = time.perf_counter()

    for name, (x_tr, y_tr, x_te, y_te) in _CACHED["cls"]:
        model = BenchClassifier(x_tr.shape[1], seed=seed, profile=profile, use_profile=True)
        _train_classifier_online(model, x_tr, y_tr, profile, x_tr.shape[1])
        acc = float(evaluate_classification(model, x_te, y_te, label_fn=str).get("accuracy", 0.0))
        scores[f"cls_{name}_acc"] = acc

    for name, x_tr, y_tr, x_te, y_te in _CACHED["reg"]:
        model = BenchRegressor(x_tr.shape[1], seed=seed, profile=profile, use_profile=True)
        _train_regressor_online(model, x_tr, y_tr, profile)
        reg_scores = evaluate_regression(model, x_te, y_te)
        scores[f"reg_{name}_rmse"] = float(reg_scores.get("rmse", 999.0))
        scores[f"reg_{name}_r2"] = float(reg_scores.get("r2", -1.0))

    hog_tr, y_tr, hog_te, y_te, _ = _CACHED["mnist_hog"]
    model = BenchClassifier(hog_tr.shape[1], seed=seed, profile=profile, use_profile=True)
    _train_classifier_online(model, hog_tr, y_tr, profile, hog_tr.shape[1])
    scores["mnist_hog_acc"] = float(
        evaluate_classification(model, hog_te, y_te, label_fn=str).get("accuracy", 0.0)
    )

    scores["wall_s"] = time.perf_counter() - t0
    return scores


def _composite(scores: dict[str, float]) -> float:
    """Higher is better — bench-weighted composite."""
    acc_keys = [k for k in scores if k.endswith("_acc")]
    rmse_keys = [k for k in scores if k.endswith("_rmse")]

    acc_part = sum(scores[k] for k in acc_keys) / max(len(acc_keys), 1)
    rmse_vals = [scores[k] for k in rmse_keys]
    rmse_part = sum(1.0 / (1.0 + r) for r in rmse_vals) / max(len(rmse_vals), 1)

    r2_bonus = 0.0
    for k in scores:
        if k.endswith("_r2"):
            r2_bonus += max(scores[k], -0.5) * 0.05

    speed_pen = min(scores.get("wall_s", 0.0) / 120.0, 0.08)
    return acc_part * 0.62 + rmse_part * 0.28 + r2_bonus - speed_pen


def _minmax_aggregate(scores: dict[str, float], bounds: dict[str, tuple[float, float]]) -> float:
    acc_keys = [k for k in scores if k.endswith("_acc")]
    rmse_keys = [k for k in scores if k.endswith("_rmse")]

    def _norm(key: str, value: float, *, invert: bool = False) -> float:
        lo, hi = bounds[key]
        if hi - lo < 1e-12:
            n = 1.0
        else:
            n = (value - lo) / (hi - lo)
        return 1.0 - n if invert else n

    acc_part = sum(_norm(k, scores[k]) for k in acc_keys) / max(len(acc_keys), 1)
    rmse_part = sum(_norm(k, scores[k], invert=True) for k in rmse_keys) / max(len(rmse_keys), 1)

    r2_bonus = 0.0
    for key in scores:
        if key.endswith("_r2") and key in bounds:
            r2_bonus += _norm(key, max(scores[key], -0.5)) * 0.05

    if "wall_s" in bounds:
        speed_pen = _norm("wall_s", scores.get("wall_s", 0.0), invert=True) * 0.08
    else:
        speed_pen = min(scores.get("wall_s", 0.0) / 120.0, 0.08)
    return acc_part * 0.62 + rmse_part * 0.28 + r2_bonus - speed_pen


def _metric_bounds(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.get("scores", {}).keys())

    bounds: dict[str, tuple[float, float]] = {}
    for key in keys:
        vals = [row["scores"][key] for row in rows if key in row.get("scores", {})]
        if vals:
            bounds[key] = (min(vals), max(vals))
    return bounds


def _apply_objective(rows: list[dict[str, Any]], objective: str) -> None:
    for row in rows:
        if row.get("error"):
            row["composite"] = -1e9
            continue
        scores = row.get("scores", {})
        passes = _passes_floors(scores)
        if objective == "composite":
            row["composite"] = _composite(scores)
        elif objective == "constrained":
            row["composite"] = _composite(scores) if passes else -1e9
        elif not passes:
            row["composite"] = -1e9

    if objective != "minmax":
        return

    eligible = [row for row in rows if row["composite"] != -1e9]
    if not eligible:
        return

    bounds = _metric_bounds(eligible)
    for row in eligible:
        row["composite"] = _minmax_aggregate(row["scores"], bounds)


def _init_worker(mnist_train: int, mnist_test: int, cal_train: int) -> None:
    _cache_datasets(mnist_train, mnist_test, cal_train)


def _eval_one(combo_id: int, profile: dict[str, Any], seed: int) -> dict[str, Any]:
    try:
        raw = _eval_profile(profile, seed=seed + combo_id)
        return {
            "combo_id": combo_id,
            "composite": 0.0,
            "scores": raw,
            "profile": profile,
            "error": None,
        }
    except Exception as exc:
        return {
            "combo_id": combo_id,
            "composite": -1e9,
            "scores": {},
            "profile": profile,
            "error": str(exc),
        }


def _run_swarm(
    profiles: list[dict[str, Any]],
    *,
    combos: int,
    jobs: int,
    seed: int,
    gpu_burn: int,
    mnist_train: int,
    mnist_test: int,
    cal_train: int,
    objective: str,
    top_k: int,
    source: str,
    write_profile: bool = True,
    stamp_prefix: str = "bench_swarm",
    regime: str | None = None,
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warmup_cuda()
    gpu_ok = cuda_gemm_usable()
    if gpu_ok and gpu_burn > 0:
        cupy_gemm_burn(gpu_burn, n=2048, d=2048, k=2048, seed=seed)

    _cache_datasets(mnist_train, mnist_test, cal_train)
    n_cls = len(_CACHED["cls"])
    n_reg = len(_CACHED["reg"])
    print(
        f"Bench swarm ({stamp_prefix}): {combos} combos, jobs={jobs}, cuda={gpu_ok}, "
        f"objective={regime or objective}, tasks={n_cls} cls + {n_reg} reg + mnist_hog",
        flush=True,
    )

    t0 = time.perf_counter()
    if Parallel is not None and jobs > 1:
        rows = Parallel(
            n_jobs=jobs,
            backend="loky",
            verbose=10,
            initializer=_init_worker,
            initargs=(mnist_train, mnist_test, cal_train),
        )(delayed(_eval_one)(i, profiles[i], seed) for i in range(combos))
    else:
        rows = [_eval_one(i, profiles[i], seed) for i in range(combos)]

    if regime is not None:
        _apply_regime_objective(rows, regime)
    else:
        _apply_objective(rows, objective)
    rows.sort(key=lambda r: r["composite"], reverse=True)
    elapsed = time.perf_counter() - t0
    floor_fn = (lambda s: _regime_passes_floors(regime, s)) if regime else _passes_floors
    eligible = [
        r for r in rows
        if not r.get("error") and floor_fn(r.get("scores", {}))
    ]
    eligible.sort(key=lambda r: r["composite"], reverse=True)
    if regime == "vision" and not eligible:
        eligible = sorted(
            [r for r in rows if not r.get("error")],
            key=lambda r: float((r.get("scores") or {}).get("mnist_hog_acc", 0.0)),
            reverse=True,
        )[:top_k]
    best = eligible[0] if eligible else rows[0]
    errors = sum(1 for r in rows if r.get("error"))
    floor_failures = sum(
        1 for r in rows if not r.get("error") and not floor_fn(r.get("scores", {}))
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    leaderboard_path = OUT_DIR / f"{stamp_prefix}_{stamp}.json"
    payload = {
        "combos": combos,
        "jobs": jobs,
        "objective": regime or objective,
        "regime": regime,
        "top_k": top_k,
        "source": source,
        "floor_constraints": FLOOR_CONSTRAINTS,
        "gpu_cuda_usable": gpu_ok,
        "elapsed_s": round(elapsed, 2),
        "errors": errors,
        "floor_failures": floor_failures,
        "top10": eligible[:10],
        "top_k_candidates": eligible[:top_k],
        "all_combos": eligible[:50],
    }
    leaderboard_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    profile_path = PROFILE_PATH
    if write_profile:
        profile_out = {
            "source": source,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_cuda_usable": gpu_ok,
            "sweep": {
                "combos": combos,
                "jobs": jobs,
                "objective": regime or objective,
                "regime": regime,
                "elapsed_s": round(elapsed, 2),
                "composite_best": best["composite"],
                "leaderboard_path": str(leaderboard_path),
            },
            **{k: v for k, v in best["profile"].items()},
            "validation_metrics": best["scores"],
        }
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(json.dumps(profile_out, indent=2), encoding="utf-8")

    print(f"Done in {elapsed:.1f}s — best composite={best['composite']:.4f}", flush=True)
    print(f"Leaderboard: {leaderboard_path}", flush=True)
    if write_profile:
        print(f"Profile: {PROFILE_PATH}", flush=True)
    if errors:
        print(f"Warning: {errors} combos failed", flush=True)
    if floor_failures:
        print(f"Warning: {floor_failures} combos failed floor constraints", flush=True)

    return {
        "profile_path": profile_path,
        "leaderboard_path": leaderboard_path,
        "best": best,
        "eligible": eligible,
        "gpu_cuda_usable": gpu_ok,
        "elapsed_s": round(elapsed, 2),
    }


def run_full_swarm(
    combos: int = 400,
    jobs: int = 6,
    seed: int = 42,
    gpu_burn: int = 16,
    mnist_train: int = 800,
    mnist_test: int = 300,
    cal_train: int = 3000,
    objective: str = "minmax",
    top_k: int = 30,
    write_profile: bool = True,
) -> dict[str, Any]:
    ranges = ParamRanges()
    rng = random.Random(seed)
    profiles = [_sample_profile(rng, ranges) for _ in range(combos)]
    return _run_swarm(
        profiles,
        combos=combos,
        jobs=jobs,
        seed=seed,
        gpu_burn=gpu_burn,
        mnist_train=mnist_train,
        mnist_test=mnist_test,
        cal_train=cal_train,
        objective=objective,
        top_k=top_k,
        source=f"cypha_bench/tuning/bench_full_swarm.py combos={combos}",
        write_profile=write_profile,
        stamp_prefix="bench_swarm",
    )


def run_narrow_swarm(
    center_profile: dict[str, Any],
    combos: int = 300,
    jobs: int = 6,
    seed: int = 42,
    gpu_burn: int = 16,
    mnist_train: int = 800,
    mnist_test: int = 300,
    cal_train: int = 3000,
    objective: str = "minmax",
    top_k: int = 30,
    write_profile: bool = False,
) -> dict[str, Any]:
    ranges = ParamRanges()
    rng = random.Random(seed + 17)
    profiles = [sample_profile_narrow(ranges, center_profile, rng) for _ in range(combos)]
    return _run_swarm(
        profiles,
        combos=combos,
        jobs=jobs,
        seed=seed,
        gpu_burn=gpu_burn,
        mnist_train=mnist_train,
        mnist_test=mnist_test,
        cal_train=cal_train,
        objective=objective,
        top_k=top_k,
        source=f"cypha_bench/tuning/bench_iterate_swarm.py narrow combos={combos}",
        write_profile=write_profile,
        stamp_prefix="bench_swarm_narrow",
    )


def run_regime_swarm(
    regime: str,
    center_profile: dict[str, Any],
    combos: int = 300,
    jobs: int = 6,
    seed: int = 42,
    gpu_burn: int = 16,
    mnist_train: int = 800,
    mnist_test: int = 300,
    cal_train: int = 3000,
    top_k: int = 30,
    write_profile: bool = False,
) -> dict[str, Any]:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime: {regime!r}; expected one of {REGIMES}")

    ranges = ParamRanges()
    rng = random.Random(seed + 31)
    profiles = [
        sample_profile_narrow_regime(ranges, center_profile, regime, rng)
        for _ in range(combos)
    ]
    return _run_swarm(
        profiles,
        combos=combos,
        jobs=jobs,
        seed=seed,
        gpu_burn=gpu_burn,
        mnist_train=mnist_train,
        mnist_test=mnist_test,
        cal_train=cal_train,
        objective="regime",
        top_k=top_k,
        source=f"cypha_bench/tuning/bench_full_swarm.py regime={regime} combos={combos}",
        write_profile=write_profile,
        stamp_prefix=f"bench_swarm_{regime}",
        regime=regime,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full bench-aligned parameter swarm")
    parser.add_argument("--combos", type=int, default=int(os.environ.get("BENCH_SWARM_COMBOS", "400")))
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("BENCH_SWARM_JOBS", max(1, (os.cpu_count() or 4) - 1))))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-burn", type=int, default=16)
    parser.add_argument("--mnist-train", type=int, default=800)
    parser.add_argument("--mnist-test", type=int, default=300)
    parser.add_argument("--objective", choices=OBJECTIVES, default="minmax")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--regime", choices=REGIMES, default=None)
    parser.add_argument("--center", type=str, default=None, help="Center profile JSON for --regime")
    args = parser.parse_args()

    if args.regime:
        if not args.center:
            parser.error("--center is required when --regime is set")
        center_path = Path(args.center)
        center_profile = json.loads(center_path.read_text(encoding="utf-8"))
        run_regime_swarm(
            regime=args.regime,
            center_profile=center_profile,
            combos=args.combos,
            jobs=args.jobs,
            seed=args.seed,
            gpu_burn=args.gpu_burn,
            mnist_train=args.mnist_train,
            mnist_test=args.mnist_test,
            top_k=args.top_k,
        )
        return

    run_full_swarm(
        combos=args.combos,
        jobs=args.jobs,
        seed=args.seed,
        gpu_burn=args.gpu_burn,
        mnist_train=args.mnist_train,
        mnist_test=args.mnist_test,
        objective=args.objective,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
