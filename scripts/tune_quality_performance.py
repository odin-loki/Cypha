#!/usr/bin/env python3
"""
Brute-force (grid) hyperparameter search for **quality** + record **performance**,
with GPU status and optional **cProfile** of the best configs.

Uses the same real data as `download_profile_e2e.py` (OpenML 1464 + California housing).
Regression metrics (R², MAE, RMSE) use **raw target units** (only X is standardized), matching that script.
Classification **val_accuracy** uses the same accuracy rule as `download_profile_e2e` (batch_infer vs ``str(int(y))``),
but default **train_frac** here is 0.82 vs 0.85/0.75 there, and this script evaluates on the **full** val split
(e2e may cap test rows). Generation **gen_match_rate** is defined in this script only (e2e does not report it);
see ``note_generation`` in the summary JSON.
CuPy/GPU is used automatically when available (`cypha_accel`); this script reports
`cuda_gemm_usable` and optional CUDA sync timing.

Examples
--------
  python scripts/tune_quality_performance.py --preset coarse
  python scripts/tune_quality_performance.py --preset coarse --include-generation
  python scripts/tune_quality_performance.py --task gen --preset coarse
  python scripts/tune_quality_performance.py --preset medium --max-combos 120 --jobs 4
  python scripts/tune_quality_performance.py --preset fine --task cls --profile-top 5

Outputs (under --out-dir, default ./artifacts/tuning):
  - tuning_<timestamp>_results.csv   (every grid point + metrics + wall times)
  - tuning_<timestamp>_summary.json (grid spec, bests, GPU info)
  - tuning_<timestamp>_profile.txt  (cProfile cumtime for top cls/reg/gen, if --profile-top > 0)

Tasks
-----
  both (default) : classification + regression grids; add --include-generation for generate() grid
  cls / reg      : single task
  gen            : only generation grid (uses default cls hyperparameters for the preset)

Presets shrink the grid:
  coarse  : small grid, quick sanity (~10–40 combos per task)
  medium  : broader poll (~100–400 combos; use --max-combos to cap)
  fine    : larger brute-force (use --max-combos strongly recommended)

GPU load (CuPy + CUDA)
----------------------
  --gpu-burn-passes N   : raw device fp64 GEMMs before the grid (default 48; 0=off)
  Per tuning cell (cls/reg): --gpu-batch-n × --gpu-stress-repeats loops of
  score_matrix + world_gate + batch_infer (cls) or chunked predict_batch (reg).
  Use `make tune-gpu-heavy` or crank --gpu-batch-n / --gpu-stress-repeats on a CUDA box.
  Without CUDA, burn and stress no-op quickly (wall_gpu_stress_s ≈ 0).

  --gen-samples N : sample N random (temperature, max_candidates, n_calls, n_per_call) settings;
  trains one classifier (fixed seed) then evaluates generation only — for large N (e.g. 1000+).
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import random
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from Cypha import CyphaDIF, DIFRegressor, TieredContextBuffer, VectorEncoder  # noqa: E402
from cypha_accel.gpu_stress import cupy_gemm_burn  # noqa: E402

try:
    from joblib import Parallel, delayed  # noqa: WPS433
except ImportError:
    Parallel = None  # type: ignore
    delayed = None  # type: ignore

_DATA_CACHE = _ROOT / "data_cache"


def _load_classification_xy() -> Tuple[np.ndarray, np.ndarray, str]:
    from sklearn.datasets import fetch_openml, load_iris
    from sklearn.preprocessing import LabelEncoder

    _DATA_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        bunch = fetch_openml(
            data_id=1464,
            as_frame=False,
            parser="liac-arff",
            data_home=str(_DATA_CACHE),
        )
        X = np.asarray(bunch.data, dtype=np.float64)
        y_enc = LabelEncoder().fit_transform(np.asarray(bunch.target).ravel())
        return X, y_enc.astype(np.int64), "openml_1464_blood_transfusion"
    except Exception:
        iris = __import__("sklearn.datasets", fromlist=["load_iris"]).load_iris()
        return iris.data.astype(np.float64), iris.target.astype(np.int64), "sklearn_iris"


def _load_regression_xy() -> Tuple[np.ndarray, np.ndarray, str]:
    from sklearn.datasets import fetch_california_housing

    _DATA_CACHE.mkdir(parents=True, exist_ok=True)
    bunch = fetch_california_housing(data_home=str(_DATA_CACHE), download_if_missing=True)
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.float64).ravel()
    return X, y, "california_housing"


def _split(
    X: np.ndarray, y: np.ndarray, train_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_tr = max(1, int(n * train_frac))
    tr, te = idx[:n_tr], idx[n_tr:]
    return X[tr], y[tr], X[te], y[te]


def _acc(clf: CyphaDIF, X: np.ndarray, y: np.ndarray) -> float:
    if len(X) == 0:
        return float("nan")
    out = clf.batch_infer([X[i] for i in range(len(X))])
    pred = np.array([out[i][0] for i in range(len(out))])
    y_str = np.array([str(int(v)) for v in y])
    return float(np.mean(pred == y_str))


def _reg_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return mae, rmse, r2


def _field_dim(d_feat: int, spec: int) -> int:
    return max(int(spec), d_feat)


@dataclass
class GPUStressConfig:
    """Large-batch Cypha paths that hit fused_score_llr / project_features on GPU."""

    enabled: bool = True
    batch_n: int = 8192
    repeats: int = 16
    reg_chunk: int = 512


GPU_STRESS_OFF = GPUStressConfig(enabled=False)


def _cypha_gpu_stress_clf(
    clf: CyphaDIF, X_tr: np.ndarray, cfg: GPUStressConfig, seed: int
) -> float:
    if not cfg.enabled or len(X_tr) == 0:
        return 0.0
    from cypha_accel.cuda_util import cuda_gemm_usable

    if not cuda_gemm_usable():
        return 0.0

    base_n = min(512, len(X_tr))
    H0 = clf.batch_encode([X_tr[i] for i in range(base_n)])
    rep = int(np.ceil(cfg.batch_n / base_n))
    H_big = np.ascontiguousarray(np.tile(H0, (rep, 1))[: cfg.batch_n], dtype=np.float64)
    t0 = time.perf_counter()
    for _ in range(cfg.repeats):
        clf.score_matrix(H_big, use_field=True)
        clf.world_gate_vector(H_big, use_field=True)
        clf.batch_infer(H_big)
    return time.perf_counter() - t0


def _cypha_gpu_stress_reg(
    reg: DIFRegressor, X_tr: np.ndarray, cfg: GPUStressConfig, seed: int
) -> float:
    if not cfg.enabled or len(X_tr) == 0:
        return 0.0
    from cypha_accel.cuda_util import cuda_gemm_usable

    if not cuda_gemm_usable():
        return 0.0

    n0 = min(256, len(X_tr))
    X0 = X_tr[:n0]
    rep = int(np.ceil(cfg.batch_n / n0))
    X_big = np.ascontiguousarray(np.tile(X0, (rep, 1))[: cfg.batch_n], dtype=np.float64)
    t0 = time.perf_counter()
    ch = max(32, cfg.reg_chunk)
    for _ in range(cfg.repeats):
        for i in range(0, len(X_big), ch):
            blk = [X_big[j].copy() for j in range(i, min(i + ch, len(X_big)))]
            reg.predict_batch(blk)
    return time.perf_counter() - t0


@dataclass
class ClsParams:
    world_lr: float
    delta_lr: float
    enc_lr: float
    mdl_lambda: float
    temperature: float
    field_dim: int
    context_win: int
    n_epochs: int
    train_passes_cap: int


@dataclass
class RegParams:
    world_lr: float
    delta_lr: float
    enc_lr: float
    mdl_lambda: float
    temperature: float
    field_dim: int
    context_win: int
    target_lr: float
    n_experts: int
    n_epochs: int
    train_passes_cap: int


def preset_grids(name: str) -> Tuple[List[ClsParams], List[RegParams]]:
    """Return full Cartesian grids for classification and regression."""
    if name == "coarse":
        cls_world = (0.01, 0.02)
        cls_delta = (0.06, 0.12)
        cls_enc = (0.002,)
        cls_mdl = (0.002,)
        cls_temp = (1.0,)
        cls_fd = (64, 96)
        cls_ctx = (24,)
        cls_ep = (3,)
        cls_cap = (350,)
        reg_world = (0.01, 0.02)
        reg_delta = (0.06, 0.10)
        reg_enc = (0.002,)
        reg_mdl = (0.002,)
        reg_temp = (1.0,)
        reg_fd = (96, 128)
        reg_ctx = (24,)
        reg_tgt = (0.05, 0.08)
        reg_ne = (6,)
        reg_ep = (1,)
        reg_cap = (500,)
    elif name == "medium":
        cls_world = (0.008, 0.015, 0.025)
        cls_delta = (0.05, 0.08, 0.12)
        cls_enc = (0.001, 0.002)
        cls_mdl = (0.001, 0.002)
        cls_temp = (0.9, 1.0, 1.15)
        cls_fd = (64, 96, 128)
        cls_ctx = (16, 32)
        cls_ep = (3, 4)
        cls_cap = (400, 600)
        reg_world = (0.01, 0.02)
        reg_delta = (0.05, 0.08, 0.12)
        reg_enc = (0.001, 0.002)
        reg_mdl = (0.001, 0.002)
        reg_temp = (0.95, 1.05)
        reg_fd = (96, 128, 160)
        reg_ctx = (16, 32)
        reg_tgt = (0.04, 0.06, 0.09)
        reg_ne = (4, 6, 8)
        reg_ep = (1, 2)
        reg_cap = (600, 900)
    else:  # fine
        cls_world = (0.006, 0.01, 0.015, 0.022, 0.03)
        cls_delta = (0.04, 0.07, 0.10, 0.14)
        cls_enc = (0.001, 0.002, 0.003)
        cls_mdl = (0.001, 0.002, 0.004)
        cls_temp = (0.85, 1.0, 1.1, 1.2)
        cls_fd = (64, 96, 128, 160)
        cls_ctx = (12, 24, 48)
        cls_ep = (3, 5)
        cls_cap = (400, 700, 1000)
        reg_world = (0.008, 0.012, 0.02, 0.03)
        reg_delta = (0.04, 0.07, 0.10, 0.14)
        reg_enc = (0.001, 0.002)
        reg_mdl = (0.001, 0.002, 0.003)
        reg_temp = (0.9, 1.0, 1.1)
        reg_fd = (96, 128, 160)
        reg_ctx = (16, 32, 48)
        reg_tgt = (0.03, 0.05, 0.07, 0.10)
        reg_ne = (4, 6, 8, 10)
        reg_ep = (1, 2, 3)
        reg_cap = (500, 800, 1200)

    cls_list: List[ClsParams] = []
    for tup in product(
        cls_world,
        cls_delta,
        cls_enc,
        cls_mdl,
        cls_temp,
        cls_fd,
        cls_ctx,
        cls_ep,
        cls_cap,
    ):
        cls_list.append(
            ClsParams(
                world_lr=tup[0],
                delta_lr=tup[1],
                enc_lr=tup[2],
                mdl_lambda=tup[3],
                temperature=tup[4],
                field_dim=tup[5],
                context_win=tup[6],
                n_epochs=tup[7],
                train_passes_cap=tup[8],
            )
        )

    reg_list: List[RegParams] = []
    for tup in product(
        reg_world,
        reg_delta,
        reg_enc,
        reg_mdl,
        reg_temp,
        reg_fd,
        reg_ctx,
        reg_tgt,
        reg_ne,
        reg_ep,
        reg_cap,
    ):
        reg_list.append(
            RegParams(
                world_lr=tup[0],
                delta_lr=tup[1],
                enc_lr=tup[2],
                mdl_lambda=tup[3],
                temperature=tup[4],
                field_dim=tup[5],
                context_win=tup[6],
                target_lr=tup[7],
                n_experts=tup[8],
                n_epochs=tup[9],
                train_passes_cap=tup[10],
            )
        )

    return cls_list, reg_list


def _subsample(rows: List[Any], max_combos: int, seed: int) -> List[Any]:
    if max_combos <= 0 or len(rows) <= max_combos:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_combos)


def _cls_dict(p: ClsParams) -> Dict[str, Any]:
    return {
        "world_lr": p.world_lr,
        "delta_lr": p.delta_lr,
        "enc_lr": p.enc_lr,
        "mdl_lambda": p.mdl_lambda,
        "temperature": p.temperature,
        "field_dim": p.field_dim,
        "context_win": p.context_win,
        "n_epochs": p.n_epochs,
        "train_passes_cap": p.train_passes_cap,
    }


def _reg_dict(p: RegParams) -> Dict[str, Any]:
    d = _cls_dict(
        ClsParams(
            p.world_lr,
            p.delta_lr,
            p.enc_lr,
            p.mdl_lambda,
            p.temperature,
            p.field_dim,
            p.context_win,
            p.n_epochs,
            p.train_passes_cap,
        )
    )
    d["target_lr"] = p.target_lr
    d["n_experts"] = p.n_experts
    return d


def train_eval_cls(
    p: ClsParams,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    d: int,
    seed: int,
    gpu_stress: GPUStressConfig,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    fd = _field_dim(d, p.field_dim)
    clf = CyphaDIF(
        encoder=VectorEncoder(d),
        field_dim=fd,
        enc_lr=p.enc_lr,
        delta_lr=p.delta_lr,
        world_lr=p.world_lr,
        mdl_lambda=p.mdl_lambda,
        context_win=p.context_win,
        rng=np.random.default_rng(seed + 11),
    )
    clf.temperature = float(p.temperature)

    passes = min(len(X_tr), p.train_passes_cap)
    for ep in range(p.n_epochs):
        order = rng.permutation(len(X_tr))
        for i in order[:passes]:
            clf.train_step(X_tr[i], str(int(y_tr[i])))

    t_train = time.perf_counter() - t0
    t1 = time.perf_counter()
    tr_acc = _acc(clf, X_tr[: min(200, len(X_tr))], y_tr[: min(200, len(y_tr))])
    va_acc = _acc(clf, X_va, y_va)
    xs = [X_tr[i] for i in range(min(64, len(X_tr)))]
    clf.batch_infer(xs)
    H = clf.batch_encode(xs)
    clf.score_matrix(H, use_field=True)
    t_infer = time.perf_counter() - t1

    t_gpu = _cypha_gpu_stress_clf(clf, X_tr, gpu_stress, seed)

    row = {**_cls_dict(p), "task": "classification"}
    row["train_acc_sample"] = tr_acc
    row["val_accuracy"] = va_acc
    row["wall_train_s"] = round(t_train, 6)
    row["wall_infer_stress_s"] = round(t_infer, 6)
    row["wall_gpu_stress_s"] = round(t_gpu, 6)
    row["wall_total_s"] = round(t_train + t_infer + t_gpu, 6)
    return row


def train_eval_reg(
    p: RegParams,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    d: int,
    seed: int,
    gpu_stress: GPUStressConfig,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    fd = _field_dim(d, p.field_dim)
    reg = DIFRegressor(
        encoder=VectorEncoder(d),
        field_dim=fd,
        n_experts=p.n_experts,
        target_lr=p.target_lr,
        rng=np.random.default_rng(seed + 13),
    )
    reg.clf.enc_lr = p.enc_lr
    reg.clf.delta_lr = p.delta_lr
    reg.clf.world_lr = p.world_lr
    reg.clf.mdl_lambda = p.mdl_lambda
    reg.clf.context = TieredContextBuffer(short_window=p.context_win)
    reg.clf.temperature = float(p.temperature)

    passes = min(len(X_tr), p.train_passes_cap)
    for ep in range(p.n_epochs):
        order = rng.permutation(len(X_tr))
        for i in order[:passes]:
            reg.train_step(X_tr[i], float(y_tr[i]))

    t_train = time.perf_counter() - t0
    t1 = time.perf_counter()
    yp_va, _ = reg.predict_batch([X_va[i] for i in range(len(X_va))])
    yp_va = yp_va.ravel()
    mae, rmse, r2 = _reg_metrics(y_va, yp_va)
    xs = [X_tr[i] for i in range(min(64, len(X_tr)))]
    reg.predict_batch(xs)
    t_infer = time.perf_counter() - t1

    t_gpu = _cypha_gpu_stress_reg(reg, X_tr, gpu_stress, seed)

    row = {**_reg_dict(p), "task": "regression"}
    row["val_mae"] = mae
    row["val_rmse"] = rmse
    row["val_r2"] = r2
    row["wall_train_s"] = round(t_train, 6)
    row["wall_infer_stress_s"] = round(t_infer, 6)
    row["wall_gpu_stress_s"] = round(t_gpu, 6)
    row["wall_total_s"] = round(t_train + t_infer + t_gpu, 6)
    return row


def _gpu_info() -> Dict[str, Any]:
    from cypha_accel.cuda_util import cuda_gemm_usable

    info: Dict[str, Any] = {"cuda_gemm_usable": bool(cuda_gemm_usable())}
    if not info["cuda_gemm_usable"]:
        return info
    try:
        import cupy as cp  # type: ignore

        t0 = time.perf_counter()
        a = cp.random.standard_normal((1024, 256), dtype=cp.float64)
        b = cp.random.standard_normal((256, 128), dtype=cp.float64)
        c = a @ b
        cp.cuda.Stream.null.synchronize()
        info["gpu_warmup_gemm_ms"] = round((time.perf_counter() - t0) * 1000, 4)
        del a, b, c
    except Exception as e:
        info["gpu_warmup_error"] = str(e)
    return info


def _profile_best_cls(
    p: ClsParams, X_tr: np.ndarray, y_tr: np.ndarray, d: int, seed: int
) -> str:
    pr = cProfile.Profile()

    def work():
        train_eval_cls(
            p, X_tr, y_tr, X_tr[:1], y_tr[:1], d, seed, GPU_STRESS_OFF
        )

    pr.enable()
    work()
    pr.disable()
    buf = io.StringIO()
    import pstats

    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumtime").print_stats(40)
    return buf.getvalue()


def _profile_best_reg(
    p: RegParams, X_tr: np.ndarray, y_tr: np.ndarray, d: int, seed: int
) -> str:
    pr = cProfile.Profile()

    def work():
        train_eval_reg(
            p, X_tr, y_tr, X_tr[:1], y_tr[:1], d, seed, GPU_STRESS_OFF
        )

    pr.enable()
    work()
    pr.disable()
    buf = io.StringIO()
    import pstats

    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumtime").print_stats(40)
    return buf.getvalue()


def _csv_header(keys: Sequence[str]) -> str:
    return ",".join(keys) + "\n"


def _csv_row(row: Dict[str, Any], keys: Sequence[str]) -> str:
    parts = []
    for k in keys:
        v = row.get(k, "")
        if isinstance(v, float):
            parts.append(f"{v:.8g}" if np.isfinite(v) else "")
        else:
            parts.append(str(v).replace(",", ";"))
    return ",".join(parts) + "\n"


def _train_cls_model(
    p: ClsParams, X_tr: np.ndarray, y_tr: np.ndarray, d: int, seed: int
) -> CyphaDIF:
    rng = np.random.default_rng(seed)
    fd = _field_dim(d, p.field_dim)
    clf = CyphaDIF(
        encoder=VectorEncoder(d),
        field_dim=fd,
        enc_lr=p.enc_lr,
        delta_lr=p.delta_lr,
        world_lr=p.world_lr,
        mdl_lambda=p.mdl_lambda,
        context_win=p.context_win,
        rng=np.random.default_rng(seed + 11),
    )
    clf.temperature = float(p.temperature)
    passes = min(len(X_tr), p.train_passes_cap)
    for ep in range(p.n_epochs):
        order = rng.permutation(len(X_tr))
        for i in order[:passes]:
            clf.train_step(X_tr[i], str(int(y_tr[i])))
    return clf


def default_cls_baseline(preset: str) -> ClsParams:
    """Reasonable fixed classifier hyperparameters when no cls search was run."""
    if preset == "coarse":
        return ClsParams(0.02, 0.06, 0.002, 0.002, 1.0, 96, 24, 3, 350)
    if preset == "medium":
        return ClsParams(0.015, 0.08, 0.002, 0.002, 1.0, 96, 32, 4, 500)
    return ClsParams(0.012, 0.07, 0.002, 0.002, 1.0, 128, 24, 5, 700)


def row_to_cls_params(row: Dict[str, Any]) -> ClsParams:
    return ClsParams(
        float(row["world_lr"]),
        float(row["delta_lr"]),
        float(row["enc_lr"]),
        float(row["mdl_lambda"]),
        float(row["temperature"]),
        int(float(row["field_dim"])),
        int(float(row["context_win"])),
        int(float(row["n_epochs"])),
        int(float(row["train_passes_cap"])),
    )


def preset_generation_grid(preset: str) -> List[Tuple[float, int, int, int]]:
    """(temperature, max_candidates, n_calls, n_per_call) for rejection sampling."""
    if preset == "coarse":
        temps = (1.2, 1.6)
        mcs = (8, 14)
        n_calls, n_per = 6, 6
    elif preset == "medium":
        temps = (1.0, 1.3, 1.6)
        mcs = (8, 12, 16)
        n_calls, n_per = 8, 8
    else:
        temps = (0.9, 1.2, 1.5, 1.9)
        mcs = (6, 10, 14, 20)
        n_calls, n_per = 10, 8
    return [(t, mc, n_calls, n_per) for t, mc in product(temps, mcs)]


def random_generation_grid(n: int, rng_seed: int) -> List[Tuple[float, int, int, int]]:
    """Sample n unique (temperature, max_candidates, n_calls, n_per_call) tuples."""
    rng = random.Random(rng_seed)
    seen: set[Tuple[float, int, int, int]] = set()
    out: List[Tuple[float, int, int, int]] = []
    attempts = 0
    cap = max(n * 100, 10_000)
    while len(out) < n and attempts < cap:
        attempts += 1
        t = round(rng.uniform(0.45, 2.35), 5)
        mc = rng.randint(4, 56)
        nc = rng.randint(2, 16)
        np_ = rng.randint(2, 16)
        key = (t, mc, nc, np_)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    while len(out) < n:
        extra = (round(rng.uniform(0.45, 2.35), 5), rng.randint(4, 56), rng.randint(2, 16), rng.randint(2, 16))
        out.append(extra)
    return out[:n]


def eval_generation_with_clf(
    clf: CyphaDIF,
    p_cls: ClsParams,
    temperature: float,
    max_candidates: int,
    n_calls: int,
    n_per_call: int,
    seed: int,
    wall_train_s: float,
) -> Dict[str, Any]:
    """Run generate + match-rate eval on an already-trained classifier."""
    with clf.memory._lock:
        labels = list(clf.memory._label_order)
    if not labels:
        return {
            "task": "generation",
            "val_accuracy_proxy": float("nan"),
            "wall_gen_s": 0.0,
            "wall_train_s": wall_train_s,
            "gen_temperature": temperature,
            "gen_max_candidates": max_candidates,
            "gen_base_label_rule": "sorted_lexicographic",
            "skipped": True,
        }
    # Lexicographic min so base_label is stable across training order (not first-seen class).
    lab = sorted(labels)[0]

    t1 = time.perf_counter()
    match = 0
    total = 0
    for j in range(n_calls):
        samples = clf.generate(
            lab,
            n=n_per_call,
            temperature=temperature,
            rejection_sampling=True,
            max_candidates=max_candidates,
            rng=np.random.default_rng(seed + j),
        )
        for h in samples:
            pred, _ = clf.infer(h)
            total += 1
            if pred == lab:
                match += 1
    t_gen = time.perf_counter() - t1

    row: Dict[str, Any] = {
        "task": "generation",
        "base_label": lab,
        "gen_base_label_rule": "sorted_lexicographic",
        "gen_temperature": temperature,
        "gen_max_candidates": max_candidates,
        "gen_n_calls": n_calls,
        "gen_n_per_call": n_per_call,
        "gen_match_rate": float(match / max(total, 1)),
        "wall_train_s": wall_train_s,
        "wall_gen_s": round(t_gen, 6),
        "wall_total_s": round(wall_train_s + t_gen, 6),
        "skipped": False,
    }
    row.update(_cls_dict(p_cls))
    return row


def train_eval_generation(
    p_cls: ClsParams,
    temperature: float,
    max_candidates: int,
    n_calls: int,
    n_per_call: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    d: int,
    seed: int,
    cls_train_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Train with cls params, then stress CyphaDIF.generate (rejection path)."""
    t0 = time.perf_counter()
    ts = cls_train_seed if cls_train_seed is not None else seed
    clf = _train_cls_model(p_cls, X_tr, y_tr, d, ts)
    t_train = round(time.perf_counter() - t0, 6)
    return eval_generation_with_clf(
        clf, p_cls, temperature, max_candidates, n_calls, n_per_call, seed, t_train
    )


def _profile_generation(
    p_cls: ClsParams,
    temperature: float,
    max_candidates: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    d: int,
    seed: int,
) -> str:
    pr = cProfile.Profile()

    def work():
        clf = _train_cls_model(p_cls, X_tr, y_tr, d, seed)
        with clf.memory._lock:
            labs = list(clf.memory._label_order)
            lab = sorted(labs)[0] if labs else "0"
        for _ in range(3):
            clf.generate(
                lab,
                n=4,
                temperature=temperature,
                rejection_sampling=True,
                max_candidates=max_candidates,
            )

    pr.enable()
    work()
    pr.disable()
    buf = io.StringIO()
    import pstats

    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumtime").print_stats(40)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=("coarse", "medium", "fine"), default="coarse")
    ap.add_argument(
        "--task",
        choices=("both", "cls", "reg", "gen"),
        default="both",
        help="'gen' = only generation grid (uses default cls hyperparameters for preset)",
    )
    ap.add_argument(
        "--include-generation",
        action="store_true",
        help="After cls+reg, tune generate() using best cls hyperparameters (adds rows, task=generation)",
    )
    ap.add_argument(
        "--gen-max-combos",
        type=int,
        default=0,
        help="Subsample generation grid (0 = full preset cross-product)",
    )
    ap.add_argument(
        "--gen-samples",
        type=int,
        default=0,
        help="If >0, ignore preset gen grid: sample this many random (temp,mc,n_calls,n_per) "
        "configs (uses fixed cls train seed per run batch so match_rate compares gen settings)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.82)
    ap.add_argument("--max-combos", type=int, default=0, help="0 = full grid")
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers (needs joblib)")
    ap.add_argument("--profile-top", type=int, default=3, help="cProfile top-N cls/reg configs (0=skip)")
    ap.add_argument(
        "--out-dir",
        default=str(_ROOT / "artifacts" / "tuning"),
        help="Directory for CSV, JSON, profile txt",
    )
    ap.add_argument(
        "--no-gpu-stress",
        action="store_true",
        help="Skip large-batch Cypha GPU loops (classification/regression)",
    )
    ap.add_argument("--gpu-batch-n", type=int, default=8192, help="Rows per GPU stress tensor")
    ap.add_argument(
        "--gpu-stress-repeats",
        type=int,
        default=16,
        help="Repeat score_matrix+gate+batch_infer per tuning cell (cls) or predict chunks (reg)",
    )
    ap.add_argument("--gpu-reg-chunk", type=int, default=512, help="predict_batch chunk size in reg GPU stress")
    ap.add_argument(
        "--gpu-burn-passes",
        type=int,
        default=48,
        help="Raw CuPy fp64 GEMM passes before grid (0=skip)",
    )
    ap.add_argument("--gpu-burn-n", type=int, default=8192)
    ap.add_argument("--gpu-burn-d", type=int, default=512)
    ap.add_argument("--gpu-burn-k", type=int, default=256)
    args = ap.parse_args()

    if args.jobs > 1 and Parallel is None:
        print("joblib not installed; running sequentially. pip install joblib", file=sys.stderr)
        args.jobs = 1

    gpu_stress_cfg = GPUStressConfig(
        enabled=not args.no_gpu_stress,
        batch_n=max(256, args.gpu_batch_n),
        repeats=max(1, args.gpu_stress_repeats),
        reg_chunk=max(32, args.gpu_reg_chunk),
    )
    if args.jobs > 1 and gpu_stress_cfg.enabled:
        print(
            "WARNING: --jobs>1 with GPU stress may serialize on GPU or OOM; prefer --jobs 1 for max GPU duty.",
            file=sys.stderr,
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"tuning_{ts}"

    from sklearn.preprocessing import StandardScaler

    print("Loading datasets...", flush=True)
    Xc, yc, cls_name = _load_classification_xy()
    Xc_tr, yc_tr, Xc_va, yc_va = _split(Xc, yc, args.train_frac, args.seed)
    sx_cls = StandardScaler()
    Xc_tr = sx_cls.fit_transform(Xc_tr)
    Xc_va = sx_cls.transform(Xc_va)
    d_cls = Xc_tr.shape[1]

    Xr_tr = Xr_va = yr_tr = yr_va = np.array([])
    d_reg = 0
    reg_name = "(not loaded)"
    if args.task != "gen":
        Xr, yr_raw, reg_name = _load_regression_xy()
        Xr_tr, yr_tr, Xr_va, yr_va = _split(Xr, yr_raw, args.train_frac, args.seed + 99)
        sx_reg = StandardScaler()
        Xr_tr = sx_reg.fit_transform(Xr_tr)
        Xr_va = sx_reg.transform(Xr_va)
        # Raw y (original target units), same as scripts/download_profile_e2e.py regression.
        # Scaling y made val_r2 / MAE incomparable to typical housing benchmarks and other tools.
        d_reg = Xr_tr.shape[1]

    cls_grid: List[ClsParams] = []
    reg_grid: List[RegParams] = []
    if args.task != "gen":
        cls_grid, reg_grid = preset_grids(args.preset)
        cls_grid = _subsample(cls_grid, args.max_combos, args.seed + 1)
        reg_grid = _subsample(reg_grid, args.max_combos, args.seed + 2)

    gpu = _gpu_info()
    if args.gpu_burn_passes > 0:
        print(
            f"CuPy GEMM burn: {args.gpu_burn_passes} passes "
            f"({args.gpu_burn_n}x{args.gpu_burn_d})@({args.gpu_burn_d}x{args.gpu_burn_k})...",
            flush=True,
        )
        burn = cupy_gemm_burn(
            args.gpu_burn_passes,
            n=args.gpu_burn_n,
            d=args.gpu_burn_d,
            k=args.gpu_burn_k,
            seed=args.seed,
        )
        gpu = {**gpu, **burn}
        print(json.dumps(burn, indent=2), flush=True)

    print(
        json.dumps(
            {
                "gpu": gpu,
                "gpu_stress": gpu_stress_cfg.__dict__,
                "cls_dataset": cls_name,
                "reg_dataset": reg_name,
            },
            indent=2,
        )
    )

    all_rows: List[Dict[str, Any]] = []

    def _generation_combo_list() -> Tuple[List[Tuple[float, int, int, int]], str]:
        if args.gen_samples > 0:
            return random_generation_grid(args.gen_samples, args.seed + 33), "random_samples"
        gl = preset_generation_grid(args.preset)
        gl = _subsample(gl, args.gen_max_combos, args.seed + 4)
        return gl, "preset"

    def _run_generation_sweep(
        gen_list: List[Tuple[float, int, int, int]],
        p_cls: ClsParams,
        row_seed_base: int,
        title: str,
    ) -> List[Dict[str, Any]]:
        cls_ts: Optional[int] = args.seed + 88001 if args.gen_samples > 0 else None
        n = len(gen_list)
        print(f"{title}: {n} combos", flush=True)
        if n == 0:
            return []

        # One trained clf, many (temp, mc, calls) — avoids N× redundant training for --gen-samples.
        if cls_ts is not None:
            t0 = time.perf_counter()
            clf = _train_cls_model(p_cls, Xc_tr, yc_tr, d_cls, cls_ts)
            wall_train = round(time.perf_counter() - t0, 6)
            out: List[Dict[str, Any]] = []
            step = max(1, n // 20)
            for i, (gt, gmc, nc, np_) in enumerate(gen_list):
                row = eval_generation_with_clf(
                    clf, p_cls, gt, gmc, nc, np_, row_seed_base + i, wall_train
                )
                out.append(row)
                if (i + 1) % step == 0 or i == n - 1:
                    print(
                        f"  [{i+1}/{n}] match_rate={row.get('gen_match_rate', 0):.3f} "
                        f"temp={gt} mc={gmc}",
                        flush=True,
                    )
            best = max(out, key=lambda r: (r.get("gen_match_rate", 0), -r.get("wall_gen_s", 0)))
            print(
                f"  best match_rate={best.get('gen_match_rate', 0):.4f} "
                f"temp={best.get('gen_temperature')} mc={best.get('gen_max_candidates')} "
                f"calls={best.get('gen_n_calls')} per={best.get('gen_n_per_call')}",
                flush=True,
            )
            return out

        def one(i: int, gt: float, gmc: int, nc: int, np_: int) -> Dict[str, Any]:
            return train_eval_generation(
                p_cls,
                gt,
                gmc,
                nc,
                np_,
                Xc_tr,
                yc_tr,
                d_cls,
                row_seed_base + i,
                cls_train_seed=None,
            )

        if args.jobs <= 1:
            out2: List[Dict[str, Any]] = []
            step = max(1, n // 20)
            for i, (gt, gmc, nc, np_) in enumerate(gen_list):
                row = one(i, gt, gmc, nc, np_)
                out2.append(row)
                if (i + 1) % step == 0 or i == n - 1:
                    print(
                        f"  [{i+1}/{n}] match_rate={row.get('gen_match_rate', 0):.3f} "
                        f"temp={gt} mc={gmc}",
                        flush=True,
                    )
            return out2

        rows = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(one)(i, gt, gmc, nc, np_)
            for i, (gt, gmc, nc, np_) in enumerate(gen_list)
        )
        best = max(rows, key=lambda r: (r.get("gen_match_rate", 0), -r.get("wall_gen_s", 0)))
        print(
            f"  done — best match_rate={best.get('gen_match_rate', 0):.4f} "
            f"temp={best.get('gen_temperature')} mc={best.get('gen_max_candidates')} "
            f"calls={best.get('gen_n_calls')} per={best.get('gen_n_per_call')}",
            flush=True,
        )
        return list(rows)

    if args.task == "gen":
        gen_list, gen_mode = _generation_combo_list()
        p0 = default_cls_baseline(args.preset)
        tag = f"generation-only ({gen_mode}, preset={args.preset})"
        gen_rows_extra = _run_generation_sweep(gen_list, p0, args.seed + 200_000, tag)
        all_rows.extend(gen_rows_extra)

    if args.task in ("both", "cls"):
        print(f"Classification grid: {len(cls_grid)} combos (preset={args.preset})", flush=True)
        if args.jobs <= 1:
            for i, p in enumerate(cls_grid):
                row = train_eval_cls(
                    p,
                    Xc_tr,
                    yc_tr,
                    Xc_va,
                    yc_va,
                    d_cls,
                    args.seed + i,
                    gpu_stress_cfg,
                )
                all_rows.append(row)
                if (i + 1) % max(1, len(cls_grid) // 10) == 0 or i == len(cls_grid) - 1:
                    print(f"  [{i+1}/{len(cls_grid)}] val_acc={row['val_accuracy']:.4f}", flush=True)
        else:
            rows = Parallel(n_jobs=args.jobs, backend="loky")(
                delayed(train_eval_cls)(
                    p,
                    Xc_tr,
                    yc_tr,
                    Xc_va,
                    yc_va,
                    d_cls,
                    args.seed + i,
                    gpu_stress_cfg,
                )
                for i, p in enumerate(cls_grid)
            )
            all_rows.extend(rows)

    if args.task in ("both", "reg"):
        print(f"Regression grid: {len(reg_grid)} combos (preset={args.preset})", flush=True)
        if args.jobs <= 1:
            for i, p in enumerate(reg_grid):
                row = train_eval_reg(
                    p,
                    Xr_tr,
                    yr_tr,
                    Xr_va,
                    yr_va,
                    d_reg,
                    args.seed + 1000 + i,
                    gpu_stress_cfg,
                )
                all_rows.append(row)
                if (i + 1) % max(1, len(reg_grid) // 10) == 0 or i == len(reg_grid) - 1:
                    print(
                        f"  [{i+1}/{len(reg_grid)}] val_r2={row['val_r2']:.4f} mae={row['val_mae']:.4f}",
                        flush=True,
                    )
        else:
            rows = Parallel(n_jobs=args.jobs, backend="loky")(
                delayed(train_eval_reg)(
                    p,
                    Xr_tr,
                    yr_tr,
                    Xr_va,
                    yr_va,
                    d_reg,
                    args.seed + 1000 + i,
                    gpu_stress_cfg,
                )
                for i, p in enumerate(reg_grid)
            )
            all_rows.extend(rows)

    if args.include_generation and args.task in ("both", "cls"):
        cls_rows_tmp = [r for r in all_rows if r.get("task") == "classification"]
        best_c = max(cls_rows_tmp, key=lambda r: r["val_accuracy"]) if cls_rows_tmp else None
        p_base = row_to_cls_params(best_c) if best_c else default_cls_baseline(args.preset)
        gen_list, gen_mode = _generation_combo_list()
        cls_tag = "best cls row" if best_c else "default cls baseline"
        title = f"Generation add-on ({cls_tag}, {gen_mode})"
        gen_rows_extra = _run_generation_sweep(gen_list, p_base, args.seed + 300_000, title)
        all_rows.extend(gen_rows_extra)

    cls_rows = [r for r in all_rows if r.get("task") == "classification"]
    reg_rows = [r for r in all_rows if r.get("task") == "regression"]
    gen_rows = [r for r in all_rows if r.get("task") == "generation"]

    best_cls = max(cls_rows, key=lambda r: r["val_accuracy"]) if cls_rows else None
    best_reg = max(reg_rows, key=lambda r: r["val_r2"]) if reg_rows else None
    def _gen_sort_key(r: Dict[str, Any]) -> Tuple[float, int, float]:
        mr = float(r.get("gen_match_rate", 0.0))
        ntot = int(r.get("gen_n_calls", 0)) * int(r.get("gen_n_per_call", 0))
        return (mr, ntot, -float(r.get("wall_gen_s", 0.0)))

    best_gen = max(gen_rows, key=_gen_sort_key) if gen_rows else None
    fastest_cls = min(cls_rows, key=lambda r: r["wall_total_s"]) if cls_rows else None
    fastest_reg = min(reg_rows, key=lambda r: r["wall_total_s"]) if reg_rows else None

    summary = {
        "timestamp_utc": ts,
        "preset": args.preset,
        "task": args.task,
        "train_frac": args.train_frac,
        "max_combos": args.max_combos,
        "gen_samples": args.gen_samples,
        "datasets": {"classification": cls_name, "regression": reg_name},
        "note_regression_y": "Target y in original units (X standardized only); same convention as download_profile_e2e regression.",
        "note_classification": (
            "val_accuracy = mean(batch_infer(x)==str(int(y))) on the validation split; same formula as "
            "download_profile_e2e _metrics_classification. This script default train_frac=0.82 (e2e uses 0.85 "
            "or 0.75), hyperparameters differ (grid search vs fixed e2e recipe), and the full val set is used "
            "(e2e caps rows with max_test). Dataset name in 'datasets.classification' — Iris fallback if OpenML fails."
        ),
        "note_generation": (
            "gen_match_rate = fraction of generated latents (rejection_sampling) for which infer(pred)==base_label. "
            "base_label is the lexicographically smallest class name among trained labels (gen_base_label_rule). "
            "download_profile_e2e does not compute this metric. Best row: max match_rate, then max "
            "gen_n_calls*gen_n_per_call (more samples), then lower wall_gen_s."
        ),
        "gpu": gpu,
        "gpu_stress_config": gpu_stress_cfg.__dict__,
        "counts": {
            "classification": len(cls_rows),
            "regression": len(reg_rows),
            "generation": len(gen_rows),
        },
        "best_by_quality": {
            "classification_val_accuracy": best_cls,
            "regression_val_r2": best_reg,
            "generation_match_rate": best_gen,
        },
        "best_by_speed": {
            "classification_min_wall_total_s": fastest_cls,
            "regression_min_wall_total_s": fastest_reg,
        },
    }

    if all_rows:
        keys = sorted({k for r in all_rows for k in r.keys()})
        csv_path = base.with_name(base.name + "_results.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(_csv_header(keys))
            for r in all_rows:
                f.write(_csv_row(r, keys))
        print(f"Wrote {csv_path}")

    json_path = base.with_name(base.name + "_summary.json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")

    prof_parts: List[str] = []
    if args.profile_top > 0 and cls_rows:
        cls_sorted = sorted(cls_rows, key=lambda r: r["val_accuracy"], reverse=True)[
            : args.profile_top
        ]
        for i, row in enumerate(cls_sorted):
            p = ClsParams(
                float(row["world_lr"]),
                float(row["delta_lr"]),
                float(row["enc_lr"]),
                float(row["mdl_lambda"]),
                float(row["temperature"]),
                int(float(row["field_dim"])),
                int(float(row["context_win"])),
                int(float(row["n_epochs"])),
                int(float(row["train_passes_cap"])),
            )
            prof_parts.append(
                f"=== PROFILE classification rank {i+1} val_acc={row['val_accuracy']:.6f} ===\n"
                + _profile_best_cls(p, Xc_tr, yc_tr, d_cls, args.seed + 5000 + i)
            )

    if args.profile_top > 0 and reg_rows:
        reg_sorted = sorted(reg_rows, key=lambda r: r["val_r2"], reverse=True)[
            : args.profile_top
        ]
        for i, row in enumerate(reg_sorted):
            p = RegParams(
                float(row["world_lr"]),
                float(row["delta_lr"]),
                float(row["enc_lr"]),
                float(row["mdl_lambda"]),
                float(row["temperature"]),
                int(float(row["field_dim"])),
                int(float(row["context_win"])),
                float(row["target_lr"]),
                int(float(row["n_experts"])),
                int(float(row["n_epochs"])),
                int(float(row["train_passes_cap"])),
            )
            prof_parts.append(
                f"=== PROFILE regression rank {i+1} val_r2={row['val_r2']:.6f} ===\n"
                + _profile_best_reg(p, Xr_tr, yr_tr, d_reg, args.seed + 6000 + i)
            )

    if args.profile_top > 0 and best_gen and not best_gen.get("skipped"):
        p_cls = (
            row_to_cls_params(best_gen)
            if "world_lr" in best_gen
            else default_cls_baseline(args.preset)
        )
        gt = float(best_gen["gen_temperature"])
        gmc = int(float(best_gen["gen_max_candidates"]))
        prof_parts.append(
            f"=== PROFILE generation (best match_rate={best_gen.get('gen_match_rate', 0):.4f}) ===\n"
            + _profile_generation(p_cls, gt, gmc, Xc_tr, yc_tr, d_cls, args.seed + 7000)
        )

    if prof_parts:
        prof_path = base.with_name(base.name + "_profile.txt")
        prof_path.write_text("\n\n".join(prof_parts), encoding="utf-8")
        print(f"Wrote {prof_path}")

    _CLS_KEYS = (
        "world_lr",
        "delta_lr",
        "enc_lr",
        "mdl_lambda",
        "temperature",
        "field_dim",
        "context_win",
        "n_epochs",
        "train_passes_cap",
    )
    _REG_KEYS = _CLS_KEYS + ("target_lr", "n_experts")

    print("\nBest quality:")
    if best_cls:
        pk = {k: best_cls[k] for k in _CLS_KEYS}
        pk["val_accuracy"] = best_cls["val_accuracy"]
        print(f"  cls {json.dumps(pk, indent=2)}")
    if best_reg:
        pk = {k: best_reg[k] for k in _REG_KEYS}
        pk["val_r2"] = best_reg["val_r2"]
        pk["val_mae"] = best_reg["val_mae"]
        print(f"  reg {json.dumps(pk, indent=2)}")
    if best_gen:
        print(
            f"  gen match_rate={best_gen.get('gen_match_rate', 0):.4f} "
            f"temp={best_gen.get('gen_temperature')} mc={best_gen.get('gen_max_candidates')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
