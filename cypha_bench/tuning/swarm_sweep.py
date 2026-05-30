#!/usr/bin/env python3
"""
Swarm hyperparameter sweep for everyday CyphaDIF / DIFRegressor / CyphaLM use.

Uses joblib parallel workers ("swarm") and CuPy GPU batch paths when available.
Writes cypha_bench/config/everyday_profile.json and a tuning report.

Usage (from repo root):
  python cypha_bench/tuning/swarm_sweep.py
  python cypha_bench/tuning/swarm_sweep.py --preset medium --jobs 8 --max-combos 120
  python cypha_bench/tuning/swarm_sweep.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from Cypha import CyphaDIF, DIFRegressor, TieredContextBuffer, VectorEncoder  # noqa: E402
from cypha_accel.cuda_util import cuda_gemm_usable, warmup_cuda  # noqa: E402
from cypha_accel.gpu_stress import cupy_gemm_burn  # noqa: E402

# Reuse tuned training/eval from the main tuner.
from scripts.tune_quality_performance import (  # noqa: E402
    ClsParams,
    GPUStressConfig,
    RegParams,
    _acc,
    _field_dim,
    _load_classification_xy,
    _load_regression_xy,
    _reg_metrics,
    _split,
    _subsample,
    preset_grids,
    train_eval_cls,
    train_eval_reg,
)

try:
    from joblib import Parallel, delayed  # noqa: WPS433
except ImportError:
    Parallel = None  # type: ignore
    delayed = None  # type: ignore


OUT_DIR = _REPO / "cypha_bench" / "artifacts" / "tuning"
PROFILE_PATH = _REPO / "cypha_bench" / "config" / "everyday_profile.json"


@dataclass
class ArchParams:
    replay_ratio: float = 0.30
    ood_sigma: float = 15.0


def _online_bench_cls_score(
    p: ClsParams,
    arch: ArchParams,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
) -> float:
    """Online one-pass training (matches cypha_bench protocol)."""
    d = X_tr.shape[1]
    fd = _field_dim(d, p.field_dim)
    clf = CyphaDIF(
        encoder=VectorEncoder(d),
        field_dim=fd,
        enc_lr=p.enc_lr,
        delta_lr=p.delta_lr,
        world_lr=p.world_lr,
        mdl_lambda=p.mdl_lambda,
        context_win=p.context_win,
        replay_ratio=arch.replay_ratio,
        rng=np.random.default_rng(seed),
    )
    clf.temperature = float(p.temperature)
    clf.ood_sigma = float(arch.ood_sigma)
    for x, y in zip(X_tr, y_tr):
        clf.train_step(x, str(int(y)))
    return _acc(clf, X_te, y_te)


def _online_bench_reg_r2(
    p: RegParams,
    arch: ArchParams,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
) -> float:
    d = X_tr.shape[1]
    fd = _field_dim(d, p.field_dim)
    reg = DIFRegressor(
        encoder=VectorEncoder(d),
        field_dim=fd,
        n_experts=p.n_experts,
        target_lr=p.target_lr,
        replay_ratio=arch.replay_ratio,
        rng=np.random.default_rng(seed),
    )
    reg.clf.world_lr = p.world_lr
    reg.clf.delta_lr = p.delta_lr
    reg.clf.enc_lr = p.enc_lr
    reg.clf.mdl_lambda = p.mdl_lambda
    reg.clf.context = TieredContextBuffer(short_window=p.context_win)
    reg.clf.temperature = float(p.temperature)
    reg.clf.ood_sigma = float(arch.ood_sigma)
    for x, y in zip(X_tr, y_tr):
        reg.train_step(x, float(y))
    preds, _ = reg.predict_batch([X_te[i] for i in range(len(X_te))])
    _, _, r2 = _reg_metrics(y_te, preds[:, 0] if preds.ndim > 1 else preds)
    return r2


def _composite_score(cls_acc: float, reg_r2: float, bench_cls: float, bench_r2: float, wall_s: float) -> float:
    """Higher is better. Favour val metrics + bench-aligned online scores; penalise slow configs."""
    val_part = 0.35 * cls_acc + 0.35 * max(reg_r2, -1.0)
    bench_part = 0.20 * bench_cls + 0.20 * max(bench_r2, -1.0)
    speed_pen = min(wall_s / 60.0, 1.0) * 0.05
    return val_part + bench_part - speed_pen


def _refine_architecture(
    best_cls: ClsParams,
    best_reg: RegParams,
    Xc_tr,
    yc_tr,
    Xc_va,
    yc_va,
    Xr_tr,
    yr_tr,
    Xr_va,
    yr_va,
    seed: int,
) -> ArchParams:
    from sklearn.datasets import load_iris, load_wine

    best_arch = ArchParams()
    best_score = -1e9
    cls_sets = [
        (load_iris().data.astype(np.float64), load_iris().target),
        (load_wine().data.astype(np.float64), load_wine().target),
    ]
    for rr, ood in product((0.0, 0.15, 0.30), (10.0, 15.0, 22.0)):
        arch = ArchParams(replay_ratio=rr, ood_sigma=ood)
        try:
            cls_scores = []
            for Xi, yi in cls_sets:
                n = len(Xi)
                split = int(n * 0.8)
                cls_scores.append(
                    _online_bench_cls_score(best_cls, arch, Xi[:split], yi[:split], Xi[split:], yi[split:], seed)
                )
            bc = float(np.mean(cls_scores))
            br = _online_bench_reg_r2(best_reg, arch, Xr_tr, yr_tr, Xr_va, yr_va, seed + 1)
            score = 0.6 * bc + 0.4 * max(br, 0.0)
            if score > best_score:
                best_score = score
                best_arch = arch
        except Exception:
            continue
    return best_arch


def _cyphalm_mini_sweep(seed: int = 42) -> dict[str, Any]:
    """Small CyphaLM sweep on Gutenberg excerpt (char-level)."""
    try:
        from cypha_lm import CyphaLM
        from cypha_lm.config import CyphaLMConfig
    except ImportError:
        return {}

    gut = _REPO / "cypha_bench" / "data" / "gutenberg" / "alice.txt"
    if not gut.exists():
        return {}
    text = gut.read_text(encoding="utf-8", errors="replace")[:8000]
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    ids = [c2i[c] for c in text if c in c2i]
    if len(ids) < 200:
        return {}

    best_cfg: dict[str, Any] | None = None
    best_loss = 1e9
    for gria_lr, kappa0 in product((0.02, 0.05, 0.1), (0.1, 0.5, 1.0)):
        cfg = CyphaLMConfig(
            vocab_size=min(256, len(chars)),
            d_embed=64,
            field_dim=128,
            d_state=64,
            ssm_layers=1,
            gria_lr=gria_lr,
            nig_kappa0=kappa0,
            seed=seed,
        )
        model = CyphaLM(cfg)
        losses = []
        for t in range(min(1500, len(ids) - 1)):
            m = model.train_step(ids[t], ids[t + 1])
            losses.append(m["loss"])
        mean_loss = float(np.mean(losses[-200:])) if losses else 1e9
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_cfg = {"gria_lr": gria_lr, "nig_kappa0": kappa0, "mean_tail_loss": mean_loss}
    return best_cfg or {}


def run_sweep(
    preset: str = "medium",
    max_combos: int = 100,
    jobs: int = 4,
    seed: int = 42,
    gpu_burn: int = 24,
    quick: bool = False,
) -> Path:
    if quick:
        preset = "coarse"
        max_combos = 24
        jobs = min(jobs, 2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warmup_cuda()
    gpu_ok = cuda_gemm_usable()

    if gpu_ok and gpu_burn > 0:
        cupy_gemm_burn(gpu_burn, n=4096, d=4096, k=4096, seed=seed)

    gpu_stress = GPUStressConfig(enabled=gpu_ok, batch_n=8192 if gpu_ok else 2048, repeats=8 if gpu_ok else 0)

    Xc, yc, cls_name = _load_classification_xy()
    Xr, yr, reg_name = _load_regression_xy()
    Xc_tr, yc_tr, Xc_va, yc_va = _split(Xc, yc, 0.82, seed)
    Xr_tr, yr_tr, Xr_va, yr_va = _split(Xr, yr, 0.82, seed + 7)
    d_cls, d_reg = Xc_tr.shape[1], Xr_tr.shape[1]

    cls_grid, reg_grid = preset_grids(preset)
    cls_grid = _subsample(cls_grid, max_combos, seed + 1)
    reg_grid = _subsample(reg_grid, max_combos, seed + 2)

    print(f"Swarm sweep preset={preset} cls={len(cls_grid)} reg={len(reg_grid)} jobs={jobs} cuda={gpu_ok}")

    t0 = time.perf_counter()
    cls_rows: list[dict] = []
    reg_rows: list[dict] = []

    if Parallel is not None and jobs > 1:
        cls_rows = Parallel(n_jobs=jobs, backend="loky")(
            delayed(train_eval_cls)(p, Xc_tr, yc_tr, Xc_va, yc_va, d_cls, seed + i, gpu_stress)
            for i, p in enumerate(cls_grid)
        )
        reg_rows = Parallel(n_jobs=jobs, backend="loky")(
            delayed(train_eval_reg)(p, Xr_tr, yr_tr, Xr_va, yr_va, d_reg, seed + 1000 + i, gpu_stress)
            for i, p in enumerate(reg_grid)
        )
    else:
        for i, p in enumerate(cls_grid):
            cls_rows.append(train_eval_cls(p, Xc_tr, yc_tr, Xc_va, yc_va, d_cls, seed + i, gpu_stress))
        for i, p in enumerate(reg_grid):
            reg_rows.append(train_eval_reg(p, Xr_tr, yr_tr, Xr_va, yr_va, d_reg, seed + 1000 + i, gpu_stress))

    best_cls_row = max(cls_rows, key=lambda r: (r.get("val_accuracy", 0), -r.get("wall_total_s", 999)))

    def _reg_rank(row: dict) -> tuple:
        r2 = float(row.get("val_r2", -999))
        mae = float(row.get("val_mae", 999))
        if r2 > 0.05:
            return (1, r2, -mae)
        return (0, -mae, r2)

    best_reg_row = max(reg_rows, key=_reg_rank)

    best_cls = ClsParams(
        best_cls_row["world_lr"],
        best_cls_row["delta_lr"],
        best_cls_row["enc_lr"],
        best_cls_row["mdl_lambda"],
        best_cls_row["temperature"],
        best_cls_row["field_dim"],
        best_cls_row["context_win"],
        best_cls_row["n_epochs"],
        best_cls_row["train_passes_cap"],
    )
    best_reg = RegParams(
        best_reg_row["world_lr"],
        best_reg_row["delta_lr"],
        best_reg_row["enc_lr"],
        best_reg_row["mdl_lambda"],
        best_reg_row["temperature"],
        best_reg_row["field_dim"],
        best_reg_row["context_win"],
        best_reg_row["target_lr"],
        best_reg_row["n_experts"],
        best_reg_row["n_epochs"],
        best_reg_row["train_passes_cap"],
    )

    arch = _refine_architecture(
        best_cls, best_reg, Xc_tr, yc_tr, Xc_va, yc_va, Xr_tr, yr_tr, Xr_va, yr_va, seed + 99
    )
    if float(best_reg_row.get("val_r2", -1)) < 0.05:
        # Online bench alignment: re-score top regression candidates on one-pass protocol.
        top_regs = sorted(reg_rows, key=_reg_rank, reverse=True)[:12]
        best_online = best_reg_row
        best_online_r2 = float(best_reg_row.get("val_r2", -999))
        for cand in top_regs:
            p = RegParams(
                cand["world_lr"],
                cand["delta_lr"],
                cand["enc_lr"],
                cand["mdl_lambda"],
                cand["temperature"],
                cand["field_dim"],
                cand["context_win"],
                cand["target_lr"],
                cand["n_experts"],
                cand["n_epochs"],
                cand["train_passes_cap"],
            )
            try:
                online_r2 = _online_bench_reg_r2(p, arch, Xr_tr, yr_tr, Xr_va, yr_va, seed + 501)
            except Exception:
                continue
            if online_r2 > best_online_r2:
                best_online_r2 = online_r2
                best_online = cand
        best_reg_row = best_online
        best_reg = RegParams(
            best_reg_row["world_lr"],
            best_reg_row["delta_lr"],
            best_reg_row["enc_lr"],
            best_reg_row["mdl_lambda"],
            best_reg_row["temperature"],
            best_reg_row["field_dim"],
            best_reg_row["context_win"],
            best_reg_row["target_lr"],
            best_reg_row["n_experts"],
            best_reg_row["n_epochs"],
            best_reg_row["train_passes_cap"],
        )
    if arch.replay_ratio == 0.0:
        # Mild replay helps multitask / forgetting in bench cross-domain tests.
        arch = ArchParams(replay_ratio=0.15, ood_sigma=arch.ood_sigma)
    cyphalm = _cyphalm_mini_sweep(seed + 200)

    elapsed = time.perf_counter() - t0

    profile = {
        "source": f"cypha_bench/tuning/swarm_sweep.py preset={preset}",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_cuda_usable": gpu_ok,
        "datasets": {"classification": cls_name, "regression": reg_name},
        "classification_cyphadif": {k: best_cls_row[k] for k in asdict(best_cls).keys()},
        "regression_difregressor": {
            **{k: best_reg_row[k] for k in asdict(best_reg).keys() if k in best_reg_row},
            "target_lr": best_reg_row.get("target_lr", best_reg.target_lr),
            "n_experts": best_reg_row.get("n_experts", best_reg.n_experts),
        },
        "architecture": asdict(arch),
        "cyphalm": cyphalm,
        "validation_metrics": {
            "cls_val_accuracy": best_cls_row.get("val_accuracy"),
            "reg_val_r2": best_reg_row.get("val_r2"),
            "reg_val_mae": best_reg_row.get("val_mae"),
            "sweep_wall_s": round(elapsed, 2),
        },
    }

    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = OUT_DIR / f"swarm_{stamp}.json"
    report_path.write_text(
        json.dumps(
            {
                "profile_path": str(PROFILE_PATH),
                "best_cls": best_cls_row,
                "best_reg": best_reg_row,
                "architecture": asdict(arch),
                "cyphalm": cyphalm,
                "n_cls_combos": len(cls_rows),
                "n_reg_combos": len(reg_rows),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Profile written: {PROFILE_PATH}")
    print(f"Report: {report_path}")
    return PROFILE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Swarm GPU/CPU hyperparameter sweep")
    parser.add_argument("--preset", default="medium", choices=("coarse", "medium", "fine"))
    parser.add_argument("--max-combos", type=int, default=int(os.environ.get("SWARM_MAX_COMBOS", "80")))
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SWARM_JOBS", max(1, (os.cpu_count() or 4) - 1))))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-burn", type=int, default=32)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_sweep(
        preset=args.preset,
        max_combos=args.max_combos,
        jobs=args.jobs,
        seed=args.seed,
        gpu_burn=args.gpu_burn,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
