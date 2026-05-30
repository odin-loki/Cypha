#!/usr/bin/env python3
"""
Convergence orchestrator: regime swarms + validation until all key domains beat baseline.

Loops until every validation metric is at least as good as the frozen baseline
(accuracy delta >= 0, rmse delta <= 0) or max_iterations is reached.

Usage:
  python cypha_bench/tuning/converge_swarm.py --max-iter 5 --jobs 6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.tuning.bench_full_swarm import (  # noqa: E402
    run_full_swarm,
    run_regime_swarm,
)
from cypha_bench.tuning.bench_iterate_swarm import (  # noqa: E402
    _merge_regime_profile,
)
from cypha_bench.tuning.validate_profile import DOMAINS, _extract_rows  # noqa: E402

OUT_DIR = _REPO / "cypha_bench" / "artifacts" / "tuning"
PROFILE_PATH = _REPO / "cypha_bench" / "config" / "everyday_profile.json"
VALIDATION_PATH = OUT_DIR / "validation_compare.json"
PYTHON = sys.executable


def _metric_from_row(row: dict) -> tuple[str, float | None, float | None]:
    name = str(
        row.get("dataset")
        or row.get("task")
        or row.get("name")
        or row.get("encoding")
        or "unknown"
    )
    scores = row.get("cypha_scores") or row.get("scores") or {}
    if "accuracy" in scores:
        return name, float(scores["accuracy"]), None
    if "rmse" in scores:
        return name, None, float(scores["rmse"])
    if "r2" in scores:
        return name, float(scores.get("r2", 0.0)), None
    return name, None, None


def _collect_domain_metrics(side: dict) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for key, (_, list_key) in DOMAINS.items():
        block = side.get(key, {})
        for row in _extract_rows(block, list_key):
            name, acc, rmse = _metric_from_row(row)
            metric_key = f"{key}/{name}"
            if acc is not None:
                out[metric_key] = ("accuracy", acc)
            elif rmse is not None:
                out[metric_key] = ("rmse", rmse)
    return out


def _compute_deltas(
    baseline: dict[str, tuple[str, float]],
    tuned: dict[str, tuple[str, float]],
) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for key in sorted(set(baseline) & set(tuned)):
        metric, base_val = baseline[key]
        tuned_metric, tuned_val = tuned[key]
        if metric != tuned_metric:
            continue
        if metric == "accuracy":
            delta = tuned_val - base_val
            favorable = delta >= 0.0
        else:
            delta = tuned_val - base_val
            favorable = delta <= 0.0
        deltas.append(
            {
                "key": key,
                "metric": metric,
                "baseline": base_val,
                "tuned": tuned_val,
                "delta": delta,
                "favorable": favorable,
            }
        )
    return deltas


KEY_METRIC_PREFIXES = ("d02/", "d03/", "d08/hog")


def _is_key_metric(key: str) -> bool:
    return any(key.startswith(p) for p in KEY_METRIC_PREFIXES)


def _all_deltas_favorable(deltas: list[dict[str, Any]]) -> bool:
    key_deltas = [d for d in deltas if _is_key_metric(d["key"])]
    return bool(key_deltas) and all(d["favorable"] for d in key_deltas)


def _run_subprocess(script: str, *args: str) -> None:
    cmd = [PYTHON, str(_REPO / script), *args]
    print(f"Running: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env.setdefault("CYPHA_BENCH_FAST", "1")
    subprocess.run(cmd, cwd=_REPO, env=env, check=True)


def _load_profile() -> dict[str, Any]:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _flat_to_regimes(flat: dict[str, Any], *, source: str) -> dict[str, Any]:
    cls = dict(flat.get("classification_cyphadif", {}))
    reg = dict(flat.get("regression_difregressor", {}))
    arch = dict(flat.get("architecture", {}))
    vision = dict(cls)
    for key in ("replay_ratio", "ood_sigma"):
        if key in arch:
            vision[key] = arch[key]
    return {
        "source": source,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "default_regime": "tabular",
        "regimes": {
            "tabular": cls,
            "vision": vision,
            "regression": reg,
        },
        "architecture": arch,
    }


def _ensure_regimes_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if profile.get("regimes"):
        return profile
    return _flat_to_regimes(
        profile,
        source="cypha_bench/tuning/converge_swarm.py flat-to-regimes",
    )


def _write_combined_leaderboard(*swarm_results: dict[str, Any]) -> Path:
    rows: list[dict[str, Any]] = []
    for result in swarm_results:
        data = json.loads(result["leaderboard_path"].read_text(encoding="utf-8"))
        rows.extend(data.get("top_k_candidates") or data.get("top10") or [])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"converge_merge_{stamp}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source": "cypha_bench/tuning/converge_swarm.py combined regime swarms",
                "top_k_candidates": rows,
                "top10": rows[:10],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def _ensure_initial_profile(
    *,
    combos: int,
    jobs: int,
    seed: int,
    gpu_burn: int,
) -> dict[str, Any]:
    if PROFILE_PATH.exists():
        profile = _ensure_regimes_profile(_load_profile())
        if not _load_profile().get("regimes"):
            PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        return profile

    print("=== No profile found: initial full swarm (minmax) ===", flush=True)
    r1 = run_full_swarm(
        combos=combos,
        jobs=jobs,
        seed=seed,
        gpu_burn=gpu_burn,
        objective="minmax",
        top_k=30,
        write_profile=False,
    )
    profile = _flat_to_regimes(
        r1["best"]["profile"],
        source=f"cypha_bench/tuning/converge_swarm.py initial full swarm combos={combos}",
    )
    profile["converge_initial_leaderboard"] = str(r1["leaderboard_path"])
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Wrote initial regimes profile: {PROFILE_PATH}", flush=True)
    return profile


def _load_or_create_baseline() -> dict[str, tuple[str, float]]:
    if VALIDATION_PATH.exists():
        data = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
        baseline = _collect_domain_metrics(data.get("baseline", {}))
        if baseline:
            return baseline

    print("=== Establishing validation baseline ===", flush=True)
    _run_subprocess("cypha_bench/tuning/validate_profile.py")
    data = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    return _collect_domain_metrics(data["baseline"])


def run_converge(
    max_iterations: int = 5,
    jobs: int = 6,
    seed: int = 42,
    gpu_burn: int = 16,
    initial_combos: int = 500,
    vision_combos: int = 200,
    tabular_combos: int = 150,
    regression_combos: int = 150,
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "max_iterations": max_iterations,
        "jobs": jobs,
        "initial_combos": initial_combos,
        "vision_combos": vision_combos,
        "tabular_combos": tabular_combos,
        "regression_combos": regression_combos,
        "iterations": [],
    }

    profile = _ensure_initial_profile(
        combos=initial_combos,
        jobs=jobs,
        seed=seed,
        gpu_burn=gpu_burn,
    )
    baseline_metrics = _load_or_create_baseline()
    log["baseline_metrics"] = {
        key: {"metric": metric, "value": value}
        for key, (metric, value) in baseline_metrics.items()
    }

    success = False
    for iteration in range(1, max_iterations + 1):
        print(f"=== Converge iteration {iteration}/{max_iterations} ===", flush=True)
        profile = _ensure_regimes_profile(_load_profile())
        iter_log: dict[str, Any] = {"iteration": iteration}

        print("--- Regime swarm: vision ---", flush=True)
        vision = run_regime_swarm(
            "vision",
            profile,
            combos=vision_combos,
            jobs=jobs,
            seed=seed + iteration * 10,
            gpu_burn=gpu_burn,
            write_profile=False,
        )
        iter_log["vision_leaderboard"] = str(vision["leaderboard_path"])

        print("--- Regime swarm: tabular ---", flush=True)
        tabular = run_regime_swarm(
            "tabular",
            profile,
            combos=tabular_combos,
            jobs=jobs,
            seed=seed + iteration * 10 + 1,
            gpu_burn=gpu_burn,
            write_profile=False,
        )
        iter_log["tabular_leaderboard"] = str(tabular["leaderboard_path"])

        print("--- Regime swarm: regression ---", flush=True)
        regression = run_regime_swarm(
            "regression",
            profile,
            combos=regression_combos,
            jobs=jobs,
            seed=seed + iteration * 10 + 2,
            gpu_burn=gpu_burn,
            write_profile=False,
        )
        iter_log["regression_leaderboard"] = str(regression["leaderboard_path"])

        combined_lb = _write_combined_leaderboard(vision, tabular, regression)
        iter_log["combined_leaderboard"] = str(combined_lb)

        flat_center = {
            "classification_cyphadif": profile.get("regimes", {}).get("tabular", {}),
            "regression_difregressor": profile.get("regimes", {}).get("regression", {}),
            "architecture": profile.get("architecture", {}),
        }
        merged = _merge_regime_profile(combined_lb, flat_center)
        merged["source"] = f"cypha_bench/tuning/converge_swarm.py iteration={iteration}"
        merged["generated_utc"] = datetime.now(timezone.utc).isoformat()
        merged["converge_iteration"] = iteration
        PROFILE_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        iter_log["merged_profile"] = str(PROFILE_PATH)

        print("--- Validation (CYPHA_BENCH_FAST=1) ---", flush=True)
        _run_subprocess("cypha_bench/tuning/validate_profile.py")

        validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
        tuned_metrics = _collect_domain_metrics(validation.get("tuned", {}))
        deltas = _compute_deltas(baseline_metrics, tuned_metrics)
        iter_log["deltas"] = deltas
        iter_log["all_favorable"] = _all_deltas_favorable(deltas)
        unfavorable = [d for d in deltas if not d["favorable"]]
        if unfavorable:
            print(
                f"Not converged: {len(unfavorable)} metric(s) still below baseline",
                flush=True,
            )
            for item in unfavorable[:8]:
                print(
                    f"  {item['key']} {item['metric']} delta={item['delta']:+.4f}",
                    flush=True,
                )
        else:
            print("All validation metrics beat or match baseline.", flush=True)

        log["iterations"].append(iter_log)
        if iter_log["all_favorable"]:
            success = True
            break

    log["success"] = success
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    print("=== Write report ===", flush=True)
    _run_subprocess("cypha_bench/tuning/write_report.py")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"converge_swarm_{stamp}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"Converge log: {log_path}", flush=True)
    print(f"Profile: {PROFILE_PATH}", flush=True)
    print(f"Success: {success}", flush=True)
    log["log_path"] = str(log_path)
    return log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regime swarm convergence loop vs validation baseline",
    )
    parser.add_argument("--max-iter", type=int, default=5, dest="max_iterations")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("BENCH_SWARM_JOBS", "6")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-burn", type=int, default=16)
    parser.add_argument("--initial-combos", type=int, default=500)
    parser.add_argument("--vision-combos", type=int, default=200)
    parser.add_argument("--tabular-combos", type=int, default=150)
    parser.add_argument("--regression-combos", type=int, default=150)
    args = parser.parse_args()
    result = run_converge(
        max_iterations=args.max_iterations,
        jobs=args.jobs,
        seed=args.seed,
        gpu_burn=args.gpu_burn,
        initial_combos=args.initial_combos,
        vision_combos=args.vision_combos,
        tabular_combos=args.tabular_combos,
        regression_combos=args.regression_combos,
    )
    print(result["log_path"])


if __name__ == "__main__":
    main()
