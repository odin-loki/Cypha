#!/usr/bin/env python3
"""
Multi-round bench swarm orchestrator.

Round 1: full swarm (minmax objective)
Round 2: validate top-K, narrow sweep centered on winner
Round 3: merge regime-specific bests into everyday_profile.json regimes
Final: validate_profile + write_report

Usage:
  python cypha_bench/tuning/bench_iterate_swarm.py
  python cypha_bench/tuning/bench_iterate_swarm.py --rounds 3 --combos-round1 500 --jobs 6
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
    run_narrow_swarm,
)
from cypha_bench.tuning.validate_top_k import validate_top_k  # noqa: E402

OUT_DIR = _REPO / "cypha_bench" / "artifacts" / "tuning"
PROFILE_PATH = _REPO / "cypha_bench" / "config" / "everyday_profile.json"
PYTHON = sys.executable


def _tabular_cls_score(row: dict[str, Any]) -> float:
    scores = row.get("scores") or {}
    acc_keys = [k for k in scores if k.startswith("cls_") and k.endswith("_acc")]
    if not acc_keys:
        return -1.0
    return sum(float(scores[k]) for k in acc_keys) / len(acc_keys)


def _merge_regime_profile(
    round2_leaderboard: Path,
    winner_profile: dict[str, Any],
    validation_winner: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = json.loads(round2_leaderboard.read_text(encoding="utf-8"))
    rows = data.get("top_k_candidates") or data.get("top10") or data.get("all_combos") or []
    if not rows:
        rows = [data.get("top10", [{}])[0]] if data.get("top10") else []

    best_cls = max(rows, key=_tabular_cls_score)
    best_mnist = max(rows, key=lambda r: float((r.get("scores") or {}).get("mnist_hog_acc", -1.0)))
    best_reg = max(rows, key=lambda r: float((r.get("scores") or {}).get("reg_california_r2", -999.0)))

    cls_profile = dict(
        best_cls.get("profile", {}).get("classification_cyphadif")
        or winner_profile.get("classification_cyphadif", {})
    )
    vision_profile = dict(
        best_mnist.get("profile", {}).get("classification_cyphadif") or cls_profile
    )
    reg_profile = dict(
        best_reg.get("profile", {}).get("regression_difregressor")
        or winner_profile.get("regression_difregressor", {})
    )
    arch = dict(
        winner_profile.get("architecture") or best_cls.get("profile", {}).get("architecture", {})
    )
    arch.setdefault("replay_ratio", 0.15)
    arch.setdefault("ood_sigma", 15.0)
    mnist_arch = best_mnist.get("profile", {}).get("architecture") or {}
    if "replay_ratio" in mnist_arch:
        vision_profile["replay_ratio"] = mnist_arch["replay_ratio"]
    if "ood_sigma" in mnist_arch:
        vision_profile["ood_sigma"] = mnist_arch["ood_sigma"]

    merged = {
        "source": "cypha_bench/tuning/bench_iterate_swarm.py regime merge round 3",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "default_regime": "tabular",
        "regimes": {
            "tabular": cls_profile,
            "vision": vision_profile,
            "regression": reg_profile,
        },
        "architecture": arch,
        "regime_merge_sources": {
            "tabular_cls_combo_id": best_cls.get("combo_id"),
            "tabular_cls_score": _tabular_cls_score(best_cls),
            "mnist_hog_combo_id": best_mnist.get("combo_id"),
            "mnist_hog_acc": float((best_mnist.get("scores") or {}).get("mnist_hog_acc", 0.0)),
            "california_r2_combo_id": best_reg.get("combo_id"),
            "reg_california_r2": float((best_reg.get("scores") or {}).get("reg_california_r2", 0.0)),
        },
    }
    if validation_winner:
        merged["validation_winner"] = {
            "combo_id": validation_winner.get("combo_id"),
            "validation_score": validation_winner.get("validation_score"),
        }
    return merged


def _run_subprocess(script: str, *args: str) -> None:
    cmd = [PYTHON, str(_REPO / script), *args]
    print(f"Running: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env.setdefault("CYPHA_BENCH_FAST", "1")
    subprocess.run(cmd, cwd=_REPO, env=env, check=True)


def run_iterate(
    rounds: int = 3,
    combos_round1: int = 500,
    combos_round2: int = 300,
    validate_top_round1: int = 20,
    validate_top_round2: int = 12,
    jobs: int = 6,
    seed: int = 42,
    gpu_burn: int = 16,
    skip_final_validation: bool = False,
    outer_loops: int = 1,
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "rounds": rounds,
        "combos_round1": combos_round1,
        "combos_round2": combos_round2,
    }

    print("=== Round 1: full swarm (minmax) ===", flush=True)
    r1 = run_full_swarm(
        combos=combos_round1,
        jobs=jobs,
        seed=seed,
        gpu_burn=gpu_burn,
        objective="minmax",
        top_k=max(validate_top_round1, 30),
        write_profile=False,
    )
    log["round1_leaderboard"] = str(r1["leaderboard_path"])

    print(f"=== Validate top {validate_top_round1} from round 1 (hybrid) ===", flush=True)
    v1 = validate_top_k(
        top=validate_top_round1,
        leaderboard=r1["leaderboard_path"],
        mode="hybrid",
        full_top=3,
    )
    winner1 = v1["winner"]
    log["round1_validation"] = str(v1["out_path"])
    log["round1_winner_combo_id"] = winner1["combo_id"]

    round2_leaderboard = r1["leaderboard_path"]
    if rounds >= 2:
        print(f"=== Round 2: narrow sweep ({combos_round2} combos) ===", flush=True)
        r2 = run_narrow_swarm(
            center_profile=winner1["profile"],
            combos=combos_round2,
            jobs=jobs,
            seed=seed + 1,
            gpu_burn=gpu_burn,
            objective="minmax",
            top_k=max(validate_top_round2, 30),
            write_profile=False,
        )
        round2_leaderboard = r2["leaderboard_path"]
        log["round2_leaderboard"] = str(round2_leaderboard)

        print(f"=== Validate top {validate_top_round2} from round 2 (hybrid) ===", flush=True)
        v2 = validate_top_k(
            top=validate_top_round2,
            leaderboard=round2_leaderboard,
            mode="hybrid",
            full_top=3,
        )
        winner2 = v2["winner"]
        log["round2_validation"] = str(v2["out_path"])
        log["round2_winner_combo_id"] = winner2["combo_id"]
        center_profile = winner2["profile"]
        validation_winner = winner2
    else:
        center_profile = winner1["profile"]
        validation_winner = winner1

    if rounds >= 3:
        print("=== Round 3: merge regime-specific profiles ===", flush=True)
        merged = _merge_regime_profile(round2_leaderboard, center_profile, validation_winner)
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        log["merged_profile"] = str(PROFILE_PATH)
        print(f"Wrote merged profile: {PROFILE_PATH}", flush=True)

    for loop_i in range(1, outer_loops):
        if rounds < 3 or not PROFILE_PATH.exists():
            break
        print(f"=== Outer loop {loop_i + 1}/{outer_loops}: narrow from merged profile ===", flush=True)
        merged_profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        flat_center = {
            "classification_cyphadif": merged_profile.get("regimes", {}).get("tabular")
            or merged_profile.get("classification_cyphadif", {}),
            "regression_difregressor": merged_profile.get("regimes", {}).get("regression")
            or merged_profile.get("regression_difregressor", {}),
            "architecture": merged_profile.get("architecture", {}),
        }
        r_extra = run_narrow_swarm(
            center_profile=flat_center,
            combos=max(combos_round2 // 2, 150),
            jobs=jobs,
            seed=seed + 100 + loop_i,
            gpu_burn=gpu_burn,
            objective="minmax",
            top_k=20,
            write_profile=False,
        )
        v_extra = validate_top_k(top=12, leaderboard=r_extra["leaderboard_path"], mode="hybrid", full_top=3)
        round2_leaderboard = r_extra["leaderboard_path"]
        merged = _merge_regime_profile(round2_leaderboard, v_extra["winner"]["profile"], v_extra["winner"])
        PROFILE_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        log[f"outer_loop_{loop_i + 1}_leaderboard"] = str(r_extra["leaderboard_path"])

    if rounds >= 3 and outer_loops >= 1:
        pass  # merged already written above
    elif rounds < 3:
        profile_out = {
            "source": "cypha_bench/tuning/bench_iterate_swarm.py validation winner",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            **center_profile,
            "validation_score": validation_winner.get("validation_score"),
            "combo_id": validation_winner.get("combo_id"),
        }
        PROFILE_PATH.write_text(json.dumps(profile_out, indent=2), encoding="utf-8")
        log["merged_profile"] = str(PROFILE_PATH)

    if not skip_final_validation:
        print("=== Final validation (validate_profile) ===", flush=True)
        _run_subprocess("cypha_bench/tuning/validate_profile.py")
        print("=== Write report ===", flush=True)
        _run_subprocess("cypha_bench/tuning/write_report.py")

    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"bench_iterate_{stamp}.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Orchestration log: {log_path}", flush=True)
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-round bench swarm orchestrator")
    parser.add_argument("--rounds", type=int, default=3, help="1=swarm only, 2=+narrow, 3=+regime merge")
    parser.add_argument("--combos-round1", type=int, default=int(os.environ.get("BENCH_SWARM_COMBOS", "500")))
    parser.add_argument("--combos-round2", type=int, default=300)
    parser.add_argument("--validate-top-round1", type=int, default=20)
    parser.add_argument("--validate-top-round2", type=int, default=12)
    parser.add_argument("--outer-loops", type=int, default=2, help="Repeat narrow+merge+validate from merged profile")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("BENCH_SWARM_JOBS", max(1, (os.cpu_count() or 4) - 1))))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-burn", type=int, default=16)
    parser.add_argument("--skip-final-validation", action="store_true")
    args = parser.parse_args()
    run_iterate(
        rounds=args.rounds,
        combos_round1=args.combos_round1,
        combos_round2=args.combos_round2,
        validate_top_round1=args.validate_top_round1,
        validate_top_round2=args.validate_top_round2,
        jobs=args.jobs,
        seed=args.seed,
        gpu_burn=args.gpu_burn,
        skip_final_validation=args.skip_final_validation,
        outer_loops=args.outer_loops,
    )


if __name__ == "__main__":
    main()
