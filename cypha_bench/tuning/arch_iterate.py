#!/usr/bin/env python3
"""
Architecture tuning orchestrator for Cypha bench profiles.

Iterates arch_swarm sweeps, validates top candidates on the full bench
(d01–d17 + cross-domain), and converges when min-ratio score reaches target.

Usage:
  python cypha_bench/tuning/arch_iterate.py
  python cypha_bench/tuning/arch_iterate.py --max-iter 5 --swarm-size 400 --jobs 6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.tuning.arch_swarm import run_arch_swarm, run_narrow_arch_swarm  # noqa: E402
from cypha_bench.tuning.bench_metrics import (  # noqa: E402
    BASELINE_PATH,
    SWARM_DOMAINS,
    collect_run_metrics,
    ensure_swarm_baseline,
    filter_metrics_for_swarm,
    load_baseline_metrics,
    load_swarm_baseline_metrics,
    save_baseline_snapshot,
    score_vs_baseline,
)

OUT_DIR = _REPO / "cypha_bench" / "artifacts" / "tuning"
PROFILE_PATH = _REPO / "cypha_bench" / "config" / "everyday_profile.json"
REPORT_PATH = _REPO / "cypha_bench" / "ARCH_TUNING_REPORT.md"
PYTHON = sys.executable

DEFAULT_TARGET_SCORE = 0.92
DEFAULT_NARROW_SIZE = 200
FULL_VALIDATE_TOP = 3


def ensure_baseline(*, fast: bool = True, full: bool = True) -> Path:
    """Ensure baseline_metrics.json exists; use full run_all snapshot when *full* is True."""
    if BASELINE_PATH.exists():
        data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        n = len(data.get("metrics", {}))
        if not full or n >= 80:
            return BASELINE_PATH
        print(f"Baseline has only {n} keys — regenerating full snapshot...", flush=True)
    return save_baseline_snapshot(fast=fast, domains=None if full else list(SWARM_DOMAINS.keys()))


def load_current_profile() -> dict[str, Any]:
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"Everyday profile not found: {PROFILE_PATH}")
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


@contextmanager
def profile_env(profile_path: Path) -> Iterator[None]:
    saved = {
        "CYPHA_BENCH_PROFILE_PATH": os.environ.get("CYPHA_BENCH_PROFILE_PATH"),
        "CYPHA_BENCH_PROFILE_JSON": os.environ.get("CYPHA_BENCH_PROFILE_JSON"),
        "CYPHA_BENCH_USE_PROFILE": os.environ.get("CYPHA_BENCH_USE_PROFILE"),
        "CYPHA_BENCH_FAST": os.environ.get("CYPHA_BENCH_FAST"),
    }
    os.environ["CYPHA_BENCH_PROFILE_PATH"] = str(profile_path)
    os.environ.pop("CYPHA_BENCH_PROFILE_JSON", None)
    os.environ["CYPHA_BENCH_USE_PROFILE"] = "1"
    os.environ["CYPHA_BENCH_FAST"] = "1"
    try:
        yield
    finally:
        for key, val in saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def _run_subprocess(script: str, *args: str, env_overrides: dict[str, str | None] | None = None) -> None:
    cmd = [PYTHON, str(_REPO / script), *args]
    print(f"Running: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    if env_overrides:
        for key, val in env_overrides.items():
            if val is None:
                env.pop(key, None)
            else:
                env[key] = val
    subprocess.run(cmd, cwd=_REPO, env=env, check=True)


def run_full_bench_with_profile(profile: dict[str, Any]) -> dict[str, tuple[str, float]]:
    """Run full cypha_bench/run_all.py with a candidate profile in FAST mode."""
    with tempfile.TemporaryDirectory(prefix="cypha_arch_iterate_") as tmp:
        profile_path = Path(tmp) / "candidate_profile.json"
        profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        with profile_env(profile_path):
            _run_subprocess("cypha_bench/run_all.py")
        return collect_run_metrics(from_tables=True)


def merge_winner_profile(current: dict[str, Any], winner: dict[str, Any]) -> dict[str, Any]:
    """Merge tuned architecture / algorithm fields into the current everyday profile."""
    merged = deepcopy(current)
    for key in (
        "regimes",
        "architecture",
        "algorithm_variants",
        "default_regime",
        "classification_cyphadif",
        "regression_difregressor",
    ):
        if key in winner:
            merged[key] = deepcopy(winner[key])
    merged["source"] = "cypha_bench/tuning/arch_iterate.py"
    merged["generated_utc"] = datetime.now(timezone.utc).isoformat()
    return merged


def validate_top_candidates(
    candidates: list[dict[str, Any]],
    baseline: dict[str, tuple[str, float]],
    *,
    top_n: int = FULL_VALIDATE_TOP,
) -> dict[str, Any]:
    """Run full bench for each top candidate and return the best scored row."""
    ranked: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates[:top_n]):
        combo_id = candidate.get("combo_id", idx)
        profile = candidate.get("profile")
        if not isinstance(profile, dict):
            continue
        print(f"Full bench validate {idx + 1}/{min(top_n, len(candidates))} combo_id={combo_id}", flush=True)
        run_full_bench_with_profile(profile)
        tuned_metrics = filter_metrics_for_swarm(
            collect_run_metrics(from_tables=True)
        )
        score, ratios, deltas = score_vs_baseline(
            tuned_metrics, baseline, aggregate="mean", scorable_only=True
        )
        ranked.append(
            {
                "combo_id": combo_id,
                "validation_score": score,
                "normalized_ratios": ratios,
                "deltas": deltas,
                "profile": profile,
                "metrics": {k: {"metric": m, "value": v} for k, (m, v) in tuned_metrics.items()},
            }
        )

    if not ranked:
        raise RuntimeError("No valid candidates to validate on full bench")

    ranked.sort(key=lambda row: row["validation_score"], reverse=True)
    return {"winner": ranked[0], "ranked": ranked}


def write_winning_profile(profile: dict[str, Any]) -> Path:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Wrote winning profile: {PROFILE_PATH}", flush=True)
    return PROFILE_PATH


def write_arch_tuning_report(log: dict[str, Any], *, winner: dict[str, Any] | None = None) -> Path:
    lines = [
        "# Cypha Bench Architecture Tuning Report",
        "",
        f"Generated: {log.get('finished_utc') or datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Target score: {log.get('target_score', DEFAULT_TARGET_SCORE):.2f}",
        f"- Success: {log.get('success', False)}",
        f"- Iterations run: {len(log.get('iterations', []))}",
        f"- Final score: {log.get('final_score')}",
        "",
    ]

    if winner:
        lines.extend(
            [
                "## Winning profile",
                "",
                "```json",
                json.dumps(
                    {
                        "architecture": winner.get("profile", {}).get("architecture"),
                        "algorithm_variants": winner.get("profile", {}).get("algorithm_variants"),
                        "combo_id": winner.get("combo_id"),
                        "validation_score": winner.get("validation_score"),
                    },
                    indent=2,
                ),
                "```",
                "",
            ]
        )

    lines.append("## Iterations")
    lines.append("")
    for item in log.get("iterations", []):
        lines.append(
            f"- Iteration {item.get('iteration')}: swarm={item.get('swarm_mode')} "
            f"best_validation={item.get('best_validation_score')} "
            f"converged={item.get('converged', False)}"
        )
        unfavorable = [
            d for d in item.get("deltas", [])
            if isinstance(d, dict) and not d.get("favorable")
        ]
        if not unfavorable and isinstance(item.get("deltas"), dict):
            unfavorable = [
                k for k, v in item["deltas"].items()
                if isinstance(v, (int, float)) and v < 0
            ]
        if unfavorable:
            lines.append(f"  - Unfavorable metrics: {len(unfavorable)}")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {REPORT_PATH}", flush=True)
    return REPORT_PATH


def run_final_validate(*, fast: bool = False) -> None:
    env_overrides = {
        "CYPHA_BENCH_USE_PROFILE": "1",
        "CYPHA_BENCH_FAST": "1" if fast else "0",
    }
    _run_subprocess("cypha_bench/tuning/validate_profile.py", env_overrides=env_overrides)


def run_arch_iterate(
    max_iter: int = 5,
    swarm_size: int = 400,
    narrow_size: int = DEFAULT_NARROW_SIZE,
    jobs: int = 6,
    seed: int = 42,
    target_score: float = DEFAULT_TARGET_SCORE,
    validate_top: int = FULL_VALIDATE_TOP,
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "max_iter": max_iter,
        "swarm_size": swarm_size,
        "narrow_size": narrow_size,
        "jobs": jobs,
        "target_score": target_score,
        "iterations": [],
    }

    ensure_baseline(fast=True, full=True)
    ensure_swarm_baseline()
    baseline = load_swarm_baseline_metrics()
    center_profile = load_current_profile()

    success = False
    final_winner: dict[str, Any] | None = None
    final_score = 0.0

    for iteration in range(1, max_iter + 1):
        print(f"=== Arch iterate {iteration}/{max_iter} ===", flush=True)
        iter_log: dict[str, Any] = {"iteration": iteration}

        if iteration == 1:
            iter_log["swarm_mode"] = "full"
            swarm_result = run_arch_swarm(
                base_profile=center_profile,
                combos=swarm_size,
                jobs=jobs,
                seed=seed + iteration,
                top_k=max(validate_top, 10),
            )
        else:
            iter_log["swarm_mode"] = "narrow"
            swarm_result = run_narrow_arch_swarm(
                center_profile=center_profile,
                combos=narrow_size,
                jobs=jobs,
                seed=seed + iteration,
                top_k=max(validate_top, 10),
            )

        iter_log["leaderboard_path"] = str(swarm_result["leaderboard_path"])
        top_candidates = swarm_result["top_k_candidates"][:validate_top]

        validation = validate_top_candidates(top_candidates, baseline, top_n=validate_top)
        winner = validation["winner"]
        final_winner = winner
        final_score = float(winner["validation_score"])
        center_profile = merge_winner_profile(center_profile, winner["profile"])

        iter_log["best_validation_score"] = final_score
        iter_log["winner_combo_id"] = winner.get("combo_id")
        iter_log["deltas"] = winner.get("deltas", [])
        iter_log["converged"] = final_score >= target_score
        log["iterations"].append(iter_log)

        print(f"Iteration {iteration} best score={final_score:.4f} (target={target_score:.2f})", flush=True)
        if final_score >= target_score:
            success = True
            break

    log["success"] = success
    log["final_score"] = final_score
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()

    if final_winner is None:
        final_winner = {"profile": center_profile, "validation_score": final_score, "combo_id": None}

    write_winning_profile(center_profile)
    print("=== Final validation (full mode) ===", flush=True)
    run_final_validate(fast=False)
    write_arch_tuning_report(log, winner=final_winner)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"arch_iterate_{stamp}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    log["log_path"] = str(log_path)
    print(f"Orchestration log: {log_path}", flush=True)
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture tuning orchestrator loop")
    parser.add_argument("--max-iter", type=int, default=5, dest="max_iter")
    parser.add_argument("--swarm-size", type=int, default=400, dest="swarm_size")
    parser.add_argument("--narrow-size", type=int, default=DEFAULT_NARROW_SIZE, dest="narrow_size")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("BENCH_SWARM_JOBS", "6")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE, dest="target_score")
    parser.add_argument("--validate-top", type=int, default=FULL_VALIDATE_TOP, dest="validate_top")
    args = parser.parse_args()

    result = run_arch_iterate(
        max_iter=args.max_iter,
        swarm_size=args.swarm_size,
        narrow_size=args.narrow_size,
        jobs=args.jobs,
        seed=args.seed,
        target_score=args.target_score,
        validate_top=args.validate_top,
    )
    print(result["log_path"])


if __name__ == "__main__":
    main()
