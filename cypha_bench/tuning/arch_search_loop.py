#!/usr/bin/env python3
"""
Iterative architecture search: narrow swarms -> rescore -> validate -> repeat.

Starts from everyday_profile.json (combo #7 rescore winner). Uses 10 parallel
workers by default and CUDA when available.

Usage:
  python cypha_bench/tuning/arch_search_loop.py --rounds 5 --combos 200 --jobs 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.tuning.arch_iterate import merge_winner_profile, write_winning_profile  # noqa: E402
from cypha_bench.tuning.arch_swarm import run_arch_swarm, run_narrow_arch_swarm  # noqa: E402
from cypha_bench.tuning.bench_metrics import ensure_swarm_baseline, load_swarm_baseline_metrics, score_vs_baseline  # noqa: E402
from cypha_bench.tuning.rescore_arch_swarm import rescore  # noqa: E402

OUT_DIR = _REPO / "cypha_bench" / "artifacts" / "tuning"
PROFILE_PATH = _REPO / "cypha_bench" / "config" / "everyday_profile.json"
PYTHON = sys.executable
LOG_PATH = OUT_DIR / "arch_search_loop.log"


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_center() -> dict[str, Any]:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _fast_validate(profile: dict[str, Any]) -> float:
    """Score profile on d02+d03+d08 via validate_profile micro path."""
    import tempfile

    baseline = load_swarm_baseline_metrics()
    with tempfile.TemporaryDirectory(prefix="cypha_fast_val_") as tmp:
        p = Path(tmp) / "candidate.json"
        p.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        env = os.environ.copy()
        env["CYPHA_BENCH_PROFILE_PATH"] = str(p)
        env["CYPHA_BENCH_USE_PROFILE"] = "1"
        env["CYPHA_BENCH_FAST"] = "1"
        env.pop("CYPHA_BENCH_PROFILE_JSON", None)
        from cypha_bench.tuning.bench_metrics import SWARM_DOMAINS, collect_run_metrics, filter_metrics_for_swarm

        side: dict[str, dict] = {}
        for key in ("d02", "d03", "d08"):
            mod_path = SWARM_DOMAINS[key][0]
            mod = __import__(mod_path, fromlist=["run"])
            side[key] = mod.run()
        tuned = filter_metrics_for_swarm(collect_run_metrics(domain_results=side))
        score, _, _ = score_vs_baseline(tuned, baseline, aggregate="mean", scorable_only=True)
        return float(score)


def run_search_loop(
    *,
    rounds: int = 5,
    combos: int = 200,
    jobs: int = 10,
    seed: int = 42,
    min_improve: float = 0.005,
) -> dict[str, Any]:
    ensure_swarm_baseline()
    center = _load_center()
    best_score = _fast_validate(center)
    _log(f"Start center score={best_score:.4f}")

    log: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "rounds": rounds,
        "combos": combos,
        "jobs": jobs,
        "round_results": [],
        "best_score": best_score,
    }

    for rnd in range(1, rounds + 1):
        _log(f"=== Round {rnd}/{rounds} ===")
        try:
            if rnd == 1:
                swarm = run_arch_swarm(
                    base_profile=center,
                    combos=combos,
                    jobs=jobs,
                    seed=seed + rnd,
                    top_k=30,
                )
            else:
                swarm = run_narrow_arch_swarm(
                    center_profile=center,
                    combos=max(combos // 2, 80),
                    jobs=jobs,
                    seed=seed + rnd,
                    top_k=20,
                )
        except Exception as exc:
            _log(f"Round {rnd} swarm FAILED: {exc}")
            log["round_results"].append({"round": rnd, "error": str(exc)})
            continue

        res = rescore(apply_best=False)
        best = res.get("best")
        if not best or not isinstance(best.get("profile"), dict):
            _log(f"Round {rnd} rescore returned no candidate")
            continue

        candidate = best["profile"]
        rescore_score = float(best.get("new_score", 0.0))
        fast_score = _fast_validate(candidate)
        _log(
            f"Round {rnd} rescore={rescore_score:.4f} fast_d020308={fast_score:.4f} "
            f"combo_id={best.get('combo_id')} source={best.get('source')}"
        )

        improved = fast_score > best_score + min_improve
        if improved:
            best_score = fast_score
            center = merge_winner_profile(center, candidate)
            write_winning_profile(center)
            _log(f"Round {rnd} NEW BEST fast_score={best_score:.4f} — profile updated")
        else:
            _log(f"Round {rnd} no improvement (best stays {best_score:.4f})")

        log["round_results"].append(
            {
                "round": rnd,
                "leaderboard": str(swarm.get("leaderboard_path")),
                "rescore_score": rescore_score,
                "fast_score": fast_score,
                "improved": improved,
                "combo_id": best.get("combo_id"),
            }
        )
        log["best_score"] = best_score

    _log("=== Final rescore + validate_profile ===")
    rescore(apply_best=False)
    write_winning_profile(center)
    subprocess.run(
        [PYTHON, str(_REPO / "cypha_bench" / "tuning" / "validate_profile.py")],
        cwd=_REPO,
        env={**os.environ, "CYPHA_BENCH_USE_PROFILE": "1", "CYPHA_BENCH_FAST": "0"},
        check=False,
    )

    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"arch_search_loop_{stamp}.json"
    out.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    _log(f"Done. Log: {out}")
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative arch search loop")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--combos", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("ARCH_SWARM_JOBS", "10")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-improve", type=float, default=0.005, dest="min_improve")
    args = parser.parse_args()
    run_search_loop(
        rounds=args.rounds,
        combos=args.combos,
        jobs=args.jobs,
        seed=args.seed,
        min_improve=args.min_improve,
    )


if __name__ == "__main__":
    main()
