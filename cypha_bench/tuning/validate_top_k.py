#!/usr/bin/env python3
"""
Fast validation of top swarm candidates on bench domains d01/d02/d03/d08.

Loads top_k_candidates (or top10) from a bench_swarm_*.json leaderboard,
runs each profile with CYPHA_BENCH_FAST=1 via CYPHA_BENCH_PROFILE_PATH,
scores by min normalized metric vs validation_compare baseline, picks winner.

Usage:
  python cypha_bench/tuning/validate_top_k.py --top 25
  python cypha_bench/tuning/validate_top_k.py --top 15 --leaderboard path/to/bench_swarm.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.tuning.bench_full_swarm import (  # noqa: E402
    _cache_datasets,
    _eval_profile,
    _init_worker,
    _minmax_aggregate,
    _passes_floors,
)
from cypha_bench.tuning.validate_profile import DOMAINS, _extract_rows  # noqa: E402

OUT_DIR = _REPO / "cypha_bench" / "artifacts" / "tuning"
BASELINE_PATH = OUT_DIR / "validation_compare.json"
EVERYDAY_PROFILE = _REPO / "cypha_bench" / "config" / "everyday_profile.json"


def _metric_from_row(row: dict) -> tuple[str, str, float | None]:
    """Return (row_label, metric_name, value)."""
    name = str(row.get("dataset") or row.get("task") or row.get("name") or row.get("encoding") or "unknown")
    scores = row.get("cypha_scores") or row.get("scores") or {}
    if "accuracy" in scores:
        return name, "accuracy", float(scores["accuracy"])
    if "rmse" in scores:
        return name, "rmse", float(scores["rmse"])
    if "r2" in scores:
        return name, "r2", float(scores["r2"])
    return name, "unknown", None


def _collect_domain_metrics(domain_key: str, domain_result: dict, list_key: str) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for row in _extract_rows(domain_result, list_key):
        label, metric, value = _metric_from_row(row)
        if value is None or metric == "unknown":
            continue
        out[f"{domain_key}/{label}"] = (metric, value)
    return out


def _collect_all_metrics(side: dict) -> dict[str, tuple[str, float]]:
    metrics: dict[str, tuple[str, float]] = {}
    for key, (_, list_key) in DOMAINS.items():
        block = side.get(key, {})
        metrics.update(_collect_domain_metrics(key, block, list_key))
    return metrics


def _normalize_ratio(metric: str, candidate: float, baseline: float) -> float:
    if metric in ("accuracy", "r2"):
        if baseline > 1e-12:
            return candidate / baseline
        return 1.0 if candidate >= baseline else 0.0
    if metric == "rmse":
        if candidate <= 1e-12:
            return 0.0
        if baseline <= 1e-12:
            return 1.0 if candidate <= baseline else 0.0
        return baseline / candidate
    return 1.0


def _min_score_vs_baseline(
    tuned: dict[str, tuple[str, float]],
    baseline: dict[str, tuple[str, float]],
) -> tuple[float, dict[str, float]]:
    ratios: dict[str, float] = {}
    for key, (metric, base_val) in baseline.items():
        if key not in tuned:
            continue
        _, cand_val = tuned[key]
        ratios[key] = _normalize_ratio(metric, cand_val, base_val)
    if not ratios:
        return 0.0, ratios
    return min(ratios.values()), ratios


def _find_leaderboard(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Leaderboard not found: {path}")
        return path
    matches = sorted(OUT_DIR.glob("bench_swarm*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No bench_swarm*.json under {OUT_DIR}")
    return matches[0]


def _load_candidates(leaderboard: Path, top: int) -> list[dict[str, Any]]:
    data = json.loads(leaderboard.read_text(encoding="utf-8"))
    rows = data.get("top_k_candidates") or data.get("top10")
    if not rows:
        rows = data.get("all_combos") or []
    if not rows:
        raise ValueError(f"No candidates in {leaderboard}")
    return rows[:top]


def _profile_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile")
    if isinstance(profile, dict) and profile:
        return profile
    keys = ("classification_cyphadif", "regression_difregressor", "architecture")
    extracted = {k: candidate[k] for k in keys if k in candidate}
    if extracted:
        return extracted
    raise ValueError("Candidate row has no profile block")


@contextmanager
def _profile_env(profile_path: Path) -> Iterator[None]:
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


def _run_domains() -> dict[str, dict]:
    results: dict[str, dict] = {}
    for key, (mod_path, _) in DOMAINS.items():
        print(f"  Running {key}...", flush=True)
        mod = __import__(mod_path, fromlist=["run"])
        results[key] = mod.run()
    return results


def _micro_score_profiles(
    candidates: list[dict[str, Any]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Fast micro-suite scoring (~3s per profile)."""
    _cache_datasets(800, 300, 3000)
    micro_rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        profile = _profile_payload(candidate)
        raw = _eval_profile(profile, seed=seed + idx)
        micro_rows.append({**candidate, "micro_scores": raw, "passes_floors": _passes_floors(raw)})
    bounds: dict[str, tuple[float, float]] = {}
    for row in micro_rows:
        for key, val in row["micro_scores"].items():
            bounds.setdefault(key, (val, val))
            lo, hi = bounds[key]
            bounds[key] = (min(lo, val), max(hi, val))
    for row in micro_rows:
        row["micro_score"] = (
            _minmax_aggregate(row["micro_scores"], bounds) if row["passes_floors"] else -1e9
        )
    return micro_rows


def validate_top_k(
    top: int = 25,
    leaderboard: Path | None = None,
    baseline_path: Path | None = None,
    write_winner: bool = False,
    mode: str = "hybrid",
    full_top: int = 3,
) -> dict[str, Any]:
    leaderboard_path = _find_leaderboard(leaderboard)
    candidates = _load_candidates(leaderboard_path, top)
    baseline_file = baseline_path or BASELINE_PATH

    baseline_metrics: dict[str, tuple[str, float]] = {}
    if baseline_file.exists():
        baseline_data = json.loads(baseline_file.read_text(encoding="utf-8"))
        baseline_metrics = _collect_all_metrics(baseline_data.get("baseline", {}))
        print(f"Loaded baseline from {baseline_file} ({len(baseline_metrics)} metrics)", flush=True)
    else:
        print(f"No baseline at {baseline_file}; scoring by raw accuracy/r2 only", flush=True)

    if mode in ("micro", "hybrid"):
        print(f"Micro-suite pre-score on {len(candidates)} candidates...", flush=True)
        micro_ranked = _micro_score_profiles(candidates)
        micro_ranked.sort(key=lambda r: r["micro_score"], reverse=True)
        if mode == "micro":
            winner_row = micro_ranked[0]
            winner = {
                "combo_id": winner_row.get("combo_id"),
                "validation_score": winner_row["micro_score"],
                "profile": _profile_payload(winner_row),
                "micro_scores": winner_row.get("micro_scores"),
            }
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = OUT_DIR / f"validate_top_k_{stamp}.json"
            out_path.write_text(
                json.dumps(
                    {
                        "mode": "micro",
                        "winner_combo_id": winner["combo_id"],
                        "winner_validation_score": winner["validation_score"],
                        "ranked": [
                            {"combo_id": r.get("combo_id"), "micro_score": r["micro_score"]}
                            for r in micro_ranked[:10]
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return {"out_path": out_path, "winner": winner, "ranked": micro_ranked, "leaderboard_path": leaderboard_path}
        candidates = micro_ranked[: max(full_top, 1)]
        print(f"Hybrid: full bench validate top {len(candidates)} after micro filter", flush=True)

    ranked: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cypha_validate_top_k_") as tmp:
        tmp_dir = Path(tmp)
        for idx, candidate in enumerate(candidates):
            combo_id = candidate.get("combo_id", idx)
            profile = _profile_payload(candidate)
            profile_path = tmp_dir / f"candidate_{combo_id}.json"
            profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

            print(f"Candidate {idx + 1}/{len(candidates)} combo_id={combo_id}", flush=True)
            with _profile_env(profile_path):
                domain_results = _run_domains()

            tuned_metrics = _collect_all_metrics(domain_results)
            if baseline_metrics:
                score, ratios = _min_score_vs_baseline(tuned_metrics, baseline_metrics)
            else:
                acc_vals = [v for m, v in tuned_metrics.values() if m == "accuracy"]
                r2_vals = [v for m, v in tuned_metrics.values() if m == "r2"]
                rmse_vals = [v for m, v in tuned_metrics.values() if m == "rmse"]
                score = 0.0
                if acc_vals:
                    score += sum(acc_vals) / len(acc_vals)
                if r2_vals:
                    score += sum(r2_vals) / len(r2_vals)
                if rmse_vals:
                    score += sum(1.0 / (1.0 + r) for r in rmse_vals) / len(rmse_vals)
                ratios = {}

            ranked.append(
                {
                    "rank_input": idx,
                    "combo_id": combo_id,
                    "swarm_composite": candidate.get("composite"),
                    "validation_score": score,
                    "normalized_ratios": ratios,
                    "metrics": {k: {"metric": m, "value": v} for k, (m, v) in tuned_metrics.items()},
                    "profile": profile,
                    "domain_results": domain_results,
                }
            )

    ranked.sort(key=lambda r: r["validation_score"], reverse=True)
    winner = ranked[0]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"validate_top_k_{stamp}.json"
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "leaderboard_path": str(leaderboard_path),
        "baseline_path": str(baseline_file) if baseline_file.exists() else None,
        "top_requested": top,
        "winner_combo_id": winner["combo_id"],
        "winner_validation_score": winner["validation_score"],
        "winner_profile": winner["profile"],
        "ranked": [
            {
                "combo_id": r["combo_id"],
                "validation_score": r["validation_score"],
                "swarm_composite": r["swarm_composite"],
                "normalized_ratios": r["normalized_ratios"],
            }
            for r in ranked
        ],
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if write_winner:
        profile_out = {
            "source": f"cypha_bench/tuning/validate_top_k.py from {leaderboard_path.name}",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "validation_score": winner["validation_score"],
            "combo_id": winner["combo_id"],
            **winner["profile"],
        }
        EVERYDAY_PROFILE.parent.mkdir(parents=True, exist_ok=True)
        EVERYDAY_PROFILE.write_text(json.dumps(profile_out, indent=2), encoding="utf-8")
        print(f"Wrote winner profile to {EVERYDAY_PROFILE}", flush=True)

    print(f"Winner combo_id={winner['combo_id']} score={winner['validation_score']:.4f}", flush=True)
    print(f"Wrote {out_path}", flush=True)
    return {"out_path": out_path, "winner": winner, "ranked": ranked, "leaderboard_path": leaderboard_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast-validate top swarm candidates")
    parser.add_argument("--top", type=int, default=25, help="Number of candidates to validate")
    parser.add_argument("--leaderboard", type=Path, default=None, help="bench_swarm JSON (default: latest)")
    parser.add_argument("--baseline", type=Path, default=None, help="validation_compare.json for baseline")
    parser.add_argument("--write-winner", action="store_true", help="Write winner to everyday_profile.json")
    parser.add_argument("--mode", choices=("full", "micro", "hybrid"), default="hybrid")
    parser.add_argument("--full-top", type=int, default=3, help="In hybrid mode, full-validate this many micro winners")
    args = parser.parse_args()
    validate_top_k(
        top=args.top,
        leaderboard=args.leaderboard,
        baseline_path=args.baseline,
        write_winner=args.write_winner,
        mode=args.mode,
        full_top=args.full_top,
    )


if __name__ == "__main__":
    main()
