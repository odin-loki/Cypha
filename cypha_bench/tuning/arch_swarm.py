#!/usr/bin/env python3
"""
Parallel architecture / algorithm-variant swarm for Cypha bench profiles.

Samples variants from everyday_profile regimes plus algorithm_variants, scores
each candidate with bench_metrics min-ratio vs baseline on fast domain evals
(d01, d02, d03, d08, d12, d16) via CYPHA_BENCH_PROFILE_PATH.

Usage:
  python cypha_bench/tuning/bench_metrics.py  # optional: save_baseline_snapshot first
  python cypha_bench/tuning/arch_swarm.py --combos 400 --jobs 6 --save-baseline
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.config.algorithm_variants import (  # noqa: E402
    VARIANT_KEYS,
    apply_algorithm_variants,
    merge_algorithm_variants,
)
from cypha_bench.config.load_profile import load_profile  # noqa: E402
from cypha_bench.tuning.bench_full_swarm import (  # noqa: E402
    ParamRanges,
    _flat_center_for_regime,
    sample_profile_narrow,
)
from cypha_bench.tuning.bench_metrics import (  # noqa: E402
    BASELINE_PATH,
    SWARM_BASELINE_PATH,
    SWARM_DOMAINS,
    collect_run_metrics,
    ensure_swarm_baseline,
    filter_metrics_for_swarm,
    load_swarm_baseline_metrics,
    save_baseline_snapshot,
    score_vs_baseline,
)

try:
    from cypha_accel.cuda_util import cuda_gemm_usable, warmup_cuda  # noqa: WPS433
except ImportError:
    def warmup_cuda() -> None:
        return None

    def cuda_gemm_usable() -> bool:
        return False

try:
    from joblib import Parallel, delayed  # noqa: WPS433
except ImportError:
    Parallel = None  # type: ignore
    delayed = None  # type: ignore

OUT_DIR = _REPO / "cypha_bench" / "artifacts" / "tuning"
PROFILE_PATH = _REPO / "cypha_bench" / "config" / "everyday_profile.json"

VARIANT_CHOICES: dict[str, tuple[Any, ...]] = {
    "cold_start_steps": (10, 15, 20, 25, 30, 40),
    "min_experts_floor": (2, 3, 4, 6, 8),
    "online_passes_extra": (0, 0, 1, 1, 2),
    "temperature_scale": (0.85, 0.92, 0.97, 1.0, 1.05, 1.12, 1.2),
    "mdl_lambda_scale": (0.5, 0.75, 1.0, 1.25, 1.5),
    "replay_ratio_scale": (0.75, 0.9, 1.0, 1.1, 1.25),
    "target_lr_scale": (0.8, 0.9, 1.0, 1.1, 1.2),
    "deliberation_lo": (0.35, 0.4, 0.45, 0.5),
    "deliberation_hi": (0.5, 0.55, 0.6, 0.65, 0.7),
    "reg_hash_routing": (False, False, False, True),
}


def _near_choice(center_val: Any, choices: tuple[Any, ...], rng: random.Random) -> Any:
    if center_val in choices:
        idx = choices.index(center_val)
        lo = max(0, idx - 1)
        hi = min(len(choices) - 1, idx + 1)
        return rng.choice(choices[lo : hi + 1])
    return rng.choice(choices)


def _sample_algorithm_variants(
    base: dict[str, Any],
    rng: random.Random,
    *,
    center: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = merge_algorithm_variants(base)
    out = dict(merged)
    for key in VARIANT_KEYS:
        choices = VARIANT_CHOICES.get(key, (merged[key],))
        if center is not None and key in center:
            out[key] = _near_choice(center[key], choices, rng)
        else:
            out[key] = rng.choice(choices)
    lo = float(out["deliberation_lo"])
    hi = float(out["deliberation_hi"])
    if lo >= hi:
        out["deliberation_lo"] = min(lo, hi - 0.05)
        out["deliberation_hi"] = max(hi, lo + 0.05)
    return out


def _flat_to_regimes(flat: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    cls = dict(flat.get("classification_cyphadif", {}))
    reg = dict(flat.get("regression_difregressor", {}))
    arch = {**out.get("architecture", {}), **flat.get("architecture", {})}
    vision = dict(cls)
    for key in ("replay_ratio", "ood_sigma"):
        if key in arch:
            vision[key] = arch[key]
    out["regimes"] = {"tabular": cls, "vision": vision, "regression": reg}
    out["architecture"] = arch
    out.setdefault("default_regime", "tabular")
    return out


def sample_variant_profile(
    base_profile: dict[str, Any],
    rng: random.Random,
    *,
    center_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample regime hyperparams (narrow) + algorithm_variants, merged into profile."""
    center = center_profile or base_profile
    flat_center = _flat_center_for_regime(center, "tabular")
    flat = sample_profile_narrow(ParamRanges(), flat_center, rng)
    if base_profile.get("regimes") or center.get("regimes"):
        profile = _flat_to_regimes(flat, copy.deepcopy(base_profile))
    else:
        profile = copy.deepcopy(flat)

    center_variants = merge_algorithm_variants(center)
    algo = _sample_algorithm_variants(profile, rng, center=center_variants)
    return apply_algorithm_variants(profile, algo)


def _eval_variant(
    combo_id: int,
    profile: dict[str, Any],
    baseline_metrics: dict[str, tuple[str, float]],
    profile_dir: str,
) -> dict[str, Any]:
    profile_path = Path(profile_dir) / f"variant_{combo_id}.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

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
    t0 = time.perf_counter()
    try:
        side: dict[str, dict] = {}
        for key, (mod_path, _) in SWARM_DOMAINS.items():
            mod = __import__(mod_path, fromlist=["run"])
            side[key] = mod.run()
        tuned_metrics = filter_metrics_for_swarm(
            collect_run_metrics(domain_results=side)
        )
        score, ratios, deltas = score_vs_baseline(
            tuned_metrics, baseline_metrics, aggregate="mean", scorable_only=True
        )
        return {
            "combo_id": combo_id,
            "composite": score,
            "validation_score": score,
            "normalized_ratios": ratios,
            "deltas": deltas,
            "profile": profile,
            "metrics": {k: {"metric": m, "value": v} for k, (m, v) in tuned_metrics.items()},
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": None,
        }
    except Exception as exc:
        return {
            "combo_id": combo_id,
            "composite": -1.0,
            "validation_score": -1.0,
            "normalized_ratios": {},
            "deltas": {},
            "profile": profile,
            "metrics": {},
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": str(exc),
        }
    finally:
        for key, val in saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def _run_swarm(
    profiles: list[dict[str, Any]],
    *,
    combos: int,
    jobs: int,
    seed: int,
    top_k: int,
    baseline_metrics: dict[str, tuple[str, float]],
    source: str,
    stamp_prefix: str,
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warmup_cuda()
    gpu_ok = cuda_gemm_usable()
    print(
        f"Arch swarm ({stamp_prefix}): {combos} combos, jobs={jobs}, cuda={gpu_ok}, "
        f"eval_domains={list(SWARM_DOMAINS.keys())}",
        flush=True,
    )

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cypha_arch_swarm_") as tmp_dir:
        if Parallel is not None and jobs > 1:
            rows = Parallel(n_jobs=jobs, backend="loky", verbose=10)(
                delayed(_eval_variant)(i, profiles[i], baseline_metrics, tmp_dir)
                for i in range(combos)
            )
        else:
            rows = [_eval_variant(i, profiles[i], baseline_metrics, tmp_dir) for i in range(combos)]

    rows.sort(key=lambda r: r["composite"], reverse=True)
    eligible = [r for r in rows if not r.get("error")]
    elapsed = time.perf_counter() - t0
    best = eligible[0] if eligible else rows[0]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    leaderboard_path = OUT_DIR / f"{stamp_prefix}_{stamp}.json"
    payload = {
        "combos": combos,
        "jobs": jobs,
        "seed": seed,
        "top_k": top_k,
        "source": source,
        "baseline_path": str(SWARM_BASELINE_PATH),
        "eval_domains": list(SWARM_DOMAINS.keys()),
        "gpu_cuda_usable": gpu_ok,
        "elapsed_s": round(elapsed, 2),
        "errors": sum(1 for r in rows if r.get("error")),
        "top10": eligible[:10],
        "top50": eligible[:50],
        "all_eligible": eligible,
        "top_k_candidates": eligible[:top_k],
        "best": best,
    }
    leaderboard_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Done in {elapsed:.1f}s — best score={best['composite']:.4f}", flush=True)
    print(f"Leaderboard: {leaderboard_path}", flush=True)
    return {
        "leaderboard_path": leaderboard_path,
        "top_k_candidates": eligible[:top_k],
        "best": best,
        "eligible": eligible,
        "elapsed_s": round(elapsed, 2),
    }


def run_arch_swarm(
    base_profile: dict[str, Any] | None = None,
    combos: int = 400,
    jobs: int = 6,
    seed: int = 42,
    top_k: int = 30,
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    if base_profile is None:
        base_profile = load_profile() if PROFILE_PATH.exists() else {}
    ensure_swarm_baseline()
    baseline_metrics = load_swarm_baseline_metrics()
    rng = random.Random(seed)
    profiles = [sample_variant_profile(base_profile, rng) for _ in range(combos)]
    return _run_swarm(
        profiles,
        combos=combos,
        jobs=jobs,
        seed=seed,
        top_k=top_k,
        baseline_metrics=baseline_metrics,
        source=f"cypha_bench/tuning/arch_swarm.py combos={combos}",
        stamp_prefix="arch_swarm",
    )


def run_narrow_arch_swarm(
    center_profile: dict[str, Any],
    combos: int = 200,
    jobs: int = 6,
    seed: int = 42,
    top_k: int = 20,
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    baseline_metrics = load_swarm_baseline_metrics()
    rng = random.Random(seed + 17)
    profiles = [
        sample_variant_profile(center_profile, rng, center_profile=center_profile)
        for _ in range(combos)
    ]
    return _run_swarm(
        profiles,
        combos=combos,
        jobs=jobs,
        seed=seed,
        top_k=top_k,
        baseline_metrics=baseline_metrics,
        source=f"cypha_bench/tuning/arch_swarm.py narrow combos={combos}",
        stamp_prefix="arch_swarm_narrow",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture / algorithm variant swarm")
    parser.add_argument("--combos", type=int, default=int(os.environ.get("ARCH_SWARM_COMBOS", "400")))
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("ARCH_SWARM_JOBS", max(1, (os.cpu_count() or 4) - 1))))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=30, dest="top_k")
    parser.add_argument("--save-baseline", action="store_true", help="Write baseline_metrics.json before swarm")
    args = parser.parse_args()

    if args.save_baseline or not BASELINE_PATH.exists():
        save_baseline_snapshot(domains=list(SWARM_DOMAINS.keys()))
        print(f"Baseline: {BASELINE_PATH}", flush=True)

    base = json.loads(PROFILE_PATH.read_text(encoding="utf-8")) if PROFILE_PATH.exists() else load_profile()
    result = run_arch_swarm(
        base_profile=base,
        combos=args.combos,
        jobs=args.jobs,
        seed=args.seed,
        top_k=args.top_k,
    )
    print(result["leaderboard_path"])


if __name__ == "__main__":
    main()
