#!/usr/bin/env python3
"""Rescore saved arch_swarm leaderboards with corrected mean-ratio scorer."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.tuning.bench_metrics import (  # noqa: E402
    SWARM_BASELINE_PATH,
    ensure_swarm_baseline,
    filter_metrics_for_swarm,
    load_swarm_baseline_metrics,
    score_vs_baseline,
)

OUT_DIR = _REPO / "cypha_bench" / "artifacts" / "tuning"
PROFILE_PATH = _REPO / "cypha_bench" / "config" / "everyday_profile.json"
REPORT_PATH = _REPO / "cypha_bench" / "ARCH_RESCORE_REPORT.md"


def _row_metrics(row: dict[str, Any]) -> dict[str, tuple[str, float]]:
    raw = row.get("metrics") or {}
    out: dict[str, tuple[str, float]] = {}
    for key, item in raw.items():
        if isinstance(item, dict) and "metric" in item and "value" in item:
            out[str(key)] = (str(item["metric"]), float(item["value"]))
    return filter_metrics_for_swarm(out)


def _collect_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, Any]] = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.name
        for bucket in ("all_eligible", "top50", "top10", "top_k_candidates", "best"):
            block = data.get(bucket)
            if isinstance(block, list):
                for row in block:
                    if isinstance(row, dict) and row.get("profile"):
                        key = (source, row.get("combo_id"))
                        if key in seen:
                            continue
                        seen.add(key)
                        rows.append({**row, "_source": source})
            elif isinstance(block, dict) and block.get("profile"):
                key = (source, block.get("combo_id"))
                if key not in seen:
                    seen.add(key)
                    rows.append({**block, "_source": source})
    return rows


def rescore(*, apply_best: bool = False) -> dict[str, Any]:
    ensure_swarm_baseline(force=not SWARM_BASELINE_PATH.exists())
    baseline = load_swarm_baseline_metrics()

    paths = sorted(OUT_DIR.glob("arch_swarm*.json"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No arch_swarm*.json under {OUT_DIR}")

    rescored: list[dict[str, Any]] = []
    for row in _collect_rows(paths):
        metrics = _row_metrics(row)
        if not metrics:
            continue
        old = float(row.get("composite") or row.get("validation_score") or 0.0)
        score, ratios, deltas = score_vs_baseline(
            metrics, baseline, aggregate="mean", scorable_only=True
        )
        rescored.append(
            {
                "combo_id": row.get("combo_id"),
                "source": row.get("_source"),
                "old_score": old,
                "new_score": score,
                "n_ratios": len(ratios),
                "normalized_ratios": ratios,
                "deltas": deltas,
                "profile": row.get("profile"),
            }
        )

    rescored.sort(key=lambda r: r["new_score"], reverse=True)
    best = rescored[0] if rescored else None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"arch_rescore_{stamp}.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_keys": len(baseline),
        "leaderboards": [str(p) for p in paths],
        "candidates_rescored": len(rescored),
        "best": best,
        "top10": rescored[:10],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Architecture Swarm Rescore Report",
        "",
        f"Generated: {payload['generated_utc']}",
        f"Candidates rescored: {len(rescored)}",
        f"Output: `{out_path.name}`",
        "",
    ]
    if best:
        lines.extend(
            [
                "## Best (corrected scorer)",
                "",
                f"- combo_id: {best.get('combo_id')}",
                f"- old_score: {best.get('old_score'):.4f}",
                f"- **new_score: {best.get('new_score'):.4f}**",
                f"- source: {best.get('source')}",
                "",
            ]
        )
        if apply_best and isinstance(best.get("profile"), dict):
            prof = dict(best["profile"])
            prof["source"] = "cypha_bench/tuning/rescore_arch_swarm.py"
            prof["generated_utc"] = payload["generated_utc"]
            prof["rescore_score"] = best["new_score"]
            PROFILE_PATH.write_text(json.dumps(prof, indent=2), encoding="utf-8")
            lines.append(f"Applied to `{PROFILE_PATH}`")
            lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    print(f"Wrote {REPORT_PATH}", flush=True)
    if best:
        print(
            f"Best combo_id={best.get('combo_id')} "
            f"old={best.get('old_score'):.4f} new={best.get('new_score'):.4f}",
            flush=True,
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore arch swarm leaderboards")
    parser.add_argument("--apply-best", action="store_true", help="Write best profile to everyday_profile.json")
    args = parser.parse_args()
    rescore(apply_best=args.apply_best)


if __name__ == "__main__":
    main()
