#!/usr/bin/env python3
"""Validate tuned everyday profile vs library defaults on key bench domains."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.config.load_profile import load_profile  # noqa: E402

DOMAINS = {
    "d01": ("cypha_bench.domains.d01_statistical_baselines", "tasks"),
    "d02": ("cypha_bench.domains.d02_regression", "datasets"),
    "d03": ("cypha_bench.domains.d03_classification", "datasets"),
    "d08": ("cypha_bench.domains.d08_computer_vision", "experiments"),
}


def _run_domain(module_path: str, use_profile: bool) -> dict:
    os.environ["CYPHA_BENCH_USE_PROFILE"] = "1" if use_profile else "0"
    mod = __import__(module_path, fromlist=["run"])
    return mod.run()


def _extract_rows(result: dict, list_key: str) -> list[dict]:
    rows = result.get(list_key) or result.get("tasks") or result.get("datasets") or []
    if isinstance(rows, dict):
        rows = list(rows.values())
    return rows if isinstance(rows, list) else []


def _metric_from_row(row: dict) -> tuple[str, float | None, float | None]:
    name = str(row.get("dataset") or row.get("task") or row.get("name") or row.get("encoding") or "unknown")
    scores = row.get("cypha_scores") or {}
    if "accuracy" in scores:
        return name, float(scores["accuracy"]), None
    if "rmse" in scores:
        return name, None, float(scores["rmse"])
    if "r2" in scores:
        return name, float(scores.get("r2", 0)), None
    return name, None, None


def _summarize(side: dict) -> list[str]:
    lines: list[str] = []
    for key, (_, list_key) in DOMAINS.items():
        block = side.get(key, {})
        for row in _extract_rows(block, list_key):
            name, acc, rmse = _metric_from_row(row)
            if acc is not None:
                lines.append(f"- **{key}/{name}** accuracy={acc:.4f}")
            elif rmse is not None:
                lines.append(f"- **{key}/{name}** rmse={rmse:.4f}")
    return lines


def main() -> None:
    out_dir = _REPO / "cypha_bench" / "artifacts" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = load_profile()
    results: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "profile_path": str(_REPO / "cypha_bench" / "config" / "everyday_profile.json"),
        "profile": profile,
        "baseline": {},
        "tuned": {},
    }

    for key, (mod, _) in DOMAINS.items():
        print(f"Running {key} baseline...", flush=True)
        results["baseline"][key] = _run_domain(mod, use_profile=False)
        print(f"Running {key} tuned...", flush=True)
        results["tuned"][key] = _run_domain(mod, use_profile=True)

    out_path = out_dir / "validation_compare.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    lines = [
        "# CyphaDIF Everyday Profile — Validation",
        "",
        f"Generated: {results['generated_utc']}",
        "",
        "## Selected profile",
        "",
        "```json",
        json.dumps(
            {
                "classification": profile.get("classification_cyphadif"),
                "regression": profile.get("regression_difregressor"),
                "architecture": profile.get("architecture"),
            },
            indent=2,
        ),
        "```",
        "",
        "## Baseline (library defaults)",
        "",
        *_summarize(results["baseline"]),
        "",
        "## Tuned (everyday profile)",
        "",
        *_summarize(results["tuned"]),
        "",
        "## Notes",
        "",
        "- Baseline: `CYPHA_BENCH_USE_PROFILE=0`",
        "- Tuned: profile from `cypha_bench/config/everyday_profile.json` (fallback `config/profiled_medium.json`)",
        "",
    ]

    report = _REPO / "cypha_bench" / "TUNING_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
