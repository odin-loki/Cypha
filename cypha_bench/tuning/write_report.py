"""Regenerate TUNING_REPORT.md from validation_compare.json and everyday_profile.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from cypha_bench.config.load_profile import load_profile
from cypha_bench.tuning.validate_profile import _summarize

VALIDATION = _REPO / "cypha_bench/artifacts/tuning/validation_compare.json"
REPORT = _REPO / "cypha_bench/TUNING_REPORT.md"


def _parse_metrics(side: dict) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for line in _summarize(side):
        # line like "- **d03/iris** accuracy=0.8667"
        body = line.strip("- ").split("**", 2)
        if len(body) < 3:
            continue
        key = body[1]
        rest = body[2].strip()
        metric, val = rest.split("=", 1)
        out[key] = (metric, float(val))
    return out


def main() -> None:
    data = json.loads(VALIDATION.read_text(encoding="utf-8"))
    profile = load_profile()
    base = _parse_metrics(data["baseline"])
    tuned = _parse_metrics(data["tuned"])

    sweep = profile.get("sweep", {})
    micro = profile.get("validation_metrics", {})
    cls_block = profile.get("classification_cyphadif") or profile.get("regimes", {}).get("tabular", {})
    reg_block = profile.get("regression_difregressor") or profile.get("regimes", {}).get("regression", {})
    vis_block = profile.get("regimes", {}).get("vision", cls_block)
    arch_block = profile.get("architecture", {})

    lines = [
        "# CyphaDIF Everyday Profile — Validation",
        "",
        f"Generated: {data['generated_utc']}",
        "",
        "## Sweep summary",
        "",
        f"- Source: `{profile.get('source', 'unknown')}`",
        f"- Combos evaluated: **{sweep.get('combos', '?')}** (bench-aligned micro-suite)",
        f"- Sweep wall time: {sweep.get('elapsed_s', '?')}s",
        f"- Best composite score: {sweep.get('composite_best', '?')}",
        f"- GPU CUDA usable: {profile.get('gpu_cuda_usable', False)}",
        f"- Leaderboard: `{sweep.get('leaderboard_path', '')}`",
        "",
        "### Micro-benchmark scores (winner on sweep suite)",
        "",
    ]
    if micro:
        for k, v in sorted(micro.items()):
            if k == "wall_s":
                continue
            lines.append(f"- `{k}`: {v:.4f}" if isinstance(v, float) else f"- `{k}`: {v}")
        lines.append("")

    lines += [
        "## Selected profile",
        "",
        "```json",
        json.dumps(
            {
                "tabular": cls_block,
                "vision": vis_block,
                "regression": reg_block,
                "architecture": arch_block,
            },
            indent=2,
        ),
        "```",
        "",
        "## Baseline vs tuned (full bench domains, CYPHA_BENCH_FAST=1)",
        "",
        "| Domain | Metric | Baseline | Tuned | Delta |",
        "|--------|--------|----------|-------|-------|",
    ]

    keys = sorted(set(base) | set(tuned))
    for key in keys:
        if key not in base or key not in tuned:
            continue
        m0, v0 = base[key]
        m1, v1 = tuned[key]
        if m0 != m1:
            continue
        if m0 == "accuracy":
            delta = v1 - v0
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {key} | acc | {v0:.4f} | {v1:.4f} | {sign}{delta:.4f} |")
        else:
            delta = v1 - v0
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {key} | rmse | {v0:.4f} | {v1:.4f} | {sign}{delta:.4f} |")

    lines += [
        "",
        "## Notes",
        "",
        "- Baseline: `CYPHA_BENCH_USE_PROFILE=0`",
        "- Tuned: `cypha_bench/config/everyday_profile.json`",
        "- Re-run sweep: `python cypha_bench/tuning/bench_full_swarm.py --combos 400 --jobs 6`",
        "- Re-run validation: `$env:CYPHA_BENCH_FAST='1'; python cypha_bench/tuning/validate_profile.py`",
        "",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
