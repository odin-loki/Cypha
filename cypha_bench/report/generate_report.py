"""Assemble BASELINE_REPORT.md from saved JSON tables."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.common.paths import BENCH_ROOT, TABLES_DIR


def json_safe(obj):
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return str(obj)


def load_all_domain_tables() -> dict[str, dict]:
    tables: dict[str, dict] = {}
    if not TABLES_DIR.exists():
        return tables
    for path in sorted(TABLES_DIR.glob("*.json")):
        stem = path.stem
        if not (stem.startswith("d") or stem.startswith("cross_")):
            continue
        if stem.startswith("d") and "_" in stem and stem.split("_")[0] in tables:
            continue
        try:
            tables[stem.split("_")[0] if stem.startswith("d") and "_" in stem else stem] = json.loads(
                path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            continue
    return tables


def _flatten_metrics(obj, prefix: str = "") -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                if all(not isinstance(vv, (dict, list)) for vv in v.values()):
                    for sk, sv in v.items():
                        rows.append((f"{key}.{sk}", sv))
                else:
                    rows.extend(_flatten_metrics(v, key))
            elif isinstance(v, list) and v and not isinstance(v[0], (dict, list)):
                rows.append((key, v))
            elif not isinstance(v, (dict, list)):
                rows.append((key, v))
    return rows


def _format_value(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 0.001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"
    if isinstance(v, list) and len(v) > 6:
        return f"[{len(v)} items]"
    return str(v)


def build_markdown(tables: dict | None = None) -> str:
    tables = tables or load_all_domain_tables()
    lines = [
        "# Cypha Bench Baseline Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Default parameters only — no hyperparameter tuning.",
        "",
    ]

    domain_ids = sorted(k for k in tables if k.startswith("d") and k[1:].isdigit())
    cross_ids = sorted(k for k in tables if k.startswith("cross_"))

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Domains run: **{len(domain_ids)}**")
    lines.append(f"- Cross-domain analyses: **{len(cross_ids)}**")
    lines.append("")

    for domain_id in domain_ids:
        payload = tables[domain_id]
        lines.append(f"## {domain_id.upper()}")
        lines.append("")
        ts = payload.get("timestamp", "")
        if ts:
            lines.append(f"*Timestamp:* {ts}")
            lines.append("")
        experiments = payload.get("experiments", {})
        if isinstance(experiments, list):
            experiments = {
                str(item.get("encoding", item.get("task", f"run_{i}"))): item
                for i, item in enumerate(experiments)
                if isinstance(item, dict)
            }
        if not experiments and payload.get("tasks"):
            experiments = {
                t.get("task", f"task_{i}"): {**t.get("scores", {}), **t.get("cypha_metrics", {})}
                for i, t in enumerate(payload["tasks"])
            }
        if not experiments:
            lines.append("_No experiments recorded._")
            lines.append("")
            continue
        for exp_name, metrics in experiments.items():
            if not isinstance(metrics, dict):
                continue
            lines.append(f"### {exp_name}")
            lines.append("")
            if not isinstance(metrics, dict):
                lines.append(f"- result: {_format_value(metrics)}")
                lines.append("")
                continue
            if metrics.get("skipped"):
                lines.append(f"- **skipped:** {metrics.get('reason', 'unknown')}")
                lines.append("")
                continue
            flat = _flatten_metrics(metrics)
            if not flat:
                lines.append("_Empty metrics._")
            else:
                lines.append("| Metric | Value |")
                lines.append("| --- | --- |")
                for key, val in sorted(flat, key=lambda x: x[0])[:30]:
                    lines.append(f"| `{key}` | {_format_value(val)} |")
            lines.append("")

    if cross_ids:
        lines.append("## Cross-Domain Analyses")
        lines.append("")
        for cid in cross_ids:
            payload = tables[cid]
            lines.append(f"### {cid}")
            lines.append("")
            summary = payload.get("summary") or payload.get("experiments") or payload
            flat = _flatten_metrics(summary)
            if flat:
                lines.append("| Metric | Value |")
                lines.append("| --- | --- |")
                for key, val in sorted(flat, key=lambda x: x[0])[:40]:
                    lines.append(f"| `{key}` | {_format_value(val)} |")
            lines.append("")

    return "\n".join(lines)


def build_report(output_path: Path | None = None) -> Path:
    output_path = output_path or (BENCH_ROOT / "BASELINE_REPORT.md")
    md = build_markdown()
    output_path.write_text(md, encoding="utf-8")
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate Cypha Bench markdown report")
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCH_ROOT / "BASELINE_REPORT.md",
        help="Output markdown path",
    )
    args = parser.parse_args()
    out = build_report(args.output)
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
