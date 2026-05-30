"""Cross-domain forgetting resistance from saved domain tables."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.bench_common import finalize_domain, json_safe, load_all_domain_tables


def _scan_forgetting(domain_id: str, exp_name: str, metrics: dict) -> dict | None:
    if "forgetting_score" in metrics:
        return json_safe(
            {
                "domain": domain_id,
                "experiment": exp_name,
                "forgetting_score": metrics["forgetting_score"],
                "accuracy_before": metrics.get("task_a_accuracy_before"),
                "accuracy_after": metrics.get("task_a_accuracy_after"),
            }
        )
    if "task_a_accuracy_before" in metrics and "task_a_accuracy_after" in metrics:
        before = float(metrics["task_a_accuracy_before"])
        after = float(metrics["task_a_accuracy_after"])
        score = (before - after) / max(before, 1e-6)
        return json_safe(
            {
                "domain": domain_id,
                "experiment": exp_name,
                "forgetting_score": score,
                "accuracy_before": before,
                "accuracy_after": after,
            }
        )
    return None


def run() -> dict:
    tables = load_all_domain_tables()
    rows = []
    for domain_id, payload in sorted(tables.items()):
        if domain_id.startswith("cross_"):
            continue
        experiments = payload.get("experiments", {})
        if isinstance(experiments, list):
            continue
        for exp_name, metrics in experiments.items():
            if not isinstance(metrics, dict):
                continue
            row = _scan_forgetting(domain_id, exp_name, metrics)
            if row:
                rows.append(row)
    mean_forgetting = (
        float(sum(r["forgetting_score"] for r in rows) / len(rows)) if rows else float("nan")
    )
    result = {"forgetting_rows": rows, "mean_forgetting_score": mean_forgetting}
    finalize_domain("cross_forgetting_resistance", result)
    return result


if __name__ == "__main__":
    print(run())
