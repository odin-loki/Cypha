"""Cross-domain online adaptation metrics from saved domain tables."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.bench_common import finalize_domain, json_safe, load_all_domain_tables

ADAPTATION_HINTS = (
    "adaptation",
    "drift",
    "online",
    "cross_asset",
    "10D",
    "17D",
    "detection_latency",
)


def _extract_adaptation(domain_id: str, exp_name: str, metrics: dict) -> dict | None:
    name_l = exp_name.lower()
    if not any(h in name_l for h in ADAPTATION_HINTS):
        if domain_id == "d17" and "bpc_improvement" in metrics:
            pass
        elif "detection_latency_steps" not in metrics:
            return None

    row = {"domain": domain_id, "experiment": exp_name}
    if "detection_latency_steps" in metrics:
        row["T_adapt_steps"] = metrics["detection_latency_steps"]
    if "bpc_improvement" in metrics:
        row["T_adapt_steps"] = metrics.get("bpc_improvement")
        row["metric"] = "bpc_improvement"
    if "final_attack_acc" in metrics:
        row["recovery_accuracy"] = metrics["final_attack_acc"]
    if "bpc_ood_before_adapt" in metrics and "bpc_ood_after_adapt" in metrics:
        row["pre_drift_metric"] = metrics["bpc_ood_before_adapt"]
        row["post_adapt_metric"] = metrics["bpc_ood_after_adapt"]
    return json_safe(row) if len(row) > 2 else None


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
            row = _extract_adaptation(domain_id, exp_name, metrics)
            if row:
                rows.append(row)
    result = {"adaptation_rows": rows, "n_domains_with_adaptation_signal": len(rows)}
    finalize_domain("cross_online_adaptation", result)
    return result


if __name__ == "__main__":
    print(run())
