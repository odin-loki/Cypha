"""Cross-domain uncertainty calibration from saved domain tables."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.bench_common import finalize_domain, json_safe, load_all_domain_tables

CLASSIFICATION_KEYS = ("accuracy", "f1_macro", "mean_epistemic_var", "mean_confidence")
REGRESSION_KEYS = ("rmse", "r2", "mean_epistemic_var")
OOD_KEYS = ("ood_auroc", "cypha_ood_auroc", "extrapolation_auroc")


def _collect_rows(tables: dict) -> list[dict]:
    rows = []
    for domain_id, payload in sorted(tables.items()):
        if domain_id.startswith("cross_"):
            continue
        experiments = payload.get("experiments", {})
        if isinstance(experiments, list):
            continue
        for exp_name, metrics in experiments.items():
            if not isinstance(metrics, dict) or exp_name.startswith("_"):
                continue
            row = {
                "domain": domain_id,
                "experiment": exp_name,
                "task_type": "unknown",
            }
            if any(k in metrics for k in CLASSIFICATION_KEYS):
                row["task_type"] = "classification"
                row["ece_proxy"] = metrics.get("mean_confidence")
                if row["ece_proxy"] is not None and "accuracy" in metrics:
                    row["ece_proxy"] = abs(float(row["ece_proxy"]) - float(metrics["accuracy"]))
            elif any(k in metrics for k in REGRESSION_KEYS):
                row["task_type"] = "regression"
            for k in OOD_KEYS:
                if k in metrics:
                    row["ood_auroc"] = metrics[k]
            if "mean_epistemic_var" in metrics:
                row["mean_epistemic_var"] = metrics["mean_epistemic_var"]
            if "cypha_mean_epistemic_var" in metrics:
                row["mean_epistemic_var"] = metrics["cypha_mean_epistemic_var"]
            if "uncertainty_rank_correlation" in metrics:
                row["uncertainty_rank_correlation"] = metrics["uncertainty_rank_correlation"]
            rows.append(json_safe(row))
    return rows


def run() -> dict:
    tables = load_all_domain_tables()
    rows = _collect_rows(tables)
    ood_vals = [r["ood_auroc"] for r in rows if r.get("ood_auroc") is not None]
    summary = {
        "n_experiments": len(rows),
        "mean_ood_auroc": float(sum(ood_vals) / len(ood_vals)) if ood_vals else float("nan"),
    }
    result = {"calibration_rows": rows, "summary": summary}
    finalize_domain("cross_uncertainty_calibration", result)
    return result


if __name__ == "__main__":
    print(run())
