"""Grand Unified Law alpha spectrum across saved domain tables."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.bench_common import finalize_domain, json_safe, load_all_domain_tables

ALPHA_KEYS = (
    "mean_alpha",
    "mean_alpha_proxy",
    "fraction_edge_of_chaos",
    "mean_alpha_text",
    "mean_alpha_binary",
)


def _walk_metrics(domain_id: str, exp_name: str, metrics: dict, prefix: str = "") -> list[dict]:
    rows = []
    if "mean_alpha" in metrics or "mean_alpha_proxy" in metrics:
        alpha = metrics.get("mean_alpha", metrics.get("mean_alpha_proxy"))
        frac = metrics.get("fraction_edge_of_chaos")
        rows.append(
            json_safe(
                {
                    "domain": domain_id,
                    "experiment": f"{prefix}{exp_name}".strip("_"),
                    "mean_alpha": alpha,
                    "fraction_edge_of_chaos": frac,
                }
            )
        )
    if "per_equation" in metrics and isinstance(metrics["per_equation"], dict):
        for eq_name, eq_metrics in metrics["per_equation"].items():
            if isinstance(eq_metrics, dict):
                rows.extend(_walk_metrics(domain_id, eq_name, eq_metrics, prefix=f"{exp_name}_"))
    if "gzip_ratios" in metrics and "mean_alpha_proxy" in metrics:
        for i, (ratio, alpha) in enumerate(
            zip(metrics["gzip_ratios"], metrics.get("mean_alpha_proxy", []))
        ):
            rows.append(
                json_safe(
                    {
                        "domain": domain_id,
                        "experiment": f"{exp_name}_file_{i}",
                        "mean_alpha": alpha,
                        "gzip_ratio": ratio,
                    }
                )
            )
    return rows


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
            rows.extend(_walk_metrics(domain_id, exp_name, metrics))

    alphas = []
    for r in rows:
        val = r.get("mean_alpha")
        if val is not None:
            try:
                alphas.append(float(val))
            except (TypeError, ValueError):
                pass
    summary = {
        "n_measurements": len(rows),
        "global_mean_alpha": float(np.mean(alphas)) if alphas else float("nan"),
        "global_std_alpha": float(np.std(alphas)) if alphas else float("nan"),
        "within_gul_band_fraction": float(np.mean(np.abs(np.array(alphas) - 0.5) < 0.15))
        if alphas
        else float("nan"),
    }
    result = {"alpha_rows": rows, "summary": summary}
    finalize_domain("cross_alpha_spectrum_global", result)
    return result


if __name__ == "__main__":
    print(run())
