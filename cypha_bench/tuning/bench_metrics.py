"""Collect, baseline, and score Cypha bench run metrics."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.common.paths import TABLES_DIR  # noqa: E402

ARTIFACTS_DIR = _REPO / "cypha_bench" / "artifacts"
BASELINE_PATH = ARTIFACTS_DIR / "baseline_metrics.json"
PYTHON = sys.executable

SWARM_DOMAINS: dict[str, tuple[str, str | None]] = {
    "d01": ("cypha_bench.domains.d01_statistical_baselines", "tasks"),
    "d02": ("cypha_bench.domains.d02_regression", "datasets"),
    "d03": ("cypha_bench.domains.d03_classification", "datasets"),
    "d08": ("cypha_bench.domains.d08_computer_vision", "experiments"),
    "d12": ("cypha_bench.domains.d12_anomaly_detection", "experiments"),
    "d16": ("cypha_bench.domains.d16_multitask", "experiments"),
}

HIGHER_IS_BETTER = frozenset(
    {"accuracy", "r2", "f1_macro", "cypha_ood_auroc", "final_attack_acc", "routing_ari"}
)
LOWER_IS_BETTER = frozenset({"rmse", "detection_latency_steps", "forgetting_score"})

# Primary task metrics used for swarm / arch scoring (excludes cypha_metrics noise).
SCORABLE_METRIC_NAMES = frozenset(
    HIGHER_IS_BETTER | LOWER_IS_BETTER | {"mean_epistemic_attack"}
)
SWARM_BASELINE_PATH = ARTIFACTS_DIR / "baseline_metrics_swarm.json"


def _metric_direction(metric: str) -> str:
    if metric in HIGHER_IS_BETTER or "auroc" in metric or metric.endswith("_acc"):
        return "higher"
    if metric in LOWER_IS_BETTER or "rmse" in metric or "latency" in metric:
        return "lower"
    return "higher"


def _primary_from_scores(scores: dict[str, Any]) -> tuple[str, float] | None:
    for metric in ("accuracy", "r2", "rmse", "cypha_ood_auroc", "final_attack_acc", "f1_macro", "routing_ari"):
        if metric in scores and scores[metric] is not None:
            return metric, float(scores[metric])
    return None


def _metric_from_row(row: dict[str, Any]) -> list[tuple[str, str, float]]:
    label = str(
        row.get("dataset")
        or row.get("task")
        or row.get("name")
        or row.get("encoding")
        or row.get("experiment")
        or "unknown"
    )
    out: list[tuple[str, str, float]] = []
    scores = row.get("cypha_scores") or row.get("scores") or {}
    if isinstance(scores, dict):
        for key, val in scores.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out.append((label, key, float(val)))
    primary = _primary_from_scores(scores if isinstance(scores, dict) else {})
    if primary and not out:
        metric, value = primary
        out.append((label, metric, value))
    for key, val in row.items():
        if key in (
            "cypha_scores",
            "scores",
            "cypha_metrics",
            "baselines",
            "sgd_online",
            "name",
            "dataset",
            "task",
            "encoding",
            "experiment",
            "domain",
            "task_type",
            "expected_bound",
            "n_train",
            "n_test",
            "data_source",
        ):
            continue
        if key not in SCORABLE_METRIC_NAMES:
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out.append((label, key, float(val)))
    return out


def _metrics_from_experiments(domain_key: str, experiments: dict[str, Any]) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for name, exp in experiments.items():
        if not isinstance(exp, dict):
            continue
        row = {"name": name, **exp}
        for label, metric, value in _metric_from_row(row):
            if value != value:
                continue
            key = f"{domain_key}/{label}/{metric}" if "/" not in label else f"{domain_key}/{label}"
            out[key] = (metric, value)
    return out


def _metrics_from_domain_table(domain_key: str, data: dict[str, Any]) -> dict[str, tuple[str, float]]:
    metrics: dict[str, tuple[str, float]] = {}
    experiments = data.get("experiments")
    if isinstance(experiments, dict):
        metrics.update(_metrics_from_experiments(domain_key, experiments))
    elif isinstance(experiments, list):
        for exp in experiments:
            if isinstance(exp, dict):
                row = dict(exp)
                enc = row.get("encoding")
                if enc:
                    row.setdefault("name", str(enc))
                for label, metric, value in _metric_from_row(row):
                    if value != value:
                        continue
                    metrics[f"{domain_key}/{label}/{metric}"] = (metric, value)
    list_key = SWARM_DOMAINS.get(domain_key, (None, None))[1]
    if list_key and list_key != "experiments":
        rows = data.get(list_key) or []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    for label, metric, value in _metric_from_row(row):
                        if value != value:
                            continue
                        metrics[f"{domain_key}/{label}/{metric}"] = (metric, value)
    return metrics


def _metrics_from_domain_result(domain_key: str, result: dict[str, Any]) -> dict[str, tuple[str, float]]:
    metrics: dict[str, tuple[str, float]] = {}
    list_key = SWARM_DOMAINS.get(domain_key, (None, None))[1]
    if list_key:
        rows = result.get(list_key) or result.get("tasks") or result.get("datasets") or []
        if isinstance(rows, dict):
            rows = list(rows.values())
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for label, metric, value in _metric_from_row(row):
                    if value != value:
                        continue
                    metrics[f"{domain_key}/{label}/{metric}"] = (metric, value)

    experiments = result.get("experiments")
    if isinstance(experiments, dict):
        metrics.update(_metrics_from_experiments(domain_key, experiments))
    return metrics


def _read_domain_tables(domain_keys: list[str] | None = None) -> dict[str, tuple[str, float]]:
    keys = domain_keys or list(SWARM_DOMAINS.keys())
    metrics: dict[str, tuple[str, float]] = {}
    for domain_key in keys:
        candidates = [
            TABLES_DIR / f"{domain_key}.json",
            TABLES_DIR / f"{domain_key}_{SWARM_DOMAINS[domain_key][0].split('.')[-1]}.json",
        ]
        for path in candidates:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            metrics.update(_metrics_from_domain_table(domain_key, data))
            break
    return metrics


def _run_domain(module_path: str, use_profile: bool) -> dict:
    os.environ["CYPHA_BENCH_USE_PROFILE"] = "1" if use_profile else "0"
    os.environ.setdefault("CYPHA_BENCH_FAST", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mod = __import__(module_path, fromlist=["run"])
        return mod.run()


def _run_domains(
    domains: dict[str, tuple[str, str | None]],
    *,
    use_profile: bool,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for key, (mod_path, _) in domains.items():
        results[key] = _run_domain(mod_path, use_profile)
    return results


def collect_run_metrics(
    *,
    domains: list[str] | None = None,
    from_tables: bool = False,
    use_profile: bool = True,
    domain_results: dict[str, dict] | None = None,
) -> dict[str, tuple[str, float]]:
    """
    Collect flattened metrics keyed as ``dXX/label/metric``.

    Reads ``report/tables/d*.json`` when *from_tables* is True, otherwise runs
    domain modules (``run_all``-style import/run) or flattens *domain_results*.
    """
    keys = domains or list(SWARM_DOMAINS.keys())
    selected = {k: SWARM_DOMAINS[k] for k in keys if k in SWARM_DOMAINS}

    if from_tables:
        return _read_domain_tables(list(selected.keys()))

    if domain_results is not None:
        metrics: dict[str, tuple[str, float]] = {}
        for key, result in domain_results.items():
            if key in selected:
                metrics.update(_metrics_from_domain_result(key, result))
        return metrics

    side = _run_domains(selected, use_profile=use_profile)
    metrics = {}
    for key, result in side.items():
        metrics.update(_metrics_from_domain_result(key, result))
    return metrics


def _serialize_metrics(metrics: dict[str, tuple[str, float]]) -> dict[str, dict[str, Any]]:
    return {key: {"metric": metric, "value": value} for key, (metric, value) in metrics.items()}


def _deserialize_metrics(payload: dict[str, Any]) -> dict[str, tuple[str, float]]:
    raw = payload.get("metrics", payload)
    out: dict[str, tuple[str, float]] = {}
    if not isinstance(raw, dict):
        return out
    for key, item in raw.items():
        if isinstance(item, dict) and "metric" in item and "value" in item:
            out[str(key)] = (str(item["metric"]), float(item["value"]))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out[str(key)] = (str(item[0]), float(item[1]))
    return out


def load_baseline_metrics(path: Path | str | None = None) -> dict[str, tuple[str, float]]:
    baseline_path = Path(path) if path is not None else BASELINE_PATH
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline metrics not found: {baseline_path}")
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    return _deserialize_metrics(data)


def _normalize_ratio(metric: str, candidate: float, baseline: float) -> float:
    direction = _metric_direction(metric)
    if direction == "higher":
        if baseline > 1e-12:
            return candidate / baseline
        return 1.0 if candidate >= baseline else 0.0
    if candidate <= 1e-12:
        return 0.0
    if baseline <= 1e-12:
        return 1.0 if candidate <= baseline else 0.0
    return baseline / candidate


def _signed_delta(metric: str, candidate: float, baseline: float) -> float:
    if _metric_direction(metric) == "higher":
        return candidate - baseline
    return baseline - candidate


def _is_scorable_key(key: str, metric: str) -> bool:
    if metric not in SCORABLE_METRIC_NAMES:
        return False
    domain = key.split("/", 1)[0]
    return domain in SWARM_DOMAINS or domain.startswith("d")


def _aggregate_ratios(ratios: dict[str, float], mode: str) -> float:
    if not ratios:
        return 0.0
    vals = list(ratios.values())
    if mode == "min":
        return min(vals)
    if mode == "median":
        vals.sort()
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0
    return sum(vals) / len(vals)


def score_vs_baseline(
    tuned: dict[str, tuple[str, float]],
    baseline: dict[str, tuple[str, float]],
    *,
    aggregate: str = "mean",
    scorable_only: bool = True,
    domains: list[str] | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Return ``(aggregate_ratio, ratios, deltas)`` — higher is better."""
    ratios: dict[str, float] = {}
    deltas: dict[str, float] = {}
    domain_set = frozenset(domains) if domains else None
    for key, (metric, base_val) in baseline.items():
        if domain_set is not None and key.split("/", 1)[0] not in domain_set:
            continue
        if scorable_only and not _is_scorable_key(key, metric):
            continue
        if key not in tuned:
            continue
        tuned_metric, tuned_val = tuned[key]
        if tuned_metric != metric:
            continue
        ratios[key] = _normalize_ratio(metric, tuned_val, base_val)
        deltas[key] = _signed_delta(metric, tuned_val, base_val)
    if not ratios:
        return 0.0, ratios, deltas
    return _aggregate_ratios(ratios, aggregate), ratios, deltas


def filter_metrics_for_swarm(
    metrics: dict[str, tuple[str, float]],
) -> dict[str, tuple[str, float]]:
    """Keep scorable keys for SWARM_DOMAINS only."""
    out: dict[str, tuple[str, float]] = {}
    for key, (metric, value) in metrics.items():
        domain = key.split("/", 1)[0]
        if domain not in SWARM_DOMAINS:
            continue
        if metric not in SCORABLE_METRIC_NAMES:
            continue
        out[key] = (metric, value)
    return out


def ensure_swarm_baseline(*, fast: bool = True, force: bool = False) -> Path:
    """Baseline snapshot for arch swarm (SWARM_DOMAINS only, scorable metrics)."""
    if SWARM_BASELINE_PATH.exists() and not force:
        return SWARM_BASELINE_PATH
    save_baseline_snapshot(fast=fast, path=SWARM_BASELINE_PATH, domains=list(SWARM_DOMAINS.keys()))
    raw = load_baseline_metrics(SWARM_BASELINE_PATH)
    filtered = filter_metrics_for_swarm(raw)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fast_mode": fast,
        "use_profile": False,
        "domains": list(SWARM_DOMAINS.keys()),
        "metrics": _serialize_metrics(filtered),
    }
    SWARM_BASELINE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Swarm baseline: {SWARM_BASELINE_PATH} ({len(filtered)} scorable keys)", flush=True)
    return SWARM_BASELINE_PATH


def load_swarm_baseline_metrics() -> dict[str, tuple[str, float]]:
    ensure_swarm_baseline()
    return load_baseline_metrics(SWARM_BASELINE_PATH)


def save_baseline_snapshot(
    *,
    fast: bool = True,
    path: Path | str | None = None,
    domains: list[str] | None = None,
) -> Path:
    """Run bench domains with library defaults (``CYPHA_BENCH_USE_PROFILE=0``) and save JSON."""
    out_path = Path(path) if path is not None else BASELINE_PATH
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CYPHA_BENCH_USE_PROFILE"] = "0"
    env["CYPHA_BENCH_FAST"] = "1" if fast else "0"
    env.pop("CYPHA_BENCH_PROFILE_PATH", None)
    env.pop("CYPHA_BENCH_PROFILE_JSON", None)

    if domains:
        prev_use = os.environ.get("CYPHA_BENCH_USE_PROFILE")
        prev_fast = os.environ.get("CYPHA_BENCH_FAST")
        os.environ["CYPHA_BENCH_USE_PROFILE"] = "0"
        os.environ["CYPHA_BENCH_FAST"] = "1" if fast else "0"
        try:
            metrics = collect_run_metrics(domains=domains, use_profile=False)
        finally:
            if prev_use is None:
                os.environ.pop("CYPHA_BENCH_USE_PROFILE", None)
            else:
                os.environ["CYPHA_BENCH_USE_PROFILE"] = prev_use
            if prev_fast is None:
                os.environ.pop("CYPHA_BENCH_FAST", None)
            else:
                os.environ["CYPHA_BENCH_FAST"] = prev_fast
    else:
        cmd = [PYTHON, str(_REPO / "cypha_bench" / "run_all.py")]
        print(f"Saving baseline snapshot (fast={fast}): {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=_REPO, env=env, check=True)
        metrics = collect_run_metrics(from_tables=True)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fast_mode": fast,
        "use_profile": False,
        "domains": domains,
        "metrics": _serialize_metrics(metrics),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote baseline metrics: {out_path} ({len(metrics)} keys)", flush=True)
    return out_path
