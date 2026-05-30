from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from cypha_bench.common.paths import FIGURES_DIR, TABLES_DIR


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_table(name: str, metrics: dict[str, Any]) -> Path:
    """Save domain metrics as dXX.json with a standard experiments envelope."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    domain_id = name
    if name.startswith("d") and "_" in name:
        domain_id = name.split("_")[0]

    payload: dict[str, Any] = {
        "domain": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if "experiments" in metrics:
        exp = metrics["experiments"]
        if isinstance(exp, list):
            payload["experiments"] = {
                str(item.get("encoding", item.get("task", f"run_{i}"))): item
                for i, item in enumerate(exp)
                if isinstance(item, dict)
            }
        else:
            payload["experiments"] = exp
        for key, val in metrics.items():
            if key != "experiments":
                payload[key] = val
    elif "tasks" in metrics:
        experiments = {}
        for task in metrics["tasks"]:
            tname = task.get("task", "task")
            experiments[tname] = {
                **task.get("scores", {}),
                **{f"cypha_{k}": v for k, v in task.get("cypha_metrics", {}).items()},
                **{f"sgd_{k}": v for k, v in task.get("sgd_online", {}).items()},
            }
        if metrics.get("assertions"):
            experiments["_assertions"] = metrics["assertions"]
        payload["experiments"] = experiments
        payload["summary"] = {
            k: v for k, v in metrics.items() if k not in ("tasks",)
        }
    else:
        payload["experiments"] = {"summary": metrics}

    path = TABLES_DIR / f"{domain_id}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)

    alias = TABLES_DIR / f"{name}.json"
    if alias != path:
        with open(alias, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)
    return path


def save_figure(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def online_train_classifier(
    model, X: np.ndarray, y, label_fn=None, passes: int = 1, max_total_steps: int = 0
) -> None:
    """Online training with optional total-step cap.

    ``max_total_steps > 0`` hard-caps the total number of ``train_step`` calls
    across all passes (honours ``train_passes_cap`` in the bench profile).
    """
    label_fn = label_fn or (lambda v: str(v))
    total = 0
    for _ in range(passes):
        for x, target in zip(X, y):
            if max_total_steps > 0 and total >= max_total_steps:
                return
            model.train_step(x, label_fn(target))
            total += 1


def online_train_regressor(model, X: np.ndarray, y: np.ndarray, passes: int = 1) -> None:
    for _ in range(passes):
        for x, target in zip(X, y):
            model.train_step(x, float(target) if np.ndim(target) == 0 else target)
    fin = getattr(model, "finalize_training", None)
    if fin is not None:
        fin(X, y)


def evaluate_classification(model, X: np.ndarray, y, label_fn=None) -> dict[str, float]:
    label_fn = label_fn or (lambda v: str(v))
    preds = []
    for x, target in zip(X, y):
        pred, _, _ = model.predict(x)
        preds.append(pred)
    y_true = [label_fn(t) for t in y]
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
    }


def evaluate_regression(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    preds = []
    for x in X:
        pred, _, _ = model.predict(x)
        preds.append(float(np.asarray(pred).ravel()[0]))
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    preds_arr = np.asarray(preds, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(y_arr, preds_arr)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_arr, preds_arr)),
        "r2": float(r2_score(y_arr, preds_arr)),
    }


def cypha_metrics(model, X_test: np.ndarray, y_test, task: Literal["classification", "regression"] = "classification") -> dict[str, Any]:
    epistemic_vars, aleatoric_vars, predictions = [], [], []

    for x, y in zip(X_test, y_test):
        if task == "classification":
            pred, _, ep_var = model.predict(x)
            al_var = 0.0
            if hasattr(model, "aleatoric_var"):
                al_var = float(model.aleatoric_var(x))
        else:
            pred, ep_var, al_var = model.predict(x)
        epistemic_vars.append(float(ep_var))
        aleatoric_vars.append(float(al_var))
        predictions.append(pred)

    ep = np.asarray(epistemic_vars, dtype=np.float64)
    al = np.asarray(aleatoric_vars, dtype=np.float64)

    alpha = model.alpha_per_expert() if hasattr(model, "alpha_per_expert") else np.array([])
    return {
        "mean_epistemic_var": float(ep.mean()) if ep.size else 0.0,
        "std_epistemic_var": float(ep.std()) if ep.size else 0.0,
        "mean_aleatoric_var": float(al.mean()) if al.size else 0.0,
        "expert_count": int(model.expert_count()),
        "mean_alpha": float(model.mean_alpha()),
        "alpha_distribution": alpha.tolist(),
        "fraction_edge_of_chaos": float(np.mean(np.abs(alpha - 0.5) < 0.1)) if alpha.size else 0.0,
        "uncertainty_rank_correlation": _uncertainty_vs_error_correlation(
            np.asarray(predictions), np.asarray(y_test), ep, task
        ),
    }


def _uncertainty_vs_error_correlation(predictions, targets, epistemic_vars, task: str) -> float:
    if epistemic_vars.size == 0:
        return 0.0
    if task == "classification":
        errors = (predictions.astype(str) != targets.astype(str)).astype(float)
    else:
        pred_vals = np.asarray([float(np.ravel(p)[0]) for p in predictions], dtype=np.float64)
        errors = np.abs(pred_vals - np.asarray(targets, dtype=np.float64).ravel())
    if errors.size < 2:
        return 0.0
    if float(np.std(errors)) < 1e-12 or float(np.std(epistemic_vars)) < 1e-12:
        return 0.0
    with np.errstate(invalid="ignore"):
        rho, _ = spearmanr(epistemic_vars, errors)
    return float(rho) if rho is not None and np.isfinite(rho) else 0.0


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std
