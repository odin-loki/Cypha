"""Shared utilities for cypha_bench domain scripts."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parent
TABLES_DIR = BENCH_ROOT / "report" / "tables"
FIGURES_DIR = BENCH_ROOT / "report" / "figures"
DATA_DIR = BENCH_ROOT / "data"

# Default CyphaDIF parameters — no tuning.
DEFAULT_FIELD_DIM = 128
DEFAULT_SEED = 42
DEFAULT_FINANCIAL_TICKERS = ["SPY", "QQQ", "GLD", "TLT"]
DEFAULT_FINANCIAL_PERIOD = "5y"
DEFAULT_FINANCIAL_WINDOW = 20


def ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_classifier(input_dim: int, seed: int = DEFAULT_SEED):
    ensure_repo_on_path()
    from Cypha import CyphaDIF, VectorEncoder

    return CyphaDIF(
        encoder=VectorEncoder(input_dim),
        field_dim=DEFAULT_FIELD_DIM,
        rng=np.random.default_rng(seed),
    )


def make_regressor(input_dim: int, seed: int = DEFAULT_SEED):
    ensure_repo_on_path()
    from Cypha import DIFRegressor, VectorEncoder

    return DIFRegressor(
        encoder=VectorEncoder(input_dim),
        field_dim=DEFAULT_FIELD_DIM,
        rng=np.random.default_rng(seed),
    )


def train_classifier_online(clf, X: np.ndarray, y, label_fmt=str, passes: int = 1) -> None:
    for p in range(passes):
        order = np.random.default_rng(DEFAULT_SEED + p).permutation(len(X))
        for i in order:
            clf.train_step(X[i], label_fmt(y[i]))


def train_regressor_online(reg, X: np.ndarray, y: np.ndarray, passes: int = 1) -> None:
    for p in range(passes):
        order = np.random.default_rng(DEFAULT_SEED + p).permutation(len(X))
        for i in order:
            reg.train_step(X[i], float(y[i]))


def clf_predict_label(clf, x: np.ndarray) -> str:
    label, _ = clf.infer(x)
    return str(label)


def clf_epistemic(clf, x: np.ndarray) -> float:
    info = clf.infer_full(x)
    return float(info.get("anomaly_score", info.get("entropy", 0.0)))


def safe_spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return 0.0
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    with np.errstate(invalid="ignore"):
        result = spearmanr(a, b)
    rho = result.correlation if hasattr(result, "correlation") else result[0]
    return float(rho) if rho is not None and np.isfinite(rho) else 0.0


def clf_metrics(clf, X: np.ndarray, y) -> dict[str, Any]:
    preds, epistemic, confidences = [], [], []
    for x, yt in zip(X, y):
        pred = clf_predict_label(clf, x)
        preds.append(pred)
        epistemic.append(clf_epistemic(clf, x))
        _, conf = clf.infer(x)
        confidences.append(float(conf))

    y_str = np.array([str(v) for v in y])
    p_str = np.array(preds)
    correct = y_str == p_str
    errors = (~correct).astype(float)
    rho = safe_spearman(epistemic, errors)
    diag = clf.diagnostics()
    return {
        "accuracy": float(accuracy_score(y_str, p_str)),
        "f1_macro": float(f1_score(y_str, p_str, average="macro", zero_division=0)),
        "mean_epistemic_var": float(np.mean(epistemic)),
        "mean_confidence": float(np.mean(confidences)),
        "expert_count": int(diag.get("n_classes", len(clf.memory._classes))),
        "uncertainty_rank_correlation": rho,
    }


def reg_metrics(reg, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    preds, unc = [], []
    for x in X:
        y_hat, u = reg.predict(x)
        preds.append(float(np.ravel(y_hat)[0]))
        unc.append(float(u))
    preds_arr = np.asarray(preds, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    errors = np.abs(preds_arr - y_arr)
    rho = safe_spearman(unc, errors)
    diag = reg.diagnostics()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_arr, preds_arr))),
        "mae": float(mean_absolute_error(y_arr, preds_arr)),
        "r2": float(r2_score(y_arr, preds_arr)),
        "mean_epistemic_var": float(np.mean(unc) ** 2),
        "expert_count": int(diag.get("n_experts", 0)),
        "uncertainty_rank_correlation": rho,
    }


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_domain_table(domain_id: str, metrics: dict[str, Any]) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "domain": domain_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **json_safe(metrics),
    }
    out = TABLES_DIR / f"{domain_id}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_domain_table(domain_id: str) -> dict[str, Any] | None:
    path = TABLES_DIR / f"{domain_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_domain_tables() -> dict[str, dict[str, Any]]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tables: dict[str, dict[str, Any]] = {}
    for path in sorted(TABLES_DIR.glob("*.json")):
        stem = path.stem
        if not (stem.startswith("d") or stem.startswith("cross_")):
            continue
        key = stem.split("_")[0] if stem.startswith("d") and "_" in stem[2:] else stem
        if key in tables:
            continue
        try:
            tables[key] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return tables


def finalize_domain(domain_id: str, experiments: dict[str, Any]) -> dict[str, Any]:
    result = {"experiments": experiments}
    save_domain_table(domain_id, result)
    return result
