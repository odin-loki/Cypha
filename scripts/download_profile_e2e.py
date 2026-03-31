#!/usr/bin/env python3
"""
Download real tabular data, exercise classification + regression + generation,
and write cProfile reports (cumtime) per phase.

  pip install -r requirements-verify.txt
  python scripts/download_profile_e2e.py
  python scripts/download_profile_e2e.py --fast -o artifacts/profiles/profile_e2e_download.txt

Data (cached under data_cache/ via sklearn data_home):
  - OpenML 1464: Blood Transfusion Service Center (classification, numeric)
  - sklearn.datasets.fetch_california_housing (regression, downloads once)

Fallback if OpenML fails: sklearn iris (bundled).

Training hyperparameters align with ``config/profiled_medium.json`` (classification /
regression / generation blocks).
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# sklearn emits FutureWarning on OpenML parsers in some versions
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from Cypha import CyphaDIF, DIFRegressor, VectorEncoder  # noqa: E402

_DATA_CACHE = _ROOT / "data_cache"


def _ensure_cache() -> None:
    _DATA_CACHE.mkdir(parents=True, exist_ok=True)


def _load_classification_xy() -> Tuple[np.ndarray, np.ndarray, str]:
    """Returns X float64 (N,d), y int labels (N,), name."""
    from sklearn.datasets import fetch_openml, load_iris
    from sklearn.preprocessing import LabelEncoder

    _ensure_cache()
    try:
        bunch = fetch_openml(
            data_id=1464,
            as_frame=False,
            parser="liac-arff",  # no pandas required (parser='auto' needs pandas for dense)
            data_home=str(_DATA_CACHE),
        )
        X = np.asarray(bunch.data, dtype=np.float64)
        y_enc = LabelEncoder().fit_transform(np.asarray(bunch.target).ravel())
        return X, y_enc.astype(np.int64), "openml_1464_blood_transfusion"
    except Exception as e:
        print(f"OpenML 1464 unavailable ({e}); using iris.")
        iris = load_iris()
        return iris.data.astype(np.float64), iris.target.astype(np.int64), "sklearn_iris"


def _load_regression_xy() -> Tuple[np.ndarray, np.ndarray, str]:
    from sklearn.datasets import fetch_california_housing

    _ensure_cache()
    bunch = fetch_california_housing(data_home=str(_DATA_CACHE), download_if_missing=True)
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.float64).ravel()
    return X, y, "california_housing"


def _split(
    X: np.ndarray, y: np.ndarray, train_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_tr = max(1, int(n * train_frac))
    tr, te = idx[:n_tr], idx[n_tr:]
    return X[tr], y[tr], X[te], y[te]


def _metrics_classification(
    clf: CyphaDIF, X_te: np.ndarray, y_te: np.ndarray, max_test: int
) -> Dict[str, float]:
    if len(X_te) == 0:
        return {"accuracy": float("nan"), "n_test": 0.0}
    if len(X_te) > max_test:
        X_te, y_te = X_te[:max_test], y_te[:max_test]
    out = clf.batch_infer([X_te[i] for i in range(len(X_te))])
    pred = np.array([out[i][0] for i in range(len(out))])
    y_str = np.array([str(int(v)) for v in y_te])
    acc = float(np.mean(pred == y_str))
    return {"accuracy": acc, "n_test": float(len(X_te))}


def _metrics_regression(
    reg: DIFRegressor, X_te: np.ndarray, y_te: np.ndarray, max_test: int
) -> Dict[str, float]:
    if len(X_te) == 0:
        return {"mae": float("nan"), "n_test": 0.0}
    if len(X_te) > max_test:
        X_te, y_te = X_te[:max_test], y_te[:max_test]
    yp, _ = reg.predict_batch([X_te[i] for i in range(len(X_te))])
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)
    ye = y_te.reshape(-1, 1) if y_te.ndim == 1 else y_te
    mae = float(np.mean(np.abs(yp.ravel() - ye.ravel())))
    return {"mae": mae, "n_test": float(len(X_te))}


def workload_classification(
    fast: bool,
    seed: int,
    scaler,
    max_test: int,
    xy: Tuple[np.ndarray, np.ndarray, str] | None = None,
) -> Tuple[Dict[str, Any], CyphaDIF]:
    X, y, name = xy if xy is not None else _load_classification_xy()
    X_tr, y_tr, X_te, y_te = _split(X, y, 0.85 if not fast else 0.75, seed)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    d = X_tr.shape[1]
    n_classes = int(y.max()) + 1
    rng = np.random.default_rng(seed)
    clf = CyphaDIF(
        encoder=VectorEncoder(d),
        field_dim=128,
        world_lr=0.008,
        delta_lr=0.05,
        enc_lr=0.002,
        mdl_lambda=0.001,
        context_win=32,
        rng=np.random.default_rng(seed + 1),
    )
    clf.temperature = 1.15

    n_epochs = 2 if fast else 3
    passes = min(len(X_tr), 120 if fast else 400)
    for ep in range(n_epochs):
        order = rng.permutation(len(X_tr))
        for i in order[:passes]:
            clf.train_step(X_tr[i], str(int(y_tr[i])))

    # stress inference
    xs_tr = [X_tr[i] for i in range(min(passes, len(X_tr)))]
    n_rep = 2 if fast else 5
    for _ in range(n_rep):
        clf.batch_infer(xs_tr)
        for x in xs_tr[: min(24, len(xs_tr))]:
            clf.infer(x)
        H = clf.batch_encode(xs_tr)
        clf.score_matrix(H, use_field=True)
        clf.world_gate_vector(H, use_field=True)

    metrics = {
        "dataset": name,
        "d": d,
        "n_classes": n_classes,
        "train_points": float(passes * n_epochs),
        **_metrics_classification(clf, X_te, y_te, max_test),
    }
    return metrics, clf


def workload_regression(
    fast: bool,
    seed: int,
    scaler,
    max_test: int,
    xy: Tuple[np.ndarray, np.ndarray, str] | None = None,
) -> Dict[str, Any]:
    X, y, name = xy if xy is not None else _load_regression_xy()
    X_tr, y_tr, X_te, y_te = _split(X, y, 0.85 if not fast else 0.75, seed + 7)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    d = X_tr.shape[1]
    reg = DIFRegressor(
        encoder=VectorEncoder(d),
        field_dim=128,
        n_experts=8,
        target_lr=0.06,
        rng=np.random.default_rng(seed + 2),
    )

    passes = min(len(X_tr), 200 if fast else 900)
    rng = np.random.default_rng(seed + 3)
    order = rng.permutation(len(X_tr))
    for i in order[:passes]:
        reg.train_step(X_tr[i], float(y_tr[i]))

    xs_tr = [X_tr[i] for i in range(min(passes, len(X_tr)))]
    n_rep = 2 if fast else 4
    for _ in range(n_rep):
        reg.predict_batch(xs_tr)
        for x in xs_tr[: min(32, len(xs_tr))]:
            reg.predict(x)

    metrics = {
        "dataset": name,
        "d": d,
        "train_points": float(passes),
        **_metrics_regression(reg, X_te, y_te, max_test),
    }
    return metrics


def workload_generation(clf: CyphaDIF, fast: bool) -> Dict[str, Any]:
    with clf.memory._lock:
        labels = list(clf.memory._label_order)
    if not labels:
        return {"n_samples": 0.0, "skipped": True}

    lab = labels[0]
    n_per_call = 4 if fast else 8
    n_calls = 2 if fast else 8
    temp = 1.0
    max_cand = 16
    # rejection path exercises fused_score_llr on candidate batches (profiled medium)
    for _ in range(n_calls):
        clf.generate(
            lab,
            n=n_per_call,
            temperature=temp,
            rejection_sampling=True,
            max_candidates=max_cand,
        )
    # simple path
    for _ in range(1 if fast else 2):
        clf.generate(lab, n=min(8, n_per_call), temperature=0.8, rejection_sampling=False)

    return {
        "n_samples": float(n_per_call * n_calls),
        "label": lab,
        "skipped": False,
    }


def _profile_section(title: str, fn: Callable[[], None]) -> str:
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    buf = io.StringIO()
    import pstats

    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumtime").print_stats(50)
    return f"{'=' * 72}\n{title}\n{'=' * 72}\n" + buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="smaller training / fewer repeats")
    ap.add_argument(
        "-o",
        "--output",
        default=str(_ROOT / "artifacts" / "profiles" / "profile_e2e_download.txt"),
        help="Combined profile + metrics (text)",
    )
    ap.add_argument(
        "--metrics-json",
        default="",
        help="Optional path to write metrics JSON",
    )
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler

    seed = 42
    max_test = 400 if args.fast else 5000
    clf_holder: Dict[str, Any] = {}

    print("Loading / caching datasets (OpenML + California housing; may use network once)...", flush=True)
    cls_xy = _load_classification_xy()
    reg_xy = _load_regression_xy()
    print(f"  classification: {cls_xy[2]}  X={cls_xy[0].shape}", flush=True)
    print(f"  regression:     {reg_xy[2]}  X={reg_xy[0].shape}", flush=True)

    def _cls():
        m, clf = workload_classification(
            args.fast, seed, StandardScaler(), max_test, xy=cls_xy
        )
        clf_holder["clf"] = clf
        clf_holder["metrics_cls"] = m

    def _reg():
        clf_holder["metrics_reg"] = workload_regression(
            args.fast, seed, StandardScaler(), max_test, xy=reg_xy
        )

    def _gen():
        clf = clf_holder.get("clf")
        if clf is None:
            return
        clf_holder["metrics_gen"] = workload_generation(clf, args.fast)

    sections: List[str] = []
    sections.append(_profile_section("1) CLASSIFICATION (train + batch_infer + score_matrix + gate)", _cls))
    sections.append(_profile_section("2) REGRESSION (DIFRegressor train + predict_batch)", _reg))
    sections.append(_profile_section("3) GENERATION (CyphaDIF.generate rejection + simple)", _gen))

    summary = {
        "classification": clf_holder.get("metrics_cls"),
        "regression": clf_holder.get("metrics_reg"),
        "generation": clf_holder.get("metrics_gen"),
        "fast": args.fast,
    }
    header = (
        "# Cypha end-to-end profile (downloaded / cached real data)\n"
        f"# fast={args.fast}\n"
        f"# metrics:\n{json.dumps(summary, indent=2)}\n\n"
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n\n".join(sections), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote profile: {out}")
    if args.metrics_json:
        mp = Path(args.metrics_json)
        mp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote metrics: {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
