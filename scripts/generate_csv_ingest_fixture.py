#!/usr/bin/env python3
"""
Emit ``parity_fixtures/csv_ingest/`` for native ``cypha::load_csv_dense`` vs ``CSVDataset.from_file``.

Writes ``cases.json`` with **column names** (``target_col_name``, ``feature_col_names``) and/or **indices**
(``target_col_index``, ``feature_col_indices``) so native resolves headers like Python when names are used.

Includes a **multiline quoted field** (newline inside quotes in the target column) to match RFC4180 / Python
``csv.reader``. Self-checks: chunked ``read_chunk_rows`` loads match full-buffer loads for the numeric fixtures.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cypha_studio.core.dataset import CSVDataset

_OUT = _ROOT / "parity_fixtures" / "csv_ingest"


def _resolve_column_layout(
    path: Path,
    *,
    target_col: Union[str, int] = -1,
    feature_cols: Optional[List[Union[str, int]]] = None,
    has_header: bool = True,
    delimiter: str = ",",
) -> tuple[int, List[int], int]:
    """Mirror ``CSVDataset.from_file`` index rules; returns ``target_idx``, ``feat_indices``, ``ncols``."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header: Optional[List[str]] = None
        if has_header:
            header = next(reader)
        first: Optional[List[str]] = None
        for row in reader:
            if row:
                first = row
                break
        if first is None:
            raise ValueError(f"No data in {path}")
        ncols = len(first)
        if isinstance(target_col, str):
            if header is None:
                raise ValueError("target_col as string requires has_header=True")
            target_idx = header.index(target_col)
        else:
            target_idx = target_col if target_col >= 0 else ncols + target_col

        if feature_cols is not None:
            feat_indices = []
            for c in feature_cols:
                if isinstance(c, str):
                    assert header is not None
                    feat_indices.append(header.index(c))
                else:
                    feat_indices.append(c)
        else:
            feat_indices = [i for i in range(ncols) if i != target_idx]
    return target_idx, feat_indices, ncols


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)

    cls_path = _OUT / "classify.csv"
    cls_path.write_text(
        "f0,f1,f2,label\n"
        '1.0,2.0,3.0,"cat,spam"\n'
        "0.5,-1.25,0.0,dog\n"
        "2.0,2.0,2.0,cat\n",
        encoding="utf-8",
    )
    ds_c = CSVDataset.from_file(
        cls_path, target_col=-1, feature_cols=None, has_header=True, delimiter=",", task="classification"
    )
    for chunk in (1, 2, 100):
        ds_ch = CSVDataset.from_file(
            cls_path,
            target_col=-1,
            feature_cols=None,
            has_header=True,
            delimiter=",",
            task="classification",
            read_chunk_rows=chunk,
        )
        assert np.array_equal(ds_ch.X, ds_c.X) and list(ds_ch.y) == list(ds_c.y)
    t_idx, feat_idx, _ = _resolve_column_layout(cls_path, target_col=-1, feature_cols=None)
    assert feat_idx == [0, 1, 2] and t_idx == 3
    exp_c = {
        "n_rows": int(ds_c.X.shape[0]),
        "n_features": int(ds_c.X.shape[1]),
        "x_rowmajor": ds_c.X.ravel(order="C").tolist(),
        "y_class": [str(x) for x in ds_c.y.tolist()],
    }
    (_OUT / "classify_expected.json").write_text(json.dumps(exp_c, indent=2), encoding="utf-8")

    reg_path = _OUT / "regression.csv"
    reg_path.write_text(
        "x0,x1,junk,y\n"
        "1.0,2.0,999,3.5\n"
        "4.0,5.0,1000,-2.25\n"
        "-1.0,0.0,0,0.0\n",
        encoding="utf-8",
    )
    ds_r = CSVDataset.from_file(
        reg_path,
        target_col="y",
        feature_cols=[0, 1],
        has_header=True,
        delimiter=",",
        task="regression",
    )
    for chunk in (1, 3):
        ds_rch = CSVDataset.from_file(
            reg_path,
            target_col="y",
            feature_cols=[0, 1],
            has_header=True,
            delimiter=",",
            task="regression",
            read_chunk_rows=chunk,
        )
        assert np.array_equal(ds_rch.X, ds_r.X) and list(ds_rch.y) == list(ds_r.y)
    exp_r = {
        "n_rows": int(ds_r.X.shape[0]),
        "n_features": int(ds_r.X.shape[1]),
        "x_rowmajor": ds_r.X.ravel(order="C").tolist(),
        "y_regression": [float(x) for x in ds_r.y.tolist()],
    }
    (_OUT / "regression_expected.json").write_text(json.dumps(exp_r, indent=2), encoding="utf-8")

    mlp = _OUT / "multiline_label.csv"
    # Force LF-only newlines (including inside the quoted cell) for stable parity vs native binary read.
    mlp.write_text(
        "f0,f1,label\n"
        "1.0,2.0,alpha\n"
        '3.0,4.0,"beta\n'
        'gamma"\n'
        "0.0,0.0,delta\n",
        encoding="utf-8",
        newline="\n",
    )
    ds_ml = CSVDataset.from_file(
        mlp,
        target_col="label",
        feature_cols=["f0", "f1"],
        has_header=True,
        delimiter=",",
        task="classification",
    )
    exp_ml = {
        "n_rows": int(ds_ml.X.shape[0]),
        "n_features": int(ds_ml.X.shape[1]),
        "x_rowmajor": ds_ml.X.ravel(order="C").tolist(),
        "y_class": [str(x) for x in ds_ml.y.tolist()],
    }
    assert exp_ml["y_class"][1] == "beta\ngamma", exp_ml["y_class"]
    (_OUT / "multiline_label_expected.json").write_text(json.dumps(exp_ml, indent=2), encoding="utf-8")

    cases = {
        "fixture_schema": 3,
        "cases": [
            {
                "description": "header names for target + features (classification)",
                "csv": "classify.csv",
                "has_header": True,
                "delimiter": ",",
                "target_col_name": "label",
                "feature_col_names": ["f0", "f1", "f2"],
                "task": "classification",
                "expected": "classify_expected.json",
            },
            {
                "description": "header names for target + features (regression)",
                "csv": "regression.csv",
                "has_header": True,
                "delimiter": ",",
                "target_col_name": "y",
                "feature_col_names": ["x0", "x1"],
                "task": "regression",
                "expected": "regression_expected.json",
            },
            {
                "description": "integer target_col_index=-1, default features (all except target)",
                "csv": "classify.csv",
                "has_header": True,
                "delimiter": ",",
                "target_col_index": -1,
                "task": "classification",
                "expected": "classify_expected.json",
            },
            {
                "description": "quoted field with embedded newline (RFC4180-style label cell)",
                "csv": "multiline_label.csv",
                "has_header": True,
                "delimiter": ",",
                "target_col_name": "label",
                "feature_col_names": ["f0", "f1"],
                "task": "classification",
                "expected": "multiline_label_expected.json",
            },
        ],
    }
    (_OUT / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}/")


if __name__ == "__main__":
    main()
