"""
``preprocess_train_classify_parity`` vs ``parity_fixtures/csv_preprocess_classify_hotpath/``.

Same goldens as ``studio_trainer_preprocess_classify_hotpath/``, but training rows come from
``train.csv`` via native ``load_csv_dense`` (``csv_spec``) before ``transform_one`` and
``dif_train_classify_sequence``.

CTest: ``native_csv_preprocess_classify_hotpath``.
Override: ``CYPHA_PREPROCESS_TRAIN_CLASSIFY_PARITY_BIN``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures" / "csv_preprocess_classify_hotpath"
_SIDE = _FIX / "sidecar.json"


def test_csv_preprocess_classify_hotpath_sidecar_geometry():
    if not _SIDE.is_file():
        pytest.skip("run scripts/generate_csv_preprocess_classify_hotpath_fixture.py")
    j = json.loads(_SIDE.read_text(encoding="utf-8"))
    d_raw = int(j["d_raw"])
    n, d_in, k = int(j["n"]), int(j["d_in"]), int(j["K"])
    x = np.asarray(j["x_rowmajor"], dtype=np.float64)
    llr = np.asarray(j["expected_llr_rowmajor"], dtype=np.float64)
    assert x.size == n * d_in
    assert llr.size == n * k
    assert len(j["label_order"]) == k
    assert int(j["n_steps"]) == int(j["n_epochs"]) * n
    assert j["csv"] == "train.csv"
    assert "steps" not in j
    cs = j["csv_spec"]
    assert cs["has_header"] is True
    assert cs["delimiter"] == ","
    if cs.get("target_col_name"):
        assert cs["target_col_name"] == "label"
        assert cs["feature_col_names"] == [f"f{i}" for i in range(d_raw)]
    else:
        assert cs["target_col_index"] == d_raw
        assert cs["feature_col_indices"] == list(range(d_raw))
    exp = j["expected_step_losses"]
    assert len(exp) == int(j["n_steps"])
    train_csv = _FIX / "train.csv"
    assert train_csv.is_file()
    lines = train_csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == int(j["n_steps"]) + 1


def test_csv_preprocess_classify_hotpath_subprocess():
    if not _SIDE.is_file() or not (_FIX / "before.cypha").is_file() or not (_FIX / "preprocessor.json").is_file():
        pytest.skip("run scripts/generate_csv_preprocess_classify_hotpath_fixture.py")
    r = run_native_executable(
        "preprocess_train_classify_parity",
        [_FIX],
        timeout=120,
        env_override="CYPHA_PREPROCESS_TRAIN_CLASSIFY_PARITY_BIN",
    )
    if r is None:
        pytest.skip("preprocess_train_classify_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
