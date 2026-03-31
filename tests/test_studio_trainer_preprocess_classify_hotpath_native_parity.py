"""
``preprocess_train_classify_parity`` vs ``parity_fixtures/studio_trainer_preprocess_classify_hotpath/``.

Raw rows + ``preprocessor.json`` -> ``transform_one`` -> multi-step train + ``batch_llr_from_x``.

CTest: ``native_studio_trainer_preprocess_classify_hotpath``.
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

_FIX = _ROOT / "parity_fixtures" / "studio_trainer_preprocess_classify_hotpath"
_SIDE = _FIX / "sidecar.json"


def test_preprocess_train_classify_hotpath_sidecar_geometry():
    if not _SIDE.is_file():
        pytest.skip("run scripts/generate_studio_trainer_preprocess_classify_hotpath_fixture.py")
    j = json.loads(_SIDE.read_text(encoding="utf-8"))
    d_raw = int(j["d_raw"])
    n, d_in, k = int(j["n"]), int(j["d_in"]), int(j["K"])
    x = np.asarray(j["x_rowmajor"], dtype=np.float64)
    llr = np.asarray(j["expected_llr_rowmajor"], dtype=np.float64)
    assert x.size == n * d_in
    assert llr.size == n * k
    assert len(j["label_order"]) == k
    assert len(j["steps"]) == int(j["n_steps"])
    assert len(j["expected_step_losses"]) == len(j["steps"])
    assert int(j["n_steps"]) == int(j["n_epochs"]) * n
    for st in j["steps"]:
        assert len(st["x_raw"]) == d_raw


def test_preprocess_train_classify_hotpath_subprocess():
    if not _SIDE.is_file() or not (_FIX / "before.cypha").is_file() or not (_FIX / "preprocessor.json").is_file():
        pytest.skip("run scripts/generate_studio_trainer_preprocess_classify_hotpath_fixture.py")
    r = run_native_executable(
        "preprocess_train_classify_parity",
        [_FIX],
        timeout=120,
        env_override="CYPHA_PREPROCESS_TRAIN_CLASSIFY_PARITY_BIN",
    )
    if r is None:
        pytest.skip("preprocess_train_classify_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
