"""
``quantile_dif_train_parity`` vs ``parity_fixtures/dif_train_replay/`` (``replay_u01`` + ``replay_ratio>0``).

CTest: ``native_dif_train_replay``. Override: ``CYPHA_DIF_TRAIN_REPLAY_PARITY_BIN`` (same binary as quantile harness).
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

_FIX = _ROOT / "parity_fixtures" / "dif_train_replay"
_SIDE = _FIX / "sidecar.json"


def test_dif_train_replay_sidecar_has_stream():
    if not _SIDE.is_file():
        pytest.skip("run scripts/generate_dif_train_replay_fixture.py")
    j = json.loads(_SIDE.read_text(encoding="utf-8"))
    assert float(j.get("replay_ratio", 0)) > 0
    ru = j.get("replay_u01")
    assert isinstance(ru, list) and len(ru) >= 1
    n, d_in, k = int(j["n"]), int(j["d_in"]), int(j["K"])
    x = np.asarray(j["x_rowmajor"], dtype=np.float64)
    llr = np.asarray(j["expected_llr_rowmajor"], dtype=np.float64)
    assert x.size == n * d_in
    assert llr.size == n * k


def test_dif_train_replay_parity_subprocess():
    if not _SIDE.is_file() or not (_FIX / "before.cypha").is_file():
        pytest.skip("run scripts/generate_dif_train_replay_fixture.py")
    r = run_native_executable(
        "quantile_dif_train_parity",
        [_FIX],
        timeout=120,
        env_override="CYPHA_DIF_TRAIN_REPLAY_PARITY_BIN",
    )
    if r is None:
        pytest.skip("quantile_dif_train_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
