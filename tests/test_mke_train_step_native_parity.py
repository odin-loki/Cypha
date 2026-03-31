"""
``mke_train_step_parity`` vs ``parity_fixtures/mke_train_step/`` and ``mke_train_extended/``.

End-to-end MKERegressor scalar step: RFF, expert RLS, router ``dif_train_step_vector``.
CTest: ``native_mke_train_step``, ``native_mke_train_extended``.
Override: ``CYPHA_MKE_TRAIN_STEP_PARITY_BIN``.
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

from Cypha import MKERegressor

_FIX = _ROOT / "parity_fixtures" / "mke_train_step"
_SIDE = _FIX / "sidecar.json"
_FIX_EXT = _ROOT / "parity_fixtures" / "mke_train_extended"
_SIDE_EXT = _FIX_EXT / "sidecar.json"


def test_mke_train_step_sidecar_geometry():
    if not _SIDE.is_file():
        pytest.skip("run scripts/generate_mke_train_step_fixture.py")
    j = json.loads(_SIDE.read_text(encoding="utf-8"))
    d_in = int(j["d_in"])
    d_rff = int(j["D_rff"])
    k = int(j["K"])
    rw = j["rff_W_rowmajor"]
    assert len(rw) == d_rff * d_in
    assert len(j["rff_b"]) == d_rff
    assert len(j["routing_labs"]) == k
    assert len(j["routing_probs"]) == k
    assert len(j["gh_scales"]) == k


def test_mke_train_step_parity_subprocess():
    if not _SIDE.is_file() or not (_FIX / "before.cypha").is_file():
        pytest.skip("run scripts/generate_mke_train_step_fixture.py")
    r = run_native_executable(
        "mke_train_step_parity",
        [_FIX],
        timeout=120,
        env_override="CYPHA_MKE_TRAIN_STEP_PARITY_BIN",
    )
    if r is None:
        pytest.skip("mke_train_step_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_mke_train_extended_sidecar_geometry():
    if not _SIDE_EXT.is_file():
        pytest.skip("run scripts/generate_mke_train_extended_fixture.py")
    j = json.loads(_SIDE_EXT.read_text(encoding="utf-8"))
    d_in = int(j["d_in"])
    d_rff = int(j["D_rff"])
    steps = j["steps"]
    assert isinstance(steps, list) and len(steps) >= 1
    s0 = steps[0]
    rw = j["rff_W_rowmajor"]
    assert len(rw) == d_rff * d_in
    assert len(j["rff_b"]) == d_rff
    labs = s0["routing_labs"]
    k = len(labs)
    assert len(s0["routing_probs"]) == k
    assert len(s0["gh_scales"]) == k


def test_mke_train_extended_parity_subprocess():
    if not _SIDE_EXT.is_file() or not (_FIX_EXT / "before.cypha").is_file():
        pytest.skip("run scripts/generate_mke_train_extended_fixture.py")
    r = run_native_executable(
        "mke_train_step_parity",
        [_FIX_EXT],
        timeout=300,
        env_override="CYPHA_MKE_TRAIN_STEP_PARITY_BIN",
    )
    if r is None:
        pytest.skip("mke_train_step_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_mkeregressor_from_data_replay_rng_forwards_to_clf():
    """``from_data(..., replay_rng=)`` wires ``CyphaDIF`` replay sampling (native parity fixtures)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((24, 2))
    y = X[:, 0] * 0.5 + rng.standard_normal(24) * 0.1
    rr = np.random.default_rng(999)
    mke = MKERegressor.from_data(X, y, K=2, D=12, field_dim=16, rng_seed=3, replay_rng=rr)
    assert mke.clf._replay_rng is rr
