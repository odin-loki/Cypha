"""
Regression tests against committed parity_fixtures/.

Native ports: load reference.cypha, replay x_input, compare to expected.npz.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary

_ROOT = Path(__file__).resolve().parents[1]
_FIX = _ROOT / "parity_fixtures"


def _require_fixtures():
    if not (_FIX / "manifest.json").is_file():
        pytest.skip("parity_fixtures/ missing — run: python scripts/generate_parity_fixtures.py")


@pytest.fixture(scope="module")
def manifest():
    _require_fixtures()
    return json.loads((_FIX / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def expected():
    _require_fixtures()
    return np.load(_FIX / "expected.npz")


@pytest.fixture(scope="module")
def clf(manifest):
    state = cypha_load_binary(str(_FIX / "reference.cypha"))
    m = manifest["model"]
    enc = VectorEncoder(int(m["input_dim"]))
    model = CyphaDIF(
        encoder=enc,
        field_dim=int(m["field_dim"]),
        rng=np.random.default_rng(0),
    )
    model.load_state(state)
    return model


def test_manifest_fixture_schema(manifest):
    assert manifest.get("fixture_schema") == 2


def test_reference_fixture_restores_tier1_for_native_cypha_parity(clf):
    """Committed ``reference.cypha`` includes Tier-1 (``ctx_hist_packed``); native ``cypha_parity`` loads it via ``from_root``."""
    with clf.context._lock:
        assert clf.context._t1_total > 0.0
        assert len(clf.context._history) > 0
        assert clf.context._last_label is not None


def test_train_hparams_json_for_native_rest():
    _require_fixtures()
    p = _FIX / "train_hparams.json"
    assert p.is_file()
    j = json.loads(p.read_text(encoding="utf-8"))
    for key in ("world_lr", "delta_lr", "ood_sigma", "enc_lr", "replay_ratio"):
        assert key in j
        assert isinstance(j[key], (int, float))
    assert "replay_cap" in j
    assert isinstance(j["replay_cap"], int)
    assert "align_every" in j and isinstance(j["align_every"], int) and j["align_every"] > 0
    assert "temp_recalib_every" in j and isinstance(j["temp_recalib_every"], int)


def test_mke_train_step_sidecar_for_native():
    """Committed ``mke_train_step/sidecar.json`` keys for ``mke_train_step_parity`` / CTest ``native_mke_train_step``."""
    _require_fixtures()
    p = _FIX / "mke_train_step" / "sidecar.json"
    if not p.is_file():
        pytest.skip("run scripts/generate_mke_train_step_fixture.py")
    j = json.loads(p.read_text(encoding="utf-8"))
    for key in (
        "d_in",
        "D_rff",
        "total_steps_start",
        "expected_phi",
        "routing_labs",
        "expected_err_sq",
        "router_train_label",
        "expected_router_loss",
        "temperature",
        "world_lr",
        "delta_lr",
        "w_before",
        "w_after",
        "P_before",
        "P_after",
    ):
        assert key in j
    assert isinstance(j["routing_labs"], list) and len(j["routing_labs"]) >= 1


def test_mke_train_extended_sidecar_for_native():
    """Committed ``mke_train_extended/sidecar.json`` for ``mke_train_step_parity`` extended mode / CTest ``native_mke_train_extended``."""
    _require_fixtures()
    p = _FIX / "mke_train_extended" / "sidecar.json"
    if not p.is_file():
        pytest.skip("run scripts/generate_mke_train_extended_fixture.py")
    j = json.loads(p.read_text(encoding="utf-8"))
    assert int(j.get("fixture_schema", 0)) >= 2
    assert int(j.get("n_extended_steps", len(j.get("steps", [])))) == len(j["steps"])
    assert "steps" in j and isinstance(j["steps"], list) and len(j["steps"]) >= 1
    assert "replay_warmup" in j and isinstance(j["replay_warmup"], list) and len(j["replay_warmup"]) >= 10
    ru = j["replay_u01"]
    assert isinstance(ru, list) and len(ru) >= 1
    for key in (
        "d_in",
        "D_rff",
        "total_steps_start",
        "enc_update_count_start",
        "temperature",
        "world_lr",
        "delta_lr",
    ):
        assert key in j
    assert float(j["enc_lr"]) > 0.0
    assert float(j["replay_ratio"]) > 0.0
    s0 = j["steps"][0]
    for sk in (
        "x",
        "y",
        "expected_phi",
        "routing_labs",
        "expected_err_sq",
        "router_train_label",
        "expected_router_loss",
        "w_before",
        "w_after",
        "P_before",
        "P_after",
        "enc_w_rowmajor",
    ):
        assert sk in s0


def test_train_step_vector_sidecar_for_native():
    _require_fixtures()
    manifest = json.loads((_FIX / "manifest.json").read_text(encoding="utf-8"))
    p = _FIX / "train_step_vector" / "sidecar.json"
    assert p.is_file()
    j = json.loads(p.read_text(encoding="utf-8"))
    for key in (
        "x",
        "label",
        "expected_loss",
        "total_steps_before",
        "world_lr",
        "delta_lr",
        "ood_sigma",
        "enc_lr",
    ):
        assert key in j
    assert isinstance(j["x"], list)
    assert len(j["x"]) == int(manifest["model"]["input_dim"])


def test_manifest_labels_match_score_matrix(clf, manifest, expected):
    H = clf.batch_encode([expected["x_input"][i] for i in range(len(expected["x_input"]))])
    LLR, labels = clf.score_matrix(H, use_field=True)
    assert labels == manifest["labels"]
    np.testing.assert_allclose(LLR, expected["llr"], rtol=0, atol=1e-12)


def test_softmax_and_gate_match_expected(clf, expected):
    H = clf.batch_encode([expected["x_input"][i] for i in range(len(expected["x_input"]))])
    LLR, _ = clf.score_matrix(H, use_field=True)
    T = float(expected["temperature"][0])
    eps = float(expected["eps"][0])
    from Cypha import _softmax_batch

    probs = _softmax_batch(LLR / (T + eps))
    gates = clf.world_gate_vector(H, use_field=True)
    np.testing.assert_allclose(probs, expected["probs"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(gates, expected["gates"], rtol=0, atol=1e-12)


def test_batch_infer_matches_fixture(clf, expected):
    xs = [expected["x_input"][i] for i in range(len(expected["x_input"]))]
    out = clf.batch_infer(xs, use_field=True)
    man = json.loads((_FIX / "manifest.json").read_text(encoding="utf-8"))
    order = man["labels"]
    for i, (lab, conf) in enumerate(out):
        assert lab == order[int(expected["pred_idx"][i])]
        np.testing.assert_allclose(conf, expected["conf_batch"][i], rtol=0, atol=1e-10)


def test_batch_infer_full_matches_fixture(clf, expected, manifest):
    """M1: ``batch_infer_full`` rows (LLR/prob dicts, entropy, anomaly) vs ``score_matrix`` pipeline."""
    from Cypha import _EPS

    xs = [expected["x_input"][i] for i in range(len(expected["x_input"]))]
    rows = clf.batch_infer_full(xs, use_field=True)
    order = manifest["labels"]
    H = clf.batch_encode(xs)
    LLR, labels = clf.score_matrix(H, use_field=True)
    assert labels == order
    probs = expected["probs"]
    gates = expected["gates"]
    for i, row in enumerate(rows):
        assert row["label"] == order[int(expected["pred_idx"][i])]
        np.testing.assert_allclose(row["confidence"], expected["conf_batch"][i], rtol=0, atol=1e-10)
        for j, lbl in enumerate(order):
            np.testing.assert_allclose(row["llrs"][lbl], LLR[i, j], rtol=0, atol=1e-12)
            np.testing.assert_allclose(row["probs"][lbl], probs[i, j], rtol=0, atol=1e-12)
        ent = float(-np.sum(probs[i] * np.log(probs[i] + _EPS)))
        np.testing.assert_allclose(row["entropy"], ent, rtol=0, atol=1e-12)
        np.testing.assert_allclose(row["anomaly_score"], float(1.0 - gates[i]), rtol=0, atol=1e-12)


def test_infer_serial_matches_fixture(clf, expected, manifest):
    order = manifest["labels"]
    for i in range(len(expected["x_input"])):
        pred, conf = clf.infer(expected["x_input"][i])
        assert pred == order[int(expected["pred_idx"][i])]
        np.testing.assert_allclose(conf, expected["serial_conf"][i], rtol=0, atol=1e-10)


def test_batch_infer_matches_infer(clf, expected):
    xs = [expected["x_input"][i] for i in range(len(expected["x_input"]))]
    batch = clf.batch_infer(xs, use_field=True)
    for i, x in enumerate(xs):
        p, c = clf.infer(x)
        assert batch[i][0] == p
        np.testing.assert_allclose(batch[i][1], c, rtol=0, atol=1e-10)


def test_regression_head_fixture_covers_manifest_labels(manifest):
    """Committed ``regression_head.json`` keys cover every routing label in ``manifest.json``."""
    p = _FIX / "regression_head.json"
    if not p.is_file():
        pytest.skip("parity_fixtures/regression_head.json missing")
    head = json.loads(p.read_text(encoding="utf-8"))
    ex = head.get("experts")
    assert isinstance(ex, dict)
    for lbl in manifest["labels"]:
        assert lbl in ex, f"missing expert for label {lbl!r}"
        row = ex[lbl]
        assert isinstance(row.get("mu"), (int, float)), lbl
        assert isinstance(row.get("var_ema", 0.0), (int, float)), lbl
