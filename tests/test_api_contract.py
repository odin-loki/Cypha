"""Stable REST surface for Qt / C++ clients (route names + schema classes)."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="FastAPI not installed (cypha_studio/requirements.txt)")
pytest.importorskip("pydantic", reason="Pydantic not installed (cypha_studio/requirements.txt)")

from cypha_studio.server import api as api_mod


def test_module_exposes_default_asgi_app():
    assert hasattr(api_mod, "app")
    paths = [getattr(r, "path", "") for r in api_mod.app.routes]
    assert "/metrics" in paths


def test_module_default_app_has_registry_from_cypha_registry_root():
    """Default ``api:app`` (``uvicorn cypha_studio.server.api:app``) uses ``ModelRegistry(registry_root())``."""
    from cypha_studio.env_config import registry_root_expanded

    reg = api_mod.app.state.registry
    assert reg is not None
    assert reg.root.resolve() == registry_root_expanded().resolve()


def test_create_app_without_arguments_leaves_registry_none():
    """Embedded servers that call ``create_app()`` with defaults do not attach a registry unless passed."""
    app = api_mod.create_app()
    assert app.state.registry is None


def test_default_module_app_get_models_returns_200():
    """``GET /models`` on ``uvicorn … api:app`` scans ``CYPHA_REGISTRY_ROOT`` (may be empty)."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    from fastapi.testclient import TestClient

    client = TestClient(api_mod.app)
    r = client.get("/models")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {"models"}
    assert isinstance(body["models"], list)


def test_default_module_app_metrics_includes_registry_model_count():
    """``GET /metrics`` reports ``registry_model_count`` when the default app has a registry."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    from fastapi.testclient import TestClient

    client = TestClient(api_mod.app)
    r = client.get("/metrics")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "registry_model_count" in data
    assert isinstance(data["registry_model_count"], int)
    assert data["registry_model_count"] >= 0


def test_create_app_exposes_expected_routes():
    app = api_mod.create_app(engine=None, registry=None, session=None)
    paths = [getattr(r, "path", "") for r in app.routes]
    session_methods = {
        tuple(sorted(getattr(r, "methods", ()) or ()))
        for r in app.routes
        if getattr(r, "path", "") == "/session"
    }
    assert any("DELETE" in m for m in session_methods), session_methods
    assert "/health" in paths
    assert "/ready" in paths
    assert "/metrics" in paths
    assert "/predict" in paths
    assert "/update" in paths
    assert "/load" in paths
    assert "/models" in paths
    assert "/session" in paths
    assert "/session/rng" in paths
    assert "/classes" in paths
    assert "/adapt_temperature" in paths
    assert "/register" in paths


def test_create_app_post_routes_include_predict_update_load_adapt():
    """Inference/registry POST surfaces are registered (method + path)."""
    app = api_mod.create_app(engine=None, registry=None, session=None)
    post_paths = {
        getattr(r, "path", "")
        for r in app.routes
        if "POST" in (getattr(r, "methods", ()) or ())
    }
    for p in ("/predict", "/update", "/load", "/register", "/adapt_temperature"):
        assert p in post_paths, post_paths


def test_predict_request_schema_fields():
    fields = set(api_mod.PredictRequest.model_fields.keys())
    assert fields >= {"input", "use_gh", "return_explanation"}


def test_predict_response_schema_fields():
    fields = set(api_mod.PredictResponse.model_fields.keys())
    assert fields >= {
        "label",
        "confidence",
        "all_scores",
        "anomaly_score",
        "is_ood",
        "latency_ms",
        "regression_val",
        "uncertainty",
    }


def test_predict_post_roundtrip_with_engine():
    """FastAPI TestClient + tiny CyphaDIF (needs httpx — requirements-verify.txt)."""
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(42)
    clf = CyphaDIF(VectorEncoder(4), field_dim=48, rng=rng)
    for i in range(40):
        x = rng.standard_normal(4)
        clf.train_step(x, str(i % 3))

    eng = InferenceEngine(clf, None)
    sess = InferenceSession(eng)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=sess)
    client = TestClient(app)
    r = client.post(
        "/predict",
        json={"input": [0.1, 0.2, -0.1, 0.3], "use_gh": False, "return_explanation": False},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "label" in data and "confidence" in data
    assert isinstance(data.get("all_scores"), dict)


@pytest.mark.parametrize("route", ["/predict", "/update", "/adapt_temperature"])
def test_predict_update_adapt_malformed_json_returns_422(route):
    """Starlette/FastAPI reject invalid JSON before the route runs (native ``cypha_rest`` → **400** ``detail``)."""
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(1)
    clf = CyphaDIF(VectorEncoder(3), field_dim=24, rng=rng)
    for i in range(10):
        clf.train_step(rng.standard_normal(3), str(i % 2))
    eng = InferenceEngine(clf, None)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=InferenceSession(eng))
    client = TestClient(app)
    r = client.post(
        route,
        content="{",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422, route
    body = r.json()
    assert "detail" in body
    assert body["detail"], route


def test_predict_update_adapt_wrong_input_dim_returns_400():
    """Encoder length mismatch → **400** + same ``detail`` string as native ``cypha_rest``."""
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(21)
    clf = CyphaDIF(VectorEncoder(4), field_dim=36, rng=rng)
    for i in range(14):
        clf.train_step(rng.standard_normal(4), str(i % 3))
    eng = InferenceEngine(clf, None)
    sess = InferenceSession(eng)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=sess)
    client = TestClient(app)
    want = {"detail": "input dim mismatch after preprocessor"}

    pr = client.post(
        "/predict",
        json={"input": [0.0, 0.0, 0.0], "use_gh": False, "return_explanation": False},
    )
    assert pr.status_code == 400
    assert pr.json() == want

    ur = client.post(
        "/update",
        json={"input": [0.0, 0.0], "correct_label": "0", "use_gh": False},
    )
    assert ur.status_code == 400
    assert ur.json() == want

    ar = client.post(
        "/adapt_temperature",
        json={
            "calibration": [{"input": [1.0, 2.0], "correct_label": "0"}],
            "n_grid": 5,
            "T_min": 0.5,
            "T_max": 2.0,
            "n_bins": 5,
        },
    )
    assert ar.status_code == 400
    assert ar.json() == want


def test_adapt_temperature_request_schema_fields():
    cal_fields = set(api_mod.AdaptTemperatureCalibrationRow.model_fields.keys())
    assert cal_fields >= {"input", "correct_label"}
    req_fields = set(api_mod.AdaptTemperatureRequest.model_fields.keys())
    assert req_fields >= {"calibration", "n_grid", "T_min", "T_max", "n_bins"}
    resp_fields = set(api_mod.AdaptTemperatureResponse.model_fields.keys())
    assert resp_fields >= {"temperature", "n_used"}


def test_adapt_temperature_post_roundtrip_with_engine():
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(44)
    clf = CyphaDIF(VectorEncoder(4), field_dim=40, rng=rng)
    for i in range(30):
        clf.train_step(rng.standard_normal(4), str(i % 3))
    eng = InferenceEngine(clf, None)
    sess = InferenceSession(eng)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=sess)
    client = TestClient(app)
    xs = [rng.standard_normal(4) for _ in range(12)]
    body = {
        "calibration": [{"input": x.tolist(), "correct_label": str(i % 3)} for i, x in enumerate(xs)],
        "n_grid": 12,
        "T_min": 0.4,
        "T_max": 4.0,
        "n_bins": 10,
    }
    r = client.post("/adapt_temperature", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "temperature" in data and "n_used" in data
    assert data["n_used"] == 12
    assert np.isfinite(data["temperature"]) and float(data["temperature"]) > 0


def test_update_post_roundtrip_with_engine():
    """POST /update applies online correction via InferenceEngine."""
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(43)
    clf = CyphaDIF(VectorEncoder(4), field_dim=48, rng=rng)
    for i in range(50):
        clf.train_step(rng.standard_normal(4), str(i % 3))

    eng = InferenceEngine(clf, None)
    assert eng.n_corrections == 0
    sess = InferenceSession(eng)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=sess)
    client = TestClient(app)
    r = client.post(
        "/update",
        json={
            "input": [0.5, -0.2, 0.1, 0.0],
            "correct_label": "1",
            "use_gh": False,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert set(data.keys()) == {"loss", "n_corrections"}
    assert isinstance(data["loss"], (int, float)) and np.isfinite(float(data["loss"]))
    assert isinstance(data["n_corrections"], int) and data["n_corrections"] >= 1
    assert eng.n_corrections >= 1


def test_update_regression_y_requires_mke_block():
    """``regression_y`` without ``mke`` in regression head returns 400 (same as native)."""
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(44)
    clf = CyphaDIF(VectorEncoder(4), field_dim=48, rng=rng)
    for i in range(10):
        clf.train_step(rng.standard_normal(4), str(i % 2))
    eng = InferenceEngine(clf, None)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=InferenceSession(eng))
    client = TestClient(app)
    base = {"input": [0.1, 0.2, 0.3, 0.4], "correct_label": "0", "use_gh": False}
    r = client.post("/update", json={**base, "regression_y": 1.0})
    assert r.status_code == 400
    assert "mke block" in r.json()["detail"]


def test_update_replay_u01_classification_ok():
    """``replay_u01`` on classification ``/update`` is accepted (no longer 501)."""
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(45)
    clf = CyphaDIF(VectorEncoder(4), field_dim=48, rng=rng, replay_ratio=0.3)
    for i in range(50):
        clf.train_step(rng.standard_normal(4), str(i % 3))
    eng = InferenceEngine(clf, None)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=InferenceSession(eng))
    client = TestClient(app)
    r = client.post(
        "/update",
        json={
            "input": [0.5, -0.2, 0.1, 0.0],
            "correct_label": "1",
            "use_gh": False,
            "replay_u01": [0.99, 0.01, 0.5, 0.5],
        },
    )
    assert r.status_code == 200, r.text
    assert np.isfinite(float(r.json()["loss"]))


def test_update_router_train_label_ignored_without_regression_y():
    """``router_train_label`` without ``regression_y`` follows plain classification (native ignores it)."""
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(46)
    clf = CyphaDIF(VectorEncoder(4), field_dim=48, rng=rng)
    for i in range(20):
        clf.train_step(rng.standard_normal(4), str(i % 2))
    eng = InferenceEngine(clf, None)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=InferenceSession(eng))
    client = TestClient(app)
    r = client.post(
        "/update",
        json={
            "input": [0.1, 0.2, 0.3, 0.4],
            "correct_label": "0",
            "use_gh": False,
            "router_train_label": "_e_0",
        },
    )
    assert r.status_code == 200, r.text


def test_fastapi_mke_update_router_loss_matches_fixture(tmp_path):
    """FastAPI ``POST /update`` with ``regression_y`` + ``mke`` matches ``mke_train_step`` sidecar."""
    import json
    from pathlib import Path

    import numpy as np
    from Cypha import CyphaDIF, RFFEncoder, cypha_load_binary
    from fastapi.testclient import TestClient

    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry
    from tests.test_cypha_rest_smoke import _mke_regression_head_from_sidecar

    fix = Path(__file__).resolve().parents[1] / "parity_fixtures" / "mke_train_step"
    side_path = fix / "sidecar.json"
    if not side_path.is_file() or not (fix / "before.cypha").is_file():
        pytest.skip("mke_train_step fixture missing")

    side = json.loads(side_path.read_text(encoding="utf-8"))
    reg_path = tmp_path / "regression_head.json"
    reg_path.write_text(json.dumps(_mke_regression_head_from_sidecar(side)), encoding="utf-8")

    enc = RFFEncoder(int(side["d_in"]), D=int(side["D_rff"]), gamma=1.0, seed=901)
    clf = CyphaDIF(
        enc,
        field_dim=24,
        rng=np.random.default_rng(997),
        enc_lr=0.0,
        replay_ratio=0.0,
    )
    clf.load_state(cypha_load_binary(str(fix / "before.cypha")))

    eng = InferenceEngine(clf, None)
    app = api_mod.create_app(
        engine=eng,
        registry=ModelRegistry(),
        session=InferenceSession(eng),
        regression_head_path=str(reg_path),
    )
    client = TestClient(app)
    r = client.post(
        "/update",
        json={
            "input": side["x"],
            "correct_label": side["router_train_label"],
            "use_gh": True,
            "regression_y": side["y"],
            "router_train_label": side["router_train_label"],
        },
    )
    assert r.status_code == 200, r.text
    np.testing.assert_allclose(
        float(r.json()["loss"]),
        float(side["expected_router_loss"]),
        rtol=0,
        atol=1e-8,
    )


def test_register_post_no_registry_returns_503():
    """``POST /register`` requires ``app.state.registry`` (same as native without ``--registry``)."""
    from fastapi.testclient import TestClient

    app = api_mod.create_app(engine=None, registry=None, session=None)
    client = TestClient(app)
    r = client.post(
        "/register",
        json={
            "name": "x",
            "version": "1.0.0",
            "model_cypha": "/tmp/m.cypha",
            "card_json": "/tmp/c.json",
            "overwrite": False,
        },
    )
    assert r.status_code == 503


def test_register_post_copies_bundle_when_registry_configured(tmp_path):
    """FastAPI ``POST /register`` mirrors native ``registry_register_bundle`` (file copy into registry tree)."""
    from fastapi.testclient import TestClient

    from cypha_studio.core.registry import ModelRegistry

    src_cypha = tmp_path / "src.cypha"
    src_card = tmp_path / "src_card.json"
    src_pre = tmp_path / "pre.json"
    src_cypha.write_bytes(b"fake cypha bytes")
    src_card.write_text('{"name":"n","version":"1.0.0"}', encoding="utf-8")
    src_pre.write_text('{"schema":1}', encoding="utf-8")

    reg = ModelRegistry(str(tmp_path / "registry_root"))
    app = api_mod.create_app(engine=None, registry=reg, session=None)
    client = TestClient(app)
    r = client.post(
        "/register",
        json={
            "name": "bundle_test",
            "version": "2.1.0",
            "model_cypha": str(src_cypha),
            "card_json": str(src_card),
            "preprocessor_json": str(src_pre),
            "overwrite": False,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("registered") is True
    assert "model_dir" in data
    dest = tmp_path / "registry_root" / "bundle_test" / "2.1.0"
    assert dest.is_dir()
    assert (dest / "model.cypha").read_bytes() == b"fake cypha bytes"
    assert "name" in (dest / "card.json").read_text(encoding="utf-8")
    assert (dest / "preprocessor.json").read_text(encoding="utf-8") == '{"schema":1}'

    r2 = client.post(
        "/register",
        json={
            "name": "bundle_test",
            "version": "2.1.0",
            "model_cypha": str(src_cypha),
            "card_json": str(src_card),
            "overwrite": False,
        },
    )
    assert r2.status_code == 400

    r3 = client.post(
        "/register",
        json={
            "name": "bundle_test",
            "version": "2.1.0",
            "model_cypha": str(src_cypha),
            "card_json": str(src_card),
            "overwrite": True,
        },
    )
    assert r3.status_code == 200, r3.text
    assert (dest / "model.cypha").is_file()

    r4 = client.post(
        "/register",
        json={
            "name": "missing",
            "version": "1.0.0",
            "model_cypha": str(tmp_path / "nope.cypha"),
            "card_json": str(src_card),
            "overwrite": False,
        },
    )
    assert r4.status_code == 400
    assert "cypha" in r4.json().get("detail", "").lower()


def test_metrics_registry_model_count_updates_after_register(tmp_path):
    """``GET /metrics`` → ``registry_model_count`` scans the same ``ModelRegistry`` tree as ``POST /register``."""
    from fastapi.testclient import TestClient

    from cypha_studio.core.registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path / "reg"))
    app = api_mod.create_app(engine=None, registry=reg, session=None)
    client = TestClient(app)

    assert client.get("/metrics").json()["registry_model_count"] == 0

    src_cypha = tmp_path / "one.cypha"
    src_card = tmp_path / "one_card.json"
    src_cypha.write_bytes(b"m")
    src_card.write_text('{"name":"m","version":"1.0.0"}', encoding="utf-8")

    r = client.post(
        "/register",
        json={
            "name": "metrics_reg",
            "version": "1.0.0",
            "model_cypha": str(src_cypha),
            "card_json": str(src_card),
            "overwrite": False,
        },
    )
    assert r.status_code == 200, r.text
    assert client.get("/metrics").json()["registry_model_count"] == 1


def test_post_load_success_returns_full_modelcard_keys(tmp_path):
    """``POST /load`` → ``loaded`` matches ``ModelCard`` field names (native clients rely on stable keys)."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from dataclasses import fields
    from fastapi.testclient import TestClient

    from cypha_studio.core.registry import ModelCard, ModelRegistry

    rng = np.random.default_rng(21)
    clf = CyphaDIF(VectorEncoder(3), field_dim=24, rng=rng)
    for i in range(10):
        clf.train_step(rng.standard_normal(3), str(i % 2))
    reg = ModelRegistry(str(tmp_path))
    card = ModelCard(
        name="contract-load",
        version="1.0.0",
        task="classification",
        model_type="CyphaDIF",
        encoder_type="VectorEncoder",
        input_dim=3,
    )
    reg.register(clf, card)
    app = api_mod.create_app(engine=None, registry=reg, session=None)
    client = TestClient(app)
    r = client.post("/load", json={"name": "contract-load", "version": "1.0.0"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {"loaded"}
    want = {f.name for f in fields(ModelCard)}
    assert set(body["loaded"].keys()) == want


def test_predict_update_adapt_classes_503_detail_without_engine():
    """Matches native ``cypha_rest`` JSON for missing model (``503`` + ``detail``)."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    from fastapi.testclient import TestClient

    app = api_mod.create_app(engine=None, registry=None, session=None)
    client = TestClient(app)
    detail = {"detail": "No model loaded"}

    pr = client.post(
        "/predict",
        json={"input": [0.0, 0.0], "use_gh": False, "return_explanation": False},
    )
    assert pr.status_code == 503
    assert pr.json() == detail

    ur = client.post(
        "/update",
        json={"input": [0.0, 0.0], "correct_label": "0", "use_gh": False},
    )
    assert ur.status_code == 503
    assert ur.json() == detail

    ar = client.post(
        "/adapt_temperature",
        json={"calibration": [{"input": [0.0, 0.0], "correct_label": "0"}]},
    )
    assert ar.status_code == 503
    assert ar.json() == detail

    cr = client.get("/classes")
    assert cr.status_code == 503
    assert cr.json() == detail


def test_session_get_without_inference_session_returns_empty_summary():
    """``session=None`` on ``create_app``: ``GET /session`` is still **200** with zeroed summary."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    from cypha_studio.core.inference import InferenceEngine
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(33)
    clf = CyphaDIF(VectorEncoder(2), field_dim=20, rng=rng)
    for i in range(8):
        clf.train_step(rng.standard_normal(2), str(i % 2))
    eng = InferenceEngine(clf, None)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=None)
    client = TestClient(app)
    r = client.get("/session")
    assert r.status_code == 200
    d = r.json()
    assert d["n_predictions"] == 0
    assert d["n_corrections"] == 0
    assert d["correction_accuracy"] == 0.0
    assert d["mean_confidence"] == 0.0
    assert d["mean_anomaly"] == 0.0
    assert d["n_ood_flagged"] == 0
    assert d["label_distribution"] == {}
    assert d["session_duration_s"] == 0.0


def test_metrics_session_null_when_inference_session_not_attached():
    """``/metrics`` → ``session: null`` when no ``InferenceSession`` on the app."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    from cypha_studio.core.inference import InferenceEngine
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(34)
    clf = CyphaDIF(VectorEncoder(2), field_dim=18, rng=rng)
    for i in range(6):
        clf.train_step(rng.standard_normal(2), str(i % 2))
    eng = InferenceEngine(clf, None)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=None)
    client = TestClient(app)
    m = client.get("/metrics").json()
    assert m.get("model_loaded") is True
    assert m.get("session") is None


def test_delete_session_noop_when_inference_session_not_attached():
    """``DELETE /session`` is always **200** with ``cleared`` (no-op if there is no ``InferenceSession``)."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    from cypha_studio.core.inference import InferenceEngine
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(36)
    clf = CyphaDIF(VectorEncoder(2), field_dim=20, rng=rng)
    for i in range(5):
        clf.train_step(rng.standard_normal(2), str(i % 2))
    eng = InferenceEngine(clf, None)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=None)
    client = TestClient(app)
    r = client.delete("/session")
    assert r.status_code == 200
    assert r.json() == {"cleared": True}


def test_predict_increments_engine_metrics_without_inference_session():
    """``POST /predict`` updates engine counters in ``/metrics``; ``GET /session`` stays zeroed without ``InferenceSession``."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    from cypha_studio.core.inference import InferenceEngine
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(35)
    clf = CyphaDIF(VectorEncoder(2), field_dim=22, rng=rng)
    for i in range(8):
        clf.train_step(rng.standard_normal(2), str(i % 2))
    eng = InferenceEngine(clf, None)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=None)
    client = TestClient(app)
    assert client.get("/metrics").json().get("n_predictions") == 0
    pr = client.post(
        "/predict",
        json={"input": [0.1, -0.2], "use_gh": False, "return_explanation": False},
    )
    assert pr.status_code == 200, pr.text
    m = client.get("/metrics").json()
    assert m.get("n_predictions") == 1
    assert m.get("n_corrections") == 0
    assert m.get("session") is None
    assert client.get("/session").json()["n_predictions"] == 0


def test_delete_session_when_no_engine_returns_cleared():
    """``DELETE /session`` is **200** + ``cleared`` even when there is no loaded model."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    from fastapi.testclient import TestClient

    app = api_mod.create_app(engine=None, registry=None, session=None)
    r = TestClient(app).delete("/session")
    assert r.status_code == 200
    assert r.json() == {"cleared": True}


def test_load_missing_name_returns_422():
    """``POST /load`` body must include ``name`` (``version`` defaults to ``latest``)."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    from fastapi.testclient import TestClient

    from cypha_studio.core.registry import ModelRegistry

    app = api_mod.create_app(engine=None, registry=ModelRegistry(), session=None)
    r = TestClient(app).post("/load", json={})
    assert r.status_code == 422
    body = r.json()
    assert "detail" in body


def test_load_malformed_json_returns_422():
    """Invalid JSON on ``POST /load`` → **422** (native ``cypha_rest`` uses **400** ``{"detail":"bad json"}``)."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    from fastapi.testclient import TestClient

    from cypha_studio.core.registry import ModelRegistry

    app = api_mod.create_app(engine=None, registry=ModelRegistry(), session=None)
    r = TestClient(app).post(
        "/load",
        content="{",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert "detail" in r.json()


def test_load_with_valid_body_but_no_registry_returns_503():
    """``registry=None`` on ``create_app``: ``POST /load`` is **503** before any disk lookup."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    from fastapi.testclient import TestClient

    app = api_mod.create_app(engine=None, registry=None, session=None)
    r = TestClient(app).post("/load", json={"name": "any", "version": "1.0.0"})
    assert r.status_code == 503
    assert r.json() == {"detail": "No registry configured"}


def test_metrics_and_ready_endpoints():
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    from fastapi.testclient import TestClient

    app = api_mod.create_app(engine=None, registry=None, session=None)
    client = TestClient(app)

    m = client.get("/metrics")
    assert m.status_code == 200
    data = m.json()
    assert data.get("model_loaded") is False
    assert "uptime_seconds" in data
    assert data.get("registry_model_count") == 0
    assert data.get("regression_head_loaded") is False

    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json() == {"ready": False, "reason": "no_model_loaded"}


def test_ready_ok_with_engine():
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(0)
    clf = CyphaDIF(VectorEncoder(3), field_dim=32, rng=rng)
    for i in range(12):
        clf.train_step(rng.standard_normal(3), str(i % 2))
    eng = InferenceEngine(clf, None)
    sess = InferenceSession(eng)
    app = api_mod.create_app(
        engine=eng, registry=ModelRegistry(), session=sess,
        cors_allow_origins=["*"],
    )
    client = TestClient(app)
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body.get("ready") is True
    assert body.get("model_type") == "CyphaDIF"

    m = client.get("/metrics")
    assert m.status_code == 200
    mj = m.json()
    assert mj.get("model_loaded") is True
    assert mj.get("gh_chi_session") == 1.0
    assert mj.get("gh_psi_session") == 1.0
    sess = mj.get("session")
    assert isinstance(sess, dict)
    assert set(sess.keys()) == {
        "n_predictions",
        "n_corrections",
        "correction_accuracy",
        "mean_confidence",
        "mean_anomaly",
        "n_ood_flagged",
        "label_distribution",
        "session_duration_s",
    }
    assert sess["n_predictions"] == 0
    assert sess["n_corrections"] == 0
    assert sess["correction_accuracy"] == 0.0
    assert sess["mean_confidence"] == 0.0
    assert sess["mean_anomaly"] == 0.0
    assert sess["n_ood_flagged"] == 0
    assert sess["label_distribution"] == {}
    assert isinstance(sess["session_duration_s"], (int, float))
    assert sess["session_duration_s"] >= 0.0


def test_health_get_ok():
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    from fastapi.testclient import TestClient

    app = api_mod.create_app(engine=None, registry=None, session=None)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert body.get("model") == "none"
    assert "uptime" in body


def test_health_n_predictions_matches_metrics_after_predict():
    """``GET /health`` and ``GET /metrics`` use the same engine counter (``PORT_CONTRACT`` §3)."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(47)
    clf = CyphaDIF(VectorEncoder(3), field_dim=28, rng=rng)
    for i in range(12):
        clf.train_step(rng.standard_normal(3), str(i % 2))
    eng = InferenceEngine(clf, None)
    sess = InferenceSession(eng)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=sess)
    client = TestClient(app)
    assert client.get("/health").json()["n_predictions"] == 0
    assert client.get("/metrics").json()["n_predictions"] == 0
    pr = client.post(
        "/predict",
        json={"input": [0.1, 0.2, 0.3], "use_gh": False, "return_explanation": False},
    )
    assert pr.status_code == 200, pr.text
    h = client.get("/health").json()
    m = client.get("/metrics").json()
    assert h["n_predictions"] == m["n_predictions"] == 1


def test_get_models_empty_registry_returns_empty_lists(tmp_path):
    """Fresh registry root with no saved models → ``models: []`` (full and summary)."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    from fastapi.testclient import TestClient

    from cypha_studio.core.registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    app = api_mod.create_app(engine=None, registry=reg, session=None)
    client = TestClient(app)
    assert client.get("/models").json() == {"models": []}
    assert client.get("/models", params={"summary": True}).json() == {"models": []}
    assert client.get("/metrics").json().get("registry_model_count") == 0


def test_session_get_and_delete_roundtrip():
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(5)
    clf = CyphaDIF(VectorEncoder(3), field_dim=24, rng=rng)
    for i in range(8):
        clf.train_step(rng.standard_normal(3), str(i % 2))
    eng = InferenceEngine(clf, None)
    sess = InferenceSession(eng)
    app = api_mod.create_app(
        engine=eng, registry=ModelRegistry(), session=sess,
    )
    client = TestClient(app)
    sess.predict(np.array([0.1, 0.2, 0.3], dtype=np.float64), use_gh=False)
    s0 = client.get("/session")
    assert s0.status_code == 200
    assert s0.json().get("n_predictions", 0) >= 1
    d = client.delete("/session")
    assert d.status_code == 200
    assert d.json().get("cleared") is True
    s1 = client.get("/session")
    assert s1.json().get("n_predictions", 0) == 0


def test_classes_get_with_engine():
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(6)
    clf = CyphaDIF(VectorEncoder(2), field_dim=20, rng=rng)
    for i in range(10):
        clf.train_step(rng.standard_normal(2), str(i % 2))
    eng = InferenceEngine(clf, None)
    sess = InferenceSession(eng)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=sess)
    client = TestClient(app)
    r = client.get("/classes")
    assert r.status_code == 200
    classes = r.json().get("classes", {})
    assert isinstance(classes, dict)
    assert len(classes) >= 1


def test_models_summary_vs_full(tmp_path):
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    from cypha_studio.core.registry import ModelCard, ModelRegistry

    rng = np.random.default_rng(99)
    clf = CyphaDIF(VectorEncoder(4), field_dim=32, rng=rng)
    for i in range(6):
        clf.train_step(rng.standard_normal(4), str(i % 2))
    reg = ModelRegistry(str(tmp_path))
    card = ModelCard(
        name="api-list",
        version="1.0.0",
        task="classification",
        model_type="CyphaDIF",
        encoder_type="VectorEncoder",
        input_dim=4,
    )
    reg.register(clf, card)
    app = api_mod.create_app(engine=None, registry=reg, session=None)
    client = TestClient(app)
    sm = client.get("/models", params={"summary": True})
    assert sm.status_code == 200
    mlist = sm.json().get("models", [])
    assert len(mlist) == 1
    assert set(mlist[0].keys()) == {"name", "version"}
    assert mlist[0]["name"] == "api-list"
    full = client.get("/models")
    assert full.status_code == 200
    fl = full.json().get("models", [])
    assert len(fl) == 1
    assert "val_accuracy" in fl[0] or "model_type" in fl[0]


def test_predict_regression_head_overlay(tmp_path):
    """Optional ``regression_head.json`` fills ``regression_val`` / ``uncertainty`` on ``/predict``."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    import json

    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient

    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    head = {
        "experts": {
            "0": {"mu": 1.0, "var_ema": 0.1},
            "1": {"mu": 2.0, "var_ema": 0.2},
        },
    }
    p = tmp_path / "rh.json"
    p.write_text(json.dumps(head), encoding="utf-8")

    rng = np.random.default_rng(7)
    clf = CyphaDIF(VectorEncoder(2), field_dim=16, rng=rng)
    for i in range(12):
        clf.train_step(rng.standard_normal(2), str(i % 2))
    eng = InferenceEngine(clf, None)
    sess = InferenceSession(eng)
    app = api_mod.create_app(
        engine=eng, registry=ModelRegistry(), session=sess,
        regression_head_path=str(p),
    )
    client = TestClient(app)
    r = client.post("/predict", json={"input": [0.0, 0.0], "use_gh": False})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get("regression_val") is not None
    assert isinstance(d["regression_val"], (int, float))
    assert float(d["uncertainty"]) >= 0.0
    met = client.get("/metrics").json()
    assert met.get("regression_head_loaded") is True


# ─── /session/rng ──────────────────────────────────────────────────────────────

def _make_trained_app(seed: int = 77, dim: int = 3, n_steps: int = 10):
    """Return a ``(app, client)`` pair with a tiny trained CyphaDIF model."""
    import numpy as np
    from Cypha import CyphaDIF, VectorEncoder
    from fastapi.testclient import TestClient
    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    rng = np.random.default_rng(seed)
    clf = CyphaDIF(VectorEncoder(dim), field_dim=24,
                   rng=rng, replay_rng=np.random.Generator(np.random.MT19937(seed)))
    for i in range(n_steps):
        clf.train_step(rng.standard_normal(dim), str(i % 2))
    eng = InferenceEngine(clf, None)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=InferenceSession(eng))
    return app, TestClient(app)


def test_session_rng_get_no_engine_returns_503():
    from fastapi.testclient import TestClient
    app = api_mod.create_app(engine=None, registry=None, session=None)
    r = TestClient(app).get("/session/rng")
    assert r.status_code == 503


def test_session_rng_get_returns_mt19937_shape():
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    app, client = _make_trained_app()
    r = client.get("/session/rng")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get("bit_generator") == "MT19937"
    assert isinstance(d.get("state"), list) and len(d["state"]) == 624
    assert isinstance(d.get("pos"), int) and 0 <= d["pos"] <= 623
    assert all(isinstance(v, int) and 0 <= v < 2**32 for v in d["state"])


def test_session_rng_post_seed_changes_state():
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    app, client = _make_trained_app(seed=10)
    s0 = client.get("/session/rng").json()["state"]
    r = client.post("/session/rng", json={"seed": 999})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get("bit_generator") == "MT19937"
    assert len(d["state"]) == 624
    # State must have changed from original (only fails if seed 999 happened to produce same state as initial)
    assert d["state"] != s0


def test_session_rng_post_no_engine_returns_503():
    from fastapi.testclient import TestClient
    app = api_mod.create_app(engine=None, registry=None, session=None)
    r = TestClient(app).post("/session/rng", json={"seed": 0})
    assert r.status_code == 503


def test_session_rng_post_bad_request_returns_400():
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    app, client = _make_trained_app()
    r = client.post("/session/rng", json={})
    assert r.status_code in (400, 422)


def test_session_rng_post_wrong_state_length_returns_400():
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    app, client = _make_trained_app()
    r = client.post("/session/rng", json={"state": [0] * 100, "pos": 0})
    assert r.status_code == 400


def test_session_rng_roundtrip_seed_to_state_matches_numpy():
    """After POST /session/rng with seed S, GET returns the same state as numpy MT19937(S)."""
    import numpy as np
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    app, client = _make_trained_app()
    seed = 424242
    r = client.post("/session/rng", json={"seed": seed})
    assert r.status_code == 200, r.text
    api_state = r.json()["state"]
    api_pos   = r.json()["pos"]

    np_rng = np.random.Generator(np.random.MT19937(seed))
    bg = np_rng.bit_generator.state
    np_state = bg["state"]["key"].tolist()
    np_pos   = int(bg["state"]["pos"])

    assert api_state == np_state, "seed-based MT19937 state mismatch between FastAPI and numpy"
    assert api_pos == np_pos, f"pos mismatch: api={api_pos}, numpy={np_pos}"


def test_session_rng_full_state_restore_roundtrip():
    """GET /session/rng → POST /session/rng with that state → GET again returns identical state."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    app, client = _make_trained_app(seed=31)
    s1 = client.get("/session/rng").json()
    # Seed to something different
    client.post("/session/rng", json={"seed": 0})
    # Restore original
    r = client.post("/session/rng", json={"state": s1["state"], "pos": s1["pos"]})
    assert r.status_code == 200, r.text
    s2 = client.get("/session/rng").json()
    assert s2["state"] == s1["state"], "state restore roundtrip failed"
    assert s2["pos"] == s1["pos"], "pos restore roundtrip failed"


def test_rng_state_response_schema_fields():
    fields = set(api_mod.RngStateResponse.model_fields.keys())
    assert fields >= {"bit_generator", "state", "pos"}


def test_rng_seed_request_schema_fields():
    fields = set(api_mod.RngSeedRequest.model_fields.keys())
    assert fields >= {"seed", "state", "pos"}
