"""
Optional subprocess tests for ``native/tools/cypha_rest.cpp``.

Skips unless ``CYPHA_REST_BIN`` is set or a built binary exists (Windows: MSVC under ``native/build/`` or MinGW cross-build ``native/build-mingw-w64/cypha_rest.exe`` from WSL).
Requires ``httpx`` and committed ``parity_fixtures/`` + ``f_field.json`` (optional ``regression_head.json`` for MoE regression fields on ``/predict``).
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")

_ROOT = Path(__file__).resolve().parents[1]
_FIX = _ROOT / "parity_fixtures"


def _cypha_rest_executable() -> Path | None:
    env = os.environ.get("CYPHA_REST_BIN", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    # Windows: MinGW cross-build from WSL (see native/scripts/build_cypha_rest_mingw_wsl.ps1) before MSVC paths.
    if sys.platform == "win32":
        candidates = [
            _ROOT / "native" / "build-mingw-w64" / "cypha_rest.exe",
            _ROOT / "native" / "build" / "Release" / "cypha_rest.exe",
            _ROOT / "native" / "build" / "Debug" / "cypha_rest.exe",
            _ROOT / "native" / "build" / "cypha_rest.exe",
        ]
    else:
        candidates = [_ROOT / "native" / "build" / "cypha_rest"]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _reference_has_embedded_f_field() -> bool:
    from Cypha import cypha_load_binary

    try:
        st = cypha_load_binary(str(_FIX / "reference.cypha"))
    except OSError:
        return False
    w = st.get("world") or {}
    return w.get("F_field") is not None


def _softmax_native_style(z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Match ``softmax_row_like_python`` / ``softmax_batch_like_python`` (k ≤ 8 path) in native infer."""
    z = np.asarray(z, dtype=np.float64)
    mx = float(np.max(z))
    e = np.exp(z - mx)
    s = float(np.sum(e)) + eps
    return (e / s).astype(np.float64)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


@pytest.fixture(scope="module")
def rest_bin():
    exe = _cypha_rest_executable()
    if exe is None:
        pytest.skip("cypha_rest not built (set CYPHA_REST_BIN or build native/)")
    if not (_FIX / "reference.cypha").is_file() or not (_FIX / "f_field.json").is_file():
        pytest.skip("parity_fixtures missing reference.cypha or f_field.json")
    return exe


@pytest.fixture
def rest_server(rest_bin):
    import httpx

    port = _free_port()
    host = "127.0.0.1"
    cmd = [
        str(rest_bin),
        "--listen",
        f"{host}:{port}",
        "--cypha",
        str(_FIX / "reference.cypha"),
        "--f-field-json",
        str(_FIX / "f_field.json"),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(_ROOT),
    )
    base = f"http://{host}:{port}"
    deadline = time.time() + 15.0
    last_err = None
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                pytest.fail(f"cypha_rest exited early ({proc.returncode}): {err[:500]}")
            try:
                r = httpx.get(f"{base}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError as e:
                last_err = e
            time.sleep(0.05)
        else:
            pytest.fail(f"cypha_rest did not become ready: {last_err}")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def rest_server_regression(rest_bin):
    """``cypha_rest`` with ``--regression-json`` (scalar mixture head for ``/predict``)."""
    if not (_FIX / "regression_head.json").is_file():
        pytest.skip("parity_fixtures/regression_head.json missing")
    import httpx

    port = _free_port()
    host = "127.0.0.1"
    cmd = [
        str(rest_bin),
        "--listen",
        f"{host}:{port}",
        "--cypha",
        str(_FIX / "reference.cypha"),
        "--f-field-json",
        str(_FIX / "f_field.json"),
        "--regression-json",
        str(_FIX / "regression_head.json"),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(_ROOT),
    )
    base = f"http://{host}:{port}"
    deadline = time.time() + 15.0
    last_err = None
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                pytest.fail(f"cypha_rest exited early ({proc.returncode}): {err[:500]}")
            try:
                r = httpx.get(f"{base}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError as e:
                last_err = e
            time.sleep(0.05)
        else:
            pytest.fail(f"cypha_rest did not become ready: {last_err}")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def rest_server_embedded_ff_only(rest_bin):
    """`cypha_rest` with only `.cypha` when `world.F_field` is inside the blob."""
    if not _reference_has_embedded_f_field():
        pytest.skip("reference.cypha has no embedded world.F_field")
    import httpx

    port = _free_port()
    host = "127.0.0.1"
    cmd = [
        str(rest_bin),
        "--listen",
        f"{host}:{port}",
        "--cypha",
        str(_FIX / "reference.cypha"),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(_ROOT),
    )
    base = f"http://{host}:{port}"
    deadline = time.time() + 15.0
    last_err = None
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                pytest.fail(f"cypha_rest exited early ({proc.returncode}): {err[:500]}")
            try:
                r = httpx.get(f"{base}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError as e:
                last_err = e
            time.sleep(0.05)
        else:
            pytest.fail(f"cypha_rest did not become ready: {last_err}")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _mke_regression_head_from_sidecar(sidecar: dict) -> dict:
    """Build ``regression_head.json`` with ``mke`` block from ``parity_fixtures/mke_train_step/sidecar.json``."""
    experts = {lbl: {"mu": 0.0, "var_ema": 0.25} for lbl in sidecar["w_before"]}
    mke = {
        "d_in": sidecar["d_in"],
        "D_rff": sidecar["D_rff"],
        "temperature": sidecar["temperature"],
        "forgetting_factor": sidecar["forgetting_factor"],
        "pi_floor": 0.02,
        "rff_W_rowmajor": sidecar["rff_W_rowmajor"],
        "rff_b": sidecar["rff_b"],
        "w": sidecar["w_before"],
        "P": sidecar["P_before"],
        "gh_scales": sidecar["gh_scales"],
    }
    return {"schema": 1, "experts": experts, "mke": mke}


@pytest.fixture
def rest_server_mke_train_step(rest_bin, tmp_path):
    """``cypha_rest`` with MKE sidecar matching ``native_mke_train_step`` fixture."""
    import httpx

    mke_dir = _FIX / "mke_train_step"
    side_path = mke_dir / "sidecar.json"
    if not side_path.is_file() or not (mke_dir / "before.cypha").is_file() or not (mke_dir / "f_field.json").is_file():
        pytest.skip("mke_train_step parity fixture missing")

    side = json.loads(side_path.read_text(encoding="utf-8"))
    reg = _mke_regression_head_from_sidecar(side)
    hp = {
        "world_lr": side["world_lr"],
        "delta_lr": side["delta_lr"],
        "ood_sigma": side["ood_sigma"],
        "enc_lr": side["enc_lr"],
        "replay_ratio": side["replay_ratio"],
        "replay_cap": side["replay_cap"],
        "align_every": side["align_every"],
        "temp_recalib_every": side["temp_recalib_every"],
    }
    (tmp_path / "regression_head.json").write_text(json.dumps(reg), encoding="utf-8")
    (tmp_path / "train_hparams.json").write_text(json.dumps(hp), encoding="utf-8")

    port = _free_port()
    host = "127.0.0.1"
    cmd = [
        str(rest_bin),
        "--listen",
        f"{host}:{port}",
        "--cypha",
        str(mke_dir / "before.cypha"),
        "--f-field-json",
        str(mke_dir / "f_field.json"),
        "--regression-json",
        str(tmp_path / "regression_head.json"),
        "--train-hparams",
        str(tmp_path / "train_hparams.json"),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(_ROOT),
    )
    base = f"http://{host}:{port}"
    deadline = time.time() + 15.0
    last_err = None
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                pytest.fail(f"cypha_rest exited early ({proc.returncode}): {err[:500]}")
            try:
                r = httpx.get(f"{base}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError as e:
                last_err = e
            time.sleep(0.05)
        else:
            pytest.fail(f"cypha_rest did not become ready: {last_err}")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def rest_server_with_registry(rest_bin, tmp_path):
    """``cypha_rest`` with ``--registry`` (initial scan may be empty until ``POST /register``)."""
    import httpx

    reg_root = tmp_path / "registry_root"
    reg_root.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    host = "127.0.0.1"
    cmd = [
        str(rest_bin),
        "--listen",
        f"{host}:{port}",
        "--cypha",
        str(_FIX / "reference.cypha"),
        "--f-field-json",
        str(_FIX / "f_field.json"),
        "--registry",
        str(reg_root),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(_ROOT),
    )
    base = f"http://{host}:{port}"
    deadline = time.time() + 15.0
    last_err = None
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                pytest.fail(f"cypha_rest exited early ({proc.returncode}): {err[:500]}")
            try:
                r = httpx.get(f"{base}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError as e:
                last_err = e
            time.sleep(0.05)
        else:
            pytest.fail(f"cypha_rest did not become ready: {last_err}")
        yield base, reg_root
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_cypha_rest_post_register_lists_model(rest_server_with_registry):
    """``POST /register`` + ``GET /models`` (native only)."""
    import httpx

    base, _reg_root = rest_server_with_registry
    c = httpx.Client(base_url=base, timeout=15.0)
    body = {
        "name": "rest_reg_smoke",
        "version": "2.0.0",
        "model_cypha": str(_FIX / "reference.cypha"),
        "card_json": str(_FIX / "registry_register" / "card.json"),
        "overwrite": True,
    }
    r = c.post("/register", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("registered") is True
    assert "model_dir" in data
    m = c.get("/models", params={"summary": "true"})
    assert m.status_code == 200, m.text
    rows = m.json().get("models", [])
    assert any(
        x.get("name") == "rest_reg_smoke" and x.get("version") == "2.0.0" for x in rows
    )


def test_cypha_rest_no_sidecar_f_field_json(rest_server_embedded_ff_only):
    """Embedded `world.F_field` allows starting without `--f-field-json`."""
    import httpx

    c = httpx.Client(base_url=rest_server_embedded_ff_only, timeout=5.0)
    assert c.get("/ready").status_code == 200


def test_cypha_rest_health_ready_metrics_session(rest_server):
    import httpx

    c = httpx.Client(base_url=rest_server, timeout=5.0)
    h = c.get("/health")
    assert h.status_code == 200
    assert h.json().get("status") == "ok"

    r = c.get("/ready")
    assert r.status_code == 200
    assert r.json().get("ready") is True

    m = c.get("/metrics")
    assert m.status_code == 200
    body = m.json()
    assert body.get("model_loaded") is True
    assert body.get("regression_head_loaded") is False
    assert body.get("session") is not None

    s0 = c.get("/session")
    assert s0.status_code == 200
    sj = s0.json()
    assert sj.get("n_predictions", 0) == 0
    assert set(sj.keys()) == {
        "n_predictions",
        "n_corrections",
        "correction_accuracy",
        "mean_confidence",
        "mean_anomaly",
        "n_ood_flagged",
        "label_distribution",
        "session_duration_s",
    }


def test_cypha_rest_health_n_predictions_matches_metrics(rest_server):
    """``GET /health`` and ``GET /metrics`` share the same prediction counter."""
    import httpx

    c = httpx.Client(base_url=rest_server, timeout=5.0)
    assert c.get("/health").json()["n_predictions"] == 0
    assert c.get("/metrics").json()["n_predictions"] == 0

    exp = np.load(_FIX / "expected.npz")
    x0 = exp["x_input"][0].astype(float).tolist()
    pr = c.post(
        "/predict",
        json={"input": x0, "use_gh": True, "return_explanation": False},
    )
    assert pr.status_code == 200, pr.text

    h = c.get("/health").json()
    m = c.get("/metrics").json()
    assert h["n_predictions"] == m["n_predictions"] == 1


def test_cypha_rest_predict_and_explanation(rest_server):
    import httpx

    exp = np.load(_FIX / "expected.npz")
    x0 = exp["x_input"][0].astype(float).tolist()

    c = httpx.Client(base_url=rest_server, timeout=5.0)
    p = c.post(
        "/predict",
        json={"input": x0, "use_gh": True, "return_explanation": False},
    )
    assert p.status_code == 200, p.text
    data = p.json()
    assert "label" in data and "confidence" in data
    assert data.get("explanation") is None

    pe = c.post(
        "/predict",
        json={"input": x0, "use_gh": True, "return_explanation": True},
    )
    assert pe.status_code == 200, pe.text
    ex = pe.json().get("explanation")
    assert isinstance(ex, dict)
    for key in ("label", "confidence", "all_scores", "class_details", "world_mu_distance"):
        assert key in ex
    assert isinstance(ex["class_details"], dict)
    assert len(ex["class_details"]) >= 1


def test_cypha_rest_malformed_json_posts(rest_server):
    """Invalid JSON body on POST handlers → **400** ``{"detail":"bad json"}`` (FastAPI uses **422**)."""
    import httpx

    c = httpx.Client(base_url=rest_server, timeout=5.0)
    bad = "{not-valid-json"
    hdrs = {"Content-Type": "application/json"}
    want = {"detail": "bad json"}
    for path in ("/predict", "/update", "/adapt_temperature"):
        r = c.post(path, content=bad, headers=hdrs)
        assert r.status_code == 400, (path, r.text)
        assert r.json() == want, (path, r.json())


def test_cypha_rest_predict_wrong_input_dim(rest_server):
    """Wrong feature length after preprocessor → **400** ``detail`` (parity fixture latent dim is **8**)."""
    import httpx

    c = httpx.Client(base_url=rest_server, timeout=5.0)
    r = c.post(
        "/predict",
        json={"input": [0.0] * 7, "use_gh": False, "return_explanation": False},
    )
    assert r.status_code == 400, r.text
    assert r.json() == {"detail": "input dim mismatch after preprocessor"}


def test_cypha_rest_update_returns_loss(rest_server):
    import httpx

    exp = np.load(_FIX / "expected.npz")
    x0 = exp["x_input"][0].astype(float).tolist()

    c = httpx.Client(base_url=rest_server, timeout=5.0)
    u = c.post(
        "/update",
        json={"input": x0, "correct_label": "0", "use_gh": False},
    )
    assert u.status_code == 200, u.text
    data = u.json()
    assert set(data.keys()) == {"loss", "n_corrections"}
    assert isinstance(data["loss"], (int, float))
    assert isinstance(data["n_corrections"], int) and data["n_corrections"] >= 1


def test_cypha_rest_update_use_gh_scales(rest_server):
    import httpx

    exp = np.load(_FIX / "expected.npz")
    x0 = exp["x_input"][0].astype(float).tolist()

    c = httpx.Client(base_url=rest_server, timeout=5.0)
    u = c.post(
        "/update",
        json={"input": x0, "correct_label": "1", "use_gh": True},
    )
    assert u.status_code == 200, u.text
    assert "loss" in u.json()
    m = c.get("/metrics")
    assert m.status_code == 200
    body = m.json()
    assert "gh_chi_session" in body and "gh_psi_session" in body
    assert body["gh_psi_session"] == 1.0


def test_cypha_rest_delete_session_resets_gh_nig(rest_server):
    """DELETE /session clears prediction history and GH session χ/ψ (like InferenceSession.clear)."""
    import httpx

    exp = np.load(_FIX / "expected.npz")
    x0 = exp["x_input"][0].astype(float).tolist()

    c = httpx.Client(base_url=rest_server, timeout=5.0)
    for i in range(8):
        r = c.post(
            "/update",
            json={"input": x0, "correct_label": str(i % 3), "use_gh": True},
        )
        assert r.status_code == 200, r.text
    m1 = c.get("/metrics").json()
    assert m1.get("gh_psi_session") == 1.0
    chi_after_updates = float(m1["gh_chi_session"])

    d = c.delete("/session")
    assert d.status_code == 200
    m2 = c.get("/metrics").json()
    assert m2["gh_chi_session"] == 1.0
    assert m2["gh_psi_session"] == 1.0
    # If adaptation never moved χ, the reset test is trivial but still valid.
    assert chi_after_updates >= 1.0


def test_cypha_rest_adapt_temperature_matches_python(rest_server):
    """POST /adapt_temperature matches CyphaDIF.adapt_temperature (ECE grid on fixture batch)."""
    import httpx
    from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary

    manifest = json.loads((_FIX / "manifest.json").read_text(encoding="utf-8"))
    state = cypha_load_binary(str(_FIX / "reference.cypha"))
    m = manifest["model"]
    enc = VectorEncoder(int(m["input_dim"]))
    clf = CyphaDIF(enc, field_dim=int(m["field_dim"]), rng=np.random.default_rng(0))
    clf.load_state(state)

    z = np.load(_FIX / "expected.npz")
    labels = manifest["labels"]
    cal = [(z["x_input"][i].astype(np.float64, copy=True), labels[int(z["pred_idx"][i])]) for i in range(len(z["x_input"]))]
    T_py = clf.adapt_temperature(cal, n_grid=20, T_min=0.3, T_max=8.0)

    payload = {
        "calibration": [{"input": row[0].tolist(), "correct_label": row[1]} for row in cal],
        "n_grid": 20,
        "T_min": 0.3,
        "T_max": 8.0,
        "n_bins": 10,
    }
    c = httpx.Client(base_url=rest_server, timeout=30.0)
    r = c.post("/adapt_temperature", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("n_used") == len(cal)
    np.testing.assert_allclose(float(data["temperature"]), float(T_py), rtol=0, atol=1e-9)


def test_cypha_rest_fastapi_json_shape_parity(rest_server):
    """Same fixture model: native `cypha_rest` and FastAPI share JSON key trees (empty session)."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    from cypha_studio.server import api as api_mod

    if not getattr(api_mod, "FASTAPI_AVAILABLE", False):
        pytest.skip("FastAPI not installed")

    import httpx
    from fastapi.testclient import TestClient

    from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary
    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    manifest = json.loads((_FIX / "manifest.json").read_text(encoding="utf-8"))
    state = cypha_load_binary(str(_FIX / "reference.cypha"))
    m = manifest["model"]
    enc = VectorEncoder(int(m["input_dim"]))
    clf = CyphaDIF(enc, field_dim=int(m["field_dim"]), rng=np.random.default_rng(0))
    clf.load_state(state)
    eng = InferenceEngine(clf, None)
    sess = InferenceSession(eng)
    app = api_mod.create_app(
        engine=eng, registry=ModelRegistry(), session=sess, cors_allow_origins=["*"],
    )
    fc = TestClient(app)
    nc = httpx.Client(base_url=rest_server, timeout=10.0)

    wrong_dim_p = {"input": [0.0] * 7, "use_gh": False, "return_explanation": False}
    wpf, wpn = fc.post("/predict", json=wrong_dim_p), nc.post("/predict", json=wrong_dim_p)
    assert wpf.status_code == wpn.status_code == 400
    assert wpf.json() == wpn.json() == {"detail": "input dim mismatch after preprocessor"}

    def assert_same_keys(a, b, where: str) -> None:
        assert set(a.keys()) == set(b.keys()), (where, sorted(a.keys()), sorted(b.keys()))

    hf, hn = fc.get("/health").json(), nc.get("/health").json()
    assert_same_keys(hf, hn, "health")

    rf, rn = fc.get("/ready").json(), nc.get("/ready").json()
    assert_same_keys(rf, rn, "ready")

    mf, mn = fc.get("/metrics").json(), nc.get("/metrics").json()
    assert_same_keys(mf, mn, "metrics")
    assert mf.get("regression_head_loaded") is False and mn.get("regression_head_loaded") is False
    assert_same_keys(mf["session"], mn["session"], "metrics.session")
    for k in (
        "n_predictions",
        "n_corrections",
        "correction_accuracy",
        "mean_confidence",
        "mean_anomaly",
        "n_ood_flagged",
    ):
        assert mf["session"][k] == mn["session"][k], k
    assert mf["session"]["label_distribution"] == mn["session"]["label_distribution"]

    sf, sn = fc.get("/session").json(), nc.get("/session").json()
    assert_same_keys(sf, sn, "session")

    mdf, mdn = fc.get("/models"), nc.get("/models")
    assert mdf.status_code == 200 and mdn.status_code == 200
    jmf, jmn = mdf.json(), mdn.json()
    assert_same_keys(jmf, jmn, "models")
    assert jmf.get("models") == [] == jmn.get("models")

    mss_f, mss_n = fc.get("/models", params={"summary": "true"}), nc.get("/models", params={"summary": "true"})
    assert mss_f.status_code == 200 and mss_n.status_code == 200
    jss_f, jss_n = mss_f.json(), mss_n.json()
    assert_same_keys(jss_f, jss_n, "models_summary")
    assert jss_f.get("models") == [] == jss_n.get("models")

    exp = np.load(_FIX / "expected.npz")
    x0 = exp["x_input"][0].astype(float).tolist()
    body = {"input": x0, "use_gh": True, "return_explanation": False}
    pf, pn = fc.post("/predict", json=body).json(), nc.post("/predict", json=body).json()
    assert_same_keys(pf, pn, "predict")
    assert set(pf["all_scores"].keys()) == set(pn["all_scores"].keys())

    body_e = {"input": x0, "use_gh": True, "return_explanation": True}
    pef, pen = fc.post("/predict", json=body_e).json(), nc.post("/predict", json=body_e).json()
    assert_same_keys(pef, pen, "predict_explain")
    assert pef.get("explanation") is not None and pen.get("explanation") is not None
    assert_same_keys(pef["explanation"], pen["explanation"], "predict_explain.explanation")

    lbl0 = str(manifest["labels"][0])
    ubody = {"input": x0, "correct_label": lbl0, "use_gh": False}
    uf, un = fc.post("/update", json=ubody).json(), nc.post("/update", json=ubody).json()
    assert_same_keys(uf, un, "update")

    z = np.load(_FIX / "expected.npz")
    labels = manifest["labels"]
    cal_small = [
        {"input": z["x_input"][i].astype(float).tolist(), "correct_label": labels[int(z["pred_idx"][i])]}
        for i in range(min(6, len(z["x_input"])))
    ]
    ad_body = {"calibration": cal_small, "n_grid": 12, "T_min": 0.4, "T_max": 4.0, "n_bins": 10}
    af, an = (
        fc.post("/adapt_temperature", json=ad_body).json(),
        nc.post("/adapt_temperature", json=ad_body).json(),
    )
    assert_same_keys(af, an, "adapt_temperature")

    df, dn = fc.delete("/session"), nc.delete("/session")
    assert df.status_code == 200 and dn.status_code == 200
    assert_same_keys(df.json(), dn.json(), "delete_session")

    rf_cls, rn_cls = fc.get("/classes"), nc.get("/classes")
    assert rf_cls.status_code == 200, rf_cls.text
    assert rn_cls.status_code == 200, rn_cls.text
    cjf, cjn = rf_cls.json(), rn_cls.json()
    assert_same_keys(cjf, cjn, "classes")
    assert_same_keys(cjf["classes"], cjn["classes"], "classes.classes")
    for lbl in cjf["classes"]:
        assert_same_keys(cjf["classes"][lbl], cjn["classes"][lbl], f"classes.classes[{lbl}]")


def test_cypha_rest_load_garbage_body_no_registry_still_503(rest_server):
    """No ``--registry``: invalid ``POST /load`` body still **503** (checked before JSON parse)."""
    import httpx

    c = httpx.Client(base_url=rest_server, timeout=5.0)
    r = c.post("/load", content="not-json-at-all", headers={"Content-Type": "application/json"})
    assert r.status_code == 503, r.text
    assert r.json() == {"detail": "No registry configured"}


def test_cypha_rest_fastapi_load_503_parity(rest_server):
    """``POST /load`` without registry: native and FastAPI return the same 503 JSON."""
    from cypha_studio.server import api as api_mod

    if not getattr(api_mod, "FASTAPI_AVAILABLE", False):
        pytest.skip("FastAPI not installed")

    import httpx
    from fastapi.testclient import TestClient

    app = api_mod.create_app(engine=None, registry=None, session=None)
    fc = TestClient(app)
    body = {"name": "missing", "version": "1.0.0"}
    rf = fc.post("/load", json=body)
    nc = httpx.Client(base_url=rest_server, timeout=10.0)
    rn = nc.post("/load", json=body)
    assert rf.status_code == rn.status_code == 503
    assert rf.json() == rn.json()


def test_cypha_rest_fastapi_load_success_shape_parity(rest_bin, tmp_path):
    """Registry with one parity model: successful ``POST /load`` JSON shape matches FastAPI."""
    from cypha_studio.server import api as api_mod

    if not getattr(api_mod, "FASTAPI_AVAILABLE", False):
        pytest.skip("FastAPI not installed")

    import httpx
    from fastapi.testclient import TestClient

    from cypha_studio.core.registry import ModelCard, ModelRegistry

    reg_root = tmp_path / "reg"
    ver_dir = reg_root / "pfest" / "1.0.0"
    ver_dir.mkdir(parents=True)
    shutil.copy(_FIX / "reference.cypha", ver_dir / "model.cypha")
    shutil.copy(_FIX / "f_field.json", ver_dir / "f_field.json")
    card = ModelCard(
        name="pfest",
        version="1.0.0",
        model_type="CyphaDIF",
        encoder_type="VectorEncoder",
        input_dim=8,
        field_dim=24,
        n_classes=3,
        class_labels=["1", "2", "0"],
    )
    (ver_dir / "card.json").write_text(json.dumps(asdict(card), indent=2), encoding="utf-8")

    port = _free_port()
    host = "127.0.0.1"
    base = f"http://{host}:{port}"
    cmd = [
        str(rest_bin),
        "--listen",
        f"{host}:{port}",
        "--cypha",
        str(_FIX / "reference.cypha"),
        "--f-field-json",
        str(_FIX / "f_field.json"),
        "--registry",
        str(reg_root),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(_ROOT),
    )
    try:
        deadline = time.time() + 20.0
        last_err = None
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                pytest.fail(f"cypha_rest exited early ({proc.returncode}): {err[:500]}")
            try:
                r = httpx.get(f"{base}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError as e:
                last_err = e
            time.sleep(0.05)
        else:
            pytest.fail(f"cypha_rest did not become ready: {last_err}")

        def assert_same_keys(a, b, where: str) -> None:
            assert set(a.keys()) == set(b.keys()), (where, sorted(a.keys()), sorted(b.keys()))

        nc = httpx.Client(base_url=base, timeout=30.0)
        load_body = {"name": "pfest", "version": "1.0.0"}
        jn = nc.post("/load", json=load_body).json()

        app = api_mod.create_app(engine=None, registry=ModelRegistry(str(reg_root)), session=None)
        fc = TestClient(app)
        resp = fc.post("/load", json=load_body)
        assert resp.status_code == 200, resp.text
        jf = resp.json()

        assert_same_keys(jf, jn, "load")
        assert_same_keys(jf["loaded"], jn["loaded"], "load.loaded")

        smf = fc.get("/models", params={"summary": "true"})
        smn = nc.get("/models", params={"summary": "true"})
        assert smf.status_code == smn.status_code == 200
        jsmf, jsmn = smf.json(), smn.json()
        assert_same_keys(jsmf, jsmn, "models_summary_nonempty")
        assert len(jsmf["models"]) == len(jsmn["models"]) == 1
        assert_same_keys(jsmf["models"][0], jsmn["models"][0], "models_summary_nonempty[0]")
        assert jsmf["models"][0] == jsmn["models"][0]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_cypha_rest_fastapi_load_404(rest_bin, tmp_path):
    """Unknown ``name``/``version`` (**404**); invalid JSON body — native **400** ``detail`` vs FastAPI **422**."""
    from cypha_studio.server import api as api_mod

    if not getattr(api_mod, "FASTAPI_AVAILABLE", False):
        pytest.skip("FastAPI not installed")

    import httpx
    from fastapi.testclient import TestClient

    from cypha_studio.core.registry import ModelCard, ModelRegistry

    reg_root = tmp_path / "reg404"
    ver_dir = reg_root / "pfest" / "1.0.0"
    ver_dir.mkdir(parents=True)
    shutil.copy(_FIX / "reference.cypha", ver_dir / "model.cypha")
    shutil.copy(_FIX / "f_field.json", ver_dir / "f_field.json")
    card = ModelCard(
        name="pfest",
        version="1.0.0",
        model_type="CyphaDIF",
        encoder_type="VectorEncoder",
        input_dim=8,
        field_dim=24,
        n_classes=3,
        class_labels=["1", "2", "0"],
    )
    (ver_dir / "card.json").write_text(json.dumps(asdict(card), indent=2), encoding="utf-8")

    port = _free_port()
    host = "127.0.0.1"
    base = f"http://{host}:{port}"
    cmd = [
        str(rest_bin),
        "--listen",
        f"{host}:{port}",
        "--cypha",
        str(_FIX / "reference.cypha"),
        "--f-field-json",
        str(_FIX / "f_field.json"),
        "--registry",
        str(reg_root),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(_ROOT),
    )
    try:
        deadline = time.time() + 20.0
        last_err = None
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                pytest.fail(f"cypha_rest exited early ({proc.returncode}): {err[:500]}")
            try:
                r = httpx.get(f"{base}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError as e:
                last_err = e
            time.sleep(0.05)
        else:
            pytest.fail(f"cypha_rest did not become ready: {last_err}")

        nc = httpx.Client(base_url=base, timeout=30.0)
        miss = {"name": "nosuch", "version": "9.9.9"}
        rn = nc.post("/load", json=miss)
        assert rn.status_code == 404
        assert rn.json() == {"detail": "model not found"}

        bad = nc.post("/load", content="{not-json", headers={"Content-Type": "application/json"})
        assert bad.status_code == 400
        assert bad.json() == {"detail": "bad json"}

        app = api_mod.create_app(engine=None, registry=ModelRegistry(str(reg_root)), session=None)
        rf = TestClient(app).post("/load", json=miss)
        assert rf.status_code == 404
        fd = rf.json()
        assert isinstance(fd.get("detail"), str) and fd["detail"]

        badf = TestClient(app).post(
            "/load",
            content="{",
            headers={"Content-Type": "application/json"},
        )
        assert badf.status_code == 422
        assert "detail" in badf.json()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_cypha_rest_predict_regression_sidecar(rest_server_regression):
    """Optional ``regression_head.json``: ``regression_val`` / ``uncertainty`` = softmax-routed scalar MoE."""
    import httpx
    from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary

    manifest = json.loads((_FIX / "manifest.json").read_text(encoding="utf-8"))
    head = json.loads((_FIX / "regression_head.json").read_text(encoding="utf-8"))
    labels = manifest["labels"]
    experts = head["experts"]

    state = cypha_load_binary(str(_FIX / "reference.cypha"))
    m = manifest["model"]
    enc = VectorEncoder(int(m["input_dim"]))
    clf = CyphaDIF(enc, field_dim=int(m["field_dim"]), rng=np.random.default_rng(0))
    clf.load_state(state)
    t_temp = float(clf.temperature)

    exp = np.load(_FIX / "expected.npz")
    x0 = exp["x_input"][0].astype(float).tolist()

    c = httpx.Client(base_url=rest_server_regression, timeout=10.0)
    assert c.get("/metrics").json().get("regression_head_loaded") is True
    r = c.post("/predict", json={"input": x0, "use_gh": True, "return_explanation": False})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("regression_val") is not None
    llr = np.array([data["all_scores"][lbl] for lbl in labels], dtype=np.float64)
    z = llr / (t_temp + 1e-8)
    probs = _softmax_native_style(z)
    mu = np.array([float(experts[lbl]["mu"]) for lbl in labels], dtype=np.float64)
    var = np.array([float(experts[lbl].get("var_ema", 0.0)) for lbl in labels], dtype=np.float64)
    y_exp = float(np.dot(probs, mu))
    mix_var = float(np.dot(probs, var))
    u_exp = float(np.sqrt(max(mix_var, 0.0)))
    np.testing.assert_allclose(float(data["regression_val"]), y_exp, rtol=0, atol=1e-9)
    np.testing.assert_allclose(float(data["uncertainty"]), u_exp, rtol=0, atol=1e-9)


def test_cypha_rest_fastapi_regression_numeric_parity(rest_server_regression):
    """FastAPI ``regression_head_path`` matches native ``cypha_rest`` ``/predict`` MoE numbers."""
    pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")
    from cypha_studio.server import api as api_mod

    if not getattr(api_mod, "FASTAPI_AVAILABLE", False):
        pytest.skip("FastAPI not installed")

    import httpx
    from fastapi.testclient import TestClient

    from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary
    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry

    manifest = json.loads((_FIX / "manifest.json").read_text(encoding="utf-8"))
    state = cypha_load_binary(str(_FIX / "reference.cypha"))
    m = manifest["model"]
    enc = VectorEncoder(int(m["input_dim"]))
    clf = CyphaDIF(enc, field_dim=int(m["field_dim"]), rng=np.random.default_rng(0))
    clf.load_state(state)
    eng = InferenceEngine(clf, None)
    sess = InferenceSession(eng)
    app = api_mod.create_app(
        engine=eng,
        registry=ModelRegistry(),
        session=sess,
        cors_allow_origins=["*"],
        regression_head_path=str(_FIX / "regression_head.json"),
    )
    fc = TestClient(app)
    nc = httpx.Client(base_url=rest_server_regression, timeout=10.0)

    exp = np.load(_FIX / "expected.npz")
    x0 = exp["x_input"][0].astype(float).tolist()
    body = {"input": x0, "use_gh": True, "return_explanation": False}
    mf2, mn2 = fc.get("/metrics").json(), nc.get("/metrics").json()
    assert mf2.get("regression_head_loaded") is True and mn2.get("regression_head_loaded") is True

    pf, pn = fc.post("/predict", json=body).json(), nc.post("/predict", json=body).json()
    assert pf.get("label") == pn.get("label")
    np.testing.assert_allclose(float(pf["regression_val"]), float(pn["regression_val"]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(float(pf["uncertainty"]), float(pn["uncertainty"]), rtol=0, atol=1e-9)


def test_cypha_rest_mke_update_router_loss_matches_fixture(rest_server_mke_train_step):
    """``POST /update`` with ``regression_y`` + ``mke`` sidecar matches ``native_mke_train_step`` router loss."""
    import httpx

    side = json.loads((_FIX / "mke_train_step" / "sidecar.json").read_text(encoding="utf-8"))
    c = httpx.Client(base_url=rest_server_mke_train_step, timeout=15.0)
    assert c.get("/metrics").json().get("regression_head_loaded") is True
    r = c.post(
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
    loss = float(r.json()["loss"])
    np.testing.assert_allclose(loss, float(side["expected_router_loss"]), rtol=0, atol=1e-8)


# ─── /session/rng — native parity ──────────────────────────────────────────────

def test_cypha_rest_session_rng_get_shape(rest_server):
    """GET /session/rng on native returns the correct MT19937 JSON shape."""
    import httpx
    c = httpx.Client(base_url=rest_server, timeout=5.0)
    r = c.get("/session/rng")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get("bit_generator") == "MT19937"
    assert isinstance(d.get("state"), list) and len(d["state"]) == 624
    assert isinstance(d.get("pos"), int)
    assert all(isinstance(v, int) and 0 <= v < 2**32 for v in d["state"])


def test_cypha_rest_session_rng_seed_changes_state(rest_server):
    """POST /session/rng {seed:S} changes the native MT19937 state."""
    import httpx
    c = httpx.Client(base_url=rest_server, timeout=5.0)
    s0 = c.get("/session/rng").json()["state"]
    r = c.post("/session/rng", json={"seed": 999999})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get("bit_generator") == "MT19937"
    assert len(d["state"]) == 624
    # State must have changed from the original run state
    assert d["state"] != s0, "Seed operation did not change the state"
    # Seed to a different value and confirm state differs again
    r2 = c.post("/session/rng", json={"seed": 1})
    assert r2.status_code == 200, r2.text
    assert r2.json()["state"] != d["state"], "Different seeds produced identical states"


def test_cypha_rest_session_rng_state_restore_roundtrip(rest_server):
    """GET /session/rng → POST /session/rng (restore) → GET returns same state on native."""
    import httpx
    c = httpx.Client(base_url=rest_server, timeout=5.0)
    s1 = c.get("/session/rng").json()
    # Seed to something else to dirty the state
    r_seed = c.post("/session/rng", json={"seed": 1})
    assert r_seed.status_code == 200, r_seed.text
    assert r_seed.json()["state"] != s1["state"], "Seed did not change state"
    # Restore the original state
    r_restore = c.post("/session/rng", json={"state": s1["state"], "pos": s1["pos"]})
    assert r_restore.status_code == 200, r_restore.text
    s2 = c.get("/session/rng").json()
    assert s2["state"] == s1["state"], "state restore roundtrip failed (native)"
    assert s2["pos"] == s1["pos"], "pos restore roundtrip failed (native)"


def test_cypha_rest_session_rng_cross_runtime_state_restore(rest_server):
    """MT19937 state captured from native can be restored into Python (FastAPI) and vice-versa.

    Note: Seed-based seeding is NOT bit-exact across runtimes because ``std::mt19937(seed)``
    uses init_genrand while numpy ``MT19937(seed)`` uses init_by_array.  However, the raw
    624-word state array IS the same MT19937 state regardless of runtime, so a full state
    restore (GET one side → POST the other) achieves cross-runtime determinism.
    """
    import httpx
    from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary
    from cypha_studio.server import api as api_mod
    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    from cypha_studio.core.registry import ModelRegistry
    from fastapi.testclient import TestClient

    if not getattr(api_mod, "FASTAPI_AVAILABLE", False):
        pytest.skip("FastAPI not installed")

    # ── 1. GET state from native ──────────────────────────────────────────────
    nc = httpx.Client(base_url=rest_server, timeout=5.0)
    native_state = nc.get("/session/rng").json()
    assert native_state.get("bit_generator") == "MT19937"
    assert len(native_state["state"]) == 624

    # ── 2. Build FastAPI app and restore native state into it ─────────────────
    manifest = json.loads((_FIX / "manifest.json").read_text(encoding="utf-8"))
    raw_state = cypha_load_binary(str(_FIX / "reference.cypha"))
    m = manifest["model"]
    enc = VectorEncoder(int(m["input_dim"]))
    clf = CyphaDIF(enc, field_dim=int(m["field_dim"]),
                   rng=np.random.default_rng(0),
                   replay_rng=np.random.Generator(np.random.MT19937(0)))
    clf.load_state(raw_state)
    eng = InferenceEngine(clf, None)
    app = api_mod.create_app(engine=eng, registry=ModelRegistry(), session=InferenceSession(eng))
    fc = TestClient(app)

    rf = fc.post("/session/rng",
                 json={"state": native_state["state"], "pos": native_state["pos"]})
    assert rf.status_code == 200, rf.text
    fastapi_restored = fc.get("/session/rng").json()

    # The 624-word state and pos must be bit-exact after cross-runtime restore
    assert fastapi_restored["state"] == native_state["state"], (
        "Cross-runtime state restore failed: FastAPI state differs from native snapshot"
    )
    assert fastapi_restored["pos"] == native_state["pos"]

    # ── 3. Restore FastAPI state back into native ─────────────────────────────
    rn2 = nc.post("/session/rng",
                  json={"state": fastapi_restored["state"], "pos": fastapi_restored["pos"]})
    assert rn2.status_code == 200, rn2.text
    native_restored = nc.get("/session/rng").json()
    assert native_restored["state"] == native_state["state"], (
        "Reverse cross-runtime restore failed: native state differs from FastAPI snapshot"
    )
    assert native_restored["pos"] == native_state["pos"]
