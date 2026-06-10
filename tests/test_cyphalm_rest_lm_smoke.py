"""
Native ``cypha_rest`` CyphaLM routes: ``/lm/*``, ``/generate``, ``/generate/stream``.

Skips unless ``CYPHA_REST_BIN`` or a built ``cypha_rest`` exists and checkpoint fixture is present.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("httpx", reason="httpx not installed (requirements-verify.txt)")

_ROOT = Path(__file__).resolve().parents[1]
_FIX = _ROOT / "parity_fixtures"
_CKPT = _FIX / "cyphalm_checkpoint" / "char_lstm" / "checkpoint.json"


def _cypha_rest_executable() -> Path | None:
    env = os.environ.get("CYPHA_REST_BIN", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    if sys.platform == "win32":
        candidates = [
            Path(r"C:\Temp\cypha_native_build6\cypha_rest.exe"),
            _ROOT / "native" / "build-mingw-w64" / "cypha_rest.exe",
            _ROOT / "native" / "build" / "Release" / "cypha_rest.exe",
        ]
    else:
        candidates = [_ROOT / "native" / "build" / "cypha_rest"]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


@pytest.fixture(scope="module")
def rest_lm_bin():
    exe = _cypha_rest_executable()
    if exe is None:
        pytest.skip("cypha_rest not built (set CYPHA_REST_BIN)")
    if not (_FIX / "reference.cypha").is_file() or not (_FIX / "f_field.json").is_file():
        pytest.skip("parity_fixtures missing reference.cypha or f_field.json")
    if not _CKPT.is_file():
        pytest.skip("run scripts/generate_cyphalm_checkpoint_fixture.py")
    return exe


@pytest.fixture
def rest_lm_server(rest_lm_bin):
    import httpx

    port = _free_port()
    host = "127.0.0.1"
    cmd = [
        str(rest_lm_bin),
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
    deadline = time.time() + 20.0
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                pytest.fail(f"cypha_rest exited early ({proc.returncode}): {err[:500]}")
            try:
                if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        else:
            pytest.fail("cypha_rest did not become ready")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def rest_lm_server_autoload(rest_lm_bin):
    """Server started with ``--cyphalm-checkpoint``."""
    import httpx

    port = _free_port()
    host = "127.0.0.1"
    cmd = [
        str(rest_lm_bin),
        "--listen",
        f"{host}:{port}",
        "--cypha",
        str(_FIX / "reference.cypha"),
        "--f-field-json",
        str(_FIX / "f_field.json"),
        "--cyphalm-checkpoint",
        str(_CKPT),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(_ROOT),
    )
    base = f"http://{host}:{port}"
    deadline = time.time() + 25.0
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                pytest.fail(f"cypha_rest exited early ({proc.returncode}): {err[:500]}")
            try:
                r = httpx.get(f"{base}/health", timeout=1.0)
                if r.status_code == 200 and r.json().get("lm_loaded") is True:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        else:
            pytest.fail("cypha_rest LM autoload did not become ready")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_cyphalm_rest_lm_load_predict_generate(rest_lm_server):
    import httpx

    side = json.loads((_CKPT.parent / "sidecar.json").read_text(encoding="utf-8"))
    tid = int(side["eval_ids"][0])
    c = httpx.Client(base_url=rest_lm_server, timeout=30.0)
    assert c.get("/health").json().get("lm_loaded") is False

    load = c.post("/lm/load", json={"checkpoint_path": str(_CKPT)})
    assert load.status_code == 200, load.text
    assert load.json().get("loaded") is True
    assert c.get("/health").json().get("lm_loaded") is True

    metrics = c.get("/lm/metrics")
    assert metrics.status_code == 200
    assert metrics.json().get("vocab_size", 0) > 0

    pn = c.post("/lm/predict_next", json={"token_id": tid})
    assert pn.status_code == 200, pn.text
    body = pn.json()
    assert isinstance(body.get("log_probs"), list)
    assert len(body["log_probs"]) == metrics.json()["vocab_size"]

    gen = c.post(
        "/generate",
        json={"prompt_ids": [tid], "max_tokens": 5, "strategy": "greedy"},
    )
    assert gen.status_code == 200, gen.text
    g = gen.json()
    assert len(g["generated_ids"]) == 5
    assert g.get("strategy") == "greedy"
    assert len(g["per_step"]) == 5


def test_cyphalm_rest_generate_top_p(rest_lm_server):
    import httpx

    c = httpx.Client(base_url=rest_lm_server, timeout=30.0)
    c.post("/lm/load", json={"checkpoint_path": str(_CKPT)})
    gen = c.post(
        "/generate",
        json={
            "prompt_ids": [1, 2],
            "max_tokens": 3,
            "strategy": "top_p",
            "top_p": 0.9,
            "temperature": 0.8,
            "seed": 7,
        },
    )
    assert gen.status_code == 200, gen.text
    assert len(gen.json()["generated_ids"]) == 3


def test_cyphalm_rest_generate_stream(rest_lm_server):
    import httpx

    side = json.loads((_CKPT.parent / "sidecar.json").read_text(encoding="utf-8"))
    tid = int(side["eval_ids"][0])
    c = httpx.Client(base_url=rest_lm_server, timeout=30.0)
    c.post("/lm/load", json={"checkpoint_path": str(_CKPT)})

    with c.stream(
        "POST",
        "/generate/stream",
        json={"prompt_ids": [tid], "max_tokens": 4, "strategy": "greedy"},
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        events = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    token_events = [e for e in events if not e.get("done")]
    assert len(token_events) == 4
    assert events[-1].get("done") is True


def test_cyphalm_rest_checkpoint_autoload(rest_lm_server_autoload):
    import httpx

    c = httpx.Client(base_url=rest_lm_server_autoload, timeout=15.0)
    m = c.get("/lm/metrics")
    assert m.status_code == 200, m.text
    assert m.json().get("loaded") is True
