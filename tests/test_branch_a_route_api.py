"""Tests for Branch A REST routing and Ollama fallback stub."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@pytest.fixture(autouse=True)
def _fast_branch_a_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use hashing embedder and small train set for CI speed."""
    monkeypatch.setenv("CYPHA_BRANCH_A_EMBED_BACKEND", "hashing")
    monkeypatch.setenv("CYPHA_BRANCH_A_N_TRAIN", "400")


def test_branch_a_router_route() -> None:
    from cypha_studio.core.branch_a_router import BranchARouter

    router = BranchARouter(n_train_samples=400, backend="hashing", epistemic_threshold=0.5)
    router.train()
    out = router.route("Linux kernel module compilation help")
    assert "label" in out
    assert "epistemic_var" in out
    assert out["action"] in ("cypha_route", "fallback_llm")
    assert isinstance(out["abstain"], bool)


def test_ollama_generate_mocked() -> None:
    from cypha_studio.core import ollama_client

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "hello from mistral", "done": True}
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_resp

    with patch.object(ollama_client.httpx, "Client", return_value=mock_client):
        out = ollama_client.ollama_generate("test prompt", model="mistral")
    assert out["provider"] == "ollama"
    assert out["text"] == "hello from mistral"


def test_dispatch_ollama_on_abstain() -> None:
    from cypha_studio.core.branch_a_router import BranchARouter

    router = BranchARouter(n_train_samples=400, backend="hashing", epistemic_threshold=-1.0)
    router.train()
    with patch(
        "cypha_studio.core.branch_a_router.ollama_generate",
        return_value={"provider": "ollama", "text": "fallback answer", "model": "mistral"},
    ):
        out = router.dispatch_generate("totally unknown quantum gardening topic xyz")
    assert out["route"]["abstain"] is True
    assert out["generation"]["provider"] == "ollama"
    assert out["generation"]["text"] == "fallback answer"


def test_branch_a_checkpoint_roundtrip(tmp_path: Path) -> None:
    from cypha_studio.core.branch_a_router import BranchARouter, checkpoint_paths

    base = tmp_path / "router_ckpt"
    a = BranchARouter(n_train_samples=400, backend="hashing")
    a.train()
    r1 = a.route("compile linux kernel")
    a.save_checkpoint(base)

    meta_path, npz_path = checkpoint_paths(base)
    assert meta_path.is_file()
    assert npz_path.is_file()

    b = BranchARouter(n_train_samples=400, backend="hashing", checkpoint_base=base)
    assert b.try_load_checkpoint()
    r2 = b.route("compile linux kernel")
    assert r1["label"] == r2["label"]
    assert abs(r1["epistemic_var"] - r2["epistemic_var"]) < 1e-9


def test_route_rest_endpoints() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from cypha_studio.server.api import create_app

    app = create_app()
    client = TestClient(app)

    r = client.get("/route/health")
    assert r.status_code == 200
    body = r.json()
    assert "router_trained" in body
    assert "ollama_reachable" in body

    r = client.post("/route/text", json={"text": "Windows NT driver development"})
    assert r.status_code == 200
    route = r.json()
    assert route["action"] in ("cypha_route", "fallback_llm")
    assert "latency_ms" in route

    with patch(
        "cypha_studio.core.branch_a_router.ollama_generate",
        return_value={"provider": "ollama", "text": "REST fallback", "model": "mistral", "latency_ms": 1.0},
    ):
        router = app.state.branch_a_router
        assert router is not None
        router.epistemic_threshold = -1.0
        r = client.post(
            "/route/generate",
            json={"text": "exotic OOD query for abstain path", "max_tokens": 16},
        )
    assert r.status_code == 200
    gen_body = r.json()
    assert gen_body["route"]["abstain"] is True
    assert gen_body["generation"]["text"] == "REST fallback"
