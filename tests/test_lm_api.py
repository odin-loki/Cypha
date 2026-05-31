"""Tests for CyphaLM generation utilities and CyphaStudio LM REST routes."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

cypha_lm = pytest.importorskip("cypha_lm")

from cypha_lm.config import CyphaLMConfig
from cypha_lm.model.cypha_lm import CyphaLM
from cypha_lm.model.generation import (
    autoregressive_decode,
    top_p_sample,
)


@pytest.fixture
def tiny_lm() -> CyphaLM:
    cfg = CyphaLMConfig(vocab_size=32, d_embed=16, d_state=32, field_dim=48, max_experts=16)
    model = CyphaLM(cfg)
    seq = list(range(1, 20)) * 3
    model.train_sequence(seq)
    return model


def test_predict_next_exposes_routing(tiny_lm: CyphaLM) -> None:
    pred = tiny_lm.predict_next(1)
    assert "routing_probs" in pred
    assert "dominant_expert" in pred
    assert "active_experts" in pred
    assert len(pred["routing_probs"]) >= 1


def test_top_p_sample(tiny_lm: CyphaLM) -> None:
    ids = top_p_sample(tiny_lm, [1, 2, 3], max_tokens=8, p=0.9, temperature=0.8)
    assert len(ids) == 8


def test_autoregressive_decode_trace(tiny_lm: CyphaLM) -> None:
    out = autoregressive_decode(tiny_lm, [1, 2], 5, strategy="greedy", temperature=0.0)
    assert len(out["generated_ids"]) == 5
    assert len(out["per_step"]) == 5
    assert "dominant_expert" in out["per_step"][0]


def test_save_load_roundtrip(tiny_lm: CyphaLM) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "ckpt")
        tiny_lm.save(base)
        loaded = CyphaLM.load(base)
        a = tiny_lm.predict_next(3)
        b = loaded.predict_next(3)
        np.testing.assert_allclose(a["log_probs"], b["log_probs"], rtol=0, atol=1e-9)


def test_stream_generate_chunks(tiny_lm: CyphaLM) -> None:
    chunks = list(tiny_lm.stream_generate([1, 2], max_tokens=4, temperature=0.9))
    assert len(chunks) >= 4
    assert chunks[-1].get("done") is True


def test_lm_rest_generate() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from cypha_studio.core.lm_engine import LMEngine
    from cypha_studio.server.api import create_app

    cfg = CyphaLMConfig(vocab_size=32, d_embed=16, d_state=32, field_dim=48, max_experts=16)
    model = CyphaLM(cfg)
    model.train_sequence(list(range(1, 15)) * 2)
    app = create_app(lm_engine=LMEngine(model))
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["lm_loaded"] is True

    r = client.post(
        "/generate",
        json={"prompt_ids": [1, 2, 3], "max_tokens": 5, "strategy": "greedy"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["generated_ids"]) == 5
    assert "per_step" in body

    r = client.post(
        "/lm/predict_next",
        json={"token_id": 2},
    )
    assert r.status_code == 200
    assert "routing_probs" in r.json()

    with client.stream(
        "POST",
        "/generate/stream",
        json={"prompt_ids": [1, 2], "max_tokens": 3, "strategy": "greedy"},
    ) as resp:
        assert resp.status_code == 200
        raw_lines = list(resp.iter_lines())
        assert len(raw_lines) >= 2
        assert any("data:" in (ln.decode() if isinstance(ln, bytes) else ln) for ln in raw_lines)
