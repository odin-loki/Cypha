"""Tests for full CyphaLM stack."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cypha_lm.config import CyphaLMConfig
from cypha_lm.model.cypha_lm import CyphaLM
from cypha_lm.tests.conftest import TEST_CONFIG, assert_no_torch_on_import


@pytest.fixture
def model() -> CyphaLM:
    return CyphaLM(TEST_CONFIG)


def test_train_step_returns_loss(model: CyphaLM) -> None:
    out = model.train_step(0, 1)
    assert "loss" in out
    assert np.isfinite(out["loss"])
    assert out["loss"] > 0.0


def test_train_reduces_loss(model: CyphaLM) -> None:
    pattern = [0, 1, 2, 3, 4, 5]
    losses = []
    model.reset_context()
    for _ in range(500):
        for t in range(len(pattern) - 1):
            m = model.train_step(pattern[t], pattern[t + 1])
            losses.append(m["loss"])
    assert np.mean(losses[-50:]) < np.mean(losses[:50])


def test_predict_valid_distribution(model: CyphaLM) -> None:
    pred = model.predict_next(0)
    log_probs = pred["log_probs"]
    assert log_probs.shape == (TEST_CONFIG.vocab_size,)
    assert np.all(np.isfinite(log_probs))
    probs = np.exp(log_probs)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_generate_length(model: CyphaLM) -> None:
    max_tokens = 25
    out = model.generate([0, 1, 2], max_tokens=max_tokens, temperature=0.8)
    assert len(out["generated_ids"]) == max_tokens


def test_uncertainty_threshold_stops(model: CyphaLM) -> None:
    max_tokens = 50
    out = model.generate(
        [0],
        max_tokens=max_tokens,
        temperature=1.0,
        uncertainty_threshold=0.0,
    )
    assert len(out["generated_ids"]) < max_tokens


def test_online_update(model: CyphaLM) -> None:
    seq = [2, 5, 2, 5, 2, 5, 2, 5]
    model.reset_context()
    ppl_before = _sequence_perplexity(model, seq)
    for _ in range(200):
        model.train_sequence(seq)
    model.reset_context()
    ppl_after = _sequence_perplexity(model, seq)
    assert ppl_after < ppl_before


def test_compression_profile_keys(model: CyphaLM) -> None:
    model.train_step(0, 1)
    profile = model.compression_profile()
    for key in (
        "mean_epistemic_var",
        "mean_alpha",
        "expert_alpha_spectrum",
        "n_experts",
    ):
        assert key in profile


def test_save_load_roundtrip(model: CyphaLM) -> None:
    model.train_step(1, 2)
    model.train_step(2, 3)
    pred_before = model.predict_next(1)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "cypha_checkpoint")
        model.save(path)
        loaded = CyphaLM.load(path)
    loaded.reset_context()
    model.reset_context()
    pred_after = loaded.predict_next(1)
    np.testing.assert_allclose(
        pred_before["log_probs"],
        pred_after["log_probs"],
        rtol=1e-5,
        atol=1e-5,
    )


def test_no_torch_dependency() -> None:
    assert_no_torch_on_import("from cypha_lm.model.cypha_lm import CyphaLM")


def test_context_reset(model: CyphaLM) -> None:
    for tid in [1, 2, 3, 4, 5]:
        model.predict_next(tid)
    model.reset_context()
    pred_clean = model.predict_next(0)
    fresh = CyphaLM(TEST_CONFIG)
    pred_fresh = fresh.predict_next(0)
    np.testing.assert_allclose(
        pred_clean["log_probs"],
        pred_fresh["log_probs"],
        rtol=1e-10,
        atol=1e-10,
    )


def _sequence_perplexity(model: CyphaLM, token_ids: list[int]) -> float:
    model.reset_context()
    nll = 0.0
    n = 0
    for t in range(len(token_ids) - 1):
        pred = model.predict_next(token_ids[t])
        nll += -float(pred["log_probs"][token_ids[t + 1]])
        n += 1
    return float(np.exp(nll / max(n, 1)))
