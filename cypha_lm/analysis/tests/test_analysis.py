"""Tests for analysis modules (alpha spectrum and compression profile)."""

from __future__ import annotations

import numpy as np
import pytest

from cypha_lm.analysis.alpha_spectrum import AlphaSpectrumAnalyser
from cypha_lm.analysis.compression_profile import CompressionProfiler
from cypha_lm.model.cypha_lm import CyphaLM
from cypha_lm.tests.conftest import TEST_CONFIG, assert_no_torch_on_import


@pytest.fixture
def model() -> CyphaLM:
    return CyphaLM(TEST_CONFIG)


def test_alpha_snapshot_keys(model: CyphaLM) -> None:
    analyser = AlphaSpectrumAnalyser(model)
    snap = analyser.snapshot()
    for key in (
        "gria_projection_alpha",
        "expert_alpha",
        "mean_alpha",
        "fraction_near_edge_of_chaos",
    ):
        assert key in snap
    assert snap["gria_projection_alpha"].shape == (TEST_CONFIG.vocab_size,)


def test_alpha_track_returns_dataframe(model: CyphaLM) -> None:
    analyser = AlphaSpectrumAnalyser(model)
    df = analyser.track(n_steps=20, train_data=[0, 1, 2, 3, 4, 5])
    assert len(df) == 20
    assert "mean_alpha" in df.columns
    assert "step" in df.columns


def test_compression_profiler_keys(model: CyphaLM) -> None:
    for _ in range(30):
        model.train_step(0, 1)
    profiler = CompressionProfiler()
    result = profiler.measure(model, [[0, 1, 2, 3, 4], [5, 6, 7, 8]])
    for key in (
        "lossy_fraction",
        "lossless_fraction",
        "per_token_profile",
        "compression_ratio",
    ):
        assert key in result
    assert 0.0 <= result["lossy_fraction"] <= 1.0
    assert 0.0 <= result["lossless_fraction"] <= 1.0


def test_compression_fractions_sum(model: CyphaLM) -> None:
    model.train_sequence([0, 1, 2, 3, 4, 5, 6])
    profiler = CompressionProfiler()
    result = profiler.measure(model, [[1, 2, 3, 4]])
    total = result["lossy_fraction"] + result["lossless_fraction"]
    assert total == pytest.approx(1.0, abs=0.05)


def test_no_torch_dependency() -> None:
    assert_no_torch_on_import(
        "from cypha_lm.analysis.alpha_spectrum import AlphaSpectrumAnalyser; "
        "from cypha_lm.analysis.compression_profile import CompressionProfiler"
    )
