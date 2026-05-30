"""Tests for GRIA alpha-projection layer."""

from __future__ import annotations

import numpy as np
import pytest

from cypha_lm.projection.gria_projection import GRIAProjection
from cypha_lm.tests.conftest import assert_no_torch_on_import


@pytest.fixture
def proj() -> GRIAProjection:
    return GRIAProjection(d_input=32, vocab_size=64, alpha_init=0.5, seed=42)


def test_output_shape(proj: GRIAProjection) -> None:
    v = np.random.default_rng(0).standard_normal(32)
    out = proj.forward(v)
    assert out.shape == (proj.vocab_size,)


def test_log_prob_sums(proj: GRIAProjection) -> None:
    v = np.random.default_rng(1).standard_normal(32)
    log_probs = proj.forward(v)
    probs = np.exp(log_probs)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_alpha_clipping(proj: GRIAProjection) -> None:
    huge_grad = np.full(proj.vocab_size, 1e6)
    for _ in range(10):
        proj.update_alpha(huge_grad, lr=1.0)
    assert np.all(proj.alpha >= 0.01)
    assert np.all(proj.alpha <= 0.99)


def test_unigram_prior(proj: GRIAProjection) -> None:
    counts = np.zeros(proj.vocab_size)
    counts[0] = 100
    counts[1] = 50
    counts[2] = 10
    proj.set_unigram_prior(counts)
    proj.W[:] = 0.0
    proj.alpha[:] = 0.0
    v = np.zeros(32)
    log_probs = proj.forward(v)
    probs = np.exp(log_probs)
    expected = counts + 1.0
    expected /= expected.sum()
    np.testing.assert_allclose(probs, expected, rtol=0.15, atol=0.05)


def test_alpha_init() -> None:
    p = GRIAProjection(d_input=16, vocab_size=32, alpha_init=0.42, seed=0)
    assert np.allclose(p.alpha, 0.42)


def test_gul_alpha_in_range(proj: GRIAProjection) -> None:
    rng = np.random.default_rng(2)
    activations = rng.standard_normal(100)
    outputs = rng.standard_normal(64)
    alpha = proj.grand_unified_law_alpha(activations, outputs)
    assert 0.0 <= alpha <= 1.0


def test_alpha_spectrum_keys(proj: GRIAProjection) -> None:
    spec = proj.alpha_spectrum()
    for key in ("mean", "std", "min", "max", "histogram", "histogram_edges"):
        assert key in spec


def test_high_alpha_context_sens() -> None:
    p = GRIAProjection(d_input=8, vocab_size=16, alpha_init=0.5, seed=1)
    p.alpha[:] = 0.5
    p.alpha[0] = 1.0
    p.alpha[1] = 0.01
    v1 = np.ones(8)
    v2 = -np.ones(8)
    lp1 = p.forward(v1)
    lp2 = p.forward(v2)
    change_high = abs(lp1[0] - lp2[0])
    change_low = abs(lp1[1] - lp2[1])
    assert change_high > change_low + 1e-4


def test_no_torch_dependency() -> None:
    assert_no_torch_on_import("from cypha_lm.projection.gria_projection import GRIAProjection")
