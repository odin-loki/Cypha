"""Tests for n-gram fusion module."""

from __future__ import annotations

import numpy as np

from cypha_lm.projection.ngram_fusion import NgramFusion


def test_gated_forward_shape() -> None:
    nf = NgramFusion(32, 32, 192, mode="gated", seed=0)
    fx = np.random.randn(32)
    em = np.random.randn(192)
    v = nf.forward(fx, em)
    assert v.shape == (32,)


def test_grad_field_finite() -> None:
    nf = NgramFusion(32, 32, 192, mode="gated", seed=0)
    fx = np.random.randn(32)
    em = np.random.randn(192)
    nf.forward(fx, em)
    g = np.random.randn(32)
    grad_fx = nf.grad_field_x(g)
    assert grad_fx.shape == (32,)
    assert np.all(np.isfinite(grad_fx))
