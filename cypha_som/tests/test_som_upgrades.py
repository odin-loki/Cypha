"""Unit tests for cypha_som upgrades."""

import numpy as np
import pytest

from cypha_som.config import reset_flags, set_upgrade_flags
from cypha_som.discriminative_feedback import DiscriminativeFeedback
from cypha_som.gng_expert import GNGExpertManager
from cypha_som.gria_controller import GRIAController
from cypha_som.hebbian_topology import DynamicHebbianGraph
from cypha_som.som_encoder import OnlineSOMEncoder
from cypha_som.temporal_som import TemporalSOM


def test_gng_grows_on_complex_data():
    gng = GNGExpertManager(8, lam=20, max_nodes=64)
    rng = np.random.default_rng(0)
    for _ in range(300):
        gng.step(rng.standard_normal(8))
    assert gng.node_count() >= 2


def test_som_returns_same_dim():
    som = OnlineSOMEncoder(16, k=4, T=100)
    z = np.ones(16)
    out = som.encode(z)
    assert out.shape == (16,)


def test_gria_alpha_bounded():
    g = GRIAController(window=50)
    rng = np.random.default_rng(1)
    for _ in range(100):
        g.push(rng.standard_normal(10), rng.standard_normal(5))
    a = g.alpha()
    assert 0.0 <= a <= 1.5


def test_hebbian_spectral_radius():
    graph = DynamicHebbianGraph(8)
    rng = np.random.default_rng(2)
    for _ in range(200):
        graph.update(rng.standard_normal(8))
    assert graph.spectral_radius() <= 1.01


def test_temporal_som_decay_range():
    ts = TemporalSOM(M=4, L_max=8)
    rng = np.random.default_rng(3)
    for _ in range(50):
        _, lf, ls = ts.step(rng.standard_normal(4))
    assert 0.5 < lf < 1.5
    assert 0.5 < ls < 1.5


def test_discriminative_modulate():
    fb = DiscriminativeFeedback(beta=0.1)
    dW = np.ones((4, 4))
    d = np.array([1.0, 0.0, 0.0, 0.0])
    out = fb.modulate(dW, d)
    assert out[0, 0] > dW[0, 0]


def test_config_flags():
    reset_flags()
    set_upgrade_flags("U2")
    from cypha_som import config

    assert config.USE_SOM_ENCODER
    assert not config.USE_GNG
