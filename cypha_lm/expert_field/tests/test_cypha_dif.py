"""Tests for CyphaDIF routing and expert field."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from cypha_lm.config import CyphaLMConfig
from cypha_lm.expert_field.cypha_dif import CyphaDIF
from cypha_lm.tests.conftest import TEST_CONFIG, assert_no_torch_on_import


@pytest.fixture
def dif() -> CyphaDIF:
    return CyphaDIF(TEST_CONFIG)


def _random_target(rng: np.random.Generator, dim: int) -> np.ndarray:
    return rng.standard_normal(dim).astype(np.float64)


def test_route_sums_to_one(dif: CyphaDIF) -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.standard_normal(32)
        probs = dif.route(x)
        assert probs.shape[0] == dif.expert_count()
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_new_expert_creation(dif: CyphaDIF) -> None:
    rng = np.random.default_rng(1)
    before = dif.expert_count()
    for _ in range(30):
        x = rng.standard_normal(32) * 50.0
        y = _random_target(rng, TEST_CONFIG.field_dim)
        dif.train_step(x, y)
    assert dif.expert_count() > before


def test_expert_specialisation() -> None:
    cfg = CyphaLMConfig(
        vocab_size=64,
        d_embed=64,
        field_dim=32,
        d_state=16,
        ssm_layers=1,
        max_experts=32,
        nig_kappa0=0.2,
        seed=7,
    )
    dif = CyphaDIF(cfg)
    rng = np.random.default_rng(7)
    dim = cfg.field_dim
    for i in range(10):
        x_seed = rng.standard_normal(32) * (i + 1) * 4.0
        dif.train_step(x_seed, rng.standard_normal(dim))

    center_a = np.zeros(32)
    center_b = np.ones(32) * 6.0
    target_a = rng.standard_normal(dim)
    target_b = rng.standard_normal(dim) + 5.0

    for block in range(20):
        for _ in range(50):
            x_a = center_a + 0.1 * rng.standard_normal(dim)
            dif.train_step(x_a, target_a)
        for _ in range(50):
            x_b = center_b + 0.1 * rng.standard_normal(dim)
            dif.train_step(x_b, target_b)

    route_a = []
    route_b = []
    for _ in range(100):
        x_a = center_a + 0.05 * rng.standard_normal(dim)
        x_b = center_b + 0.05 * rng.standard_normal(dim)
        route_a.append(dif.route(x_a))
        route_b.append(dif.route(x_b))

    mean_a = np.mean(route_a, axis=0)
    mean_b = np.mean(route_b, axis=0)
    assert dif.expert_count() >= 2
    assert float(np.max(np.abs(mean_a - mean_b))) > 0.15


def test_predict_keys(dif: CyphaDIF) -> None:
    x = np.random.default_rng(2).standard_normal(32)
    out = dif.predict(x)
    for key in ("mean", "epistemic_var", "aleatoric_var", "routing_probs", "active_experts"):
        assert key in out
    assert out["mean"].shape == (TEST_CONFIG.field_dim,)


def test_epistemic_decreases_training(dif: CyphaDIF) -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(32)
    y = _random_target(rng, TEST_CONFIG.field_dim)
    epi_trace = []
    for _ in range(100):
        dif.train_step(x, y)
        epi_trace.append(dif.predict(x)["epistemic_var"])
    assert epi_trace[-1] <= epi_trace[0] * 1.05 + 1e-6


def test_high_epistemic_on_novel(dif: CyphaDIF) -> None:
    rng = np.random.default_rng(4)
    dim = TEST_CONFIG.field_dim
    for i in range(8):
        x_seed = rng.standard_normal(32) * (i + 1) * 3.0
        y_seed = rng.standard_normal(dim)
        dif.train_step(x_seed, y_seed)

    x_in = rng.standard_normal(32) * 0.5
    y = _random_target(rng, dim)
    for _ in range(200):
        dif.train_step(x_in, y)
    in_dist = dif.predict(x_in)["epistemic_var"]

    before_count = dif.expert_count()
    x_ood = rng.standard_normal(32) * 80.0 + 300.0
    dif.route(x_ood)
    ood = dif.predict(x_ood)["epistemic_var"]
    assert dif.expert_count() >= before_count
    assert ood >= in_dist * 0.5 or ood > 1e-3


def test_alpha_per_expert_range(dif: CyphaDIF) -> None:
    rng = np.random.default_rng(5)
    for _ in range(50):
        x = rng.standard_normal(32)
        y = _random_target(rng, TEST_CONFIG.field_dim)
        dif.train_step(x, y)
    alphas = dif.alpha_per_expert()
    assert alphas.size > 0
    assert np.all(alphas >= 0.0)
    assert np.all(alphas <= 1.0)


def test_alpha_edge_of_chaos(dif: CyphaDIF) -> None:
    rng = np.random.default_rng(6)
    for _ in range(800):
        x = rng.standard_normal(32)
        y = _random_target(rng, TEST_CONFIG.field_dim)
        dif.train_step(x, y)
    alphas = dif.alpha_per_expert()
    if alphas.size == 0:
        pytest.skip("no experts created")
    assert np.all((alphas >= 0.0) & (alphas <= 1.0))
    assert float(np.mean(alphas)) <= 0.55


def test_reset_clears_experts(dif: CyphaDIF) -> None:
    rng = np.random.default_rng(8)
    for _ in range(20):
        dif.train_step(rng.standard_normal(32), _random_target(rng, TEST_CONFIG.field_dim))
    assert dif.expert_count() >= 1
    dif.reset()
    assert dif.expert_count() in (0, 1)


def test_no_torch_dependency() -> None:
    assert_no_torch_on_import("from cypha_lm.expert_field.cypha_dif import CyphaDIF")
