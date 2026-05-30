"""Tests for NIGExpert posterior updates."""

from __future__ import annotations

import numpy as np
import pytest

from cypha_lm.expert_field.nig_expert import NIGExpert
from cypha_lm.tests.conftest import assert_no_torch_on_import


@pytest.fixture
def expert() -> NIGExpert:
    return NIGExpert(kappa0=1.0, alpha0=2.0, beta0=1.0, mu0=0.0)


def test_prior_prediction(expert: NIGExpert) -> None:
    mean, epistemic, aleatoric = expert.predict()
    assert mean == pytest.approx(0.0)
    assert epistemic == pytest.approx(expert.beta0 / (expert.kappa0 * (expert.alpha0 - 1.0)))
    assert aleatoric == pytest.approx(expert.beta0 / (expert.alpha0 - 1.0))


def test_single_update(expert: NIGExpert) -> None:
    expert.update(3.0)
    mean, _, _ = expert.predict()
    assert mean == pytest.approx(1.5)
    assert mean != pytest.approx(0.0)


def test_many_updates_converge() -> None:
    ex = NIGExpert(kappa0=0.5, alpha0=2.0, beta0=1.0, mu0=0.0)
    rng = np.random.default_rng(42)
    obs = rng.normal(5.0, 1.0, size=1000)
    for y in obs:
        ex.update(float(y))
    mean, _, aleatoric = ex.predict()
    assert mean == pytest.approx(5.0, rel=0.05, abs=0.15)
    assert aleatoric == pytest.approx(1.0, rel=0.2, abs=0.3)


def test_epistemic_decreases(expert: NIGExpert) -> None:
    first = expert.epistemic_variance()
    rng = np.random.default_rng(0)
    for _ in range(50):
        expert.update(float(rng.normal(2.0, 0.5)))
    last = expert.epistemic_variance()
    assert last < first


def test_aleatoric_stable() -> None:
    ex = NIGExpert(kappa0=1.0, alpha0=2.0, beta0=1.0, mu0=0.0)
    rng = np.random.default_rng(0)
    for y in rng.normal(0.0, 1.0, size=500):
        ex.update(float(y))
    aleatoric = ex.aleatoric_variance()
    assert aleatoric == pytest.approx(1.0, rel=0.25, abs=0.35)


def test_log_prob_valid(expert: NIGExpert) -> None:
    lp = expert.predictive_log_prob(0.5)
    assert np.isfinite(lp)


def test_online_vs_batch() -> None:
    kappa0, alpha0, beta0, mu0 = 1.0, 2.0, 1.0, 0.0
    rng = np.random.default_rng(123)
    obs = rng.normal(2.0, 0.5, size=20)

    online = NIGExpert(kappa0, alpha0, beta0, mu0=mu0)
    for y in obs:
        online.update(float(y))

    ref = NIGExpert(kappa0, alpha0, beta0, mu0=mu0)
    for y in obs:
        ref.update(float(y))

    o_mean, o_epi, o_ale = online.predict()
    r_mean, r_epi, r_ale = ref.predict()
    assert o_mean == pytest.approx(r_mean, rel=1e-9)
    assert o_epi == pytest.approx(r_epi, rel=1e-9)
    assert o_ale == pytest.approx(r_ale, rel=1e-9)


def test_no_torch_dependency() -> None:
    assert_no_torch_on_import("from cypha_lm.expert_field.nig_expert import NIGExpert")
