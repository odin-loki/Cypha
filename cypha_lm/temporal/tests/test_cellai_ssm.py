"""Tests for CellAI multi-scale SSM."""

from __future__ import annotations

import numpy as np
import pytest

from cypha_lm.temporal.cellai_ssm import CellAISSM
from cypha_lm.tests.conftest import assert_no_torch_on_import


def _make_ssm(n_layers: int = 1, seed: int = 42, d_input: int = 16, d_state: int = 8) -> CellAISSM:
    return CellAISSM(
        d_input=d_input,
        d_state=d_state,
        tau_fast=1.0,
        tau_slow=20.0,
        n_layers=n_layers,
        seed=seed,
    )


def _structured_sequence(T: int, d_input: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(d_input)
    seq = np.zeros((T, d_input), dtype=np.float64)
    for t in range(T):
        seq[t] = base * np.sin(0.1 * t) + 0.3 * rng.standard_normal(d_input)
    return seq


def _next_step_mse(ssm: CellAISSM, seq: np.ndarray) -> float:
    contexts = ssm.process_sequence(seq)
    if seq.shape[0] < 3:
        return float("inf")
    x = contexts[:-1]
    y = seq[1:]
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    return float(np.mean((pred - y) ** 2))


@pytest.fixture
def ssm() -> CellAISSM:
    return _make_ssm()


def test_step_shape(ssm: CellAISSM) -> None:
    e = np.random.default_rng(0).standard_normal(ssm.d_input)
    ctx = ssm.step(e)
    assert ctx.shape == (ssm.context_dim,)


def test_sequence_shape(ssm: CellAISSM) -> None:
    T = 20
    seq = np.random.default_rng(1).standard_normal((T, ssm.d_input))
    out = ssm.process_sequence(seq)
    assert out.shape == (T, ssm.context_dim)


def test_online_vs_batch(ssm: CellAISSM) -> None:
    seq = np.random.default_rng(2).standard_normal((15, ssm.d_input))
    ssm.reset()
    online = np.stack([ssm.step(seq[t]) for t in range(seq.shape[0])], axis=0)
    ssm.reset()
    batch = ssm.process_sequence(seq)
    np.testing.assert_allclose(online, batch, rtol=1e-10, atol=1e-10)


def test_fast_track_decay() -> None:
    ssm = _make_ssm(seed=3)
    impulse = np.zeros(ssm.d_input)
    impulse[0] = 1.0
    ssm.step(impulse)
    h_after = ssm._h[0].copy()
    zeros = np.zeros(ssm.d_input)
    for _ in range(200):
        ssm.step(zeros)
    h_final = ssm._h[0]
    ratio = np.linalg.norm(h_final) / (np.linalg.norm(h_after) + 1e-12)
    expected = ssm.lambda_fast ** 200
    assert ratio < expected * 1.5 + 0.05


def test_slow_track_decay() -> None:
    ssm = _make_ssm(seed=4)
    impulse = np.zeros(ssm.d_input)
    impulse[0] = 1.0
    ssm.step(impulse)
    s_after = ssm._s[0].copy()
    zeros = np.zeros(ssm.d_input)
    for _ in range(200):
        ssm.step(zeros)
    s_final = ssm._s[0]
    ratio = np.linalg.norm(s_final) / (np.linalg.norm(s_after) + 1e-12)
    expected = ssm.lambda_slow ** 200
    assert ratio > expected * 0.5


def test_temporal_separation() -> None:
    ssm = _make_ssm(seed=5)
    rng = np.random.default_rng(5)
    impulse = np.zeros(ssm.d_input)
    impulse[0] = 1.0
    ssm.step(impulse)
    h_after = np.linalg.norm(ssm._h[0])
    s_after = np.linalg.norm(ssm._s[0])
    zeros = np.zeros(ssm.d_input)
    for _ in range(80):
        ssm.step(zeros)
    h_ratio = np.linalg.norm(ssm._h[0]) / (h_after + 1e-12)
    s_ratio = np.linalg.norm(ssm._s[0]) / (s_after + 1e-12)
    assert s_ratio > h_ratio


def test_reset_clears_state(ssm: CellAISSM) -> None:
    fresh = _make_ssm(seed=ssm.seed)
    e = np.random.default_rng(6).standard_normal(ssm.d_input)
    ssm.step(e)
    ssm.reset()
    out_reset = ssm.step(e)
    out_fresh = fresh.step(e)
    np.testing.assert_allclose(out_reset, out_fresh, rtol=1e-10, atol=1e-10)


def test_state_serialisation(ssm: CellAISSM) -> None:
    rng = np.random.default_rng(7)
    for _ in range(5):
        ssm.step(rng.standard_normal(ssm.d_input))
    state = ssm.get_state()
    e = rng.standard_normal(ssm.d_input)
    out_before = ssm.step(e.copy())
    ssm.set_state(state)
    out_after = ssm.step(e.copy())
    np.testing.assert_allclose(out_before, out_after, rtol=1e-10, atol=1e-10)


def test_n_layers_stacking() -> None:
    s1 = _make_ssm(n_layers=1)
    s2 = _make_ssm(n_layers=2)
    assert s1.context_dim == 2 * s1.d_state * 1
    assert s2.context_dim == 2 * s2.d_state * 2


def test_multi_layer_depth() -> None:
    seq = _structured_sequence(80, d_input=16, seed=11)
    mse_1 = _next_step_mse(_make_ssm(n_layers=1, seed=10), seq)
    mse_2 = _next_step_mse(_make_ssm(n_layers=2, seed=10), seq)
    assert mse_2 < mse_1 * 0.99 or mse_2 < mse_1 - 1e-6


def test_no_exploding_gradients() -> None:
    ssm = _make_ssm(seed=8)
    rng = np.random.default_rng(8)
    max_val = 0.0
    for _ in range(10_000):
        ctx = ssm.step(rng.standard_normal(ssm.d_input))
        max_val = max(max_val, float(np.max(np.abs(ctx))))
    assert max_val < 1e4


def test_dtype_stability() -> None:
    ssm = _make_ssm(seed=9)
    rng = np.random.default_rng(9)
    for _ in range(100_000):
        ctx = ssm.step(rng.standard_normal(ssm.d_input).astype(np.float32))
        assert np.all(np.isfinite(ctx))
    assert not np.any(np.isnan(ssm._h[0]))


def test_no_torch_dependency() -> None:
    assert_no_torch_on_import("from cypha_lm.temporal.cellai_ssm import CellAISSM")
