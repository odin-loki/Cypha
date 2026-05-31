"""CUDA backend tests for CyphaLM (skipped when GPU unavailable)."""

from __future__ import annotations

import numpy as np
import pytest

from cypha_lm.array_backend import cuda_available
from cypha_lm.config import CyphaLMConfig
from cypha_lm.model.cypha_lm import CyphaLM

pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA/CuPy not available")


@pytest.fixture
def gpu_config() -> CyphaLMConfig:
    return CyphaLMConfig(
        vocab_size=64,
        d_embed=64,
        field_dim=32,
        d_state=16,
        ssm_layers=1,
        max_experts=32,
        seed=42,
        device="cuda",
    )


def test_cuda_model_reports_device(gpu_config: CyphaLMConfig) -> None:
    model = CyphaLM(gpu_config)
    assert model.device == "cuda"


def test_cuda_matches_cpu_log_probs(gpu_config: CyphaLMConfig) -> None:
    cpu_cfg = CyphaLMConfig(**{**gpu_config.__dict__, "device": "cpu"})
    cpu = CyphaLM(cpu_cfg)
    gpu = CyphaLM(gpu_config)
    for tid in (0, 1, 2, 5):
        cpu.reset_context()
        gpu.reset_context()
        p_cpu = cpu.predict_next(tid)
        p_gpu = gpu.predict_next(tid)
        np.testing.assert_allclose(
            p_cpu["log_probs"],
            p_gpu["log_probs"],
            rtol=1e-5,
            atol=1e-5,
        )


def test_cuda_train_step_finite(gpu_config: CyphaLMConfig) -> None:
    model = CyphaLM(gpu_config)
    out = model.train_step(0, 1)
    assert np.isfinite(out["loss"])
