"""Shared fixtures and helpers for CyphaLM tests."""

from __future__ import annotations

import subprocess
import sys

import pytest

from cypha_lm.config import CyphaLMConfig

TEST_CONFIG = CyphaLMConfig(
    vocab_size=64,
    d_embed=64,
    field_dim=32,
    d_state=16,
    ssm_layers=1,
    max_experts=32,
    seed=42,
    device="cpu",
)


def assert_no_torch_on_import(module_import_stmt: str) -> None:
    """Import *module_import_stmt* in a clean subprocess; torch must stay unloaded."""
    code = (
        f"{module_import_stmt}; "
        "import sys; "
        "assert 'torch' not in sys.modules, 'torch was imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.fixture
def test_config() -> CyphaLMConfig:
    return TEST_CONFIG
