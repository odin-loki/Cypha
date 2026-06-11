"""
``generation_parity`` vs ``parity_fixtures/generation/sidecar.json``.

Covers all native generation methods:
  generate_class_gaussian (no-rejection + rejection)
  generate_conditioned
  generate_langevin
  generate_boundary
  generate_ood
  generate_mdl_ball
  generate_ancestral
  predict_next_probs
  rollout (autoregressive label/latent sequence)
  generate_from_observation
  generate_retrieval_augmented (retrieve_from_x + centroid RAG)

All cases use pre-drawn random variates stored in the sidecar so
correctness is independent of C++ vs Python RNG implementation.

CTest: ``native_generation``.
Override: ``CYPHA_GENERATION_PARITY_BIN``.

Trigger: re-run ``scripts/generate_generation_fixture.py`` whenever
generation math in Cypha.py or native/src/generation.cpp changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable  # noqa: E402

_SIDE = _ROOT / "parity_fixtures" / "generation" / "sidecar.json"

_EXPECTED_CASES = {
    "generate_gaussian_no_rejection",
    "generate_gaussian_rejection",
    "generate_conditioned",
    "generate_langevin",
    "generate_boundary",
    "generate_ood",
    "generate_mdl_ball",
    "generate_ancestral",
    "predict_next",
    "rollout",
    "generate_from_observation",
    "generate_retrieval_augmented",
}


def test_generation_fixture_exists():
    """Fixture must exist before binary tests run (CI gate)."""
    assert _SIDE.is_file(), (
        f"Missing {_SIDE} — run scripts/generate_generation_fixture.py"
    )


def test_generation_fixture_cases():
    """Fixture must contain all expected case keys."""
    if not _SIDE.is_file():
        pytest.skip("fixture missing")
    data = json.loads(_SIDE.read_text(encoding="utf-8"))
    missing = _EXPECTED_CASES - set(data.get("cases", {}).keys())
    assert not missing, f"Fixture missing cases: {missing}"


def test_generation_native_parity():
    """All generation math cases must pass natively against the Python fixture."""
    if not _SIDE.is_file():
        pytest.skip("run scripts/generate_generation_fixture.py")
    r = run_native_executable(
        "generation_parity",
        [_SIDE],
        timeout=60,
        env_override="CYPHA_GENERATION_PARITY_BIN",
    )
    if r is None:
        pytest.skip("generation_parity binary not built")
    assert r.returncode == 0, (
        "generation_parity failed:\n" + r.stdout + r.stderr
    )
    assert "All generation parity checks PASSED." in r.stdout
