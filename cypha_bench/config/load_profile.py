"""Load tuned CyphaDIF / DIFRegressor profiles for everyday use."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

BENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parent

DEFAULT_PROFILE_PATH = BENCH_ROOT / "config" / "everyday_profile.json"
FALLBACK_PROFILE_PATH = REPO_ROOT / "config" / "profiled_medium.json"

VALID_REGIMES = frozenset({"tabular", "vision", "regression"})
VISION_INPUT_DIM_THRESHOLD = 256
_ARCH_KEYS = ("replay_ratio", "ood_sigma")


def _profile_path_from_env() -> Path | None:
    raw = os.environ.get("CYPHA_BENCH_PROFILE_PATH") or os.environ.get("CYPHA_BENCH_PROFILE_JSON")
    if not raw:
        return None
    return Path(raw)


def load_profile(path: Path | None = None) -> dict[str, Any]:
    env_path = _profile_path_from_env()
    candidates = [path, env_path, DEFAULT_PROFILE_PATH, FALLBACK_PROFILE_PATH]
    for p in candidates:
        if p is None or not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            profile = json.load(f)
        from cypha_bench.config.algorithm_variants import apply_algorithm_variants

        return apply_algorithm_variants(profile)
    raise FileNotFoundError("No Cypha profile JSON found")


def uses_regimes(profile: dict[str, Any]) -> bool:
    regimes = profile.get("regimes")
    return isinstance(regimes, dict) and bool(regimes)


def select_classification_regime(input_dim: int) -> str:
    """Pick tabular vs vision regime from feature dimensionality."""
    return "vision" if int(input_dim) >= VISION_INPUT_DIM_THRESHOLD else "tabular"


def _resolve_regime(profile: dict[str, Any], regime: str) -> str:
    if regime in VALID_REGIMES:
        return regime
    default = str(profile.get("default_regime", "tabular"))
    return default if default in VALID_REGIMES else "tabular"


def regime_params(profile: dict[str, Any], regime: str) -> dict[str, Any]:
    """Return hyperparameters for a regime (tabular / vision / regression)."""
    regime = _resolve_regime(profile, regime)
    if uses_regimes(profile):
        regimes = profile["regimes"]
        return dict(regimes.get(regime, regimes.get(profile.get("default_regime", "tabular"), {})))
    if regime == "regression":
        return dict(profile.get("regression_difregressor", {}))
    return dict(profile.get("classification_cyphadif", {}))


def architecture_params(profile: dict[str, Any], regime: str | None = None) -> dict[str, Any]:
    """Shared architecture keys, with optional per-regime overrides."""
    arch = dict(profile.get("architecture", {}))
    if regime is not None:
        overrides = regime_params(profile, regime)
        for key in _ARCH_KEYS:
            if key in overrides:
                arch[key] = overrides[key]
    return arch


def classification_params(
    profile: dict[str, Any] | None = None,
    *,
    regime: str | None = None,
) -> dict[str, Any]:
    profile = profile or load_profile()
    if not uses_regimes(profile):
        return dict(profile.get("classification_cyphadif", {}))
    if regime is None:
        regime = str(profile.get("default_regime", "tabular"))
    if regime not in ("tabular", "vision"):
        regime = "tabular"
    return regime_params(profile, regime)


def regression_params(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = profile or load_profile()
    return regime_params(profile, "regression")


def cyphalm_params(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = profile or load_profile()
    return dict(profile.get("cyphalm", {}))
