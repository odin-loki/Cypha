"""Load, merge, and apply algorithm-level variant knobs from bench profiles."""

from __future__ import annotations

import copy
from typing import Any

from cypha_bench.config.load_profile import uses_regimes

DEFAULT_ALGORITHM_VARIANTS: dict[str, int | float | bool] = {
    "cold_start_steps": 20,
    "min_experts_floor": 4,
    "online_passes_extra": 0,
    "temperature_scale": 1.0,
    "mdl_lambda_scale": 1.0,
    "replay_ratio_scale": 1.0,
    "target_lr_scale": 1.0,
    "deliberation_lo": 0.45,
    "deliberation_hi": 0.55,
    "reg_hash_routing": False,
}

VARIANT_KEYS = tuple(DEFAULT_ALGORITHM_VARIANTS.keys())
_ALGO_KEYS = frozenset(VARIANT_KEYS)


def default_algorithm_variants() -> dict[str, Any]:
    return dict(DEFAULT_ALGORITHM_VARIANTS)


def load_algorithm_variants(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge ``algorithm_variants`` from *profile* with :data:`DEFAULT_ALGORITHM_VARIANTS`."""
    merged = default_algorithm_variants()
    if not profile:
        return merged
    raw = profile.get("algorithm_variants")
    if isinstance(raw, dict):
        for key in _ALGO_KEYS:
            if key in raw:
                merged[key] = raw[key]
    return merged


def merge_algorithm_variants(profile: dict[str, Any]) -> dict[str, Any]:
    """Return merged algorithm variants for *profile* (defaults + overrides)."""
    return load_algorithm_variants(profile)


def _scale_block(block: dict[str, Any], variants: dict[str, Any]) -> None:
    temp_scale = float(variants["temperature_scale"])
    mdl_scale = float(variants["mdl_lambda_scale"])
    replay_scale = float(variants["replay_ratio_scale"])
    target_scale = float(variants["target_lr_scale"])
    extra_passes = int(variants["online_passes_extra"])

    if "temperature" in block:
        block["temperature"] = float(block["temperature"]) * temp_scale
    if "mdl_lambda" in block:
        block["mdl_lambda"] = float(block["mdl_lambda"]) * mdl_scale
    if "target_lr" in block:
        block["target_lr"] = float(block["target_lr"]) * target_scale
    if "replay_ratio" in block:
        block["replay_ratio"] = min(0.95, float(block["replay_ratio"]) * replay_scale)
    if "n_epochs" in block:
        block["n_epochs"] = max(1, int(block["n_epochs"]) + extra_passes)


def apply_algorithm_variants(
    profile: dict[str, Any],
    variants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deep-copy *profile*, store variants, and apply scale/extra-pass adjustments."""
    out = copy.deepcopy(profile)
    v = load_algorithm_variants(out if variants is None else {**out, "algorithm_variants": variants})
    out["algorithm_variants"] = v

    if uses_regimes(out):
        regimes = out.setdefault("regimes", {})
        for name in ("tabular", "vision", "regression"):
            block = regimes.get(name)
            if isinstance(block, dict):
                _scale_block(block, v)
        arch = out.setdefault("architecture", {})
        if "replay_ratio" in arch:
            arch["replay_ratio"] = min(
                0.95,
                float(arch["replay_ratio"]) * float(v["replay_ratio_scale"]),
            )
        return out

    for key in ("classification_cyphadif", "regression_difregressor"):
        block = out.get(key)
        if isinstance(block, dict):
            _scale_block(block, v)
    arch = out.setdefault("architecture", {})
    if "replay_ratio" in arch:
        arch["replay_ratio"] = min(
            0.95,
            float(arch["replay_ratio"]) * float(v["replay_ratio_scale"]),
        )
    return out
