"""Self-organising upgrades for Cypha (GNG, SOM, GRIA, feedback, topology)."""

from cypha_som.config import (
    USE_DISCRIM_FEEDBACK,
    USE_DYNAMIC_TOPOLOGY,
    USE_GNG,
    USE_GRIA_CONTROLLER,
    USE_SOM_ENCODER,
    USE_TEMPORAL_SOM,
    apply_env_overrides,
    reset_flags,
    set_upgrade_flags,
)
from cypha_som.hooks import CyphaSOMHooks, wire_cellai

__all__ = [
    "USE_GNG",
    "USE_SOM_ENCODER",
    "USE_GRIA_CONTROLLER",
    "USE_DISCRIM_FEEDBACK",
    "USE_DYNAMIC_TOPOLOGY",
    "USE_TEMPORAL_SOM",
    "apply_env_overrides",
    "reset_flags",
    "set_upgrade_flags",
    "CyphaSOMHooks",
    "wire_cellai",
]
