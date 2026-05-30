"""Feature flags for self-organising Cypha upgrades (all off by default)."""

from __future__ import annotations

import os

USE_GNG: bool = False
USE_SOM_ENCODER: bool = False
USE_GRIA_CONTROLLER: bool = False
USE_DISCRIM_FEEDBACK: bool = False
USE_DYNAMIC_TOPOLOGY: bool = False
USE_TEMPORAL_SOM: bool = False


def reset_flags() -> None:
    global USE_GNG, USE_SOM_ENCODER, USE_GRIA_CONTROLLER
    global USE_DISCRIM_FEEDBACK, USE_DYNAMIC_TOPOLOGY, USE_TEMPORAL_SOM
    USE_GNG = False
    USE_SOM_ENCODER = False
    USE_GRIA_CONTROLLER = False
    USE_DISCRIM_FEEDBACK = False
    USE_DYNAMIC_TOPOLOGY = False
    USE_TEMPORAL_SOM = False


def set_upgrade_flags(upgrade: str) -> None:
    """Enable flags for a single upgrade or cumulative 'all' (integration order)."""
    reset_flags()
    u = upgrade.upper()
    if u in ("NONE", ""):
        return
    order = ["U2", "U1", "U3", "U4", "U5", "U6"]
    if u == "ALL":
        enabled = order
    elif u in order:
        # Isolated benchmark: only the requested upgrade (integration uses "all").
        enabled = [u]
    else:
        raise ValueError(f"Unknown upgrade {upgrade!r}")
    g = globals()
    g["USE_SOM_ENCODER"] = "U2" in enabled
    g["USE_GNG"] = "U1" in enabled
    g["USE_GRIA_CONTROLLER"] = "U3" in enabled
    g["USE_DISCRIM_FEEDBACK"] = "U4" in enabled
    g["USE_DYNAMIC_TOPOLOGY"] = "U5" in enabled
    g["USE_TEMPORAL_SOM"] = "U6" in enabled


def apply_env_overrides() -> None:
    g = globals()
    for key, var in [
        ("USE_GNG", "CYPHA_SOM_USE_GNG"),
        ("USE_SOM_ENCODER", "CYPHA_SOM_USE_SOM_ENCODER"),
        ("USE_GRIA_CONTROLLER", "CYPHA_SOM_USE_GRIA_CONTROLLER"),
        ("USE_DISCRIM_FEEDBACK", "CYPHA_SOM_USE_DISCRIM_FEEDBACK"),
        ("USE_DYNAMIC_TOPOLOGY", "CYPHA_SOM_USE_DYNAMIC_TOPOLOGY"),
        ("USE_TEMPORAL_SOM", "CYPHA_SOM_USE_TEMPORAL_SOM"),
    ]:
        v = os.environ.get(var)
        if v is not None:
            g[key] = v.strip().lower() in ("1", "true", "yes", "on")


apply_env_overrides()
