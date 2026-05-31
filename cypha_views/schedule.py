"""Preset view schedules and resolution helpers."""

from __future__ import annotations

from cypha_views.types import MemoryPolicy, ViewSchedule, ViewSpec

# LM view name → transform function name (see transforms.apply_transform aliases).
VIEW_TRANSFORM_NAMES: dict[str, str] = {
    "forward": "identity",
    "block_shuffle": "block_shuffle",
    "rotated": "rotate_start",
    "backward": "reverse",
}

BLOCK_SEGMENTED_VIEWS: frozenset[str] = frozenset({"forward", "block_shuffle"})

PRESET_SCHEDULES: dict[str, list[str]] = {
    "same_order": ["forward"],
    "schedule_a": ["forward", "block_shuffle"],
    "schedule_b": ["forward", "block_shuffle", "rotated"],
    "schedule_c": ["forward", "block_shuffle", "backward"],
}


def _memory_policy_for_view(name: str) -> MemoryPolicy:
    if name == "forward":
        return MemoryPolicy(
            reset_fast=False,
            carry_slow=True,
            carry_dif=True,
            carry_gria_bias=True,
        )
    if name == "block_shuffle":
        return MemoryPolicy(
            reset_fast=True,
            carry_slow=False,
            carry_dif=True,
            carry_gria_bias=True,
        )
    return MemoryPolicy(
        reset_fast=True,
        carry_slow=False,
        carry_dif=True,
        carry_gria_bias=True,
    )


def make_view_spec(name: str) -> ViewSpec:
    """Build a :class:`ViewSpec` from a canonical LM view name."""
    transform_name = VIEW_TRANSFORM_NAMES.get(name, name)
    return ViewSpec(
        name=name,
        view_id=name,
        transform_name=transform_name,
        memory_policy=_memory_policy_for_view(name),
    )


def resolve_schedule(
    name_or_list: str | list[str],
    seed: int = 42,
    train_epochs: int = 1,
) -> ViewSchedule:
    """Resolve a preset name or explicit view-name list into a :class:`ViewSchedule`."""
    if isinstance(name_or_list, str):
        key = name_or_list.strip()
        if key == "same_order":
            view_names = ["forward"] * max(1, int(train_epochs))
        else:
            view_names = PRESET_SCHEDULES.get(key, [key])
    else:
        view_names = list(name_or_list)

    views = [make_view_spec(name) for name in view_names]
    return ViewSchedule(views=views, seed=seed)
