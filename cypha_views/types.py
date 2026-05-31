"""Core types for multi-view online training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryPolicy:
    """Per-view rules for fast/slow memory carry across segments and views."""

    reset_fast: bool = True
    carry_slow: bool = False
    carry_dif: bool = True
    carry_gria_bias: bool = True


@dataclass
class ViewSpec:
    """One presentation of the corpus: transform + metadata for the trainer."""

    name: str
    view_id: str
    transform_name: str
    memory_policy: MemoryPolicy = field(default_factory=MemoryPolicy)


@dataclass
class ViewSchedule:
    """Ordered views to apply across training epochs."""

    views: list[ViewSpec]
    seed: int = 42
