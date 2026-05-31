"""Structure-preserving multi-view schedules for online Cypha training."""

from cypha_views.runner import iter_view_epochs
from cypha_views.schedule import (
    PRESET_SCHEDULES,
    BLOCK_SEGMENTED_VIEWS,
    VIEW_TRANSFORM_NAMES,
    make_view_spec,
    resolve_schedule,
)
from cypha_views.transforms import (
    apply_transform,
    block_shuffle,
    block_shuffle_blocks,
    blocks_forward,
    identity,
    reverse,
    rotate_start,
    split_blocks,
    split_blocks_by_delimiter,
)
from cypha_views.types import MemoryPolicy, ViewSchedule, ViewSpec

__all__ = [
    "MemoryPolicy",
    "ViewSpec",
    "ViewSchedule",
    "PRESET_SCHEDULES",
    "BLOCK_SEGMENTED_VIEWS",
    "VIEW_TRANSFORM_NAMES",
    "make_view_spec",
    "resolve_schedule",
    "identity",
    "reverse",
    "rotate_start",
    "split_blocks",
    "split_blocks_by_delimiter",
    "block_shuffle",
    "block_shuffle_blocks",
    "blocks_forward",
    "apply_transform",
    "iter_view_epochs",
]

__version__ = "0.1.0"
