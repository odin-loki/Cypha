"""Iterator that streams corpus segments under a multi-view schedule."""

from __future__ import annotations

from collections.abc import Iterator

from cypha_views.schedule import BLOCK_SEGMENTED_VIEWS
from cypha_views.transforms import (
    apply_transform,
    block_shuffle_blocks,
    split_blocks,
    split_blocks_by_delimiter,
)
from cypha_views.types import ViewSchedule, ViewSpec

ViewEpochItem = tuple[ViewSpec, int, list[int], bool]


def _transform_kwargs(
    view_spec: ViewSpec,
    ids: list[int],
    schedule: ViewSchedule,
    epoch_idx: int,
) -> dict:
    kwargs: dict = {"seed": schedule.seed + epoch_idx}
    if view_spec.transform_name == "rotate_start":
        kwargs["offset"] = len(ids) // 4 if ids else 0
    return kwargs


def _block_segments(
    view_spec: ViewSpec,
    ids: list[int],
    schedule: ViewSchedule,
    epoch_idx: int,
    char_newline_id: int | None,
    block_size: int = 512,
) -> list[list[int]]:
    kwargs = _transform_kwargs(view_spec, ids, schedule, epoch_idx)

    if view_spec.name == "block_shuffle":
        return block_shuffle_blocks(
            ids,
            block_size=block_size,
            seed=kwargs["seed"],
            delimiter_id=char_newline_id,
        )

    transformed = apply_transform(view_spec.transform_name, ids, **kwargs)
    if char_newline_id is not None:
        blocks, _ = split_blocks_by_delimiter(transformed, char_newline_id)
    else:
        blocks, _ = split_blocks(transformed, None, block_size=block_size)
    return blocks


def iter_view_epochs(
    ids: list[int],
    schedule: ViewSchedule,
    char_newline_id: int | None = None,
    block_size: int = 512,
) -> Iterator[ViewEpochItem]:
    """Yield ``(view_spec, epoch_idx, segment_ids, reset_context_before_segment)``."""
    for epoch_idx, view_spec in enumerate(schedule.views):
        if view_spec.name in BLOCK_SEGMENTED_VIEWS:
            segments = _block_segments(
                view_spec,
                ids,
                schedule,
                epoch_idx,
                char_newline_id,
                block_size=block_size,
            )
            reset = view_spec.memory_policy.reset_fast
            for segment in segments:
                if segment:
                    yield view_spec, epoch_idx, segment, reset
            continue

        kwargs = _transform_kwargs(view_spec, ids, schedule, epoch_idx)
        segment = apply_transform(view_spec.transform_name, ids, **kwargs)
        if segment:
            yield view_spec, epoch_idx, segment, view_spec.memory_policy.reset_fast
