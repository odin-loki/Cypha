"""Pure list[int] transforms for structure-preserving corpus reorderings."""

from __future__ import annotations

import random
from typing import Callable

TransformFn = Callable[..., list[int]]

_TRANSFORM_REGISTRY: dict[str, TransformFn] = {}


def _register(name: str, fn: TransformFn) -> TransformFn:
    _TRANSFORM_REGISTRY[name] = fn
    return fn


def identity(ids: list[int], **_kwargs) -> list[int]:
    """Return token ids unchanged."""
    return list(ids)


def reverse(ids: list[int], **_kwargs) -> list[int]:
    """Reverse token order."""
    return list(reversed(ids))


def rotate_start(ids: list[int], offset: int = 0, **_kwargs) -> list[int]:
    """Rotate sequence to begin at ``offset`` (wraps around)."""
    if not ids:
        return []
    k = offset % len(ids)
    if k == 0:
        return list(ids)
    return ids[k:] + ids[:k]


def split_blocks(
    ids: list[int],
    block_boundaries: list[int] | None = None,
    block_size: int = 512,
) -> tuple[list[list[int]], list[int]]:
    """Split *ids* into contiguous blocks.

    *block_boundaries* lists indices where a new block starts (0 is always implied).
    When *block_boundaries* is ``None``, fixed-size chunks of *block_size* are used.
    """
    if not ids:
        return [], [0]

    if block_boundaries is not None:
        starts = sorted({0, *block_boundaries})
    else:
        starts = list(range(0, len(ids), block_size))

    blocks: list[list[int]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(ids)
        if start < end:
            blocks.append(ids[start:end])

    return blocks, starts


def split_blocks_by_delimiter(
    ids: list[int],
    delimiter_id: int,
) -> tuple[list[list[int]], list[int]]:
    """Split *ids* at newline (or other delimiter) token boundaries."""
    if not ids:
        return [], [0]

    boundaries = [0]
    for i, token_id in enumerate(ids):
        if token_id == delimiter_id:
            nxt = i + 1
            if nxt < len(ids) and nxt not in boundaries:
                boundaries.append(nxt)

    blocks: list[list[int]] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(ids)
        if start < end:
            blocks.append(ids[start:end])

    return blocks, boundaries


def block_shuffle(
    ids: list[int],
    block_size: int = 512,
    seed: int = 0,
    block_boundaries: list[int] | None = None,
) -> list[int]:
    """Shuffle contiguous blocks; preserve within-block token order."""
    blocks, _ = split_blocks(ids, block_boundaries, block_size)
    rng = random.Random(seed)
    order = list(range(len(blocks)))
    rng.shuffle(order)
    return [token for idx in order for token in blocks[idx]]


def block_shuffle_blocks(
    ids: list[int],
    block_size: int = 512,
    seed: int = 0,
    block_boundaries: list[int] | None = None,
    delimiter_id: int | None = None,
) -> list[list[int]]:
    """Return shuffled blocks (one list per block) for segment-wise training."""
    if delimiter_id is not None:
        blocks, _ = split_blocks_by_delimiter(ids, delimiter_id)
    else:
        blocks, _ = split_blocks(ids, block_boundaries, block_size)
    rng = random.Random(seed)
    order = list(range(len(blocks)))
    rng.shuffle(order)
    return [blocks[idx] for idx in order]


def blocks_forward(
    ids: list[int],
    block_size: int = 512,
    block_boundaries: list[int] | None = None,
    delimiter_id: int | None = None,
) -> tuple[list[int], list[int]]:
    """Forward order with block start indices for context reset."""
    if delimiter_id is not None:
        _, boundaries = split_blocks_by_delimiter(ids, delimiter_id)
    else:
        _, boundaries = split_blocks(ids, block_boundaries, block_size)
    return list(ids), boundaries


def apply_transform(name: str, ids: list[int], **kwargs) -> list[int]:
    """Apply a named transform to token ids."""
    import inspect

    canonical = _ALIASES.get(name, name)
    if canonical == "rotate_start" and "offset" not in kwargs and ids:
        kwargs = {**kwargs, "offset": len(ids) // 4}

    try:
        fn = _TRANSFORM_REGISTRY[canonical]
    except KeyError as exc:
        raise ValueError(f"unknown transform: {name!r}") from exc

    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(ids, **filtered)


_ALIASES: dict[str, str] = {
    "forward": "identity",
    "backward": "reverse",
    "rotated": "rotate_start",
}

_register("identity", identity)
_register("reverse", reverse)
_register("rotate_start", rotate_start)
_register("block_shuffle", block_shuffle)
