"""Fixed U(0,1) replay stream for parity-style deterministic priority replay."""
from __future__ import annotations


class ListReplayRng:
    """Replays a fixed stream (must match recorded draw order)."""

    def __init__(self, xs: list[float]) -> None:
        self.xs = list(xs)
        self.i = 0

    def random(self, *args, **kwargs):  # noqa: ANN002
        if kwargs.get("out") is not None:
            out = kwargs["out"]
            n = int(args[0])
            for j in range(n):
                if self.i >= len(self.xs):
                    raise RuntimeError("ListReplayRng exhausted (batch)")
                out[j] = self.xs[self.i]
                self.i += 1
            return out
        if self.i >= len(self.xs):
            raise RuntimeError("ListReplayRng exhausted (scalar)")
        v = self.xs[self.i]
        self.i += 1
        return v
