"""Learnable (or fixed) view-slot embeddings for multi-view GRIA input."""

from __future__ import annotations

from typing import Any

import numpy as np

# Stable slot ids for schedule presets (CyphaLM Upgrade V2).
CANONICAL_VIEW_SLOTS: dict[str, int] = {
    "forward": 0,
    "block_shuffle": 1,
    "rotated": 2,
    "backward": 3,
}


class ViewEmbedding:
    """Per-view vector concatenated to GRIA input after field/ngram projection."""

    def __init__(
        self,
        n_slots: int,
        d_view: int,
        *,
        seed: int = 42,
        learnable: bool = True,
        xp: Any = None,
    ) -> None:
        self.n_slots = int(n_slots)
        self.d_view = int(d_view)
        self.learnable = bool(learnable)
        self._xp = xp if xp is not None else np
        xp_mod = self._xp
        rng = np.random.default_rng(seed)
        self.table = xp_mod.asarray(
            rng.standard_normal((self.n_slots, self.d_view)) * 0.02
        )

    def slot_for_view(self, view_name: str) -> int:
        if view_name in CANONICAL_VIEW_SLOTS:
            return int(CANONICAL_VIEW_SLOTS[view_name]) % self.n_slots
        return hash(view_name) & 0x7FFFFFFF % self.n_slots

    def forward(self, slot: int) -> Any:
        xp = self._xp
        idx = int(slot) % self.n_slots
        return xp.asarray(self.table[idx], dtype=xp.float64).ravel()

    def update(self, slot: int, grad_view: Any, lr: float) -> None:
        if not self.learnable or lr <= 0:
            return
        xp = self._xp
        idx = int(slot) % self.n_slots
        g = xp.asarray(grad_view, dtype=xp.float64).ravel()
        if g.size != self.d_view:
            g = xp.resize(g, self.d_view)
        self.table[idx] -= float(lr) * g

    def get_state(self) -> dict[str, Any]:
        from cypha_lm.array_backend import asnumpy

        return {
            "n_slots": self.n_slots,
            "d_view": self.d_view,
            "learnable": self.learnable,
            "table": asnumpy(self.table).tolist(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        xp = self._xp
        table = np.asarray(state["table"], dtype=np.float64)
        if table.shape != (self.n_slots, self.d_view):
            raise ValueError(
                f"view table shape {table.shape} != ({self.n_slots}, {self.d_view})"
            )
        self.table = xp.asarray(table, dtype=xp.float64)
        self.learnable = bool(state.get("learnable", self.learnable))
