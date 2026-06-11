"""N-gram + field fusion for GRIA input (CyphaLM Upgrade V2 Track B)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _sigmoid(x: Any, xp: Any) -> Any:
    x = xp.asarray(x, dtype=xp.float64)
    return 1.0 / (1.0 + xp.exp(-x))


class NgramFusion:
    """
    Fuse SSM field projection with n-gram embed projection.

    Modes: ``sum`` (V1), ``gated`` (V2 B1).
    """

    def __init__(
        self,
        field_dim: int,
        field_in: int,
        embed_in: int,
        *,
        mode: str = "sum",
        n_positions: int = 0,
        position_weights: bool = False,
        seed: int = 42,
        xp: Any = None,
    ) -> None:
        self.field_dim = int(field_dim)
        self.field_in = int(field_in)
        self.embed_in = int(embed_in)
        self.mode = str(mode)
        self.n_positions = int(n_positions)
        self.position_weights = bool(position_weights)
        self._xp = xp if xp is not None else np
        xp_mod = self._xp
        rng = np.random.default_rng(seed)
        scale = 0.02
        self.W_field = xp_mod.asarray(rng.standard_normal((field_dim, field_in)) * scale)
        self.W_embed = xp_mod.asarray(rng.standard_normal((field_dim, embed_in)) * scale)
        gate_in = field_in + embed_in
        self.W_gate = xp_mod.asarray(rng.standard_normal((field_dim, gate_in)) * scale)
        if position_weights and n_positions > 0:
            self.pos_weights = xp_mod.ones(n_positions, dtype=xp_mod.float64)
        else:
            self.pos_weights = None
        self._cache: dict[str, Any] = {}

    def _apply_position_weights(self, embeds: Any) -> Any:
        xp = self._xp
        if self.pos_weights is None or self.n_positions <= 0:
            return embeds
        embeds = xp.asarray(embeds, dtype=xp.float64).ravel()
        d = self.embed_in // self.n_positions if self.n_positions else self.embed_in
        if d <= 0 or embeds.size != self.embed_in:
            return embeds
        parts = []
        for i in range(self.n_positions):
            start = i * d
            parts.append(float(self.pos_weights[i]) * embeds[start : start + d])
        return xp.concatenate(parts)

    def forward(self, field_x: Any, embeds: Any) -> Any:
        xp = self._xp
        fx = xp.asarray(field_x, dtype=xp.float64).ravel()
        em = self._apply_position_weights(embeds)
        em = xp.asarray(em, dtype=xp.float64).ravel()
        field_part = self.W_field @ fx
        embed_part = self.W_embed @ em
        if self.mode == "gated":
            gate_in = xp.concatenate([fx, em])
            g = _sigmoid(self.W_gate @ gate_in, xp)
            v = g * field_part + (1.0 - g) * embed_part
            self._cache = {"g": g, "fx": fx, "em": em, "field_part": field_part, "embed_part": embed_part}
        else:
            v = field_part + embed_part
            self._cache = {"fx": fx, "em": em}
        return v

    def grad_field_x(self, grad_v: Any) -> Any:
        """Gradient w.r.t. field_x for truncated BPTT (gria_ngram path)."""
        xp = self._xp
        g = xp.asarray(grad_v, dtype=xp.float64).ravel()
        if self.mode == "gated" and self._cache:
            gate = self._cache["g"]
            grad_field_part = g * gate
            grad_fx = self.W_field.T @ grad_field_part
            xp.concatenate([self._cache["fx"], self._cache["em"]])
            grad_gate_in = self.W_gate.T @ (g * gate * (1.0 - gate) * (self._cache["field_part"] - self._cache["embed_part"]))
            grad_fx = grad_fx + grad_gate_in[: self.field_in]
            return grad_fx
        return self.W_field.T @ g
