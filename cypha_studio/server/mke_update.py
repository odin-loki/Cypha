"""
MKE scalar train step for FastAPI ``POST /update`` (parity with native ``mke_scalar_train_step``).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.replay_rng import ListReplayRng

_EPS = 1e-8


@dataclass
class MkeRestState:
    """Mutable MKE sidecar state loaded from ``regression_head.json`` ``mke`` block."""

    d_in: int
    d_rff: int
    W: np.ndarray
    b: np.ndarray
    w: Dict[str, np.ndarray] = field(default_factory=dict)
    P: Dict[str, np.ndarray] = field(default_factory=dict)
    temperature: float = 1.0
    forgetting_factor: float = 1.0
    pi_floor: float = 0.02
    gh_scales: Optional[np.ndarray] = None


def _rff_encode(x: np.ndarray, W: np.ndarray, b: np.ndarray, d_rff: int) -> np.ndarray:
    scale = math.sqrt(2.0 / float(d_rff))
    return (scale * np.cos(W @ x + b)).astype(np.float64)


def _mke_expert_rls_scalar_step(
    phi: np.ndarray,
    pi: float,
    gh_scale: float,
    err: float,
    forgetting_factor: float,
    w: np.ndarray,
    P: np.ndarray,
) -> None:
    d = phi.shape[0]
    if pi < 0.02:
        return
    if forgetting_factor > 0.0 and forgetting_factor < 1.0:
        P *= 1.0 / forgetting_factor
    Pphi = P @ phi
    denom = 1.0 + pi * float(phi @ Pphi)
    if denom <= 0.0:
        denom = 1e-30
    Kg = pi * Pphi / denom * gh_scale
    w += Kg * err
    P -= pi * np.outer(Kg, phi) @ P


def parse_mke_from_regression_file(path: str) -> Optional[MkeRestState]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    mk = raw.get("mke")
    if not isinstance(mk, dict):
        return None
    d_in = int(mk["d_in"])
    d_rff = int(mk["D_rff"])
    W_flat = np.asarray(mk["rff_W_rowmajor"], dtype=np.float64)
    W = W_flat.reshape(d_rff, d_in, order="C")
    b = np.asarray(mk["rff_b"], dtype=np.float64)
    wj = mk["w"]
    pj = mk["P"]
    if not isinstance(wj, dict) or not isinstance(pj, dict):
        raise ValueError("mke block requires object maps 'w' and 'P'")
    w = {str(k): np.asarray(v, dtype=np.float64) for k, v in wj.items()}
    p = {str(k): np.asarray(v, dtype=np.float64).reshape(d_rff, d_rff, order="C") for k, v in pj.items()}
    gh = mk.get("gh_scales")
    gh_arr = np.asarray(gh, dtype=np.float64) if gh is not None else None
    return MkeRestState(
        d_in=d_in,
        d_rff=d_rff,
        W=W,
        b=b,
        w=w,
        P=p,
        temperature=float(mk.get("temperature", 1.0)),
        forgetting_factor=float(mk.get("forgetting_factor", 1.0)),
        pi_floor=float(mk.get("pi_floor", 0.02)),
        gh_scales=gh_arr,
    )


def _apply_replay_u01(model: Any, replay_u01: Optional[List[float]]):
    if not replay_u01:
        return None
    old_rng = getattr(model, "_replay_rng", None)
    lr = ListReplayRng([float(v) for v in replay_u01])
    model._replay_rng = lr
    if hasattr(model, "replay") and hasattr(model.replay, "_rng"):
        model.replay._rng = lr
    return old_rng


def _restore_replay_rng(model: Any, old_rng: Any) -> None:
    if old_rng is None:
        return
    model._replay_rng = old_rng
    if hasattr(model, "replay") and hasattr(model.replay, "_rng"):
        model.replay._rng = old_rng


def _rff_encoder(model: Any):
    enc_fn = getattr(model, "encoder_fn", None)
    if enc_fn is not None and hasattr(enc_fn, "W") and hasattr(enc_fn, "b"):
        return enc_fn
    return None


def _sync_rff_encoder_from_mke(model: Any, state: MkeRestState):
    """Align embedded ``RFFEncoder`` weights with the sidecar (``load_state`` may not restore them)."""
    enc = _rff_encoder(model)
    if enc is None:
        return None
    old = (enc.W.copy(), enc.b.copy())
    enc.W = state.W.copy()
    enc.b = state.b.copy()
    if hasattr(enc, "_scale"):
        enc._scale = math.sqrt(2.0 / float(state.d_rff))
    return old


def _restore_rff_encoder(model: Any, old: Any) -> None:
    if old is None:
        return
    enc = _rff_encoder(model)
    if enc is None:
        return
    enc.W, enc.b = old


def mke_rest_update(
    model: Any,
    state: MkeRestState,
    x_pp: np.ndarray,
    y: float,
    *,
    router_train_label: Optional[str] = None,
    replay_u01: Optional[List[float]] = None,
    use_gh: bool = True,
) -> Tuple[float, float]:
    """
    One scalar MKE update. Returns ``(err_sq, router_loss)``.

    ``router_loss`` is returned as ``POST /update`` ``loss`` (native ``cypha_rest`` contract).
    """
    from Cypha import _softmax_batch

    x = np.asarray(x_pp, dtype=np.float64).ravel()
    if x.size != state.d_in:
        raise ValueError(f"input dim mismatch after preprocessor: got length {x.size}, expected {state.d_in}")

    phi = _rff_encode(x, state.W, state.b, state.d_rff)
    LLR, labs = model.score_matrix(phi.reshape(1, -1), use_field=True)
    labs_list = list(labs)
    p = _softmax_batch(LLR / (state.temperature + _EPS))[0]

    y_hat = 0.0
    for i, lbl in enumerate(labs_list):
        w_k = state.w.get(lbl)
        if w_k is None or w_k.size != state.d_rff:
            raise ValueError(f"mke missing w for label {lbl!r}")
        y_hat += float(p[i]) * float(w_k @ phi)
    err = float(y) - y_hat
    err_sq = err * err

    for i, lbl in enumerate(labs_list):
        pi = float(p[i])
        if pi < state.pi_floor:
            continue
        w_k = state.w.get(lbl)
        P_k = state.P.get(lbl)
        if w_k is None or P_k is None:
            raise ValueError(f"mke missing w/P for label {lbl!r}")
        gh = 1.0
        if use_gh and state.gh_scales is not None and i < state.gh_scales.size:
            gh = float(state.gh_scales[i])
        _mke_expert_rls_scalar_step(phi, pi, gh, err, state.forgetting_factor, w_k, P_k)

    if router_train_label:
        router_label = str(router_train_label)
    else:
        router_label = labs_list[int(np.argmax(p))]

    old_rng = _apply_replay_u01(model, replay_u01)
    old_enc = _sync_rff_encoder_from_mke(model, state)
    try:
        router_loss = float(model.train_step(x, router_label))
    finally:
        _restore_rff_encoder(model, old_enc)
        _restore_replay_rng(model, old_rng)

    return err_sq, router_loss
