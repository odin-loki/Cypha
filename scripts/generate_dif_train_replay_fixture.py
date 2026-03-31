#!/usr/bin/env python3
"""
Emit ``parity_fixtures/dif_train_replay/`` for ``quantile_dif_train_parity``.

Records a shared ``replay_u01`` stream (replay gate + per-slot sample uniforms) so Python and
native consume identical draws without relying on NumPy vs ``std::mt19937`` seed equivalence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary, cypha_save_binary

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "parity_fixtures" / "dif_train_replay"


class RecordingReplayRng:
    """Wraps a NumPy generator and records every scalar/batch ``random`` draw."""

    def __init__(self, inner: np.random.Generator) -> None:
        self._inner = inner
        self.values: list[float] = []

    def random(self, *args, **kwargs):  # noqa: ANN002
        if kwargs.get("out") is not None:
            out = kwargs["out"]
            n = int(args[0])
            for j in range(n):
                v = float(self._inner.random())
                self.values.append(v)
                out[j] = v
            return out
        v = float(self._inner.random())
        self.values.append(v)
        return v


class ListReplayRng:
    """Replays a fixed stream (must match recorded order)."""

    def __init__(self, xs: list[float]) -> None:
        self.xs = xs
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
        v = float(self.xs[self.i])
        self.i += 1
        return v


def main() -> None:
    d_in = 4
    field_dim = 24
    k_bins = 3
    n = 12
    seed_init = 7755
    seed_data = 7756
    replay_inner_seed = 424242
    replay_ratio = 0.3

    rng_clf = np.random.default_rng(seed_init)
    rng_data = np.random.default_rng(seed_data)
    rec = RecordingReplayRng(np.random.default_rng(replay_inner_seed))

    clf = CyphaDIF(
        VectorEncoder(d_in),
        field_dim=field_dim,
        rng=rng_clf,
        replay_rng=rec,
        enc_lr=0.0,
        replay_ratio=replay_ratio,
    )
    _OUT.mkdir(parents=True, exist_ok=True)
    cypha_save_binary(clf.save_state(), str(_OUT / "before.cypha"))
    f_field = np.ascontiguousarray(clf.memory.world.F_field, dtype=np.float64)
    (_OUT / "f_field.json").write_text(json.dumps(f_field.tolist(), indent=2), encoding="utf-8")

    x_mat = rng_data.standard_normal((n, d_in)).astype(np.float64)
    y_norm = x_mat[:, 0] * 0.5 + x_mat[:, 1] * 0.3 + rng_data.standard_normal(n) * 0.11
    quantiles = np.quantile(y_norm, np.linspace(0.0, 1.0, k_bins + 1))
    perm = rng_data.permutation(n)

    steps: list[dict] = []
    for i in perm:
        ii = int(i)
        k = int(np.searchsorted(quantiles[1:-1], float(y_norm[ii])))
        steps.append({"x": x_mat[ii].tolist(), "label": f"_ts_{k}"})

    expected_step_losses: list[float] = []
    for s in steps:
        expected_step_losses.append(
            float(clf.train_step(np.asarray(s["x"], dtype=np.float64), s["label"]))
        )

    replay_u01 = rec.values
    rows = [x_mat[j] for j in range(n)]
    h = clf.batch_encode(rows)
    llr, labs = clf.score_matrix(h, use_field=True)

    # Self-check: list replay reproduces the same LLR from a fresh load.
    st = cypha_load_binary(str(_OUT / "before.cypha"))
    clf2 = CyphaDIF(
        VectorEncoder(d_in),
        field_dim=field_dim,
        rng=rng_clf,
        replay_rng=ListReplayRng(list(replay_u01)),
        enc_lr=0.0,
        replay_ratio=replay_ratio,
    )
    clf2.load_state(st)
    for s in steps:
        clf2.train_step(np.asarray(s["x"], dtype=np.float64), s["label"])
    h2 = clf2.batch_encode(rows)
    llr2, _ = clf2.score_matrix(h2, use_field=True)
    max_d = float(np.max(np.abs(llr - llr2)))
    if max_d > 1e-12:
        raise SystemExit(f"replay_u01 self-check failed max_abs={max_d}")

    doc = {
        "fixture_schema": 1,
        "description": "Quantile labels + replay_ratio>0 with recorded replay_u01 stream; enc_lr=0",
        "d_in": d_in,
        "field_dim": field_dim,
        "n": n,
        "n_steps": len(steps),
        "K": int(llr.shape[1]),
        "label_order": labs,
        "world_lr": float(clf.world_lr),
        "delta_lr": float(clf.delta_lr),
        "enc_lr": 0.0,
        "ood_sigma": float(clf.ood_sigma),
        "temperature": float(clf.temperature),
        "replay_ratio": replay_ratio,
        "replay_cap": 10000,
        "align_every": 500,
        "temp_recalib_every": 0,
        "rng_seed": seed_init,
        "replay_u01": replay_u01,
        "expected_step_losses": expected_step_losses,
        "steps": steps,
        "x_rowmajor": x_mat.ravel(order="C").tolist(),
        "expected_llr_rowmajor": np.ascontiguousarray(llr, dtype=np.float64).ravel(order="C").tolist(),
    }
    (_OUT / "sidecar.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT / 'sidecar.json'} (+ before.cypha, f_field.json), len(replay_u01)={len(replay_u01)}")


if __name__ == "__main__":
    main()
