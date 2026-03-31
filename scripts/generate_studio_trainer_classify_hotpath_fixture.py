#!/usr/bin/env python3
"""
Emit ``parity_fixtures/studio_trainer_classify_hotpath/`` for native ``quantile_dif_train_parity``.

Mirrors ``cypha_studio.core.trainer.Trainer.fit`` online loop for ``CyphaDIF`` (classification):
preprocessed rows, ``np.random.default_rng(seed + epoch).permutation(n)`` visit order,
``train_step`` with ``enc_lr>0`` and ``replay_ratio>0``. Records ``replay_u01`` so native
does not depend on NumPy vs ``std::mt19937`` for replay draws.

Golden: per-step losses + final ``batch_llr_from_x`` over rows in original index order.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary, cypha_save_binary

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "parity_fixtures" / "studio_trainer_classify_hotpath"


class RecordingReplayRng:
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
    n_samples = 6
    n_epochs = 2
    trainer_seed = 42
    rng_data_seed = 9001
    clf_rng_seed = 8844
    replay_inner_seed = 5150
    replay_ratio = 0.25
    enc_lr = 0.002

    rng_data = np.random.default_rng(rng_data_seed)
    rng_clf = np.random.default_rng(clf_rng_seed)
    rec = RecordingReplayRng(np.random.default_rng(replay_inner_seed))

    clf = CyphaDIF(
        VectorEncoder(d_in),
        field_dim=field_dim,
        rng=rng_clf,
        replay_rng=rec,
        enc_lr=enc_lr,
        replay_ratio=replay_ratio,
    )
    _OUT.mkdir(parents=True, exist_ok=True)
    cypha_save_binary(clf.save_state(), str(_OUT / "before.cypha"))
    f_field = np.ascontiguousarray(clf.memory.world.F_field, dtype=np.float64)
    (_OUT / "f_field.json").write_text(json.dumps(f_field.tolist(), indent=2), encoding="utf-8")

    x_mat = rng_data.standard_normal((n_samples, d_in)).astype(np.float64)
    y_norm = x_mat[:, 0] * 0.5 + x_mat[:, 1] * 0.3 + rng_data.standard_normal(n_samples) * 0.11
    quantiles = np.quantile(y_norm, np.linspace(0.0, 1.0, k_bins + 1))

    steps: list[dict] = []
    expected_step_losses: list[float] = []
    for epoch in range(n_epochs):
        perm = np.random.default_rng(trainer_seed + epoch).permutation(n_samples)
        for idx in perm:
            ii = int(idx)
            k = int(np.searchsorted(quantiles[1:-1], float(y_norm[ii])))
            lab = f"_sthp_{k}"
            x = x_mat[ii]
            steps.append({"x": x.tolist(), "label": lab})
            expected_step_losses.append(float(clf.train_step(x, lab)))

    replay_u01 = rec.values

    rows = [x_mat[j] for j in range(n_samples)]
    h = clf.batch_encode(rows)
    llr, labs = clf.score_matrix(h, use_field=True)

    st = cypha_load_binary(str(_OUT / "before.cypha"))
    clf2 = CyphaDIF(
        VectorEncoder(d_in),
        field_dim=field_dim,
        rng=rng_clf,
        replay_rng=ListReplayRng(list(replay_u01)),
        enc_lr=enc_lr,
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
        "description": "Studio Trainer.fit CyphaDIF loop: epoch permutations, enc_lr>0, replay_ratio>0, replay_u01",
        "d_in": d_in,
        "field_dim": field_dim,
        "n": n_samples,
        "n_steps": len(steps),
        "n_epochs": n_epochs,
        "trainer_seed": trainer_seed,
        "K": int(llr.shape[1]),
        "label_order": labs,
        "world_lr": float(clf.world_lr),
        "delta_lr": float(clf.delta_lr),
        "enc_lr": enc_lr,
        "ood_sigma": float(clf.ood_sigma),
        "temperature": float(clf.temperature),
        "replay_ratio": replay_ratio,
        "replay_cap": 10000,
        "align_every": 500,
        "temp_recalib_every": 0,
        "rng_seed": clf_rng_seed,
        "replay_u01": replay_u01,
        "expected_step_losses": expected_step_losses,
        "steps": steps,
        "x_rowmajor": x_mat.ravel(order="C").tolist(),
        "expected_llr_rowmajor": np.ascontiguousarray(llr, dtype=np.float64).ravel(order="C").tolist(),
    }
    (_OUT / "sidecar.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(
        f"Wrote {_OUT / 'sidecar.json'} (+ before.cypha, f_field.json), "
        f"n_steps={len(steps)} len(replay_u01)={len(replay_u01)}"
    )


if __name__ == "__main__":
    main()
