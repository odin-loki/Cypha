#!/usr/bin/env python3
"""Emit ``parity_fixtures/multilabel_dif/`` for ``multilabel_dif_parity``."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_OUT = _ROOT / "parity_fixtures" / "multilabel_dif"


def main() -> None:
    from Cypha import MultiLabelDIF, VectorEncoder

    d_in = 4
    field_dim = 24
    rng_data = np.random.default_rng(8801)

    mlf = MultiLabelDIF(
        encoder=VectorEncoder(d_in),
        field_dim=field_dim,
        rng=np.random.default_rng(42),
    )
    # Freeze encoder + skip replay for deterministic native parity.
    label_rng_seeds: dict[str, int] = {}
    initial_enc_w: dict[str, list] = {}
    initial_w_inject: dict[str, list] = {}
    initial_field_w_t: dict[str, list] = {}
    initial_field_sr_vec: dict[str, list] = {}

    train_steps_spec = [
        {"x": rng_data.standard_normal(d_in), "labels": {"dog": True, "big": False}},
        {"x": rng_data.standard_normal(d_in), "labels": {"dog": False, "black": True}},
        {"x": rng_data.standard_normal(d_in), "labels": {"big": True, "black": True}},
    ]

    train_steps_out: list[dict] = []
    for spec in train_steps_spec:
        x = np.asarray(spec["x"], dtype=np.float64)
        labels = spec["labels"]
        for lbl in labels:
            if lbl not in initial_enc_w:
                clf = mlf._get_or_create(lbl)
                label_rng_seeds[lbl] = int(hash(lbl) % (2**32))
                initial_enc_w[lbl] = clf.encoder.W.astype(np.float64).tolist()
                if clf._W_inject is not None:
                    initial_w_inject[lbl] = clf._W_inject.astype(np.float64).tolist()
                initial_field_w_t[lbl] = clf.field._W_T.astype(np.float64).tolist()
                initial_field_sr_vec[lbl] = clf.field._sr_vec.astype(np.float64).tolist()
                clf.enc_lr = 0.0
                clf._replay_ratio = 0.0
        losses = mlf.train_step(x, labels)
        train_steps_out.append(
            {
                "x": x.tolist(),
                "labels": labels,
                "expected_losses": {k: float(v) for k, v in losses.items()},
            }
        )

    predict_x = rng_data.standard_normal(d_in)
    expected_predict = {k: float(v) for k, v in mlf.predict(predict_x).items()}

    batch_n = 3
    batch_x = rng_data.standard_normal((batch_n, d_in))
    batch_probs = mlf.predict_batch([batch_x[i] for i in range(batch_n)])
    expected_batch = {k: np.asarray(v, dtype=np.float64).tolist() for k, v in batch_probs.items()}

    ref_clf = mlf._get_or_create("dog")
    doc = {
        "fixture_schema": 1,
        "d_in": d_in,
        "field_dim": field_dim,
        "world_lr": float(ref_clf.world_lr),
        "delta_lr": float(ref_clf.delta_lr),
        "ood_sigma": float(ref_clf.ood_sigma),
        "enc_lr": 0.0,
        "replay_ratio": 0.0,
        "replay_cap": 10000,
        "align_every": 500,
        "label_rng_seeds": label_rng_seeds,
        "initial_enc_w": initial_enc_w,
        "initial_w_inject": initial_w_inject,
        "initial_field_w_t": initial_field_w_t,
        "initial_field_sr_vec": initial_field_sr_vec,
        "train_steps": train_steps_out,
        "predict_x": predict_x.tolist(),
        "expected_predict": expected_predict,
        "batch_n": batch_n,
        "batch_x_rowmajor": batch_x.ravel(order="C").tolist(),
        "expected_batch": expected_batch,
    }

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "sidecar.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT / 'sidecar.json'}")


if __name__ == "__main__":
    main()
