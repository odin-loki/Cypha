#!/usr/bin/env python3
"""Emit ``parity_fixtures/merge_from/`` for ``merge_from_parity``."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_OUT = _ROOT / "parity_fixtures" / "merge_from"


def main() -> None:
    from Cypha import CyphaDIF, VectorEncoder, cypha_save_binary

    d_in = 4
    field_dim = 24
    rng = np.random.default_rng(9901)

    def make_clf(seed: int) -> CyphaDIF:
        c = CyphaDIF(
            VectorEncoder(d_in),
            field_dim=field_dim,
            rng=np.random.default_rng(seed),
            replay_ratio=0.0,
            enc_lr=0.0,
        )
        return c

    self_clf = make_clf(11)
    other_clf = make_clf(22)

    xs = rng.standard_normal((6, d_in))
    for i, x in enumerate(xs[:3]):
        self_clf.train_step(x, str(i % 2))
    for i, x in enumerate(xs[3:]):
        other_clf.train_step(x, str(i % 2))
    other_clf.train_step(xs[0], "extra_only")

    _OUT.mkdir(parents=True, exist_ok=True)
    cypha_save_binary(self_clf.save_state(), str(_OUT / "self_before.cypha"))
    cypha_save_binary(other_clf.save_state(), str(_OUT / "other.cypha"))

    weight_self = 0.6
    weight_other = 0.4
    new_labels = self_clf.merge_from(other_clf, weight_self=weight_self, weight_other=weight_other)
    cypha_save_binary(self_clf.save_state(), str(_OUT / "self_after.cypha"))

    sidecar = {
        "field_dim": field_dim,
        "weight_self": weight_self,
        "weight_other": weight_other,
        "expected_new_labels": new_labels,
        "f_field_self": self_clf.memory.world.F_field.astype(np.float64).tolist(),
        "f_field_other": other_clf.memory.world.F_field.astype(np.float64).tolist(),
    }
    (_OUT / "sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}/")


if __name__ == "__main__":
    main()
