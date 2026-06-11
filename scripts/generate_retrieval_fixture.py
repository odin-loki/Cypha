#!/usr/bin/env python3
"""Emit ``parity_fixtures/retrieval/sidecar.json`` for native ``retrieval_parity``."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary, cypha_save_binary

_OUT = _ROOT / "parity_fixtures" / "retrieval"


def _retrieve_case(
    clf: CyphaDIF,
    query_x: np.ndarray,
    database_x: list[np.ndarray],
    top_k: int,
    label: str | None,
    name: str,
) -> dict:
    hits = clf.retrieve(query_x, database_x, top_k=top_k, label=label)
    return {
        "name": name,
        "query_x": np.asarray(query_x, dtype=np.float64).tolist(),
        "database_x": [np.asarray(x, dtype=np.float64).tolist() for x in database_x],
        "top_k": int(top_k),
        "label": label,
        "expected": [
            {
                "index": int(idx),
                "log_likelihood": float(ll),
                "predicted_label": str(lbl),
            }
            for idx, ll, lbl in hits
        ],
    }


def main() -> None:
    ref = _ROOT / "parity_fixtures" / "reference.cypha"
    if not ref.is_file():
        raise SystemExit(f"missing {ref} — run scripts/generate_parity_fixtures.py first")

    state = cypha_load_binary(str(ref))
    manifest = json.loads((_ROOT / "parity_fixtures" / "manifest.json").read_text(encoding="utf-8"))
    m = manifest["model"]
    enc = VectorEncoder(int(m["input_dim"]))
    clf = CyphaDIF(enc, field_dim=int(m["field_dim"]), rng=np.random.default_rng(0))
    clf.load_state(state)

    exp = np.load(_ROOT / "parity_fixtures" / "expected.npz")
    xs = exp["x_input"].astype(np.float64)
    db = [xs[i] for i in range(min(8, len(xs)))]

    cases = [
        _retrieve_case(clf, xs[0], db, top_k=5, label=None, name="retrieve_pred_label"),
        _retrieve_case(clf, xs[1], db, top_k=3, label=manifest["labels"][0], name="retrieve_fixed_label"),
        _retrieve_case(clf, xs[2], db[:3], top_k=2, label=None, name="retrieve_small_db"),
    ]

    _OUT.mkdir(parents=True, exist_ok=True)
    cypha_save_binary(clf.save_state(), str(_OUT / "reference.cypha"))
    f_field = np.ascontiguousarray(clf.memory.world.F_field, dtype=np.float64)
    (_OUT / "f_field.json").write_text(json.dumps(f_field.tolist()), encoding="utf-8")

    sidecar = {
        "fixture_schema": 1,
        "description": "CyphaDIF.retrieve parity vs native retrieve_from_x",
        "reference_cypha": "reference.cypha",
        "f_field_json": "f_field.json",
        "input_dim": int(m["input_dim"]),
        "field_dim": int(m["field_dim"]),
        "use_field": True,
        "cases": cases,
    }
    (_OUT / "sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {_OUT / 'sidecar.json'} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
