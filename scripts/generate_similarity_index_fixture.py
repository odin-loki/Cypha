#!/usr/bin/env python3
"""Emit ``parity_fixtures/similarity_index/`` for ``similarity_index_parity``."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_OUT = _ROOT / "parity_fixtures" / "similarity_index"
_REF = _ROOT / "parity_fixtures" / "reference.cypha"
_FF = _ROOT / "parity_fixtures" / "f_field.json"


def main() -> None:
    from Cypha import CyphaDIF, SimilarityIndex, VectorEncoder, cypha_load_binary

    if not _REF.is_file():
        raise SystemExit("Run scripts/generate_parity_fixtures.py first")

    state = cypha_load_binary(str(_REF))
    exp = np.load(_ROOT / "parity_fixtures" / "expected.npz")
    x_pool = exp["x_input"][:6].astype(np.float64)

    clf = CyphaDIF(VectorEncoder(8), field_dim=24, rng=np.random.default_rng(0))
    clf.load_state(state)

    idx = SimilarityIndex(clf)
    metas = [{"label": f"c{i}", "id": int(i)} for i in range(len(x_pool))]
    for x, md in zip(x_pool, metas):
        idx.add(x, metadata=md)

    sim_x1 = x_pool[0]
    sim_x2 = x_pool[1]
    expected_sim = float(idx.similarity(sim_x1, sim_x2))

    query_x = exp["x_input"][6]
    k = 3
    query_hits = idx.query(query_x, k=k)
    expected_query = [
        {"index": int(h["index"]), "similarity": float(h["similarity"])} for h in query_hits
    ]

    batch_x = exp["x_input"][7:10]
    batch_hits = idx.query_batch([batch_x[i] for i in range(len(batch_x))], k=2)
    expected_batch = [
        [{"index": int(h["index"]), "similarity": float(h["similarity"])} for h in row]
        for row in batch_hits
    ]

    _OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_REF, _OUT / "reference.cypha")
    shutil.copyfile(_FF, _OUT / "f_field.json")

    sidecar = {
        "add_examples": [{"x": x_pool[i].tolist(), "metadata": metas[i]} for i in range(len(x_pool))],
        "sim_x1": sim_x1.tolist(),
        "sim_x2": sim_x2.tolist(),
        "expected_similarity": expected_sim,
        "query_x": query_x.tolist(),
        "query_k": k,
        "expected_query": expected_query,
        "batch_query": {
            "n": len(batch_x),
            "k": 2,
            "x_rowmajor": batch_x.ravel(order="C").tolist(),
            "expected": expected_batch,
        },
    }
    (_OUT / "sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}/")


if __name__ == "__main__":
    main()
