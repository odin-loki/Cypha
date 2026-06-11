#!/usr/bin/env python3
"""Emit ``parity_fixtures/embed_table/sidecar.json`` for native ``embed_table_parity``."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from cypha_lm.embeddings.izaac_embed import IzaacEmbedding

_OUT = _ROOT / "parity_fixtures" / "embed_table"


def main() -> None:
    cases = [
        {"vocab_size": 64, "d_embed": 64, "seed": 42, "tokens": [0, 1, 5, 10, 63]},
        {"vocab_size": 17, "d_embed": 32, "seed": 7, "tokens": [0, 1, 8, 16]},
    ]
    side = {"fixture_schema": 1, "description": "Izaac GF(2^n) EmbedTable parity", "cases": []}

    for cfg in cases:
        emb = IzaacEmbedding(
            vocab_size=cfg["vocab_size"],
            d_embed=cfg["d_embed"],
            seed=cfg["seed"],
        )
        import numpy as np

        table = np.stack([emb.embed(t) for t in range(emb.vocab_size)], axis=0)
        side["cases"].append(
            {
                "vocab_size": cfg["vocab_size"],
                "d_embed": cfg["d_embed"],
                "seed": cfg["seed"],
                "n": emb.n,
                "k": emb.k,
                "a": int(emb._a),
                "b": int(emb._b),
                "table_sum": float(table.sum()),
                "tokens": {str(t): emb.embed(t).tolist() for t in cfg["tokens"]},
            }
        )

    # Primary case at top level for CTest (first config).
    primary = side["cases"][0]
    side.update({k: primary[k] for k in ("vocab_size", "d_embed", "seed", "table_sum", "tokens")})

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "sidecar.json").write_text(json.dumps(side, indent=2), encoding="utf-8")
    print(f"wrote {_OUT / 'sidecar.json'} ({len(side['cases'])} configs)")


if __name__ == "__main__":
    main()
