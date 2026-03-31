#!/usr/bin/env python3
"""Write f_field.json (2D list) for cypha_rest from a loaded CyphaDIF + rng(0) load path."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary


def main() -> None:
    cypha = _ROOT / "parity_fixtures" / "reference.cypha"
    out = _ROOT / "parity_fixtures" / "f_field.json"
    state = cypha_load_binary(str(cypha))
    clf = CyphaDIF(VectorEncoder(8), field_dim=24, rng=np.random.default_rng(0))
    clf.load_state(state)
    F = clf.memory.world.F_field.astype(float).tolist()
    out.write_text(json.dumps(F), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
