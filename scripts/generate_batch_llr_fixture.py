#!/usr/bin/env python3
"""
Emit ``parity_fixtures/batch_llr/sidecar.json`` for ``batch_llr_parity``.

Copies ``x_input`` and ``llr`` from committed ``parity_fixtures/expected.npz`` (same source as
``native_parity.bin``). Re-run after regenerating main parity fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_FIX = _ROOT / "parity_fixtures"
_OUT = _FIX / "batch_llr" / "sidecar.json"


def main() -> None:
    npz_path = _FIX / "expected.npz"
    if not npz_path.is_file():
        raise SystemExit(f"missing {npz_path} — run scripts/generate_parity_fixtures.py first")
    z = np.load(npz_path)
    x = np.asarray(z["x_input"], dtype=np.float64)
    llr = np.asarray(z["llr"], dtype=np.float64)
    n, d_in = x.shape
    K = int(llr.shape[1])
    doc = {
        "fixture_schema": 1,
        "source": "parity_fixtures/expected.npz (x_input, llr)",
        "n": int(n),
        "d_in": int(d_in),
        "K": K,
        "x_rowmajor": x.ravel(order="C").tolist(),
        "expected_llr_rowmajor": llr.ravel(order="C").tolist(),
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}")


if __name__ == "__main__":
    main()
