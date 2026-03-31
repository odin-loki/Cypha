#!/usr/bin/env python3
"""
Emit ``parity_fixtures/two_stage_pipeline/sidecar.json`` for
``regression_two_stage_pipeline_parity`` (native LLR + RFF stage-2 + ridge-style combine).

Requires: NumPy, and ``Cypha`` on ``PYTHONPATH`` (repo root).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary  # noqa: E402

_FIX = _ROOT / "parity_fixtures"
_OUT_DIR = _FIX / "two_stage_pipeline"
_OUT = _OUT_DIR / "sidecar.json"


def main() -> None:
    manifest = json.loads((_FIX / "manifest.json").read_text(encoding="utf-8"))
    m = manifest["model"]
    state = cypha_load_binary(str(_FIX / "reference.cypha"))
    enc = VectorEncoder(int(m["input_dim"]))
    model = CyphaDIF(
        encoder=enc,
        field_dim=int(m["field_dim"]),
        rng=np.random.default_rng(0),
    )
    model.load_state(state)

    x = np.load(_FIX / "expected.npz")["x_input"][0].astype(np.float64)
    d_in = int(x.shape[0])
    llr_mat, _ = model.score_matrix(model.batch_encode(x.reshape(1, -1)))
    llr = np.asarray(llr_mat[0], dtype=np.float64)
    K = int(llr.shape[0])

    rng = np.random.default_rng(99)
    w1 = rng.standard_normal(K + d_in)
    b1 = 0.07
    D2 = 14
    W2 = rng.standard_normal((D2, d_in))
    b2_rff = rng.uniform(0.0, 2.0 * math.pi, size=D2)
    w2 = rng.standard_normal(D2)
    b2 = -0.03
    y_mean, y_std = 0.25, 1.8
    scale = math.sqrt(2.0 / D2)
    phi2 = scale * np.cos(x @ W2.T + b2_rff)
    y_s1 = float(np.dot(np.concatenate([llr, x]), w1) + b1)
    y_s2 = float(np.dot(phi2, w2) + b2)
    y_exp = (y_s1 + y_s2) * y_std + y_mean

    doc = {
        "fixture_schema": 1,
        "d_in": d_in,
        "D2": D2,
        "K": K,
        "x": x.tolist(),
        "enc2_W": W2.astype(np.float64).ravel(order="C").tolist(),
        "enc2_b": b2_rff.astype(np.float64).tolist(),
        "w1": w1.astype(np.float64).tolist(),
        "b1": float(b1),
        "w2": w2.astype(np.float64).tolist(),
        "b2": float(b2),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "expected_y": float(y_exp),
        "expected_llr": llr.tolist(),
    }
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}")


if __name__ == "__main__":
    main()
