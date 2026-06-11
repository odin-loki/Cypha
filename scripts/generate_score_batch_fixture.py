#!/usr/bin/env python3
"""
Emit ``parity_fixtures/score_batch/sidecar.json`` for ``score_batch_parity``.

Goldens: ``cypha_accel.score_batch.project_features`` + ``fused_score_llr`` with
tensors from ``CyphaDIF._score_llr_tensors(use_field=True)`` — the same fused LLR
path ``CyphaDIF.score_matrix`` uses. Re-run after ``generate_parity_fixtures.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from Cypha import CyphaDIF, VectorEncoder, _BESSEL_TABLES_OK, cypha_load_binary
from cypha_accel.score_batch import fused_score_llr, project_features

_FIX = _ROOT / "parity_fixtures"
_OUT = _FIX / "score_batch" / "sidecar.json"


def main() -> None:
    if not _BESSEL_TABLES_OK:
        raise SystemExit(
            "Cypha Bessel lookup tables are unavailable (need scipy or bessel_ratios.npz next to Cypha.py)."
        )
    npz_path = _FIX / "expected.npz"
    cypha_path = _FIX / "reference.cypha"
    if not npz_path.is_file() or not cypha_path.is_file():
        raise SystemExit(f"missing {npz_path} or {cypha_path} — run scripts/generate_parity_fixtures.py first")

    z = np.load(npz_path)
    F = np.ascontiguousarray(z["x_input"], dtype=np.float64)
    H_ref = np.ascontiguousarray(z["H"], dtype=np.float64)
    llr_ref = np.ascontiguousarray(z["llr"], dtype=np.float64)
    n, d = F.shape
    K = int(llr_ref.shape[1])

    manifest_path = _FIX / "manifest.json"
    field_dim = 24
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        field_dim = int(manifest.get("model", {}).get("field_dim", field_dim))

    state = cypha_load_binary(str(cypha_path))
    clf = CyphaDIF(
        encoder=VectorEncoder(d),
        field_dim=field_dim,
        rng=np.random.default_rng(0),
    )
    clf.load_state(state)

    parts = clf._score_llr_tensors(use_field=True)
    if parts is None:
        raise SystemExit("reference model has no classes")
    _labels, D, mu0, inv_v, D_sq, u_k, ctx_arr = parts

    W = np.ascontiguousarray(clf.encoder.W, dtype=np.float64)
    H_proj = project_features(F, W)
    LLR = fused_score_llr(H_proj, mu0, inv_v, D, D_sq, u_k, ctx_arr)

    np.testing.assert_allclose(H_proj, H_ref, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(LLR, llr_ref, rtol=0.0, atol=1e-12)

    doc = {
        "fixture_schema": 1,
        "source": "cypha_accel.score_batch project_features + fused_score_llr (score_matrix tensors)",
        "n": int(n),
        "d": int(d),
        "K": K,
        "F_rowmajor": F.ravel(order="C").tolist(),
        "W_enc_rowmajor": W.ravel(order="C").tolist(),
        "mu0": mu0.ravel().tolist(),
        "inv_v": inv_v.ravel().tolist(),
        "D_rowmajor": D.ravel(order="C").tolist(),
        "D_sq": D_sq.ravel().tolist(),
        "u_k": u_k.ravel().tolist(),
        "ctx": ctx_arr.ravel().tolist(),
        "expected_H_rowmajor": H_proj.ravel(order="C").tolist(),
        "expected_LLR_rowmajor": LLR.ravel(order="C").tolist(),
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_OUT}")


if __name__ == "__main__":
    main()
