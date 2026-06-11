#!/usr/bin/env python3
"""
Emit ``parity_fixtures/gh_infer_deliberation/sidecar.json`` for native ``gh_infer_deliberation_parity``.

Covers:
- ``CyphaDIF.gh_infer`` (T_adj + classify, chi/psi threaded)
- ``CyphaDIF.infer`` with deliberation band enabled/disabled
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from Cypha import CyphaDIF, VectorEncoder, cypha_load_binary, cypha_save_binary

_OUT = _ROOT / "parity_fixtures" / "gh_infer_deliberation"
_EPS = 1e-8


def _gh_case(clf: CyphaDIF, x: np.ndarray, chi: float, psi: float) -> dict:
    pred, conf, r_eff, chi_new, psi_new = clf.gh_infer(x, chi, psi)
    return {
        "x": np.asarray(x, dtype=np.float64).tolist(),
        "use_gh": True,
        "chi": float(chi),
        "psi": float(psi),
        "expected_label": str(pred),
        "expected_confidence": float(conf),
        "expected_r_eff": float(r_eff),
        "expected_chi_new": float(chi_new),
        "expected_psi_new": float(psi_new),
    }


def _infer_case(clf: CyphaDIF, x: np.ndarray, lo: float, hi: float) -> dict:
    saved_lo, saved_hi = clf.deliberation_lo, clf.deliberation_hi
    clf.deliberation_lo = float(lo)
    clf.deliberation_hi = float(hi)
    pred, conf = clf.infer(x)
    clf.deliberation_lo, clf.deliberation_hi = saved_lo, saved_hi
    return {
        "x": np.asarray(x, dtype=np.float64).tolist(),
        "use_gh": False,
        "deliberation_lo": float(lo),
        "deliberation_hi": float(hi),
        "expected_label": str(pred),
        "expected_confidence": float(conf),
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

    cases: list[dict] = []
    chi, psi = 1.0, 1.0
    for i in range(min(4, len(xs))):
        row = _gh_case(clf, xs[i], chi, psi)
        row["name"] = f"gh_infer_row_{i}"
        cases.append(row)
        chi, psi = row["expected_chi_new"], row["expected_psi_new"]

    # Deliberation disabled (defaults) should match plain infer
    cases.append({**_infer_case(clf, xs[0], 1.0, 0.0), "name": "infer_deliberation_disabled"})

    # Mid-confidence abstention: pick x where raw conf is in band
    clf.deliberation_lo = 1.0
    clf.deliberation_hi = 0.0
    band_x = None
    band_conf = None
    for i in range(len(xs)):
        p, c = clf.infer(xs[i])
        if 0.35 <= c <= 0.65 and p != "__unknown__":
            band_x = xs[i]
            band_conf = float(c)
            break
    if band_x is None:
        band_x = xs[-1]
        _, band_conf = clf.infer(band_x)
    lo, hi = 0.35, 0.65
    if not (lo <= band_conf <= hi):
        lo = max(0.0, band_conf - 0.05)
        hi = min(1.0, band_conf + 0.05)
    cases.append({**_infer_case(clf, band_x, lo, hi), "name": "infer_deliberation_abstain"})

    _OUT.mkdir(parents=True, exist_ok=True)
    cypha_save_binary(clf.save_state(), str(_OUT / "reference.cypha"))
    f_field = np.ascontiguousarray(clf.memory.world.F_field, dtype=np.float64)
    (_OUT / "f_field.json").write_text(json.dumps(f_field.tolist()), encoding="utf-8")

    sidecar = {
        "fixture_schema": 1,
        "description": "gh_infer + deliberation parity vs Cypha.py",
        "reference_cypha": "reference.cypha",
        "f_field_json": "f_field.json",
        "input_dim": int(m["input_dim"]),
        "field_dim": int(m["field_dim"]),
        "labels": manifest["labels"],
        "nig_alpha": 0.98,
        "cases": cases,
    }
    (_OUT / "sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {_OUT / 'sidecar.json'} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
