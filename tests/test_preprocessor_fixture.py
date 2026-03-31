"""Sanity-check preprocessor parity fixtures for native `preprocessor_parity`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DIR = _ROOT / "parity_fixtures" / "preprocessor"


def _require():
    if not (_DIR / "preprocessor.json").is_file():
        pytest.skip("preprocessor fixtures missing — run: python scripts/generate_preprocessor_parity.py")


def test_preprocessor_json_fitted():
    _require()
    st = json.loads((_DIR / "preprocessor.json").read_text(encoding="utf-8"))
    assert st.get("fitted") is True
    assert st.get("input_dim", 0) > 0
    assert st.get("output_dim", 0) > 0


def test_preprocessor_sidecar_vectors():
    _require()
    sc = json.loads((_DIR / "sidecar.json").read_text(encoding="utf-8"))
    st = json.loads((_DIR / "preprocessor.json").read_text(encoding="utf-8"))
    assert len(sc["x"]) == st["input_dim"]
    assert len(sc["expected"]) == st["output_dim"]
