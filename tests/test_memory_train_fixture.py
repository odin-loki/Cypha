"""Sanity-check memory_train parity fixtures for native `memory_train_parity`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_DIR = _ROOT / "parity_fixtures" / "memory_train"

from Cypha import cypha_load_binary


def _require():
    if not (_DIR / "sidecar.json").is_file():
        pytest.skip("memory_train fixtures missing — run: python scripts/generate_memory_train_parity.py")


def test_memory_train_sidecar_schema():
    _require()
    sc = json.loads((_DIR / "sidecar.json").read_text(encoding="utf-8"))
    assert "h" in sc and "h_field" in sc and "label" in sc
    assert "context_prior" in sc and "f_field" in sc
    assert "expected_loss" in sc
    assert len(sc["h"]) == 8
    assert len(sc["h_field"]) == int(sc["field_dim"])


def test_memory_train_cypha_pair_exists():
    _require()
    assert (_DIR / "before.cypha").is_file()
    assert (_DIR / "after.cypha").is_file()


def test_memory_train_after_includes_field_a_eff_with_field_w_t():
    """Regenerate with `python scripts/generate_memory_train_parity.py` if this fails."""
    _require()
    after = cypha_load_binary(str(_DIR / "after.cypha"))
    if "field_W_T" not in after:
        pytest.skip("after.cypha has no field_W_T")
    assert "field_a_eff" in after, "run scripts/generate_memory_train_parity.py after Cypha.py save_state adds field_a_eff"
    wt = np.asarray(after["field_W_T"], dtype=np.float64)
    ae = np.asarray(after["field_a_eff"], dtype=np.float64)
    assert wt.shape == ae.shape
    assert wt.ndim == 2 and wt.shape[0] == wt.shape[1]
