#!/usr/bin/env python3
"""
Write ``bessel_ratios.npz`` next to ``Cypha.py`` for SciPy-free GH Bessel lookup.

Same grid and ratios as ``scripts/gen_native_bessel_table.py`` / ``Cypha.py`` SciPy init.
Run once (requires scipy): ``python scripts/export_bessel_ratios_npz.py``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import kv

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "bessel_ratios.npz"

_X = np.linspace(1e-6, 120.0, 16384)
_K0 = kv(0.0, _X)
_K1 = kv(1.0, _X)
_K2 = kv(2.0, _X)
_K2_K1 = (_K2 / np.maximum(_K1, 1e-300)).astype(np.float64)
_K0_K1 = (_K0 / np.maximum(_K1, 1e-300)).astype(np.float64)
np.savez_compressed(_OUT, x=_X.astype(np.float64), k2_k1=_K2_K1, k0_k1=_K0_K1)
print(f"Wrote {_OUT} ({_OUT.stat().st_size} bytes)")
