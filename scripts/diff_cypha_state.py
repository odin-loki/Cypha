#!/usr/bin/env python3
"""Compare two cypha_save_binary files (shallow recursive). Usage: diff_cypha_state.py a.cypha b.cypha"""
from __future__ import annotations

import sys

import numpy as np

from Cypha import cypha_load_binary


def diff(a, b, path: str = "root") -> bool:
    ta, tb = type(a), type(b)
    if ta != tb:
        print("type", path, ta, tb)
        return False
    if isinstance(a, dict):
        ka, kb = set(a), set(b)
        if ka != kb:
            print("keys", path, "only_a", sorted(ka - kb), "only_b", sorted(kb - ka))
            return False
        ok = True
        for k in sorted(ka):
            if not diff(a[k], b[k], f"{path}/{k}"):
                ok = False
        return ok
    if isinstance(a, float):
        if a != b and abs(a - b) > 1e-12:
            print("float", path, a, b)
            return False
        return True
    if isinstance(a, np.ndarray):
        if a.shape != b.shape:
            print("arr_shape", path, a.shape, b.shape)
            return False
        if not np.allclose(a, b, rtol=0, atol=0, equal_nan=True):
            mx = float(np.max(np.abs(a - b)))
            print("arr_data", path, "max_abs_diff", mx)
            return False
        return True
    if a != b:
        print("val", path, repr(a)[:120], repr(b)[:120])
        return False
    return True


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: diff_cypha_state.py a.cypha b.cypha", file=sys.stderr)
        return 2
    a = cypha_load_binary(sys.argv[1])
    b = cypha_load_binary(sys.argv[2])
    return 0 if diff(a, b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
