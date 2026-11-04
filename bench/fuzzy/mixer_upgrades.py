#!/usr/bin/env python3
"""Content-mixer upgrades without the GLN (docs/GENERATION_PLAN.md §13.21). Odin Loch, 2026-11-04.

On the dump (glnx_prep.sh), the online chain (exponentiated gradient) and the online final layer, as cyphalm_trie's
defaults, against:
  U1  two learning rates: the chain also run at a slow (0.005) and a fast (0.08) step; the final layer gets both
      chains' outputs as two more inputs and picks between them (PAQ runs mixers at several rates);
  U2  the final layer's 80 weight sets chosen by copy's bucket (8) x the previous byte's class (5) x InfiniGram's match
      at least 16 bytes or not (2), instead of InfiniGram's bucket x the previous byte's class;
  U3  U1 and U2 together.
Usage: mixer_upgrades.py DUMP ARM   (ARM: U0 U1 U2 U3; U0 is the default, for the same run's reference)
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from glnx import COLS  # noqa: E402
from gln_mix import bits, chain, diff_ci, final_layer, prev_class  # noqa: E402


def main():
    D = np.fromfile(sys.argv[1], np.float32).reshape(-1, len(COLS)).astype(np.float64)
    c = {n: D[:, COLS.index(n)] for n in COLS}
    half = len(D) // 2
    arm = sys.argv[2]
    cur, snaps = chain(c, "eg", 0)
    extra = ()
    if arm in ("U1", "U3"):
        slow, _ = chain(c, "eg", 0, eta=0.005)
        fast, _ = chain(c, "eg", 0, eta=0.08)
        extra = (slow, fast)
    gate = None
    if arm in ("U2", "U3"):
        gate = c["copy_b"].astype(int).clip(0, 7) * 10 + prev_class(c) * 2 + (c["ig_n"] >= 16).astype(int)
    p = final_layer(D, c, snaps, cur, gate=gate, extra=extra)
    b = bits(p, half)
    np.save(f"/root/work/glnmix/upg_{arm}.npy", b)
    print(json.dumps({"arm": arm, "bpb": round(float(b.mean()), 4), "chain": round(float(bits(cur, half).mean()), 4)}), flush=True)
    if arm != "U0":
        try:
            b0 = np.load("/root/work/glnmix/upg_U0.npy")
            print(json.dumps({"arm": arm, "minus_U0": diff_ci(b, b0)}))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
