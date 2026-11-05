#!/usr/bin/env python3
"""One row of docs/reports/K5_MCMC.md's results from an arm's files in OUT (arm.sh):
NAME v2 [lo, hi]  v3.1 [lo, hi]  surface S  copy32  verbatim13  sample seconds  wall seconds  peak MB  MCMC line  flags.
Run with python3 -I. Author: Odin Loch"""
import re
import sys

out, name, flags = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""


def pooled(path):
    for line in open(path):
        m = re.match(r"\| trie samples \(pooled\) \| \d+ \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \|(.*)", line)
        if m:
            return m.group(1), m.group(2), m.group(3), m.group(4)
    return "?", "?", "?", ""


v2, l2, h2, rest = pooled(f"{out}/{name}.scribe.md")
S = re.search(r"\| [0-9.]+ \| ([0-9.]+) \[", "|" + rest)
v3, l3, h3, _ = pooled(f"{out}/{name}.scribe31.md")
ev = open(f"{out}/{name}.sample.out").read()
cp = re.search(r"copy32 T[0-9.]+ ([0-9.]+) verbatim13 [0-9.]+ \((\d+) of", ev)
err = open(f"{out}/{name}.sample.err").read()
gs = re.search(r"generated \d+ bytes in ([0-9.]+) s", err)
mc = re.search(r"k5-mcmc: (\d+ paragraphs.*)", err)
try:
    tm = open(f"{out}/{name}.time").read().split()[-2:]
except OSError:
    tm = ["?", "0"]
print("\t".join([name, f"{v2} [{l2}, {h2}]", f"{v3} [{l3}, {h3}]", S.group(1) if S else "?", cp.group(1) if cp else "?",
                 cp.group(2) if cp else "?", gs.group(1) if gs else "?", tm[0], f"{int(tm[1]) / 1024:.0f}",
                 mc.group(1) if mc else "-", flags]))
