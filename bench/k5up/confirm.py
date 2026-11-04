#!/usr/bin/env python3
"""The confirmation rule of docs/reports/K5_UPGRADES.md: PICK against BASE over seed offsets 1000-4000.
Usage: confirm.py RESULTS.tsv BASE PICK   (rows named BASE_s1000 ... and PICK_s1000 ...)"""
import sys

rows = {}
for line in open(sys.argv[1], encoding="utf-8"):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 7 and f[1] == "dev":
        rows[f[0]] = dict(sc=float(f[2].split("[")[0]), S=float(f[3]), cp=float(f[4]), v13=int(f[5]), bpb=float(f[6]))
base, pick = sys.argv[2], sys.argv[3]
seeds = [1000, 2000, 3000, 4000]
B = [rows[f"{base}_s{s}"] for s in seeds]
P = [rows[f"{pick}_s{s}"] for s in seeds]
mean = lambda xs: sum(xs) / len(xs)
mb, mp = mean([r["sc"] for r in B]), mean([r["sc"] for r in P])
wins = sum(p["sc"] > b["sc"] for p, b in zip(P, B))
dcp = mean([r["cp"] for r in P]) - mean([r["cp"] for r in B])
dS = mean([r["S"] for r in P]) - mean([r["S"] for r in B])
v13 = sum(r["v13"] for r in P)
print(f"{base}: " + " ".join(f"{r['sc']:.3f}" for r in B) + f"  mean {mb:.3f}  S {mean([r['S'] for r in B]):.3f}  copy32 {mean([r['cp'] for r in B]):.4f}")
print(f"{pick}: " + " ".join(f"{r['sc']:.3f}" for r in P) + f"  mean {mp:.3f}  S {mean([r['S'] for r in P]):.3f}  copy32 {mean([r['cp'] for r in P]):.4f}")
ok = mp - mb >= 0.01 and wins >= 3 and v13 == 0 and dcp <= 0.02 and dS >= -0.01
print(f"diff {mp - mb:+.3f}, wins {wins}/4, copy32 {dcp:+.4f}, S {dS:+.3f}, verbatim13 {v13} -> {'ADOPT' if ok else 'not adopted'}")
