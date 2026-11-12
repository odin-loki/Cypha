#!/usr/bin/env python3
"""DECODE_LOOP.md amendment 2: a round's win shares on the judged items whose two texts differ (secondary).
  loop_diffscore.py NAME   (reads the judgments logged in bench/decode_loop/rounds.json and the arms' cut texts)
"""
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("DLOOP_RUNS", os.path.expanduser("~/dloop_runs"))
name = sys.argv[1]
ent = next(x for x in json.load(open(os.path.join(HERE, "rounds.json")))["rounds"] if x["name"] == name)
feats = {}


def text(arm, dev):
    if arm not in feats:
        feats[arm] = json.load(open(os.path.join(RUNS, f"{arm}.feats.json")))["per"]
    return feats[arm][str(dev)]["text"]


rng = random.Random(0)
for arm in ent["arms"]:
    per, same = {}, 0
    for j in ent["judgments"]:
        if j["arm"] != arm:
            continue
        if text(arm, j["dev"]) == text(ent["best"], j["dev"]):
            same += 1
            continue
        per.setdefault(j["dev"], []).append(j["win"])
    vals = [st.mean(v) for v in per.values()]
    if not vals:
        print(f"  {arm}: no differing items")
        continue
    bs = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(10000))
    print(f"  {arm:12s} differing only: {st.mean(vals):.3f} [{bs[250]:.3f}, {bs[9749]:.3f}]  prompts {len(vals)} "
          f"({same} judgments on identical texts left out)")
