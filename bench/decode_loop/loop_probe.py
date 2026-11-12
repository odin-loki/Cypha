#!/usr/bin/env python3
"""DECODE_LOOP.md amendment 1: X1's probe metrics for loop arms. Odin Loch, 2026-11-13.

  loop_probe.py LABEL [LABEL ...]   builds RUNS/probe_<first>.items.tsv from each arm's texts on X1's 30 prompts (the
                                    first 30 of subset A), cut as the raters read them; runs k5_probe --modes map; writes
                                    RUNS/LABEL.probe.json per arm: per text, splice and tail shares.
  splice  share of words whose 4-word context K5 has seen (ctx_ord 5) but never followed by the word (ans_ord < 5)
  tail    share of words of rank above 10 in K5's distribution, or P < 0.02 (WHERE_IT_BREAKS's "tail draw")
Both read under today's K5 with the overlay, whatever the arm's settings (the probe builds T8's model).
"""
import csv
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.environ.get("DLOOP_RUNS", os.path.expanduser("~/dloop_runs"))
BIN = os.environ.get("DLOOP_BIN", os.path.expanduser("~/dloop_bin"))
SETS = json.load(open(os.path.join(HERE, "prompt_sets.json")))
X1 = SETS["A"][:30]


def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def read_gen_raw(path):
    out, cur = {}, None
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("PROMPT\t"):
                cur = int(line.split("\t")[1])
            elif line.startswith("GEN\t") and cur is not None and cur not in out:
                out[cur] = line.split("\t", 2)[2]
    return out


def unesc(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] in "nt\\":
            out.append({"n": "\n", "t": "\t", "\\": "\\"}[s[i + 1]])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def cut460(t):
    t = " ".join(t.split())
    if len(t) <= 460:
        return t
    t = t[:460]
    return t[: t.rfind(" ")].rstrip(" ,;:")


def main():
    labels = sys.argv[1:]
    rows = list(csv.DictReader(open(os.path.join(ROOT, "bench/eval/gen_prompts_dev.tsv"), encoding="utf-8"),
                               delimiter="\t", quoting=csv.QUOTE_NONE))
    items = os.path.join(RUNS, f"probe_{labels[0]}.items.tsv")
    with open(items, "w", encoding="utf-8", errors="surrogateescape") as f:
        for lab in labels:
            g = read_gen_raw(os.path.join(HERE, "human.gen.txt") if lab == "human" else os.path.join(RUNS, f"{lab}.gen.txt"))
            for i in X1:
                t = " " + cut460(unesc(g[i]))
                f.write(f"{i}\t{lab}\t{rows[i]['prompt']}\t{esc(t)}\n")
    pre = os.path.join(RUNS, f"probe_{labels[0]}")
    subprocess.run([os.path.join(BIN, "k5_probe"), "--items", items, "--out", pre, "--modes", "map", "--threads", "6",
                    "--no-learner"], check=True, cwd=ROOT, stdout=open(pre + ".out", "w"), stderr=subprocess.STDOUT)
    per = {lab: {} for lab in labels}
    lt = math.log2(0.02)
    with open(pre + ".map.tsv", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["kind"] not in per:
                continue
            d = per[r["kind"]].setdefault(r["item"], [0, 0, 0])
            if r["unk"] == "1":
                d[0] += 1
                continue
            d[0] += 1
            d[1] += int(r["ctx_ord"]) >= 5 and int(r["ans_ord"]) < 5
            d[2] += int(r["rank"]) > 10 or float(r["lp_full"]) < lt
    for lab in labels:
        out = {i: dict(splice=v[1] / v[0], tail=v[2] / v[0], words=v[0]) for i, v in per[lab].items() if v[0]}
        json.dump(out, open(os.path.join(RUNS, f"{lab}.probe.json"), "w"))
        n = len(out)
        print(f"{lab}: {n} texts, splice {sum(x['splice'] for x in out.values()) / n:.3f}, "
              f"tail {sum(x['tail'] for x in out.values()) / n:.3f}")


if __name__ == "__main__":
    main()
