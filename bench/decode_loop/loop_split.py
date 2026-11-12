#!/usr/bin/env python3
"""DECODE_LOOP.md: put a loop run's PROMPT ids back to GEN-A dev numbers and split it. Odin Loch, 2026-11-13.

  loop_split.py RAW.gen.txt PREFIX    ->  PREFIX.gen.txt (all), PREFIX.A.gen.txt (subset A), PREFIX.R.gen.txt (rater prompts)
  loop_split.py --tsv                 ->  writes bench/decode_loop/loop_prompts.tsv (subset A, then the rater prompts)
  loop_split.py --human               ->  writes bench/decode_loop/human.gen.txt (the human continuations of all 150)

Row k of loop_prompts.tsv is GEN-A dev prompt ORDER[k]; `sample` numbers its PROMPT lines by row.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SETS = json.load(open(os.path.join(HERE, "prompt_sets.json")))
ORDER = SETS["A"] + SETS["R"]
DEV = os.path.join(ROOT, "bench/eval/gen_prompts_dev.tsv")


def dev_lines():
    with open(DEV, encoding="utf-8") as f:
        lines = f.read().split("\n")
    return lines[0], [x for x in lines[1:] if x]


def main():
    if sys.argv[1] == "--tsv":
        head, rows = dev_lines()
        assert len(rows) == 250
        with open(os.path.join(HERE, "loop_prompts.tsv"), "w", encoding="utf-8") as f:
            f.write(head + "\n" + "".join(rows[i] + "\n" for i in ORDER))
        return
    if sys.argv[1] == "--human":
        rows = list(csv.DictReader(open(DEV, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
        with open(os.path.join(HERE, "human.gen.txt"), "w", encoding="utf-8") as f:
            f.write("MODEL\thuman\n")
            for i in ORDER:
                f.write(f"PROMPT\t{i}\t{rows[i]['prompt']}\nGEN\thuman\t{rows[i]['continuation']}\n")
        return
    raw, pre = sys.argv[1], sys.argv[2]
    A = set(SETS["A"])
    out = {k: open(f"{pre}{k}.gen.txt", "w", encoding="utf-8", errors="surrogateescape") for k in ("", ".A", ".R")}
    seen, cur = 0, None
    with open(raw, encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            if line.startswith("MODEL\t"):
                for g in out.values():
                    g.write(line)
                continue
            if line.startswith("PROMPT\t"):
                p = line.split("\t", 2)
                cur = ORDER[int(p[1])]
                line = f"PROMPT\t{cur}\t{p[2]}"
                seen += 1
            out[""].write(line)
            out[".A" if cur in A else ".R"].write(line)
    for g in out.values():
        g.close()
    if seen != len(ORDER):
        sys.exit(f"{raw}: {seen} prompts, want {len(ORDER)}")


if __name__ == "__main__":
    main()
