#!/usr/bin/env python3
"""E2 hand-checked set (docs/reports/E2_ENTITIES.md): hand_labels.txt + hand_pool.tsv -> hand.tsv (the silver format).

    python3 -I scripts/e2_hand.py bench/e2_ner/hand_pool.tsv bench/e2_ner/hand_labels.txt bench/e2_ner/hand.tsv

Each TYPE=surface item is found in the sentence after the previous item's end (bytes); a miss is an error.
Author: Odin Loch
"""
import sys

pool = [l.rstrip("\n").split("\t") for l in open(sys.argv[1], encoding="utf-8")]
out = []
for line in open(sys.argv[2], encoding="utf-8"):
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        continue
    f = line.split("\t")
    n = int(f[0])
    title, text = pool[n - 1][0], pool[n - 1][1]
    b = text.encode("utf-8")
    pos, spans = 0, []
    if len(f) > 1 and f[1].strip():
        for item in f[1].split(" | "):
            ty, surf = item.split("=", 1)
            s = surf.encode("utf-8")
            i = b.find(s, pos)
            if i < 0:
                sys.exit(f"sentence {n}: '{surf}' not found after byte {pos}")
            spans.append(f"{i}:{i + len(s)}:{ty}")
            pos = i + len(s)
    out.append(f"{title}\t{text}\t{';'.join(spans)}\n")
open(sys.argv[3], "w", encoding="utf-8").writelines(out)
print(f"{len(out)} sentences, {sum(len(l.split(chr(9))[2].split(';')) if l.split(chr(9))[2].strip() else 0 for l in out)} entities")
