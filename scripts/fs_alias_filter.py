"""Keep the aliases whose target is a subject of the store (docs/reports/PLAN_FACTS.md, GENERATION_PLAN §13.42 Q1).

usage: python3 -I scripts/fs_alias_filter.py ALIASES_ALL.tsv STOREDIR

Reads fs_aliases' output and writes STOREDIR/aliases.tsv: its header, then the lines whose target is the subject of a
row of STOREDIR/seeds.tsv or STOREDIR/facts_f2.tsv (exact title). The planner (facts.hpp, --k5-plan-find 2) reads the
file; everything else ignores it.
Author: Odin Loch
"""
import os
import sys


def main():
    src, store = sys.argv[1:3]
    subj = set()
    for name in ("seeds.tsv", "facts_f2.tsv"):
        p = os.path.join(store, name)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="surrogateescape") as f:
            next(f, None)
            for line in f:
                subj.add(line.split("\t", 1)[0])
    n = kept = 0
    with open(src, encoding="utf-8", errors="surrogateescape") as f, \
            open(os.path.join(store, "aliases.tsv"), "w", encoding="utf-8", errors="surrogateescape", newline="") as o:
        o.write(next(f))
        for line in f:
            n += 1
            c = line.rstrip("\n").split("\t")
            if len(c) >= 6 and c[1] in subj:
                o.write(line)
                kept += 1
    print(f"aliases {n}, kept {kept} (targets that are store subjects)")


if __name__ == "__main__":
    main()
