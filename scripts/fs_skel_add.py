"""Add new relations' skeletons to the planner's bank (docs/reports/PLAN_FACTS.md, GENERATION_PLAN §13.42 Q1).

usage: python3 -I scripts/fs_skel_add.py OLD_BANK NEW_BANK OUT RELS [OLD_MODEL NEW_MODEL]

OUT is OLD_BANK unchanged, then NEW_BANK's "skel" lines of the relations RELS (comma-separated) and its "trans" lines
that name one of them. The old relations keep their skeletons; only the transition totals around the new relations
grow. With OLD_MODEL and NEW_MODEL, NEW_MODEL is OLD_MODEL (k5_planner train's mixer, floor and stats) with its "skel"
line pointing at OUT.
Author: Odin Loch
"""
import sys


def main():
    old, new, out, rels = sys.argv[1:5]
    rels = set(rels.split(","))
    n_s = n_t = 0
    with open(out, "w", encoding="utf-8", newline="") as o:
        with open(old, encoding="utf-8") as f:
            o.write(f.read())
        with open(new, encoding="utf-8") as f:
            for line in f:
                c = line.split()
                if len(c) >= 3 and c[0] == "skel" and c[1] in rels:
                    o.write(line)
                    n_s += 1
                elif len(c) == 4 and c[0] == "trans" and (c[1] in rels or c[2] in rels):
                    o.write(line)
                    n_t += 1
    print(f"added {n_s} skeletons and {n_t} transitions of {sorted(rels)}")
    if len(sys.argv) >= 7:
        with open(sys.argv[5], encoding="utf-8") as f, open(sys.argv[6], "w", encoding="utf-8", newline="") as o:
            for line in f:
                o.write(f"skel {out}\n" if line.startswith("skel ") else line)


if __name__ == "__main__":
    main()
