"""Merge two pattern_learn stores into one the content plan reads (docs/reports/FACT_STORE_SCALE.md).

usage: python3 -I scripts/fs_merge_store.py BASE_DIR ADD_DIR OUT_DIR BASE_TAG ADD_TAG

seeds.tsv, facts_f2.tsv, facts_f2.evidence.tsv, facts_wd_check.tsv and pages.tsv are BASE's rows then ADD's rows (one
header). seeds.tsv and facts_f2.tsv gain a last column `corpus` (BASE_TAG or ADD_TAG); content_plan.hpp reads only the
first 6 and 9 columns. patterns.tsv (the frames) is BASE's; ADD's is kept as patterns_add.tsv. Offsets are not changed:
enwik8 is enwik9's first 1e8 bytes, so both stores' byte_offset index enwik9.
Author: Odin Loch
"""
import os
import shutil
import sys


def merge(name, base, add, out, tags, tag_col):
    with open(os.path.join(out, name), "w", encoding="utf-8", newline="") as o:
        for i, d in enumerate((base, add)):
            p = os.path.join(d, name)
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8", errors="surrogateescape", newline="") as f:
                for j, line in enumerate(f):
                    line = line.rstrip("\n")
                    has_header = name != "pages.tsv"
                    if has_header and j == 0:
                        if i == 0:
                            o.write(line + ("\tcorpus" if tag_col else "") + "\n")
                        continue
                    o.write(line + ("\t" + tags[i] if tag_col else "") + "\n")


def main():
    base, add, out, t0, t1 = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    for n, tag in (("seeds.tsv", True), ("facts_f2.tsv", True), ("facts_f2.evidence.tsv", False),
                   ("facts_wd_check.tsv", False), ("pages.tsv", False)):
        merge(n, base, add, out, (t0, t1), tag)
    shutil.copy(os.path.join(base, "patterns.tsv"), os.path.join(out, "patterns.tsv"))
    shutil.copy(os.path.join(add, "patterns.tsv"), os.path.join(out, "patterns_add.tsv"))


if __name__ == "__main__":
    main()
