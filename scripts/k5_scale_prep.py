#!/usr/bin/env python3
"""K5 on more Wikipedia (docs/reports/K5_SCALE.md): text plumbing for scripts/k5_clean_enwik9.sh.

  k5_scale_prep.py lines IN.jsonl OUT.txt          the "text" of each JSONL line, one per line (white space in a
                                                   text, newlines included, collapsed to one space)
  k5_scale_prep.py gena BENCH_DIR OUT.txt          GEN-A's prompts and continuations, dev and test, one per line
  k5_scale_prep.py nest DOCS.txt OUT_DIR           shuffle the documents (seed 20261104) and write nested
                                                   prefixes train_7M, train_30M, train_100M, train_300M, train_all

The overlap check itself is scripts/k5_overlap.cpp.
Author: Odin Loch
"""
import json
import os
import random
import re
import sys

SEED = 20261104
# Nested sizes in bytes (UTF-8, newlines between documents). 7,053,155 is the bake-off's enwik8 train.txt.
SIZES = [("7M", 7_053_155), ("30M", 30_000_000), ("100M", 100_000_000), ("300M", 300_000_000)]
WS = re.compile(r"\s+")


def unesc(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append({"t": "\t", "n": "\n", "\\": "\\"}.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def one_line(t):
    return WS.sub(" ", t).strip()


def main():
    cmd = sys.argv[1]
    if cmd == "lines":
        n = 0
        with open(sys.argv[2], encoding="utf-8") as f, open(sys.argv[3], "w", encoding="utf-8", newline="\n") as o:
            for line in f:
                if line.strip():
                    t = one_line(json.loads(line)["text"])
                    if t:
                        o.write(t + "\n")
                        n += 1
        print(f"lines {sys.argv[3]}: {n}")
    elif cmd == "gena":
        n = 0
        with open(sys.argv[3], "w", encoding="utf-8", newline="\n") as o:
            for split in ("dev", "test"):
                with open(os.path.join(sys.argv[2], f"gen_prompts_{split}.tsv"), encoding="utf-8") as f:
                    for line in f.read().split("\n")[1:]:
                        c = line.split("\t")
                        if len(c) >= 4:
                            for t in (unesc(c[2]), unesc(c[3])):
                                # the text as it is, and white space collapsed (the cleaned documents' form)
                                o.write(t.replace("\n", " ") + "\n" + one_line(t) + "\n")
                                n += 1
        print(f"gena {sys.argv[3]}: {n} texts")
    elif cmd == "nest":
        with open(sys.argv[2], encoding="utf-8") as f:
            docs = [d for d in f.read().split("\n") if d]
        random.Random(SEED).shuffle(docs)
        out = sys.argv[3]
        files = {name: open(os.path.join(out, f"train_{name}.txt"), "w", encoding="utf-8", newline="\n") for name, _ in SIZES}
        stat = {name: [0, 0] for name, _ in SIZES}
        total = 0
        with open(os.path.join(out, "train_all.txt"), "w", encoding="utf-8", newline="\n") as fa:
            for i, d in enumerate(docs):
                b = len(d.encode("utf-8")) + (1 if i else 0)
                sep = "\n" if i else ""
                fa.write(sep + d)
                for name, cap in SIZES:
                    if name in files:
                        if total + b <= cap:
                            files[name].write(sep + d)
                            stat[name][0] += 1
                            stat[name][1] += b
                        else:
                            files.pop(name).close()
                total += b
        for f in files.values():
            f.close()
        stat["all"] = [len(docs), total]
        for k, v in stat.items():
            print(f"nest train_{k}.txt: {v[0]} docs, {v[1]} bytes")
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
