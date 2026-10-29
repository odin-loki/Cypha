#!/usr/bin/env python3
"""Build the frozen generation benchmark (docs/reports/GEN_TRACK.md, GEN-A).

Input: the JSONL written by
    scribe prompts enwik:ENWIK8:72:96 --n 100000 --seed 20261029 --set gen --out POOL
(SCRIBE's own prompt maker: a prompt of 100-300 bytes and the true human continuation of
400-600 bytes with at least 3 sentences, one per cleaned page of enwik8 [72 MiB, end)).
No CyphaLM training byte ([0, 8) MiB) and no SCRIBE pack, reference or test byte ([16, 72) MiB) is in that range.

Contamination: a row is dropped when more than 10% of its continuation's bytes, or more than 25% of its
prompt's, lie in 32-byte spans that occur verbatim in enwik8's first 8 MiB (CyphaLM's training slice and
holdout), raw or as cleaned by SCRIBE (CLEAN = `scribe docs enwik:ENWIK8:0:8 --out CLEAN`). Some pages of
enwik8 repeat text from earlier pages.

Output: OUTDIR/gen_prompts_dev.tsv and OUTDIR/gen_prompts_test.tsv. A row goes to dev when the
FNV-1a 64 hash of its prompt is even, else to test; each file keeps the first N clean rows of its half
in the pool's (seeded) order. Columns: id, doc, prompt, continuation; tab, newline and backslash
are escaped as \\t, \\n and \\\\.

Run with python3 -I (the pool is data, read as data).
Author: Odin Loch
"""
import json
import sys


def fnv1a64(s):
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


K = 32  # span length of the contamination rule


def grams(b, out):
    for i in range(0, len(b) - K + 1):
        out.add(hash(b[i:i + K]))


def covered(text, G):
    b = text.encode("utf-8")
    cov = bytearray(len(b))
    for i in range(0, len(b) - K + 1):
        if hash(b[i:i + K]) in G:
            cov[i:i + K] = b"\x01" * K
    return sum(cov) / max(1, len(b))


def main():
    if len(sys.argv) != 6:
        print("usage: gen_bench.py POOL.jsonl OUTDIR N ENWIK8 CLEAN_0_8.jsonl", file=sys.stderr)
        return 2
    pool, outdir, n, enwik, clean = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    G = set()
    with open(enwik, "rb") as f:
        grams(f.read(8388608), G)
    with open(clean, encoding="utf-8") as f:
        grams(" ".join(json.loads(line)["text"] for line in f).encode("utf-8"), G)
    halves = {"dev": [], "test": []}
    dropped = {"dev": 0, "test": 0}
    with open(pool, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            p, c = o.get("prompt", ""), o.get("ref", "")
            if not p or not c:
                continue
            half = "dev" if fnv1a64(p) % 2 == 0 else "test"
            if len(halves[half]) >= n:
                continue
            if covered(c, G) > 0.10 or covered(p, G) > 0.25:
                dropped[half] += 1
                continue
            halves[half].append((o["meta"]["doc"], p, c))
    for half, rows in halves.items():
        if len(rows) < n:
            print("only %d rows in %s, need %d" % (len(rows), half, n), file=sys.stderr)
            return 1
        with open("%s/gen_prompts_%s.tsv" % (outdir, half), "w", encoding="utf-8", newline="\n") as out:
            out.write("id\tdoc\tprompt\tcontinuation\n")
            for k, (doc, p, c) in enumerate(rows[:n]):
                out.write("%s/%04d\t%d\t%s\t%s\n" % (half, k, doc, esc(p), esc(c)))
        print("%s: %d rows written, %d dropped as contaminated before them" % (half, n, dropped[half]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
