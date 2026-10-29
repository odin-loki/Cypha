#!/usr/bin/env python3
"""Summary of one scripts/gen_eval.sh run (docs/reports/GEN_TRACK.md, GEN-A).

    gen_eval_summary.py OUTDIR LABEL SPLIT                one run
    gen_eval_summary.py --pair OUTDIR LABEL_A LABEL_B     paired continuation BPC, B minus A (same split)

Reads OUTDIR/LABEL.{scribe.md,sample.out,gen.txt,gen.txt.cont.tsv}. Run with python3 -I.
Author: Odin Loch
"""
import random
import re
import sys

RUNGS = ["held-out human", "sentences shuffled across docs", "trie samples (pooled)",
         "trie samples T=0.6", "trie samples T=0.8", "trie samples T=1.0"]


def rung_table(md):
    out = {}
    for line in md.splitlines():
        m = re.match(r"\| (.+?) \| (\d+) \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \| ([0-9.]+) \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \| (\d+) \|", line)
        if m and m.group(1) not in out:
            out[m.group(1)] = dict(n=int(m.group(2)), scribe=float(m.group(3)), lo=float(m.group(4)), hi=float(m.group(5)),
                                   S=float(m.group(7)), Slo=float(m.group(8)), Shi=float(m.group(9)), bytes=int(m.group(10)))
    return out


def axis_DI(md):
    out = {}
    for line in md.splitlines():
        m = re.match(r"\| (.+?) \| (\d+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|", line)
        if m:
            out[m.group(1)] = float(m.group(8))  # D_I
    return out


def unesc(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            c = s[i + 1]
            out.append({"n": "\n", "t": "\t", "\\": "\\"}.get(c, c))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def cont_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            k, n, b = line.rstrip("\n").split("\t")
            rows.append((int(n), float(b)))
    return rows


def one(outdir, label, split):
    md = open("%s/%s.scribe.md" % (outdir, label), encoding="utf-8").read()
    so = open("%s/%s.sample.out" % (outdir, label), encoding="utf-8").read()
    rt, di = rung_table(md), axis_DI(md)
    print("# gen_eval %s (%s split)\n" % (label, split))
    m = re.search(r"Trie rung: (\d+) GEN lines, (\d+) kept", md)
    if m:
        print("GEN lines %s, kept after SCRIBE's cleaning %s\n" % (m.group(1), m.group(2)))
    print("| rung | n | SCRIBE v2 [95%] | surface S [95%] | mean bytes | D_I |")
    print("|---|---|---|---|---|---|")
    for r in RUNGS:
        if r in rt:
            x = rt[r]
            print("| %s | %d | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %d | %.2f |" % (
                r, x["n"], x["scribe"], x["lo"], x["hi"], x["S"], x["Slo"], x["Shi"], x["bytes"], di.get(r, float("nan"))))
    for line in so.splitlines():
        if line.startswith("GEN_EVAL"):
            print("\n" + line)
    kinds = [line.split("\t")[1] for line in open("%s/%s.gen.txt" % (outdir, label), encoding="utf-8") if line.startswith("GEN\t")]
    show = "trie_T0.8" if "trie_T0.8" in kinds or not kinds else kinds[0]  # TEMPS: a run without T 0.8 shows its first kind
    print("\n## Five samples (T = %s)\n" % show[len("trie_T"):])
    prompt, shown = None, 0
    for line in open("%s/%s.gen.txt" % (outdir, label), encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if f[0] == "PROMPT":
            prompt = unesc(f[2])
        elif f[0] == "GEN" and f[1] == show and shown < 5:
            print("- **Prompt:** %s\n  **CyphaLM:** %s\n" % (prompt, unesc(f[2])[:400].replace("\n", " / ")))
            shown += 1


def pair(outdir, a, b):
    ra, rb = cont_rows("%s/%s.gen.txt.cont.tsv" % (outdir, a)), cont_rows("%s/%s.gen.txt.cont.tsv" % (outdir, b))
    assert len(ra) == len(rb) and all(x[0] == y[0] for x, y in zip(ra, rb)), "not the same rows"
    n = sum(x[0] for x in ra)
    d = (sum(y[1] for y in rb) - sum(x[1] for x in ra)) / n
    rng, bs = random.Random(20261029), []
    for _ in range(2000):
        idx = [rng.randrange(len(ra)) for _ in ra]
        nn = sum(ra[i][0] for i in idx)
        bs.append((sum(rb[i][1] for i in idx) - sum(ra[i][1] for i in idx)) / nn)
    bs.sort()
    print("cont_bpc %s minus %s: %+.4f [%+.4f, %+.4f] (paired, %d prompts)" % (b, a, d, bs[49], bs[1949], len(ra)))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--pair":
        pair(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 4:
        one(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
