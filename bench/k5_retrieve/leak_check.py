#!/usr/bin/env python3
"""K5_RETRIEVE.md: the retrieval's leak check. Odin Loch, 2026-11-12.

  python3 -I bench/k5_retrieve/leak_check.py map --titles E9_TITLES.tsv --clean clean_e9.txt --train train_all.txt --out TRAIN_TITLES.tsv
  python3 -I bench/k5_retrieve/leak_check.py check --train-titles TRAIN_TITLES.tsv --gena-titles GENA_TITLES.tsv \
      --prompts bench/eval/gen_prompts_dev.tsv --out OUT.md LABEL=RET_LOG [...]

map: each line of train_all.txt (K5_SCALE's kept text, in nest's order) is a document of clean_e9.txt, sometimes with
sentences cut (k5_overlap.cpp). A line is matched to the clean_e9 document that has a sentence starting with the line's
first 40 bytes and holds three 40-byte pieces of the line (start, middle, end); k5_ret_titles gives that document's
page title and byte offset in enwik9.

check: a retrieval log (CYPHA_K5_RET_LOG: prompt, query words, doc:score, ...) is joined to the GEN-A prompts (by the
prompt text) and to their source pages (the `doc` column: SCRIBE's page index in enwik8 [72, 96) MiB, titled by
k5_ret_titles). Counted: retrieved documents whose page title equals the prompt's page title (case and spaces folded),
whose enwik byte offset falls in the GEN-A source range [75497472, 100000000), and documents with no title match.
It decides nothing.

copy: LABEL=GEN_FILE,RET_LOG ...: each continuation against its own prompt's retrieved documents (train_all.txt
lines): the share of its bytes in 32-byte spans of them, and its longest run of words found in them.
"""
import argparse
import bisect
import csv
import sys

csv.field_size_limit(1 << 30)


def fnv64(b):
    h = 1469598103934665603
    for c in b:
        h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def read_titles(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            i, off, title, nb, h = line.rstrip("\n").split("\t")
            rows.append((int(off), title, int(nb), h))
    return rows


def cmd_map(a):
    titles = read_titles(a.titles)
    with open(a.clean, "rb") as f:
        clean = f.read().split(b"\n")
    if clean and clean[-1] == b"":
        clean.pop()
    if len(clean) != len(titles):
        sys.exit(f"clean_e9 has {len(clean)} lines, titles {len(titles)}")
    same = sum(1 for i in range(0, len(clean), 997) if f"{fnv64(clean[i]):016x}" == titles[i][3])
    print(f"hash check: {same} of {len(range(0, len(clean), 997))} sampled clean_e9 lines equal k5_ret_titles' document", file=sys.stderr)
    with open(a.train, "rb") as f:
        train = f.read().split(b"\n")
    if train and train[-1] == b"":
        train.pop()
    # train_all's lines are in nest's order, not clean_e9's: index every sentence start of clean_e9 (its first 40
    # bytes), then match a line by its first sentence and check its three pieces.
    starts = {}
    for k, c in enumerate(clean):
        p = 0
        while p < len(c):
            starts.setdefault(c[p:p + 40], []).append(k)
            q = c.find(b". ", p)
            if q < 0:
                break
            p = q + 2
    miss = 0
    big = None
    out = open(a.out, "w", encoding="utf-8")
    out.write("line\tclean_index\toffset\ttitle\tcandidates\n")
    for li, t in enumerate(train):
        n = len(t)
        pieces = [t[:40], t[max(0, n // 2 - 20):n // 2 + 20], t[max(0, n - 40):]]
        cand = []
        p, tries = 0, 0
        while not cand and p < n and tries < 20:  # the line's first sentence may have been cut: try its next ones
            cand = [k for k in starts.get(t[p:p + 40], []) if all(x in clean[k] for x in pieces)]
            q = t.find(b". ", p)
            if q < 0:
                break
            p, tries = q + 2, tries + 1
        if not cand:  # cuts inside the first sentences: find pieces of the line in all of clean_e9
            if big is None:
                big = b"\n".join(clean)
                bstart = [0]
                for c in clean[:-1]:
                    bstart.append(bstart[-1] + len(c) + 1)
            votes = {}
            for f in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
                x = t[int(n * f):int(n * f) + 40]
                if len(x) < 40:
                    continue
                o = big.find(x)
                if o >= 0:
                    k = bisect.bisect_right(bstart, o) - 1
                    votes[k] = votes.get(k, 0) + 1
            best = max(votes.items(), key=lambda kv: (kv[1], -kv[0]), default=(None, 0))
            if best[1] >= 3:
                cand = [best[0]]
        if not cand:
            miss += 1
            out.write(f"{li}\t-1\t-1\t\t0\n")
            continue
        # Several pages can hold the same text: all their titles are kept (" || " between).
        cand = sorted(set(cand))
        k = cand[0]
        names = " || ".join(titles[c][1] for c in cand)
        out.write(f"{li}\t{k}\t{titles[k][0]}\t{names}\t{len(cand)}\n")
    out.close()
    print(f"train_all lines {len(train)}, matched {len(train) - miss}, unmatched {miss}", file=sys.stderr)


def fold(s):
    return " ".join(s.replace("_", " ").split()).casefold()


def cmd_check(a):
    tt = {}
    with open(a.train_titles, encoding="utf-8") as f:
        next(f)
        for line in f:
            li, ci, off, title = line.rstrip("\n").split("\t")[:4]
            tt[int(li)] = (int(off), title)
    gt = read_titles(a.gena_titles)
    prompts = {}
    with open(a.prompts, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            p = r["prompt"].replace("\\t", " ").replace("\\n", " ").replace("\\\\", "\\")
            prompts[" ".join(p.split())] = (r["id"], int(r["doc"]))
    lines = ["| arm | prompts in log | joined to GEN-A | retrieved docs | distinct docs | same page title as the prompt | "
             "page offset in GEN-A's source range | no title found |", "|---|---|---|---|---|---|---|---|"]
    for spec in a.logs:
        label, path = spec.split("=", 1)
        n = joined = docs = same = inrange = untitled = 0
        distinct = set()
        examples = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                n += 1
                key = " ".join(parts[0].split())
                hit = prompts.get(key)
                ptitle = None
                if hit:
                    joined += 1
                    ptitle = gt[hit[1]][1]
                for ds in filter(None, parts[2].split(",")):
                    d = int(ds.split(":")[0])
                    docs += 1
                    distinct.add(d)
                    off, title = tt.get(d, (-1, ""))
                    if not title:
                        untitled += 1
                    if 75497472 <= off < 100000000:
                        inrange += 1
                    if ptitle is not None and title and any(fold(x) == fold(ptitle) for x in title.split(" || ")):
                        same += 1
                        examples.append((hit[0], ptitle, d))
        lines.append(f"| {label} | {n} | {joined} | {docs} | {len(distinct)} | {same} | {inrange} | {untitled} |")
        for e in examples[:20]:
            lines.append(f"<!-- same title: {e} -->")
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def read_gen(path):
    out, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("PROMPT\t"):
                cur = int(line.split("\t")[1])
            elif line.startswith("GEN\t") and cur is not None and cur not in out:
                out[cur] = line.split("\t", 2)[2]
    return out


def cmd_copy(a):
    """Each continuation against its own prompt's retrieved documents: the share of its bytes in 32-byte spans of
    them, and its longest run of words found in them (words split on spaces, as written)."""
    with open(a.train, "rb") as f:
        train = f.read().split(b"\n")
    prompts = {}
    with open(a.prompts, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            p = r["prompt"].replace("\\t", " ").replace("\\n", " ").replace("\\\\", "\\")
            prompts[" ".join(p.split())] = int(r["id"].split("/")[1])
    lines = ["| arm | continuations | bytes in 32-byte spans of their retrieved docs | longest word run (max) | runs of 13+ words |",
             "|---|---|---|---|---|"]
    for spec in a.arms:
        label, rest = spec.split("=", 1)
        gpath, lpath = rest.split(",", 1)
        gen = read_gen(gpath)
        docs = {}
        with open(lpath, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                i = prompts.get(" ".join(parts[0].split()))
                if i is not None and len(parts) >= 3:
                    docs[i] = [int(x.split(":")[0]) for x in parts[2].split(",") if x]
        cov = tot = 0
        longest = 0
        n13 = 0
        n = 0
        for i, g in gen.items():
            ds = docs.get(i)
            if ds is None:
                continue
            n += 1
            ref = b"\n".join(train[d] for d in ds)
            spans = {ref[j:j + 32] for j in range(len(ref) - 31)}
            gb = g.encode("utf-8")
            hit = bytearray(len(gb))
            for j in range(len(gb) - 31):
                if gb[j:j + 32] in spans:
                    hit[j:j + 32] = b"\x01" * 32
            cov += sum(hit)
            tot += len(gb)
            gw = g.split()
            rs = " " + " ".join(ref.decode("utf-8", "replace").split()) + " "
            best = 0
            for j in range(len(gw)):
                k = best + 1
                while j + k <= len(gw) and (" " + " ".join(gw[j:j + k]) + " ") in rs:
                    best = k
                    k += 1
            longest = max(longest, best)
            n13 += best >= 13
        lines.append(f"| {label} | {n} | {cov / max(tot, 1):.4f} | {longest} | {n13} |")
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("map")
    m.add_argument("--titles", required=True)
    m.add_argument("--clean", required=True)
    m.add_argument("--train", required=True)
    m.add_argument("--out", required=True)
    c = sp.add_parser("check")
    c.add_argument("--train-titles", required=True)
    c.add_argument("--gena-titles", required=True)
    c.add_argument("--prompts", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("logs", nargs="+")
    y = sp.add_parser("copy")
    y.add_argument("--train", required=True)
    y.add_argument("--prompts", required=True)
    y.add_argument("--out", required=True)
    y.add_argument("arms", nargs="+", help="LABEL=GEN_FILE,RET_LOG")
    a = ap.parse_args()
    {"map": cmd_map, "check": cmd_check, "copy": cmd_copy}[a.cmd](a)


if __name__ == "__main__":
    main()
