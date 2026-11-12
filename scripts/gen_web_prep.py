#!/usr/bin/env python3
"""GEN-W and C1 (docs/reports/GEN_WEB.md): C4 text prepared for the web benchmark and the clean prose corpus.

    gen_web_prep.py extract SHARD.jsonl OUT.jsonl SET (--docs N | --bytes B)
        C4 documents in file order: JSON decoded in full, typographic marks to ASCII, zero-width marks removed,
        exact repeats dropped. Stops after N kept documents, or once B bytes of text are kept.
    gen_web_prep.py dedup IN.jsonl OUT.jsonl [SEEN_HASHES]
        drops a document whose text equals an earlier one's (after SCRIBE's cleaner); with SEEN_HASHES, also one
        seen in an earlier file (the file is read and then extended). Line i of OUT is document i.
    gen_web_prep.py lines IN.jsonl OUT.txt
        the "text" of each line, one document per line (newlines and tabs become spaces).
    gen_web_prep.py texts OUT.txt TSV...
        every prompt and continuation of the TSVs, one per line (for the 32-byte span checks).
    gen_web_prep.py without SRC.jsonl TSV OUT.jsonl
        SRC without the documents named in the TSV's doc column (the ladder's region S).
    gen_web_prep.py join OUT.txt LIMIT IN.txt...
        the inputs in order, cross-file exact repeats dropped, cut at the first document boundary past LIMIT bytes.
    gen_web_prep.py bench POOL.jsonl OUTDIR N --train F... --scribe F... [--scan-rows R]
        the GEN-W rows (rules 1-4 of GEN_WEB.md), written as OUTDIR/gen_web_{dev,test}.tsv.
        --train: the writer's training texts (rule 2); --scribe: SCRIBE-W texts and GEN-A's texts (rule 3).

Run with python3 -I. Author: Odin Loch
"""
import hashlib
import json
import re
import sys

TYPO = {
    "\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": " - ", "\u2026": "...", "\u00a0": " ",
    "\u200b": "", "\u200c": "", "\u200d": "", "\u2060": "", "\u00ad": "", "\ufeff": "",
}
TYPO_RE = re.compile("[" + "".join(TYPO) + "]")


def normalise(t):
    return TYPO_RE.sub(lambda m: TYPO[m.group(0)], t)


def h64(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def fnv1a64(s):
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def unesc(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            out.append({"t": "\t", "n": "\n", "\\": "\\"}.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def tsv_rows(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            idx, doc, p, c = line.rstrip("\n").split("\t")
            yield idx, int(doc), unesc(p), unesc(c)


def extract(a):
    src, out, set_ = a[0], a[1], a[2]
    lim_docs = int(a[a.index("--docs") + 1]) if "--docs" in a else None
    lim_bytes = int(a[a.index("--bytes") + 1]) if "--bytes" in a else None
    seen, n, nb, rep, read = set(), 0, 0, 0, 0
    with open(src, encoding="utf-8") as f, open(out, "w", encoding="utf-8", newline="\n") as o:
        for line in f:
            read += 1
            t = normalise(json.loads(line)["text"])
            k = h64(t)
            if k in seen:
                rep += 1
                continue
            seen.add(k)
            o.write(json.dumps({"id": "%s/%07d" % (set_, n), "set": set_, "system": "human", "text": t},
                               ensure_ascii=False) + "\n")
            n += 1
            nb += len(t.encode("utf-8"))
            if (lim_docs and n >= lim_docs) or (lim_bytes and nb >= lim_bytes):
                break
    print("extract %s: %d read, %d kept (%d bytes of text), %d exact repeats dropped" % (out, read, n, nb, rep))


def dedup(a):
    src, out = a[0], a[1]
    seen_path = a[2] if len(a) > 2 else None
    seen = set()
    if seen_path:
        try:
            with open(seen_path, "rb") as f:
                b = f.read()
            seen = {b[i:i + 8] for i in range(0, len(b), 8)}
        except FileNotFoundError:
            pass
    n = rep = 0
    new = []
    with open(src, encoding="utf-8") as f, open(out, "w", encoding="utf-8", newline="\n") as o:
        for line in f:
            r = json.loads(line)
            k = h64(r["text"])
            if k in seen:
                rep += 1
                continue
            seen.add(k)
            new.append(k)
            r["id"] = "%s/%07d" % (r.get("set", "docs"), n)
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    if seen_path:
        with open(seen_path, "ab") as f:
            f.write(b"".join(new))
    print("dedup %s: %d kept, %d repeats dropped" % (out, n, rep))


def one_line(t):
    return re.sub(r"[\n\r\t]+", " ", t).strip()


def lines(a):
    n = 0
    with open(a[0], encoding="utf-8") as f, open(a[1], "w", encoding="utf-8", newline="\n") as o:
        for line in f:
            t = one_line(json.loads(line)["text"])
            if t:
                o.write(t + "\n")
                n += 1
    print("lines %s: %d" % (a[1], n))


def texts(a):
    n = 0
    with open(a[0], "w", encoding="utf-8", newline="\n") as o:
        for path in a[1:]:
            for _, _, p, c in tsv_rows(path):
                o.write(one_line(p) + "\n" + one_line(c) + "\n")
                n += 2
    print("texts %s: %d" % (a[0], n))


def without(a):
    drop = {doc for _, doc, _, _ in tsv_rows(a[1])}
    n = k = 0
    with open(a[0], encoding="utf-8") as f, open(a[2], "w", encoding="utf-8", newline="\n") as o:
        for i, line in enumerate(f):
            n += 1
            if i in drop:
                continue
            o.write(line)
            k += 1
    print("without %s: %d of %d documents kept (%d dropped)" % (a[2], k, n, n - k))


def join(a):
    out, limit, ins = a[0], int(a[1]), a[2:]
    seen, nb, n, rep = set(), 0, 0, 0
    with open(out, "w", encoding="utf-8", newline="\n") as o:
        for path in ins:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if nb >= limit:
                        break
                    t = line.rstrip("\n")
                    if not t:
                        continue
                    k = h64(t)
                    if k in seen:
                        rep += 1
                        continue
                    seen.add(k)
                    o.write(t + "\n")
                    nb += len(t.encode("utf-8")) + 1
                    n += 1
    print("join %s: %d documents, %d bytes, %d cross-file repeats dropped" % (out, n, nb, rep))


K = 32


def grams(b):
    return [hash(b[i:i + K]) for i in range(0, len(b) - K + 1)]


def scan(path, G, hit):
    with open(path, "rb") as f:
        b = f.read()
    for i in range(0, len(b) - K + 1):
        h = hash(b[i:i + K])
        if h in G:
            hit.add(h)
    return len(b)


def covered(gs, hit):
    n = len(gs) + K - 1
    if n <= 0:
        return 0.0
    cov = bytearray(n)
    for i, h in enumerate(gs):
        if h in hit:
            cov[i:i + K] = b"\x01" * K
    return sum(cov) / n


def bench(a):
    pool, outdir, n = a[0], a[1], int(a[2])
    opts, cur = {"--train": [], "--scribe": []}, None
    scan_rows = 4000
    i = 3
    while i < len(a):
        if a[i] in opts:
            cur = a[i]
        elif a[i] == "--scan-rows":
            scan_rows = int(a[i + 1])
            i += 1
        else:
            opts[cur].append(a[i])
        i += 1
    rows = []
    with open(pool, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("prompt") and o.get("ref"):
                rows.append((o["meta"]["doc"], o["prompt"], o["ref"]))
            if len(rows) >= scan_rows:
                break
    gp = [grams(p.encode("utf-8")) for _, p, _ in rows]
    gc = [grams(c.encode("utf-8")) for _, _, c in rows]
    G = set()
    for x in gp + gc:
        G.update(x)
    hits = {}
    for key in ("--train", "--scribe"):
        hit = set()
        for path in opts[key]:
            nb = scan(path, G, hit)
            print("scanned %s (%d bytes)" % (path, nb), file=sys.stderr)
        hits[key] = hit
    link = re.compile(r"http|www\.|@")
    halves = {"dev": [], "test": []}
    why = {"dev": [0, 0, 0, 0], "test": [0, 0, 0, 0]}
    kept = set()
    read = 0
    for k, (doc, p, c) in enumerate(rows):
        half = "dev" if fnv1a64(p) % 2 == 0 else "test"
        if len(halves[half]) >= n:
            if all(len(v) >= n for v in halves.values()):
                break
            continue
        read += 1
        if link.search(p) or link.search(c):
            why[half][0] += 1
            continue
        if covered(gc[k], hits["--train"]) > 0.10 or covered(gp[k], hits["--train"]) > 0.25:
            why[half][1] += 1
            continue
        if covered(gc[k], hits["--scribe"]) > 0.10 or covered(gp[k], hits["--scribe"]) > 0.25:
            why[half][2] += 1
            continue
        if any(h in kept for h in gp[k]) or any(h in kept for h in gc[k]):
            why[half][3] += 1
            continue
        kept.update(gp[k])
        kept.update(gc[k])
        halves[half].append((doc, p, c))
    for half, rs in halves.items():
        if len(rs) < n:
            print("only %d rows in %s, need %d (raise --scan-rows)" % (len(rs), half, n), file=sys.stderr)
            return 1
        with open("%s/gen_web_%s.tsv" % (outdir, half), "w", encoding="utf-8", newline="\n") as out:
            out.write("id\tdoc\tprompt\tcontinuation\n")
            for j, (doc, p, c) in enumerate(rs[:n]):
                out.write("%s/%04d\t%d\t%s\t%s\n" % (half, j, doc, esc(p), esc(c)))
        w = why[half]
        print("%s: %d rows written; dropped before them: links %d, writer's text %d, SCRIBE-W/GEN-A text %d, "
              "shared span %d" % (half, n, w[0], w[1], w[2], w[3]))
    print("pool rows scanned %d, rows read %d" % (len(rows), read))
    return 0


def main():
    cmd, a = sys.argv[1], sys.argv[2:]
    f = {"extract": extract, "dedup": dedup, "lines": lines, "texts": texts, "without": without, "join": join,
         "bench": bench}[cmd]
    return f(a) or 0


if __name__ == "__main__":
    sys.exit(main())
