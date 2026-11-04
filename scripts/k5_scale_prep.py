#!/usr/bin/env python3
"""K5 on more Wikipedia (docs/reports/K5_SCALE.md): the overlap check and the nested training texts.

Reads SCRIBE's cleaned enwik9 documents (past byte 100,000,000), drops every document holding a 32-byte span of
the forbidden texts, shuffles the rest with a fixed seed, and writes nested prefixes as train texts (one document per
line, as fuzzy_bake.py's train.txt).

Forbidden texts (a document holding any of their 32-byte spans is dropped):
  - GEN-A's prompts and continuations, dev and test (bench/eval/gen_prompts_{dev,test}.tsv);
  - SCRIBE's reference and test texts: enwik8 [48, 64) MiB, cleaned by `scribe docs`.
Reported, not dropped: spans of SCRIBE's pack (enwik8 [16, 48) MiB) and of GEN-A's whole source region
(enwik8 [72 MiB, end)), both cleaned by `scribe docs`.

Spans are hashed (64-bit polynomial hash of 32 bytes); every hash hit is confirmed on the bytes.
After writing, the largest text is checked again from scratch (a second pass over the written file).

usage: k5_scale_prep.py CLEAN_E9.jsonl SCRIBE_REFTEST.jsonl SCRIBE_PACK.jsonl GENA_SRC.jsonl BENCH_DIR OUT_DIR
Author: Odin Loch
"""
import json
import os
import random
import sys
import time

import numpy as np

K = 32
MUL = np.uint64(0x100000001B3)
SEED = 20261104
# Nested sizes in bytes (UTF-8, newlines included). 7,053,155 is the size of the bake-off's enwik8 train.txt.
SIZES = [("7M", 7_053_155), ("30M", 30_000_000), ("100M", 100_000_000), ("300M", 300_000_000)]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


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


def read_docs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def span_hashes(buf):
    """64-bit hashes of every 32-byte window of the uint8 array buf (len(buf) - 31 of them)."""
    n = len(buf) - K + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    b = buf.astype(np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(K):
            h = h * MUL + b[j:j + n]
    return h


def hash_set(texts):
    hs = [span_hashes(np.frombuffer(t.encode("utf-8"), dtype=np.uint8)) for t in texts]
    hs = [h for h in hs if len(h)]
    return np.unique(np.concatenate(hs)) if hs else np.zeros(0, dtype=np.uint64)


def byte_spans(texts):
    s = set()
    for t in texts:
        b = t.encode("utf-8")
        s.update(b[i:i + K] for i in range(len(b) - K + 1))
    return s


def hits(blob, starts, forb_sorted, forb_bytes, chunk=1 << 26):
    """Documents of blob (docs joined by b'\\n'; starts = each doc's first byte) that hold a forbidden span,
    confirmed on the bytes. Returns {doc index: confirmed hit count}."""
    arr = np.frombuffer(blob, dtype=np.uint8)
    found = {}
    if len(forb_sorted) == 0:
        return found
    for a in range(0, len(arr), chunk):
        e = min(len(arr), a + chunk + K - 1)
        h = span_hashes(arr[a:e])
        idx = np.searchsorted(forb_sorted, h)
        idx[idx == len(forb_sorted)] = 0
        pos = np.nonzero(forb_sorted[idx] == h)[0] + a
        for p in pos.tolist():
            if blob[p:p + K] in forb_bytes:
                d = int(np.searchsorted(starts, p, side="right")) - 1
                found[d] = found.get(d, 0) + 1
    return found


def join(docs):
    enc = [d.encode("utf-8") for d in docs]
    starts = np.zeros(len(enc), dtype=np.int64)
    o = 0
    for i, b in enumerate(enc):
        starts[i] = o
        o += len(b) + 1
    return b"\n".join(enc), starts


def main():
    clean_e9, reftest, pack, gena_src, bench, out = sys.argv[1:7]
    os.makedirs(out, exist_ok=True)
    rep = {}
    gena = []
    for split in ("dev", "test"):
        with open(os.path.join(bench, f"gen_prompts_{split}.tsv"), encoding="utf-8") as f:
            for line in f.read().split("\n")[1:]:
                c = line.split("\t")
                if len(c) >= 4:
                    gena += [unesc(c[2]), unesc(c[3])]
    rt = read_docs(reftest)
    log(f"GEN-A texts {len(gena)}; SCRIBE ref+test docs {len(rt)}")
    forb = gena + rt
    forb_h, forb_b = hash_set(forb), byte_spans(forb)
    log(f"forbidden spans: {len(forb_b)} distinct")
    docs = read_docs(clean_e9)
    rep["docs_cleaned"] = len(docs)
    rep["bytes_cleaned"] = sum(len(d.encode("utf-8")) + 1 for d in docs)
    log(f"enwik9 cleaned: {len(docs)} docs, {rep['bytes_cleaned']} bytes")
    blob, starts = join(docs)
    # Which forbidden source each dropped document hits (GEN-A or SCRIBE), for the report.
    g_h, g_b = hash_set(gena), byte_spans(gena)
    r_h, r_b = hash_set(rt), byte_spans(rt)
    hg = hits(blob, starts, g_h, g_b)
    hr = hits(blob, starts, r_h, r_b)
    log(f"docs with a GEN-A span: {len(hg)}; with a SCRIBE ref/test span: {len(hr)}")
    rep["docs_hit_gena"], rep["spans_hit_gena"] = len(hg), sum(hg.values())
    rep["docs_hit_scribe_reftest"], rep["spans_hit_scribe_reftest"] = len(hr), sum(hr.values())
    drop = set(hg) | set(hr)
    rep["docs_dropped"] = len(drop)
    rep["bytes_dropped"] = sum(len(docs[i].encode("utf-8")) + 1 for i in drop)
    # Reported only: the pack and GEN-A's whole source region.
    for name, path in (("scribe_pack", pack), ("gena_source_region", gena_src)):
        t = read_docs(path)
        hx = hits(blob, starts, hash_set(t), byte_spans(t))
        rep[f"docs_hit_{name}"], rep[f"spans_hit_{name}"] = len(hx), sum(hx.values())
        log(f"(report only) docs with a span of {name}: {len(hx)} ({sum(hx.values())} spans)")
        del t
    del blob
    keep = [d for i, d in enumerate(docs) if i not in drop]
    del docs
    random.Random(SEED).shuffle(keep)
    rep["shuffle_seed"] = SEED
    total = 0
    sizes = {}
    marks = list(SIZES)
    path_all = os.path.join(out, "train_all.txt")
    files = {name: open(os.path.join(out, f"train_{name}.txt"), "w", encoding="utf-8", newline="\n") for name, _ in SIZES}
    with open(path_all, "w", encoding="utf-8", newline="\n") as fa:
        n = 0
        for d in keep:
            b = len(d.encode("utf-8")) + 1
            sep = "\n" if n else ""
            fa.write(sep + d)
            for name, cap in SIZES:
                if name in files:
                    if total + b <= cap or total == 0:
                        files[name].write(sep + d)
                        sizes.setdefault(name, [0, 0])
                        sizes[name][0] += 1
                        sizes[name][1] += b
                    else:
                        files.pop(name).close()
            total += b
            n += 1
    for f in files.values():
        f.close()
    sizes["all"] = [len(keep), total]
    rep["sizes"] = {k: {"docs": v[0], "bytes": v[1] - 1} for k, v in sizes.items()}  # the last line has no newline
    log("sizes", rep["sizes"])
    # The second pass: the written train_all.txt, read back, must hold no forbidden span.
    with open(path_all, "rb") as f:
        blob = f.read()
    lines = blob.split(b"\n")
    st = np.zeros(len(lines), dtype=np.int64)
    o = 0
    for i, ln in enumerate(lines):
        st[i] = o
        o += len(ln) + 1
    h2 = hits(blob, st, forb_h, forb_b)
    rep["recheck_docs_hit_forbidden"] = len(h2)
    log(f"recheck of train_all.txt: {len(h2)} docs with a forbidden span")
    with open(os.path.join(out, "prep_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
