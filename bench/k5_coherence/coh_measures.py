#!/usr/bin/env python3
"""K5_COHERENCE.md: does a continuation stay on its prompt's subject, and does it stuff keywords? Odin Loch, 2026-11-12.

  python3 -I bench/k5_coherence/coh_measures.py [--cut 460] [--ids FILE] NAME=GEN_FILE ... [--human]

Each continuation is cut as blind_rate.py cuts it for the raters (near --cut characters, at a word). Content words:
lower case, letters only, at least 4 letters, not in a short list of function words. Per arm, over the prompts:
  on_prompt s1 / s2 / s3 / s4+  share of the continuation's sentences at that position holding at least one content
                                word of the prompt (s1 is the first sentence);
  on_prev                       share of sentences 2 and later holding a content word of the sentence before;
  stuff3                        share of content-word tokens that are a word's third or later use in the continuation
                                (keyword repetition: topic stuffing raises it);
  top_word                      mean, over continuations, of the most used content word's share of content tokens.
None of these decides anything. They are read beside the raters and my own reading.
"""
import argparse
import csv
import json
import re
import sys

STOP = set("""
that this with from were have which their there they been when also into than then them these those other about after
before would could should while where being because some such more most many much over under only very what will your
first second three during between within without through against among upon each both either well just like even
""".split())


def cut(t, n=460):
    t = " ".join(t.split())
    if len(t) <= n:
        return t
    t = t[:n]
    t = t[: t.rfind(" ")]
    return t.rstrip(" ,;:") + " …"


def words(t):
    return [w for w in re.findall(r"[A-Za-z]+", t.lower()) if len(w) >= 4 and w not in STOP]


def sents(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


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


def measure(prompt, cont, n):
    pw = set(words(prompt))
    ss = sents(cut(cont, n))
    pos = {}
    prev_hits = prev_n = 0
    for i, s in enumerate(ss):
        w = set(words(s))
        k = min(i, 3)
        pos.setdefault(k, []).append(1 if w & pw else 0)
        if i > 0:
            prev_n += 1
            prev_hits += 1 if w & set(words(ss[i - 1])) else 0
    toks = words(cut(cont, n))
    seen, late = {}, 0
    for w in toks:
        seen[w] = seen.get(w, 0) + 1
        late += seen[w] >= 3
    top = max(seen.values()) / len(toks) if toks else 0.0
    return pos, prev_hits, prev_n, late, len(toks), top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="NAME=GEN_FILE")
    ap.add_argument("--cut", type=int, default=460)
    ap.add_argument("--ids", default=None, help="JSON list of dev prompt numbers (default: every prompt in the files)")
    ap.add_argument("--prompts", default="bench/eval/gen_prompts_dev.tsv")
    ap.add_argument("--human", action="store_true")
    a = ap.parse_args()
    rows = {}
    with open(a.prompts, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)):
            rows[i] = r
    arms = [(s.split("=", 1)[0], read_gen(s.split("=", 1)[1])) for s in a.arms]
    if a.human:
        arms.append(("human", {i: r["continuation"] for i, r in rows.items()}))
    ids = None
    if a.ids:
        j = json.load(open(a.ids))
        ids = j["prompts"] if isinstance(j, dict) else j
    print(f"| arm | n | on_prompt s1 | s2 | s3 | s4+ | on_prev | stuff3 | top_word |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, g in arms:
        use = [i for i in (ids if ids is not None else sorted(g)) if i in g and i in rows]
        P = {k: [] for k in range(4)}
        ph = pn = late = nt = 0
        tops = []
        for i in use:
            pos, h, n_, l, t, top = measure(rows[i]["prompt"], g[i], a.cut)
            for k, v in pos.items():
                P[k] += v
            ph, pn, late, nt = ph + h, pn + n_, late + l, nt + t
            tops.append(top)
        f = lambda v: f"{sum(v) / len(v):.3f}" if v else "–"
        print(f"| {name} | {len(use)} | {f(P[0])} | {f(P[1])} | {f(P[2])} | {f(P[3])} | {ph / max(pn, 1):.3f} | {late / max(nt, 1):.3f} | "
              f"{sum(tops) / max(len(tops), 1):.3f} |")


if __name__ == "__main__":
    sys.exit(main())
