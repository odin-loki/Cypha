#!/usr/bin/env python3
"""K5_CONTROL.md: the properties the decoding controller steers, measured on text as the raters read it. Odin Loch, 2026-11-13.

  python3 -I bench/k5_control/ctl_measures.py --cut 460 [--human] [--ids FILE.json] NAME=GEN_FILE ...

Per source, over the GEN-A dev prompts (the continuation cut near --cut characters at a word, as blind_rate.py cuts it;
a last piece cut mid-sentence dropped; para_plan_mine.py's sentence split):
  words   words a sentence (tokens of para_plan_mine.toks that hold a letter or digit), pooled over sentences;
  link    the share of sentences 2+ that share a content word with the sentence before (para_plan_mine.content: 4+
          letters, not on its stop list); hook_screen.py's on_prev;
  link1   the same for sentence 1 against the prompt's last sentence;
  short   the share of sentences under 8 words (a proxy for verbless fragments);
  per text: the median and the quartiles of each text's own mean words a sentence and own link share.
--human adds the human continuations of the dev file. With two or more sources, each source minus the first, paired
by prompt, with a 95% bootstrap interval (2,000 resamples, seed 20261213). It decides nothing.
"""
import argparse
import csv
import importlib.util
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ppm = load("ppm", os.path.join(ROOT, "scripts/para_plan_mine.py"))


def nwords(s):
    return sum(1 for t in ppm.toks(s) if any(ch.isalnum() for ch in t))


def per_text(prompt, text, cut):
    ps = ppm.split_sents(prompt)
    cs = ppm.cont_sents_of(text, cut)
    prev = set(ppm.content(ps[-1])) if ps else set()
    x = dict(n=len(cs), wsum=0, short=0, lhit=0, ln=0, l1=None)
    for j, s in enumerate(cs):
        w = nwords(s)
        c = set(ppm.content(s))
        x["wsum"] += w
        x["short"] += w < 8
        hit = int(bool(c & prev))
        if j == 0:
            x["l1"] = hit
        else:
            x["lhit"] += hit
            x["ln"] += 1
        prev = c
    return x


def pooled(d, ids):
    n = sum(d[i]["n"] for i in ids)
    ln = sum(d[i]["ln"] for i in ids)
    l1 = [d[i]["l1"] for i in ids if d[i]["l1"] is not None]
    return dict(words=sum(d[i]["wsum"] for i in ids) / n if n else None,
                link=sum(d[i]["lhit"] for i in ids) / ln if ln else None,
                link1=sum(l1) / len(l1) if l1 else None,
                short=sum(d[i]["short"] for i in ids) / n if n else None,
                sents=n / len(ids) if ids else None)


def quart(v):
    if len(v) < 4:
        return None
    q = statistics.quantiles(v, n=4)
    return [round(q[0], 3), round(statistics.median(v), 3), round(q[2], 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("srcs", nargs="*", help="NAME=GEN_FILE")
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--prompts", default=os.path.join(ROOT, "bench/eval/gen_prompts_dev.tsv"))
    ap.add_argument("--ids", help="a JSON list of prompt ids (blind_rate.py's prompts.json); default all")
    ap.add_argument("--cut", type=int, default=460)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = {}
    with open(a.prompts, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)):
            rows[i] = r
    want = None
    if a.ids:
        with open(a.ids) as f:
            j = json.load(f)
        want = set(int(x) for x in (j["prompts"] if isinstance(j, dict) else j))
    srcs = []
    if a.human:
        srcs.append(("human", {i: r["continuation"] for i, r in rows.items()}))
    for s in a.srcs:
        n, p = s.split("=", 1)
        srcs.append((n, ppm.read_gen(p)))
    per = {}
    for name, g in srcs:
        per[name] = {i: per_text(rows[i]["prompt"], g[i], a.cut) for i in g if i in rows and (want is None or i in want)}
    ids = sorted(set.intersection(*(set(d) for d in per.values())))
    out = {"cut": a.cut, "n_prompts": len(ids), "src": {}}
    rnd = random.Random(20261213)
    first = srcs[0][0]
    print(f"cut {a.cut}, {len(ids)} prompts")
    print(f"{'source':10} {'words':>6} {'link':>6} {'link1':>6} {'short':>6} {'sents':>6}  words/text q1,med,q3   link/text q1,med,q3")
    for name, _ in srcs:
        d = per[name]
        P = pooled(d, ids)
        wt = [d[i]["wsum"] / d[i]["n"] for i in ids if d[i]["n"]]
        lt = [d[i]["lhit"] / d[i]["ln"] for i in ids if d[i]["ln"]]
        P["words_text_q"] = quart(wt)
        P["link_text_q"] = quart(lt)
        if name != first:
            keys = ("words", "link", "short")
            samp = {k: [] for k in keys}
            B = pooled(per[first], ids)
            for _ in range(a.boot):
                s = [rnd.choice(ids) for _ in ids]
                x, y = pooled(d, s), pooled(per[first], s)
                for k in keys:
                    if x[k] is not None and y[k] is not None:
                        samp[k].append(x[k] - y[k])
            P["minus_first"] = {}
            for k in keys:
                v = sorted(samp[k])
                P["minus_first"][k] = [P[k] - B[k], v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]
        out["src"][name] = P
        mf = P.get("minus_first")
        extra = ""
        if mf:
            extra = "  vs " + first + ": " + ", ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in mf.items())
        print(f"{name:10} {P['words']:6.2f} {P['link']:6.3f} {P['link1'] if P['link1'] is not None else float('nan'):6.3f} "
              f"{P['short']:6.3f} {P['sents']:6.2f}  {P['words_text_q']}  {P['link_text_q']}{extra}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
