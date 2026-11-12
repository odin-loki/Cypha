#!/usr/bin/env python3
"""PARA_PLAN.md, PP amendment: the measures of the para-plan hook's screen. Odin Loch, 2026-11-12.

  python3 -I bench/para_plan/hook_screen.py --surprise PATH/surprise_profile.py --train /mnt/d/cypha_llm/fuzzy/train.txt \
      --out bench/para_plan/runs/screen L1=bench/k5_coherence/runs/L1.gen.txt PP1=... [--cut 460]

The first arm is the baseline. Every continuation is cut as blind_rate.py cuts it (near --cut characters, at a word).
Per arm, over the GEN-A dev prompts:
  para_plan_mine.py's window labels: the orphan rate (NONE -> PRON: the share of sentences after a sentence that does not
      refer to the subject that refer to it by a pronoun), the DEF + BIRTH share of sentences, on_prev (the sentence shares
      a content word with the one before), kept_s1..s4, run, pron_s2, same_func, new_share, tokens a sentence;
  coh_measures.py's stuff3 (3rd+ use of a content word) and top_word;
  WRITING_SIGNAL.md's surp_mean (scripts/surprise_profile.py at 2124363f: word trigram Kneser-Ney over --train, bits per
      word, the prompt's last two words the first context; texts cut and uncut as its gap set);
  samples unlike the baseline (whole GEN text).
Each arm minus the baseline with a 95% paired bootstrap interval over prompts (2,000 resamples, seed 20261112).
Writes OUT.json and OUT.md. It decides nothing.
"""
import argparse
import csv
import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="NAME=GEN_FILE (the first is the baseline)")
    ap.add_argument("--surprise", required=True, help="scripts/surprise_profile.py of wip/writing-signal (2124363f)")
    ap.add_argument("--train", default="/mnt/d/cypha_llm/fuzzy/train.txt")
    ap.add_argument("--prompts", default=os.path.join(ROOT, "bench/eval/gen_prompts_dev.tsv"))
    ap.add_argument("--cut", type=int, default=460)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ppm = load("ppm", os.path.join(ROOT, "scripts/para_plan_mine.py"))
    coh = load("coh", os.path.join(ROOT, "bench/k5_coherence/coh_measures.py"))
    sp = load("sp", a.surprise)
    with open(os.path.join(ROOT, "bench/para_plan/caps.json")) as f:
        ppm.CAPS.c = json.load(f)
    rows = {}
    with open(a.prompts, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)):
            rows[i] = r
    print(f"surprise model: {a.train}", file=sys.stderr)
    kn = sp.KN3(a.train)
    arms = []
    for s in a.arms:
        n, p = s.split("=", 1)
        arms.append((n, ppm.read_gen(p), sp.read_gen(p)))
    per = {}  # arm -> prompt -> per-prompt counts
    for name, g, graw in arms:
        d = {}
        for i in sorted(g):
            if i not in rows:
                continue
            w = ppm.window(ppm.split_sents(rows[i]["prompt"]), ppm.cont_sents_of(g[i], a.cut))
            rr = w["rows"]
            x = dict(n=len(rr), none_prev=0, orph=0, defb=sum(1 for r in rr if r["func"] in (0, 2)),
                     prev_n=max(len(rr) - 1, 0), prev_hit=sum(r["on_prev"] for r in rr[1:]), w=w)
            for j in range(1, len(rr)):
                if rr[j - 1]["ref"] == 3:
                    x["none_prev"] += 1
                    x["orph"] += rr[j]["ref"] == 1
            pos, h, n_, late, nt, top = coh.measure(rows[i]["prompt"], graw[i], a.cut)  # coh_measures.py reads the raw GEN text
            x.update(late=late, nt=nt, top=top)
            t = graw.get(i, "")
            if len(t.strip()) >= 50:
                pr = " ".join(rows[i]["prompt"].split())
                tx = sp.uncut(sp.cut(t))
                toks = sp.TOK.findall(tx)
                bits = kn.surprise(sp.TOK.findall(pr), toks)
                ws = [b for tk, b in zip(toks, bits) if sp.is_word(tk)]
                if len(ws) >= 3:
                    x["surp"] = sum(ws) / len(ws)
            x["gen"] = graw.get(i)
            d[i] = x
        per[name] = d

    def stats(d, ids):
        S = {}
        nn = sum(d[i]["none_prev"] for i in ids)
        S["orphan"] = sum(d[i]["orph"] for i in ids) / nn if nn else None
        ns = sum(d[i]["n"] for i in ids)
        S["def_birth"] = sum(d[i]["defb"] for i in ids) / ns if ns else None
        pn = sum(d[i]["prev_n"] for i in ids)
        S["on_prev"] = sum(d[i]["prev_hit"] for i in ids) / pn if pn else None
        nt = sum(d[i]["nt"] for i in ids)
        S["stuff3"] = sum(d[i]["late"] for i in ids) / nt if nt else None
        S["top_word"] = sum(d[i]["top"] for i in ids) / len(ids) if ids else None
        sv = [d[i]["surp"] for i in ids if "surp" in d[i]]
        S["surp_mean"] = sum(sv) / len(sv) if sv else None
        return S

    base = arms[0][0]
    res = {}
    rnd = random.Random(20261112)
    for name, g, _ in arms:
        d = per[name]
        ids = sorted(set(d) & set(per[base]))
        sc, ag = ppm.scalars([d[i]["w"] for i in ids])
        S = stats(d, ids)
        for k in ("kept_s1", "kept_s2", "kept_s3", "kept_s4", "run", "pron_s2", "same_func", "new_share", "ntok", "func_DEF", "func_BIRTH"):
            S[k] = sc[k]
        S["n_prompts"] = len(ids)
        S["unlike_base"] = sum(1 for i in ids if d[i]["gen"] != per[base][i]["gen"])
        S["nsent_mean"] = sum(d[i]["n"] for i in ids) / len(ids)
        if name != base:
            B = stats(per[base], ids)
            keys = ["orphan", "def_birth", "on_prev", "stuff3", "surp_mean"]
            samp = {k: [] for k in keys}
            for _ in range(a.boot):
                s = [rnd.choice(ids) for _ in ids]
                x, y = stats(d, s), stats(per[base], s)
                for k in keys:
                    if x[k] is not None and y[k] is not None:
                        samp[k].append(x[k] - y[k])
            S["vs_base"] = {}
            for k in keys:
                v = sorted(samp[k])
                S["vs_base"][k] = (S[k] - B[k], v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1])
        res[name] = S
    with open(a.out + ".json", "w") as f:
        json.dump({"cut": a.cut, "baseline": base, "arms": res}, f, indent=1)
    L = [f"# Para-plan hook screen measures (cut {a.cut}; baseline {base}; `bench/para_plan/hook_screen.py`)", "",
         "| arm | orphan (NONE→PRON) | DEF+BIRTH | on_prev | stuff3 | mean surprise (bits) | kept_s1 | kept_s2 | kept_s3 | kept_s4+ | run | sentences | unlike base |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, S in res.items():
        L.append(f"| {name} | {S['orphan']:.3f} | {S['def_birth']:.3f} | {S['on_prev']:.3f} | {S['stuff3']:.4f} | {S['surp_mean']:.3f} | "
                 f"{S['kept_s1']:.3f} | {S['kept_s2']:.3f} | {S['kept_s3']:.3f} | {S['kept_s4']:.3f} | {S['run']:.2f} | {S['nsent_mean']:.2f} | {S['unlike_base']} |")
    L += ["", f"Each arm minus {base} (95% paired bootstrap over prompts, {a.boot} resamples, seed 20261112):", "",
          "| arm | orphan | DEF+BIRTH | on_prev | stuff3 | mean surprise |", "|---|---|---|---|---|---|"]
    for name, S in res.items():
        if "vs_base" in S:
            v = S["vs_base"]
            f = lambda k, p=3: f"{v[k][0]:+.{p}f} [{v[k][1]:+.{p}f}, {v[k][2]:+.{p}f}]"
            L.append(f"| {name} | {f('orphan')} | {f('def_birth')} | {f('on_prev')} | {f('stuff3', 4)} | {f('surp_mean')} |")
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
