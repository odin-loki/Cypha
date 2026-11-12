#!/usr/bin/env python3
"""DECODE_LOOP.md: the loop's metrics, its screen and its ranking. Odin Loch, 2026-11-13.

  loop_measures.py feats LABEL [LABEL ...]       per-text features of RUNS/LABEL.gen.txt -> RUNS/LABEL.feats.json
  loop_measures.py feats human                   the same for bench/decode_loop/human.gen.txt
  loop_measures.py screen --best B [--rank r1,r2,..] CAND [CAND ...]
                                                 the round table, the screen and the ranking (DECODE_LOOP.md "Screen")
  loop_measures.py agree                         metric agreement with the raters over bench/decode_loop/rounds.json

Per-text features (every text cut as blind_rate.py cuts it, 460 characters):
  surp_mean, surp_sd   mean and spread of per-word surprise (WRITING_SIGNAL's word trigram KN over fuzzy/train.txt)
  sl_mean              words per sentence (surprise_profile.py's sentences)
  on_prev              share of sentences 2+ sharing a content word with the sentence before (para_plan_mine.py window)
  verbless             share of whole sentences with no finite verb by a fixed word list (VERBS) and two suffix rules
  stuff3               3rd+ uses of a content word over content words (coh_measures.py)
  commit_text, junk, rep3   surprise_profile.py's, reported
Arm-level, read from the run's files: SCRIBE v2 and v3.1 (subset A), fact precision (subset A), copy32 and
verbatim13 (all 150 prompts).
"""
import argparse
import importlib.util
import json
import math
import os
import random
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.environ.get("DLOOP_RUNS", os.path.expanduser("~/dloop_runs"))
SETS = json.load(open(os.path.join(HERE, "prompt_sets.json")))
A, R = SETS["A"], SETS["R"]
TRAIN = "/mnt/d/cypha_llm/fuzzy/train.txt"
CUT = 460


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sp = load("sp", os.path.join(ROOT, "scripts/surprise_profile.py"))
ppm = load("ppm", os.path.join(ROOT, "scripts/para_plan_mine.py"))
coh = load("coh", os.path.join(ROOT, "bench/k5_coherence/coh_measures.py"))

# Finite verbs: be, have, do, the modals, and common irregular and regular forms; plus two rules: a word of 5+ letters
# ending in "ed", and a word ending in "s" right after he / she / it / which / that / who / this.
VERBS = set("""is was are were be am 's has have had do does did will would can could may might shall should must
became becomes become began begins begin wrote writes write made makes make took takes take gave gives give found finds
led leads won wins left leaves held holds said says went goes came comes saw sees knew knows grew grows built builds
brought brings bought fought sought taught thought told tells kept keeps met meets ran runs rose rises fell falls lost
loses paid pays sent sends set sets put puts got gets stood stands sold sells spent spends struck won wore wears drew
draws drove drives flew flies spoke speaks broke breaks chose chooses lay lies laid sat sits stayed stays seems seem
includes include contains contain refers refer means mean uses use consists consist remains remain lives live dies die
appears appear occurs occur exists exist allows allow requires require provides provide represents represent shows
show describes describe produces produce involves involve forms form serves serve plays play works work comprises""".split())
SUBJ = {"he", "she", "it", "which", "that", "who", "this", "they"}
FEATS = ["surp_mean", "surp_sd", "sl_mean", "on_prev", "verbless", "stuff3", "commit_text", "junk", "rep3"]
HUMAN_RANGE = ["surp_mean", "surp_sd", "sl_mean", "on_prev", "verbless", "stuff3"]
# The registered good direction of every metric: +1 higher reads better, -1 lower, "h" nearer the human mean.
DIRECTION = {"surp_mean": -1, "surp_sd": "h", "sl_mean": "h", "on_prev": +1, "verbless": -1, "stuff3": -1,
             "commit_text": +1, "junk": -1, "rep3": -1, "scribe_v2": +1, "scribe_v31": +1, "fact_precision": +1,
             "copy32": -1}
INITIAL_RANK = ["surp_mean", "scribe_v2", "stuff3", "verbless"]


def verbless(sentences):
    n = v = 0
    for s in sentences:
        w = [t for t in sp.TOK.findall(s) if sp.is_word(t)]
        if len(w) < 2:
            continue
        n += 1
        lw = [x.lower() for x in w]
        has = any(x in VERBS for x in lw) or any(len(x) >= 5 and x.endswith("ed") for x in lw) or \
            any(x.endswith("s") and lw[i - 1] in SUBJ for i, x in enumerate(lw) if i > 0)
        v += not has
    return v / n if n else None


def text_feats(kn, prompt, text):
    prompt = " ".join(prompt.split())
    tx = sp.uncut(sp.cut(text, CUT))
    f = sp.raw_features(kn, prompt, tx)
    out = {k: f.get(k) for k in ("surp_mean", "surp_sd", "sl_mean", "commit_text", "junk", "rep3")}
    ss = ppm.cont_sents_of(text, CUT)
    w = ppm.window(ppm.split_sents(prompt), ss)
    rr = w["rows"]
    out["on_prev"] = sum(r["on_prev"] for r in rr[1:]) / (len(rr) - 1) if len(rr) > 1 else None
    out["verbless"] = verbless(ss)
    _, _, _, late, nt, _ = coh.measure(prompt, text, CUT)
    out["stuff3"] = late / nt if nt else None
    out["n_sent"] = len(ss)
    return out


def read_prompts():
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, "bench/eval/gen_prompts_dev.tsv"), encoding="utf-8"),
                               delimiter="\t", quoting=csv.QUOTE_NONE))
    return {i: r["prompt"] for i, r in enumerate(rows)}


def gen_path(label):
    return os.path.join(HERE, "human.gen.txt") if label == "human" else os.path.join(RUNS, f"{label}.gen.txt")


def cmd_feats(a):
    kn = sp.KN3(TRAIN)
    pr = read_prompts()
    for label in a.labels:
        g = sp.read_gen(gen_path(label))
        per = {}
        for i in A + R:
            t = g.get(i, "")
            per[i] = text_feats(kn, pr[i], t) if len(t.strip()) >= 20 else {}
            per[i]["text"] = sp.cut(t, CUT)
        arm = arm_level(label) if label != "human" else {}
        json.dump({"label": label, "per": per, "arm": arm}, open(os.path.join(RUNS, f"{label}.feats.json"), "w"))
        print(f"{label}: features of {len(per)} texts", file=sys.stderr)


def arm_level(label):
    out = {}
    try:
        s = open(os.path.join(RUNS, f"{label}.scribe.md")).read()
        m = re.search(r"\| trie samples \(pooled\) \| \d+ \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", s)
        out["scribe_v2"], out["scribe_v2_lo"], out["scribe_v2_hi"] = map(float, m.groups())
    except (OSError, AttributeError):
        pass
    try:
        s = open(os.path.join(RUNS, f"{label}.scribe31.md")).read()
        m = re.search(r"\| trie samples \(pooled\) \| \d+ \| ([0-9.]+)", s)
        out["scribe_v31"] = float(m.group(1))
    except (OSError, AttributeError):
        pass
    try:
        s = open(os.path.join(RUNS, f"{label}.sample.out")).read()
        m = re.search(r"copy32 \S+ ([0-9.]+) verbatim13 ([0-9.]+) \((\d+) of (\d+)\)", s)
        out["copy32"], out["verbatim13_n"], out["n_samples"] = float(m.group(1)), int(m.group(3)), int(m.group(4))
    except (OSError, AttributeError):
        pass
    try:
        d = json.load(open(os.path.join(RUNS, f"facts_{label}", "summary.json")))["arms"][label]
        out["fact_precision"] = d["precision"]
        for k in ("precision_lo", "precision_hi", "ci_precision"):
            if k in d:
                out[k] = d[k]
    except (OSError, KeyError):
        pass
    return out


def feats(label):
    return json.load(open(os.path.join(RUNS, f"{label}.feats.json")))


def mean_on(per, ids, k):
    v = [per[str(i)][k] for i in ids if per[str(i)].get(k) is not None]
    return st.mean(v) if v else None


def human_ranges():
    h = feats("human")["per"]
    out = {}
    for k in FEATS:
        v = [h[str(i)][k] for i in A if h[str(i)].get(k) is not None]
        out[k] = (st.mean(v), st.pstdev(v))
    return out


def gap(x, hr, k):
    mu, sd = hr[k]
    return abs(x - mu) / sd if sd else 0.0


def cmd_screen(a):
    hr = human_ranges()
    labs = [a.best] + a.cands
    F = {lab: feats(lab) for lab in labs}
    rank_by = a.rank.split(",") if a.rank else INITIAL_RANK
    rows = {}
    for lab in labs:
        per, arm = F[lab]["per"], F[lab]["arm"]
        r = {k: mean_on(per, A, k) for k in FEATS}
        r.update(arm)
        r["unlike_best_R"] = sum(1 for i in R if per[str(i)]["text"] != F[a.best]["per"][str(i)]["text"])
        rows[lab] = r
    B = rows[a.best]
    print(f"# Round table (subset A, {len(A)} prompts; best {a.best}; human mean ± sd per text)\n")
    print("| metric | human | " + " | ".join(labs) + " |")
    print("|---|---|" + "---|" * len(labs))
    for k in FEATS:
        h = f"{hr[k][0]:.3f} ± {hr[k][1]:.3f}"
        print(f"| {k} | {h} | " + " | ".join(f"{rows[l][k]:.3f}" if rows[l][k] is not None else "–" for l in labs) + " |")
    for k in ("scribe_v2", "scribe_v31", "fact_precision", "copy32", "verbatim13_n", "unlike_best_R"):
        print(f"| {k} | | " + " | ".join(str(round(rows[l].get(k), 3)) if rows[l].get(k) is not None else "–" for l in labs) + " |")
    print("\n## Screen\n")
    passed = []
    for lab in a.cands:
        r, why = rows[lab], []
        if r.get("verbatim13_n", 1) != 0:
            why.append(f"verbatim13 {r.get('verbatim13_n')}")
        if r.get("fact_precision") is None or not r["fact_precision"] >= 0.874:
            why.append(f"fact precision {r.get('fact_precision')}")
        for k in HUMAN_RANGE:
            if r[k] is None or B[k] is None:
                continue
            d = gap(r[k], hr, k) - gap(B[k], hr, k)
            if d > 0.5:
                why.append(f"{k} {d:+.2f} human sd further from human")
        print(f"- {lab}: " + ("pass" if not why else "FAIL (" + "; ".join(why) + ")"))
        if not why:
            passed.append(lab)
    print(f"\n## Ranking by {', '.join(rank_by)} (rank sum, lower is better; ties: lower copy32)\n")
    if not passed:
        print("no candidate passed")
        return
    rs = {lab: 0.0 for lab in passed}
    for k in rank_by:
        d = DIRECTION[k.lstrip("-")]
        sign = -1 if k.startswith("-") else 1
        def key(lab):
            x = rows[lab][k.lstrip("-")]
            if d == "h":
                return gap(x, hr, k.lstrip("-"))
            return -d * sign * x
        order = sorted(passed, key=key)
        for j, lab in enumerate(order):
            rs[lab] += j
    final = sorted(passed, key=lambda lab: (rs[lab], rows[lab].get("copy32", 1)))
    for lab in final:
        print(f"- {lab}: rank sum {rs[lab]:.0f}")
    print(f"\nkept for the readers (at most 2): {', '.join(final[:2])}")


def boot_ci(clusters, reps=2000, seed=20261210):
    """clusters: list of lists of 0/0.5/1 agreement values; mean over judgments, bootstrap over clusters."""
    rng = random.Random(seed)
    flat = [x for c in clusters for x in c]
    if not flat:
        return None, None, None, 0
    bs = []
    for _ in range(reps):
        s = [x for c in rng.choices(clusters, k=len(clusters)) for x in c]
        bs.append(st.mean(s) if s else 0.5)
    bs.sort()
    return st.mean(flat), bs[int(0.025 * reps)], bs[int(0.975 * reps) - 1], len(flat)


def cmd_agree(a):
    """For every rated pair (arm text vs best text, one rater's choice, not a tie, texts differ): does each metric,
    read in its registered direction, pick the rater's choice? Per-text metrics per judgment (95% bootstrap over
    prompt clusters); arm-level metrics by the sign of the arm's difference on subset A against the round's win share."""
    log = json.load(open(os.path.join(HERE, "rounds.json")))
    hr = human_ranges()
    judg = {k: {} for k in FEATS}
    armlvl = {k: [0, 0] for k in ("scribe_v2", "scribe_v31", "fact_precision", "copy32") + tuple(FEATS)}
    for rnd in log["rounds"]:
        if "judgments" not in rnd:
            continue
        best = rnd["best"]
        Fb = feats(best)
        for arm, share in rnd.get("shares", {}).items():
            Fa = feats(arm)
            for k in armlvl:
                xa = Fa["arm"].get(k) if k not in FEATS else mean_on(Fa["per"], A, k)
                xb = Fb["arm"].get(k) if k not in FEATS else mean_on(Fb["per"], A, k)
                if xa is None or xb is None or xa == xb or share == 0.5:
                    continue
                d = DIRECTION[k]
                better = (gap(xa, hr, k) < gap(xb, hr, k)) if d == "h" else ((xa - xb) * d > 0)
                armlvl[k][0] += better == (share > 0.5)
                armlvl[k][1] += 1
        for j in rnd["judgments"]:  # {"dev": i, "arm": X, "rater": r, "win": 1|0|0.5}
            if j["win"] == 0.5:
                continue
            pa, pb = feats(j["arm"])["per"][str(j["dev"])], Fb["per"][str(j["dev"])]
            if pa["text"] == pb["text"]:
                continue
            for k in FEATS:
                xa, xb = pa.get(k), pb.get(k)
                if xa is None or xb is None:
                    continue
                d = DIRECTION[k]
                if d == "h":
                    sa, sb = -gap(xa, hr, k), -gap(xb, hr, k)
                else:
                    sa, sb = d * xa, d * xb
                v = 0.5 if sa == sb else float((sa > sb) == (j["win"] == 1))
                judg[k].setdefault((rnd["name"], j["dev"]), []).append(v)
    print("# Metric agreement with the raters (registered directions)\n")
    print("| metric | judgments | agreement [95%] | arm-level sign agreement |")
    print("|---|---|---|---|")
    res = {}
    for k in list(FEATS) + ["scribe_v2", "scribe_v31", "fact_precision", "copy32"]:
        m, lo, hi, n = boot_ci(list(judg[k].values())) if k in judg else (None, None, None, 0)
        al = armlvl.get(k, [0, 0])
        res[k] = dict(agree=m, lo=lo, hi=hi, n=n, arm_hits=al[0], arm_n=al[1])
        js = f"{m:.3f} [{lo:.3f}, {hi:.3f}]" if m is not None else "–"
        print(f"| {k} | {n} | {js} | {al[0]} of {al[1]} |")
    json.dump(res, open(os.path.join(HERE, "agree.json"), "w"), indent=1)
    trust = []
    for k, r in res.items():
        if r["lo"] is not None and r["lo"] > 0.5:
            trust.append(k)
        elif r["hi"] is not None and r["hi"] < 0.5:
            trust.append("-" + k)
        elif r["n"] == 0 and r["arm_n"] >= 5 and r["arm_hits"] / r["arm_n"] >= 0.8:
            trust.append(k)
        elif r["n"] == 0 and r["arm_n"] >= 5 and r["arm_hits"] / r["arm_n"] <= 0.2:
            trust.append("-" + k)
    print(f"\nmetrics that track the raters by the registered rule: {', '.join(trust) or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("feats")
    f.add_argument("labels", nargs="+")
    s = sub.add_parser("screen")
    s.add_argument("--best", required=True)
    s.add_argument("--rank", default=None)
    s.add_argument("cands", nargs="+")
    sub.add_parser("agree")
    a = ap.parse_args()
    {"feats": cmd_feats, "screen": cmd_screen, "agree": cmd_agree}[a.cmd](a)


if __name__ == "__main__":
    main()
