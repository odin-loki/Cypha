"""Results driver for the SCRIBE v3 paper (docs/reports/SCRIBE_V3_PAPER.md): every re-run analysis, v2, v3 and v3.1 side by side.

Author: Odin Loch

Reads the files written by scripts/scribe_eval/runs/v3_paper.sh under <sp>/run/v3p and writes, for each analysis,
<out>/<name>.md, .csv and .json through meta.py (the v2 paper's statistics, unchanged), plus a few tables computed here
(paired attack wins, the combination ablation, K5, stability, cost), which go to <out>/v3_extra.md and .json.

    python scripts/scribe_eval/v3_report.py [--sp SP] [--out paper/scribe/results_v2.0]
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import meta  # noqa: E402  (first: it limits BLAS threads)
import report  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402

REPO = HERE.parents[1]
DEFAULT_SP = os.environ.get("SCRIBE_SP", "/tmp/claude-0/-home-user-Cypha/c3804d0f-86ab-5ff6-9437-45a3d11e8017/scratchpad")

METRICS = [("scribe_v2", "SCRIBE v2"), ("scribe_v3", "SCRIBE v3"), ("scribe_v31", "SCRIBE v3.1"),
           ("surprise", "SURPRISE"), ("match", "MATCH"), ("surprise31", "SURPRISE 3.1"), ("match31", "MATCH 3.1"),
           ("min3", "min of 3")]
LADDER = ["human", "sent_shuffled", "word_shuffled", "random_lexicon", "random_letters", "repeated_phrase"]


def metric_args(names=None):
    out = []
    for n, lab in METRICS:
        if names is None or n in names:
            out += ["--metric", f"{n}:+:{lab}"]
    return out


def run_meta(argv, out):
    argv = list(argv) + ["--out", str(out)]
    rc = meta.main(argv, quiet=True)
    print(f"wrote {out.name}", flush=True)
    return rc


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def systems_of(path):
    seen = []
    for r in meta.read_jsonl(path):
        if r["system"] not in seen:
            seen.append(r["system"])
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp", default=DEFAULT_SP)
    ap.add_argument("--out", default=str(REPO / "paper" / "scribe" / "results_v2.0"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    sp = Path(a.sp)
    O, D = sp / "run" / "v3p", sp / "data"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = sp / "run" / "v3p_work"
    work.mkdir(parents=True, exist_ok=True)
    want = (lambda n: True) if not a.only else (lambda n: n in a.only)
    extra = {}

    # ladders exported by `scribe ladder --export`
    lad = ["--systems", ",".join(LADDER)]
    for name, d in [("fresh3_ladder", O / "fresh" / "ladder"), ("genre_web", O / "genre" / "web"),
                    ("genre_books", O / "genre" / "books"), ("genre_web_wikiref", O / "genre" / "web_wikiref"),
                    ("stab_base_ladder", O / "stab" / "base")]:
        if want(name) and (d / "scores.jsonl").exists():
            run_meta(["ladder", "--texts", str(d / "texts.jsonl"), "--scores", str(d / "scores.jsonl")] + metric_args() + lad,
                     out / f"{name}.md")

    # GPT-2 output dataset
    G = O / "gpt2out"
    if want("gpt2out") and (G / "samples.v31.jsonl").exists():
        pure = ["human", "xl-1542M", "large-762M", "medium-345M", "small-117M"]
        k40 = ["human"] + [s + "-k40" for s in pure[1:]]
        for tag, order in (("pure", pure), ("k40", k40)):
            run_meta(["models", "--texts", str(G / "samples.jsonl"), "--scores", str(G / "samples.v31.jsonl"),
                      "--systems", ",".join(order)] + metric_args(), out / f"gpt2out_{tag}.md")
        extra["gpt2_human_vs"] = paired_free_acc(G / "samples.jsonl", G / "samples.v31.jsonl", "human")

    # dose-response
    C = O / "dose"
    if want("dose") and (C / "corrupt.v31.jsonl").exists():
        run_meta(["dose", "--texts", str(C / "corrupt.jsonl"), "--scores", str(C / "corrupt.v31.jsonl")] + metric_args(),
                 out / "dose.md")

    # human ratings
    H = O / "human"
    for mode in ("open", "prompt"):
        if want("hanna") and (H / f"hanna.{mode}.jsonl").exists():
            t = work / "hanna_texts.jsonl"
            write_jsonl(t, [report.hanna_ratings(r) for r in meta.read_jsonl(D / "hanna" / "texts.jsonl")])
            run_meta(["corr", "--texts", str(t), "--scores", str(H / f"hanna.{mode}.jsonl"), "--criteria", "RE,CH,EM,SU,EG,CX",
                      "--human-system", "human"] + metric_args(), out / f"hanna_{mode}.md")
        for c in ("roc", "wp"):
            if want("mans") and (H / f"mans_{c}.{mode}.jsonl").exists():
                t = work / f"mans_{c}_texts.jsonl"
                write_jsonl(t, [report.openmeva_ratings(r) for r in meta.read_jsonl(D / "openmeva" / f"mans_{c}.jsonl")])
                run_meta(["corr", "--texts", str(t), "--scores", str(H / f"mans_{c}.{mode}.jsonl"), "--criteria", "overall"]
                         + metric_args(), out / f"mans_{c}_{mode}.md")
        if want("mauve") and (H / f"mauve_he.{mode}.jsonl").exists():
            prs = meta.read_jsonl(D / "mauve_he" / "pairs.jsonl")
            prs = report.attach_te(prs, sp / "dl" / "ds" / "mauve_he" / "mauve_human_eval.csv")
            p = work / "mauve_pairs_te.jsonl"
            write_jsonl(p, prs)
            run_meta(["pairs", "--texts", str(D / "mauve_he" / "texts.jsonl"), "--scores", str(H / f"mauve_he.{mode}.jsonl"),
                      "--pairs", str(p)] + metric_args(), out / f"mauve_he_{mode}_te25.md")
            run_meta(["pairs", "--texts", str(D / "mauve_he" / "texts.jsonl"), "--scores", str(H / f"mauve_he.{mode}.jsonl"),
                      "--pairs", str(D / "mauve_he" / "pairs.jsonl"), "--min-time", "0"] + metric_args(),
                     out / f"mauve_he_{mode}_all.md")

    # n-gram ladder
    M = O / "models"
    if want("ngram") and (M / "ngram.v31.jsonl").exists():
        run_meta(["models", "--texts", str(M / "ngram.open.jsonl"), "--scores", str(M / "ngram.v31.jsonl"),
                  "--systems", "human,ngram5,ngram4,ngram3,ngram2,ngram1"] + metric_args(), out / "ngram_ladder.md")

    # Dolly
    Q = O / "qa"
    if want("qa") and (Q / "test.v31.jsonl").exists():
        sc = work / "qa_scores.jsonl"
        v2p = {r["id"]: r["scribe_v2"] for r in meta.read_jsonl(Q / "test.v2p.jsonl")}
        rows = []
        for r in meta.read_jsonl(Q / "test.v31.jsonl"):
            r = {k: v for k, v in r.items() if not isinstance(v, dict)}
            r["scribe_v2p"] = v2p[r["id"]]
            rows.append(r)
        write_jsonl(sc, rows)
        run_meta(["ladder", "--texts", str(sp / "run" / "qa" / "test.jsonl"), "--scores", str(sc),
                  "--systems", "correct,other,echo,shuffled", "--metric", "scribe_v2p:+:SCRIBE v2p"] + metric_args(),
                 out / "qa_ladder.md")

    # v2's attack set: paired wins, as in the v2 paper's Table 9
    At = O / "attacks"
    if want("attacks") and (At / "attacks.v31.jsonl").exists():
        extra["attacks_v2set"] = paired_attacks(At / "attacks.v31.jsonl")

    # R1: the fresh battery, paired, and the combination ablation
    F = O / "fresh"
    if want("fresh") and (F / "att" / "scores.jsonl").exists():
        extra["fresh3_attacks"] = paired_attacks(F / "att" / "scores.jsonl")
        extra["fresh3_ablation"] = ablation(F / "ladder" / "scores.jsonl", F / "att" / "scores.jsonl")

    # length
    S = O / "stab"
    if want("length") and (S / "base" / "scores.jsonl").exists():
        run_meta(["length", "--texts", str(S / "base" / "texts.jsonl"), "--scores", str(S / "base" / "scores.jsonl"),
                  "--where", "system=human|human_short", "--by", "system"] + metric_args(), out / "length_wiki.md")

    # stability, K5, cost: from the tool's own reports
    if want("stability") and S.exists():
        extra["stability"] = {p.stem: parse_v3_pairs(p) for p in sorted(S.glob("*.md"))}
        extra["pack_cost"] = {p.stem: open(p).read().strip().splitlines()[-1] for p in sorted(S.glob("*.time"))}
    K = O / "k5"
    if want("k5") and K.exists():
        extra["k5"] = {p.stem: parse_model_rung(p) for p in sorted(K.glob("*.md"))}
    if "k5" in extra:
        extra["k5_means"] = k5_means(extra["k5"])
    Co = O / "cost"
    if want("cost") and Co.exists():
        extra["cost"] = {p.stem: parse_cost(p) for p in sorted(Co.glob("*.err"))}

    if extra:
        old = {}
        jp = out / "v3_extra.json"
        if jp.exists():
            old = json.loads(jp.read_text())
        old.update(extra)
        jp.write_text(json.dumps(meta.jsonable(old), indent=1, ensure_ascii=False))
        write_extra_md(old, out / "v3_extra.md")
        print("wrote v3_extra.md / .json")


# ---------------------------------------------------------------------------

def load_scores(path):
    return {r["id"]: r for r in meta.read_jsonl(path)}


def paired_attacks(scores_path):
    """Per attack: mean of each metric, and the share of prompts where the true continuation beats the attack (ties 1/2).
    Ids are <label>/<attack>/<k> (scribe attacks) or attack/<name>/<k> (attacks.py)."""
    sc = meta.read_jsonl(scores_path)
    by = {}
    for r in sc:
        parts = r["id"].split("/")
        by.setdefault(parts[-2], {})[parts[-1]] = r
    hum = by.get("human") or by.get("true")
    res = {}
    for att, rows in by.items():
        row = {"n": len(rows)}
        for m, _ in METRICS:
            vals = [x[m] for x in rows.values() if m in x and x[m] is not None]
            row[m] = float(np.mean(vals)) if vals else None
            if att != "human" and hum:
                w = [1.0 if hum[k][m] > rows[k][m] else 0.5 if hum[k][m] == rows[k][m] else 0.0 for k in rows if k in hum]
                row[m + "_win"] = float(np.mean(w)) if w else None
        res[att] = row
    return res


def auc(u, l):
    u, l = np.asarray(u, float), np.asarray(l, float)
    return float(meta.auc(u, l))


def ablation(ladder_scores, att_scores):
    """v3.1 with one judge left out, the plain mean of the v3.1 judges, and their minimum."""
    def combos(r):
        a, s, m = r["scribe_v2"], r["surprise31"], r["match31"]
        return {"v3.1": r["scribe_v31"], "no v2": math.sqrt(s * m), "no SURPRISE 3.1": math.sqrt(a * m),
                "no MATCH 3.1": math.sqrt(a * s), "plain mean": (a + s + m) / 3, "minimum": min(a, s, m)}
    out = {"ladder": {}, "attacks": {}}
    lad = {}
    for r in meta.read_jsonl(ladder_scores):
        lad.setdefault(r["system"], []).append(combos(r))
    names = list(next(iter(lad.values()))[0].keys())
    for c in names:
        accs = []
        for u, l in zip(LADDER, LADDER[1:]):
            accs.append(auc([x[c] for x in lad[u]], [x[c] for x in lad[l]]))
        out["ladder"][c] = accs
    by = {}
    for r in meta.read_jsonl(att_scores):
        p = r["id"].split("/")
        by.setdefault(p[-2], {})[p[-1]] = combos(r)
    hum = by["human"]
    for att, rows in by.items():
        if att == "human":
            continue
        # all (human, attack) pairs, ties 1/2: the accuracy of the batteries' registered rule
        out["attacks"][att] = {c: auc([h[c] for h in hum.values()], [x[c] for x in rows.values()]) for c in names}
    return out


def paired_free_acc(texts, scores, human):
    """P(human text > system text) over all pairs, per metric and system."""
    sc = load_scores(scores)
    sysv = {}
    for r in meta.read_jsonl(texts):
        s = sc.get(r["id"])
        if s:
            sysv.setdefault(r["system"], []).append(s)
    res = {}
    for name, rows in sysv.items():
        if name == human:
            continue
        res[name] = {m: auc([x[m] for x in sysv[human]], [x[m] for x in rows]) for m, _ in METRICS}
    return res


def parse_v3_pairs(path):
    """From a `scribe ladder --profile v31` report: the 'v3 parts' control-ladder table and the v3 rung table."""
    txt = path.read_text()
    res = {"pairs": {}, "means": {}}
    m = re.search(r"### v3 parts: control ladder.*?\n\n(\|.*?)\n\n", txt, re.S)
    if m:
        lines = [l for l in m.group(1).splitlines() if l.startswith("| ") and not l.startswith("| upper")]
        for l in lines:
            c = [x.strip() for x in l.strip("|").split("|")]
            res["pairs"][c[0]] = {"v2": c[1], "SURPRISE": c[3], "MATCH": c[5], "v3": c[7], "min3": c[9],
                                  "SURPRISE 3.1": c[11], "MATCH 3.1": c[13], "v3.1": c[15],
                                  "pass": {"v2": c[2], "v3": c[8], "v3.1": c[16]}}
    m = re.search(r"\| rung \| n \| v2 \| SURPRISE \| MATCH \| v3 \|.*?\n\|[-|]+\n(.*?)\n\n", txt, re.S)
    if m:
        for l in m.group(1).splitlines():
            c = [x.strip() for x in l.strip("|").split("|")]
            res["means"][c[0]] = {"v2": c[2], "v3": c[5], "v3.1": c[9], "SURPRISE 3.1": c[7], "MATCH 3.1": c[8]}
    return res


def parse_model_rung(path):
    """The trie (K5) rung's means in the v3 table of a `scribe ladder --profile v31 --gen-trie` report."""
    txt = path.read_text()
    res = {}
    m = re.search(r"\| rung \| n \| v2 \| SURPRISE \| MATCH \| v3 \|.*?\n\|[-|]+\n(.*?)\n\n", txt, re.S)
    if m:
        for l in m.group(1).splitlines():
            c = [x.strip() for x in l.strip("|").split("|")]
            res[c[0]] = {"n": c[1], "v2": c[2], "SURPRISE": c[3], "MATCH": c[4], "v3": c[5], "min3": c[6],
                         "SURPRISE 3.1": c[7], "MATCH 3.1": c[8], "v3.1": c[9], "disagree": c[10]}
    return res


def k5_means(k5):
    """Mean over the saved seeds of each K5 arm (E8, NALL, U8, C1: 4 seeds; the others 1), trie rung."""
    arms = {}
    for run, r in k5.items():
        arm = run.split("_s")[0]
        c = r.get("trie samples (pooled)")
        if c:
            arms.setdefault(arm, []).append(c)
    out = {}
    for arm, rows in arms.items():
        out[arm] = {"seeds": len(rows)}
        for k in ("v2", "SURPRISE", "MATCH", "v3", "min3", "SURPRISE 3.1", "MATCH 3.1", "v3.1"):
            out[arm][k] = float(np.mean([float(x[k].split()[0]) for x in rows]))
    return out


def parse_cost(err_path):
    txt = err_path.read_text()
    t = re.search(r"TIMING (\{.*\})", txt)
    mt = re.search(r"MEMTIME (.*)", txt)
    return {"timing": json.loads(t.group(1)) if t else None, "memtime": mt.group(1) if mt else None}


def f(v, nd=3):
    return "" if v is None else f"{v:.{nd}f}"


def write_extra_md(x, path):
    L = ["# SCRIBE v3 paper: extra tables", "", "Written by `scripts/scribe_eval/v3_report.py` from the runs of `scripts/scribe_eval/runs/v3_paper.sh`.", ""]
    for key, title in (("fresh3_attacks", "R1 fresh battery, enwik9 [900, 924) MiB"), ("attacks_v2set", "R10 v2's attack set")):
        if key in x:
            L += [f"## {title}: mean, and P(human > attack) over prompts (ties 1/2)", ""]
            ms = [m for m, _ in METRICS]
            L += ["| attack | n | " + " | ".join(lab for _, lab in METRICS) + " |", "|---|---|" + "---|" * len(ms)]
            for att, r in x[key].items():
                cells = [f(r.get(m)) + (f" ({f(r.get(m + '_win'))})" if r.get(m + "_win") is not None else "") for m in ms]
                L.append(f"| {att} | {r['n']} | " + " | ".join(cells) + " |")
            L.append("")
    if "fresh3_ablation" in x:
        ab = x["fresh3_ablation"]
        names = list(ab["ladder"].keys())
        L += ["## R1 combination ablation", "", "Ladder, adjacent-pair accuracy:", "",
              "| combination | " + " | ".join(f"{u} > {l}" for u, l in zip(LADDER, LADDER[1:])) + " |", "|---|" + "---|" * 5]
        for c in names:
            L.append(f"| {c} | " + " | ".join(f(v) for v in ab["ladder"][c]) + " |")
        L += ["", "Attacks, P(human > attack) over all (human, attack) pairs, ties 1/2 (the batteries' rule):", "", "| attack | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for att, r in ab["attacks"].items():
            L.append(f"| {att} | " + " | ".join(f(r[c]) for c in names) + " |")
        L.append("")
    if "gpt2_human_vs" in x:
        L += ["## GPT-2 output dataset: P(WebText > system) over all text pairs", "",
              "| system | " + " | ".join(lab for _, lab in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        for s, r in x["gpt2_human_vs"].items():
            L.append(f"| {s} | " + " | ".join(f(r[m]) for m, _ in METRICS) + " |")
        L.append("")
    if "stability" in x:
        L += ["## Stability ladders (scribe ladder --profile v31 reports)", "",
              "| run | pair | v2 | v3 | v3.1 | SURPRISE 3.1 | MATCH 3.1 |", "|---|---|---|---|---|---|---|"]
        for run, r in x["stability"].items():
            for pair, p in r["pairs"].items():
                L.append(f"| {run} | {pair} | {p['v2']} | {p['v3']} | {p['v3.1']} | {p['SURPRISE 3.1']} | {p['MATCH 3.1']} |")
        L.append("")
    if "k5" in x:
        L += ["## K5 (GEN-A dev ladder), trie rung", "", "| read | n | v2 | SURPRISE | MATCH | v3 | min3 | SURPRISE 3.1 | MATCH 3.1 | v3.1 | disagree |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for run, r in x["k5"].items():
            for rung, c in r.items():
                if rung.startswith("trie samples (pooled)"):
                    L.append(f"| {run} | " + " | ".join(c[k] for k in ("n", "v2", "SURPRISE", "MATCH", "v3", "min3", "SURPRISE 3.1", "MATCH 3.1", "v3.1", "disagree")) + " |")
        L.append("")
    if "k5_means" in x:
        ks = ("v2", "SURPRISE", "MATCH", "v3", "min3", "SURPRISE 3.1", "MATCH 3.1", "v3.1")
        L += ["## K5: mean over the saved seeds", "", "| arm | seeds | " + " | ".join(ks) + " |", "|---|---|" + "---|" * len(ks)]
        for arm, r in x["k5_means"].items():
            L.append(f"| {arm} | {r['seeds']} | " + " | ".join(f(r[k]) for k in ks) + " |")
        L.append("")
    if "cost" in x:
        L += ["## Cost", "", "| run | texts/s | v3 ms a text | memory and time |", "|---|---|---|---|"]
        for run, r in x["cost"].items():
            t = r["timing"] or {}
            L.append(f"| {run} | {t.get('texts_per_s', '')} | {t.get('v3_ms', '')} | {r['memtime']} |")
        L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
