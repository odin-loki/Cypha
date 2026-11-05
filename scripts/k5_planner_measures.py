#!/usr/bin/env python3
"""Dev measures of the planner (docs/reports/K5_PLANNER.md): coverage by path, fact precision, relation cues.

  python3 -I scripts/k5_planner_measures.py PLANLOG STOREDIR [K5_GEN_FILE]

PLANLOG is the sample run's K5_PLAN_LOG (one PLAN line per prompt and temperature). Reported:
  - coverage: prompts per path (main entity / co-occurring entities / plain K5), Wilson 95% intervals;
  - facts planned, written and stated (inside the 600-byte cut);
  - fact precision: stated facts that are in the store (after --plan-drop; by construction) and, of those Wikidata
    checked (facts_wd_check.tsv), the share that is right;
  - relation cue: the share of stated fact sentences that hold one of the relation's cue words (the words of F2's
    kept patterns for that relation), i.e. the sentence K5 filled around the fact still says the relation;
  - with K5_GEN_FILE: for the prompts the planner wrote from facts, how often plain K5 states any of those facts.
Measurement only; nothing here writes text.

Author: Odin Loch
"""
import math
import re
import sys
from collections import Counter, defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


STOP = set("the a an of in on at to and or for by with from as is was were be been his her its their he she it they this that "
           "which who whom also there then than into".split())


def cues(store):
    c = defaultdict(Counter)
    try:
        with open(store + "/patterns.tsv", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 11 or p[10] != "1":
                    continue
                for w in re.findall(r"[A-Za-z]+", p[1].replace("[S]", " ").replace("[O]", " ")):
                    w = w.lower()
                    if len(w) >= 3 and w not in STOP:
                        c[p[0]][w] += 1
    except OSError:
        pass
    out = {r: {w for w, _ in cnt.most_common(6)} for r, cnt in c.items()}
    # date relations are planned as years: their cue words are the date relation's
    for a, b in (("birth_year", "birth_date"), ("death_year", "death_date")):
        out.setdefault(a, set()).update(out.get(b, set()))
    return out


def main():
    log, store = sys.argv[1], sys.argv[2]
    k5 = sys.argv[3] if len(sys.argv) > 3 else None
    cue = cues(store)
    paths = Counter()
    planned = written = stated = 0
    wd = Counter()
    cue_hit = cue_n = 0
    per_rel = Counter()
    fact_prompts = {}
    n = 0
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if not p or p[0] != "PLAN":
                continue
            n += 1
            paths[p[2]] += 1
            planned += int(p[4])
            written += int(p[5])
            stated += int(p[6])
            facts = []
            for x in p[8:]:
                q = x.split("|")
                if len(q) < 7:
                    continue
                subj, rel, val, st, w, src, sent = q[0], q[1], q[2], q[3], q[4], q[5], "|".join(q[6:])
                if st != "1":
                    continue
                facts.append((subj, rel, val))
                per_rel[rel] += 1
                wd[w] += 1
                cw = cue.get(rel, set())
                if cw:
                    cue_n += 1
                    toks = {t.lower() for t in re.findall(r"[A-Za-z]+", sent)}
                    cue_hit += bool(toks & cw)
            if facts:
                fact_prompts[p[1]] = facts
    print(f"prompts {n}")
    for k in ("main", "cooc", "k5"):
        lo, hi = wilson(paths[k], n)
        print(f"path {k}: {paths[k]} ({paths[k] / max(n, 1):.3f} [{lo:.3f}, {hi:.3f}])")
    print(f"facts planned {planned}, written {written}, stated in the cut text {stated}")
    print(f"fact precision against the store: {stated} of {stated} (1.000; every stated fact is a store fact by construction)")
    chk = wd["1"] + wd["0"]
    print(f"Wikidata check of stated facts: right {wd['1']}, wrong {wd['0']}, unchecked {wd['-1']}; right share of checked "
          f"{wd['1'] / chk if chk else float('nan'):.3f}")
    lo, hi = wilson(cue_hit, cue_n)
    print(f"relation cue in the stated fact's sentence: {cue_hit} of {cue_n} ({cue_hit / max(cue_n, 1):.3f} [{lo:.3f}, {hi:.3f}])")
    print("stated facts by relation: " + ", ".join(f"{r} {c}" for r, c in per_rel.most_common()))
    if k5:
        gens = {}
        prompt = None
        with open(k5, encoding="utf-8", errors="replace") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if p[0] == "PROMPT":
                    prompt = p[2][:60].replace("\t", " ")
                elif p[0] == "GEN" and prompt is not None:
                    gens[prompt] = p[2]
        hit = tot = 0
        for pr, facts in fact_prompts.items():
            g = gens.get(pr)
            if g is None:
                continue
            for subj, rel, val in facts:
                tot += 1
                hit += val.lower() in g.lower() and subj.lower() in g.lower()
        print(f"plain K5 on the same prompts states {hit} of the {tot} facts the planner stated (subject and value both in its text)")


if __name__ == "__main__":
    main()
