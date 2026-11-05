#!/usr/bin/env python3
"""Dev measures of the planner's discourse (docs/reports/PLAN_DISCOURSE.md, GENERATION_PLAN §13.42 Q3).

  python3 -I scripts/plan_discourse_measures.py --ref Q3.plan.log [--prompts bench/eval/gen_prompts_dev.tsv] [--cut N] \
      --arm P1=P1.gen.txt,P1.plan.log --arm Q3=Q3.gen.txt,Q3.plan.log

--ref is the discourse run's K5_PLAN_LOG: its PLAN lines name each prompt's path and main entity, and its D field the
linked entities. Every arm is measured against the same entity lists. Per arm, over the prompts the planner writes
from facts (paths main and cooc in --ref) and over all prompts:
  - on subject: the share of the continuation's sentences that name the main entity, and that name the main entity
    or a linked entity (the full name or the short name the writer uses, without a trailing "(...)" or ", Region";
    case-insensitive, at word boundaries; pronouns do not count);
  - facts stated per prompt (the arm's plan log: facts whose sentence ends inside the byte limit);
  - repeated facts: stated facts that repeat an earlier stated fact of the same continuation (the same subject and
    value, or the same subject and relation), or that the prompt already holds (its subject and value both in it);
  - repeated sentences: sentences that occur twice in one continuation.
--cut N measures only the first N characters of each continuation (whitespace joined, as blind_rate.py's cut).
Measurement only; nothing here writes text.

Author: Odin Loch
"""
import argparse
import csv
import re
import statistics as stt
import sys


def unesc(s):
    """sample's printable(): \\n and \\xNN back to text."""
    b = bytearray()
    i = 0
    while i < len(s):
        if s.startswith("\\x", i) and i + 4 <= len(s):
            try:
                b.append(int(s[i + 2:i + 4], 16))
                i += 4
                continue
            except ValueError:
                pass
        if s.startswith("\\n", i):
            b += b"\n"
            i += 2
            continue
        b += s[i].encode("utf-8")
        i += 1
    return b.decode("utf-8", errors="replace")


def read_gen(path):
    out, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith("PROMPT\t"):
                cur = int(line.split("\t")[1])
            elif line.startswith("GEN\t") and cur is not None and cur not in out:
                out[cur] = unesc(line.split("\t", 2)[2])
    return out


def key60(prompt):
    b = prompt.encode("utf-8")[:60]
    return b.replace(b"\t", b" ").replace(b"\n", b" ")


def read_log(path, by_key):
    rows = {}
    with open(path, "rb") as f:
        for raw in f:
            p = raw.rstrip(b"\n").rstrip(b"\r").split(b"\t")
            if not p or p[0] != b"PLAN" or len(p) < 8:
                continue
            i = by_key.get(p[1])
            if i is None:
                continue
            facts, linked = [], []
            for x in p[8:]:
                q = x.decode("utf-8", errors="replace").split("|")
                if q[0] == "D" and len(q) >= 4:
                    linked = [e for e in q[2].split(";;") if e]
                    continue
                if len(q) < 7:
                    continue
                facts.append((q[0], q[1], q[2], q[3] == "1"))
            rows[i] = {"path": p[2].decode(), "ent": p[3].decode("utf-8", errors="replace"), "facts": facts, "linked": linked}
    return rows


def sentences(t):
    t = " ".join(t.split())
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if len(s) >= 3]


def short(n):
    """The name as the discourse writer writes it: no trailing "(...)", no ", Region" (Alcaeus (poet) -> Alcaeus)."""
    s = re.sub(r"\s*\([^()]*\)\s*$", "", n)
    m = re.search(r"(?<=[^\W\d_]),\s*(?=[^\W\d_])", s)
    if m and s[m.end()].isupper():
        s = s[: m.start()]
    return s.strip() or n


def names_re(names):
    names = [n for n in names if n and n != "-" and not re.fullmatch(r"[\d\s.,-]+", n)]
    names = names + [short(n) for n in names]
    if not names:
        return None
    alt = "|".join(re.escape(n) for n in sorted(set(names), key=len, reverse=True))
    return re.compile(r"(?<!\w)(?:" + alt + r")(?!\w)", re.IGNORECASE)


def cut(t, n):
    t = " ".join(t.split())
    if not n or len(t) <= n:
        return t
    t = t[:n]
    return t[: t.rfind(" ")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--prompts", default="bench/eval/gen_prompts_dev.tsv")
    ap.add_argument("--arm", action="append", required=True, help="NAME=GEN_FILE,PLAN_LOG")
    ap.add_argument("--cut", type=int, default=0)
    ap.add_argument("--ids", default=None, help="JSON list (or {'prompts': [...]}) of dev ids to measure only")
    a = ap.parse_args()
    prompts = {}
    with open(a.prompts, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            prompts[int(row["id"].split("/")[1])] = row["prompt"]
    by_key = {key60(p): i for i, p in prompts.items()}
    ref = read_log(a.ref, by_key)
    only = None
    if a.ids:
        import json
        j = json.load(open(a.ids))
        only = set(j["prompts"] if isinstance(j, dict) else j)
    planned = sorted(i for i, r in ref.items() if r["path"] in ("main", "cooc") and (only is None or i in only))
    every = sorted(i for i in prompts if i in ref and (only is None or i in only))
    print(f"prompts {len(every)}; planned from facts (main or cooc in --ref) {len(planned)}; cut {a.cut or 'none'}")
    print("| Arm | sentences (planned) | main entity | main or linked | facts stated / planned prompt | / all prompts | repeated facts | repeated sentences |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for spec in a.arm:
        name, files = spec.split("=", 1)
        gen_path, log_path = files.split(",", 1)
        gen = read_gen(gen_path)
        log = read_log(log_path, by_key)
        ns = nm = nl = 0
        rep_s = 0
        for i in planned:
            r = ref[i]
            m_re = names_re([r["ent"]])
            l_re = names_re([r["ent"]] + r["linked"])
            ss = sentences(cut(gen.get(i, ""), a.cut))
            seen = set()
            for s in ss:
                ns += 1
                nm += bool(m_re and m_re.search(s))
                nl += bool(l_re and l_re.search(s))
                rep_s += s in seen
                seen.add(s)
        stated = {i: [f for f in log.get(i, {}).get("facts", []) if f[3]] for i in every}
        rep_f = 0
        for i in every:
            sv, sr = set(), set()
            low = prompts[i].lower()
            for subj, rel, val, _ in stated[i]:
                if (subj, val) in sv or (subj, rel) in sr or (subj.lower() in low and val.lower() in low):
                    rep_f += 1
                sv.add((subj, val)), sr.add((subj, rel))
        tot = sum(len(v) for v in stated.values())
        per_planned = stt.mean(len(stated[i]) for i in planned) if planned else 0
        print(f"| {name} | {ns} | {nm / max(ns, 1):.3f} ({nm}) | {nl / max(ns, 1):.3f} ({nl}) | {per_planned:.2f} | {tot / max(len(every), 1):.2f} "
              f"| {rep_f} of {tot} | {rep_s} |")


if __name__ == "__main__":
    sys.exit(main())
