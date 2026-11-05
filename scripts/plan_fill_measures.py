#!/usr/bin/env python3
"""The planner's fill measures (docs/reports/PLAN_FILL.md, GENERATION_PLAN §13.42 Q2): names intact, slot type errors.

  python3 -I scripts/plan_fill_measures.py PLANLOG STOREDIR TYPES [--min-sent N] [--share S] [--examples N]

PLANLOG is the sample run's K5_PLAN_LOG; STOREDIR the fact store (its patterns.tsv and patterns_add.tsv); TYPES the
prefix of k5_planner types --table (TYPES.entities.tsv, TYPES.positions.tsv). Over the stated fact sentences:

  - names intact: both names stand in the sentence as written, neither run into a capitalised word next to it nor into
    the other name ("He Gothenburg University Gothenburg Municipality was ...", "King Francis Upritchard");
  - names side by side: the sentence holds "S O" or "O S" (case folded; MERGE_FILL.md's measure);
  - slot type errors, from the store's own types: a store pattern of relation r (train_sentences >= --min-sent) whose
    words next to its [S] or [O] (up to 3, with a content word) stand next to a name asserts that the name fills r:S or
    r:O. It is an error if the name does not fill that position in the store and its type (the EM type of
    k5_planner types) holds less than --share of the position's entities. "He was born in Gothenburg University":
    "born in" asserts birth_place:O, and almost no birth place is of the university's type.

Measurement only; nothing here writes text. Relation cues: scripts/k5_planner_measures.py.

Author: Odin Loch
"""
import math
import re
import sys
from collections import Counter, defaultdict

TOK = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[0-9]+(?:[.,][0-9]+)*|\S")  # K5's tokens (word_kn.hpp tokenize)
STOP = set("the a an of in on at to and or for by with from as is was were be been his her its their he she it they this that "
           "which who whom also there then than into after during since when while however but".split())
MONTHS = set("January February March April May June July August September October November December".split())
CANON = {"birth_date": "birth_year", "death_date": "death_year"}  # the store's canon (content_plan.hpp)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def toks(s):
    return TOK.findall(s)


def is_cap_word(t):
    return len(t) >= 2 and t[0].isupper() and t.isalpha()


def content(lit):
    return any(t.isalpha() and len(t) >= 3 and t.lower() not in STOP for t in lit)


def contexts(store, min_sent):
    """(side, literal) -> positions. side L: the literal ends just before the name; R: it starts just after."""
    ctx = defaultdict(set)
    for name in ("patterns.tsv", "patterns_add.tsv"):
        try:
            f = open(store + "/" + name, encoding="utf-8")
        except OSError:
            continue
        with f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 4 or int(p[3] or 0) < min_sent:
                    continue
                rel = CANON.get(p[0], p[0])
                t = p[1].split()
                holes = [i for i, x in enumerate(t) if x in ("[S]", "[O]", "[P]", "^")]
                for i, x in enumerate(t):
                    if x not in ("[S]", "[O]", "[P]"):
                        continue
                    role = "O" if x == "[O]" else "S"
                    prev = max([h for h in holes if h < i], default=-1)
                    nxt = min([h for h in holes if h > i], default=len(t))
                    left, right = t[prev + 1:i][-3:], t[i + 1:nxt][:3]
                    if left and content(left):
                        ctx[("L", tuple(w.lower() for w in left))].add(f"{rel}:{role}")
                    if right and content(right):
                        ctx[("R", tuple(w.lower() for w in right))].add(f"{rel}:{role}")
    return ctx


def lit_match(lit, seg):
    if len(lit) != len(seg):
        return False
    for a, b in zip(lit, seg):
        if a == "<m>":
            if b not in MONTHS:
                return False
        elif a == "#":
            if not b[:1].isdigit():
                return False
        elif a != b.lower():
            return False
    return True


def find(sent, name):
    n = len(name)
    for a in range(len(sent) - n + 1):
        if sent[a:a + n] == name:
            return a
    return -1


def main():
    args = sys.argv[1:]
    opt = {"--min-sent": "10", "--share": "0.05", "--examples": "0"}
    pos = []
    i = 0
    while i < len(args):
        if args[i] in opt:
            opt[args[i]] = args[i + 1]
            i += 2
        else:
            pos.append(args[i])
            i += 1
    log, store, types = pos[0], pos[1], pos[2]
    share_min, n_ex = float(opt["--share"]), int(opt["--examples"])
    ctx = contexts(store, int(opt["--min-sent"]))
    by_len = defaultdict(list)
    for (side, lit), ps in ctx.items():
        by_len[(side, len(lit))].append((lit, ps))
    etype, epos = {}, {}
    with open(types + ".entities.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            etype[p[0]] = int(p[1])
            epos[p[0]] = set(p[4].split(",")) if len(p) > 4 and p[4] else set()
    pshare = {}
    with open(types + ".positions.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            n = int(p[1])
            pshare[p[0]] = [int(x) / n if n else 0.0 for x in p[2:]]

    def asserted(sent, a, b):
        out = set()
        for (side, L), lst in by_len.items():
            seg = sent[a - L:a] if side == "L" else sent[b:b + L]
            if (side == "L" and a < L) or (side == "R" and b + L > len(sent)):
                continue
            for lit, ps in lst:
                if lit_match(lit, seg):
                    out |= ps
        return out

    n = intact = side = 0
    checked = errors = sent_err = 0
    err_rel = Counter()
    examples = []
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if not p or p[0] != "PLAN":
                continue
            for x in p[8:]:
                q = x.split("|")
                if len(q) < 7 or q[3] != "1":
                    continue
                subj, rel, val = q[0], q[1], q[2]
                sent = "|".join(q[6:]).split(" ")
                n += 1
                spans = []
                ok = True
                for nm in (subj, val):
                    t = toks(nm)
                    a = find(sent, t) if t else -1
                    spans.append((a, a + len(t)) if a >= 0 else None)
                    if a < 0:
                        ok = False
                        continue
                    b = a + len(t)
                    before = sent[a - 1] if a > 0 else ""
                    after = sent[b] if b < len(sent) else ""
                    if is_cap_word(after) or (is_cap_word(before) and not (a == 1 and before.lower() in STOP)):
                        ok = False
                s0, o0 = spans
                if s0 and o0 and (s0[1] == o0[0] or o0[1] == s0[0]):
                    ok = False
                low = " ".join(sent).lower()
                if subj.lower() + " " + val.lower() in low or val.lower() + " " + subj.lower() in low:
                    side += 1
                intact += ok
                bad = False
                for nm, sp in ((subj, s0), (val, o0)):
                    if sp is None or nm not in etype:
                        continue
                    ps = asserted(sent, sp[0], sp[1])
                    if not ps:
                        continue
                    checked += 1
                    t = etype[nm]
                    wrong = [pp for pp in ps if pp not in epos.get(nm, set()) and pp in pshare and pshare[pp][t] < share_min]
                    if wrong:
                        errors += 1
                        bad = True
                        for pp in wrong:
                            err_rel[pp] += 1
                        if len(examples) < n_ex:
                            examples.append(f"{nm} (type {t}) as {', '.join(sorted(wrong))}: {' '.join(sent)}")
                sent_err += bad
    print(f"stated fact sentences {n}")
    lo, hi = wilson(intact, n)
    print(f"names intact: {intact} of {n} ({intact / max(n, 1):.3f} [{lo:.3f}, {hi:.3f}])")
    lo, hi = wilson(side, n)
    print(f"names side by side: {side} of {n} ({side / max(n, 1):.3f} [{lo:.3f}, {hi:.3f}])")
    lo, hi = wilson(errors, checked)
    print(f"slot type errors: {errors} of {checked} checked names ({errors / max(checked, 1):.3f} [{lo:.3f}, {hi:.3f}]); "
          f"{sent_err} of {n} sentences ({sent_err / max(n, 1):.3f})")
    print("errors by asserted position: " + ", ".join(f"{r} {c}" for r, c in err_rel.most_common(12)))
    for e in examples:
        print("  " + e)


if __name__ == "__main__":
    main()
