#!/usr/bin/env python3
"""J2 / CT3.3 + CT2.2: how real Wikipedia paragraphs move, mined offline (docs/reports/PARA_PLAN.md). Odin Loch, 2026-11-12.

  python3 -I scripts/para_plan_mine.py mine --text TRAIN_ALL.txt --out bench/para_plan [--threads 4] [--limit-docs N]
  python3 -I scripts/para_plan_mine.py compare --out bench/para_plan --prompts bench/eval/gen_prompts_dev.tsv \
          --arm L1=bench/k5_coherence/runs/L1.gen.txt [--arm G16=...] [--long L1=bench/k5_coherence/runs/L1_long.gen.txt]
  python3 -I scripts/para_plan_mine.py debug --text TRAIN_ALL.txt --docs 20      (labels per sentence; no counts)

Every source (mined Wikipedia windows, GEN-A dev human continuations, a writer's GEN file) goes through the same
functions: split into sentences, label each sentence, measure the window. Labels (all defined below, fixed before mining):

  open   K5_STRUCTURE's opening type, ported from structure/experts.hpp open_type: 0 connective, 1 pronoun subject,
         2 "The/A/An ...", 3 date or time phrase, 4 named subject, 5 other; 6 none (no sentence before).
  func   a sentence's function (a relation cue, first match in this order): DEF (a copula definition), DEATH, BIRTH,
         EDUC, AWARD, WORK, NAME (naming, etymology), PLACE (location, population), EVENT (holds a year or month),
         OTHER (description).
  ref    how a sentence refers to the window's subject: NAME (its head word), PRON (a third-person pronoun in the first
         six tokens), DEFNP ("The/This/These/Such" + a content word of the prompt or the sentence before, in the first
         four tokens), NONE.
  subject  of a window: the prompt's main name (a run of capitalised words, mid-sentence capitalised in the corpus;
         the most frequent head in the prompt, ties to the first); with no name, the prompt's most frequent content
         word (ties to the first) as a TERM subject.
  content words  as bench/k5_coherence/coh_measures.py: lower case, letters only, 4+ letters, not a function word.
Nothing here reads GEN-A test.
"""
import argparse
import codecs
import collections
import csv
import json
import math
import os
import random
import re
import sys

# ---------------------------------------------------------------- words and sentences

STOP = set("""
that this with from were have which their there they been when also into than then them these those other about after
before would could should while where being because some such more most many much over under only very what will your
first second three during between within without through against among upon each both either well just like even
""".split())

TOK = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*|[^\w\s]", re.UNICODE)
ABBR = set("mr mrs ms dr st jr sr vs inc co ltd no gen col lt sgt prof rev mt ft ca c e.g i.e etc u.s u.k fig vol pp ed".split())
BOUND = re.compile(r"[.!?][\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
MONTHS = set("January February March April May June July August September October November December".split())

CONN = set("""However But And Also Thus Therefore Moreover Furthermore Meanwhile Later Then Although While As Because Despite
Instead Nevertheless Finally Additionally Similarly Consequently Hence Still Yet Eventually Initially Originally Subsequently
Afterwards Indeed Nonetheless Likewise Accordingly Otherwise So Or Unlike Like Besides Overall""".split())
PRON_OPEN = set("He She It They This These That Those His Her Its Their We I There Such".split())
PREP = set("In On During After Before By Since From Until Between At Around Following Circa".split())
IN_CONN = set("addition contrast particular fact general practice turn short response comparison".split())
PRON3 = set("he him his himself she her hers herself it its itself they them their theirs themselves".split())
PRON_CLASS = {**{w: 0 for w in "he him his himself".split()}, **{w: 1 for w in "she her hers herself".split()},
              **{w: 2 for w in "it its itself".split()}, **{w: 3 for w in "they them their theirs themselves".split()}}
DET_ANA = set("The This These Such That Those".split())
NAME_GLUE = set("of de del della der van von la le du da di y and the".split())

OPEN_NAMES = ["conn", "pron", "the", "date", "name", "other", "none"]
FUNC_NAMES = ["DEF", "DEATH", "BIRTH", "EDUC", "AWARD", "WORK", "NAME", "PLACE", "EVENT", "OTHER"]
REF_NAMES = ["NAME", "PRON", "DEFNP", "NONE"]

FUNC_CUES = [
    ("DEATH", r"\b(died|dies|death|killed|buried|murdered|executed|assassinated|funeral|passed away)\b"),
    ("BIRTH", r"\b(born|birth|birthplace|childhood|grew up|early life|his parents|her parents)\b"),
    ("EDUC", r"\b(educated|studied|graduated|graduate|degree|attended|student|doctorate|alumnus)\b"),
    ("AWARD", r"\b(award|awarded|awards|prize|prizes|medal|medals|honour|honor|honours|honors|won|wins|nominated|nomination)\b"),
    ("WORK", r"\b(worked|works|joined|became|appointed|elected|served|career|founded|wrote|written|published|released|"
             r"recorded|directed|played|signed|produced|composed|built|designed|developed|invented|starred|commanded|ruled|reigned)\b"),
    ("NAME", r"\b(named|renamed|called|known as|derives|derived|meaning|etymology|the name|its name|his name|her name|name of)\b"),
    ("PLACE", r"\b(located|situated|lies|population|inhabitants|area of|km|kilometres|kilometers|miles|border|borders|"
              r"northern|southern|eastern|western|river|coast|capital|census)\b"),
]
FUNC_RE = [(n, re.compile(p, re.IGNORECASE)) for n, p in FUNC_CUES]
DEF_RE = re.compile(r"\b(is|was|are|were)\s+(a|an|the|one of)\b")


def toks(s):
    return TOK.findall(s)


def content(s):
    return [w for w in re.findall(r"[A-Za-z]+", s.lower()) if len(w) >= 4 and w not in STOP]


def split_sents(t):
    """Sentences: cut after . ! ? (with closing quotes or brackets) before a capital, digit or opening mark; not after an
    abbreviation or a single capital letter (an initial); a piece under 24 bytes joins the next (K5's gate rule)."""
    t = " ".join(t.split())
    out, a, pend = [], 0, ""
    for m in BOUND.finditer(t):
        e = m.start() + 1
        w = t[a:e].rstrip(".!?\"')]").split()
        last = w[-1].lower() if w else ""
        if last in ABBR or (len(last) == 1 and last.isalpha()) or re.fullmatch(r"([a-z]\.)+[a-z]", last):
            continue
        piece = pend + t[a:m.end()].strip() if not pend else pend + " " + t[a:m.end()].strip()
        if len(piece.encode("utf-8")) < 24:
            pend = piece
        else:
            out.append(piece)
            pend = ""
        a = m.end()
    rest = t[a:].strip()
    if rest or pend:
        out.append((pend + " " + rest).strip() if pend else rest)
    return [s for s in out if s]


def ended(s):
    return bool(re.search(r"[.!?][\"')\]]*$", s))


def year_of(t):
    return len(t) == 4 and t.isascii() and t.isdigit() and 1000 <= int(t) <= 2099


# ---------------------------------------------------------------- capitalisation table (K5_STRUCTURE's caps_)

class Caps:
    """For each word, how often it is capitalised away from a sentence's start (counted on a slice of the text)."""

    def __init__(self):
        self.c = {}

    def add_doc(self, line):
        for s in split_sents(line):
            tt = toks(s)
            for t in tt[1:]:
                if t[:1].isalpha():
                    k = t.lower()
                    v = self.c.get(k)
                    if v is None:
                        v = self.c[k] = [0, 0]
                    v[0 if t[0].isupper() else 1] += 1

    def name_word(self, t, first, next_cap=False):
        """structure/experts.hpp cap_name."""
        if not t[:1].isupper() or t == "I" or t.lower() in PRON_CLASS:
            return False
        if not first:
            return True
        v = self.c.get(t.lower())
        if v is None:
            return next_cap
        return v[0] > v[1]


CAPS = Caps()


# ---------------------------------------------------------------- labels

def open_type(tt):
    if not tt:
        return 5
    a = tt[0]
    if year_of(a) or a in MONTHS:
        return 3
    if a in PREP:
        for t in tt[1:4]:
            if year_of(t) or t in MONTHS:
                return 3
        if a == "In" and len(tt) > 1 and tt[1] in IN_CONN:
            return 0
    if a in CONN:
        return 0
    if a in PRON_OPEN:
        return 1
    if a in ("The", "A", "An"):
        return 2
    if CAPS.name_word(a, True, len(tt) > 1 and CAPS.name_word(tt[1], False)):
        return 4
    return 5


def func_type(s, tt):
    head = " ".join(tt[:14])
    if DEF_RE.search(head) and open_type(tt) in (2, 4, 5):
        return 0
    for i, (n, r) in enumerate(FUNC_RE):
        if r.search(s):
            return FUNC_NAMES.index(n)
    if any(year_of(t) or t in MONTHS for t in tt):
        return 8
    return 9


def name_runs(tt):
    """Runs of name words (capitalised, mid-sentence capitalised in the corpus at a sentence's start), with glue words
    inside; a leading "The" is not part of the run. Returns lists of tokens."""
    runs, cur = [], []
    for i, t in enumerate(tt):
        nxt = tt[i + 1] if i + 1 < len(tt) else ""
        if CAPS.name_word(t, i == 0, CAPS.name_word(nxt, False) if nxt else False) and not (i == 0 and t in PRON_OPEN | CONN | PREP | {"The", "A", "An"}):
            cur.append(t)
        elif cur and t.lower() in NAME_GLUE and nxt[:1].isupper():
            cur.append(t)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        while r and r[-1].lower() in NAME_GLUE:
            r = r[:-1]
        if r:
            out.append(r)
    return out


def head_of(run):
    """The run's last word of 3+ letters that is not a glue word (the name's head: "Barbara Anderson" -> "Anderson")."""
    for t in reversed(run):
        if len(t) >= 3 and t.lower() not in NAME_GLUE and not t.isdigit():
            return t
    return run[-1]


def subject_of(prompt_sents):
    cnt, first = collections.Counter(), {}
    words = {}
    for s in prompt_sents:
        for r in name_runs(toks(s)):
            h = bare(head_of(r))
            cnt[h] += 1
            first.setdefault(h, len(first))
            words.setdefault(h, set()).update(bare(t) for t in r if len(bare(t)) >= 4 and t.lower() not in NAME_GLUE)
    if cnt:
        h = max(cnt, key=lambda k: (cnt[k], -first[k]))
        ws = words[h] | {h}
        # a full name broken by a nickname or a bracket ("Antonio Rodrigo "Minotauro" Nogueira"): when the subject's run
        # opens the prompt, the name words of its first eight tokens (before "(", "is", "was") belong to it too
        t0 = toks(prompt_sents[0]) if prompt_sents else []
        if t0 and any(bare(t) in ws for t in t0[:3]):
            for t in t0[:8]:
                if t in ("(", "is", "was", "are", "were", ","):
                    break
                if len(bare(t)) >= 4 and CAPS.name_word(t, False):
                    ws.add(bare(t))
        return ("NAME", h, ws)
    cw = collections.Counter()
    order = {}
    for s in prompt_sents:
        for w in content(s):
            cw[w] += 1
            order.setdefault(w, len(order))
    if not cw:
        return ("NONE", "", set())
    w = max(cw, key=lambda k: (cw[k], -order[k]))
    return ("TERM", w, {w})


def bare(t):
    """A token without a possessive ("Spoiler's" -> "Spoiler")."""
    return re.sub(r"['’]s?$", "", t)


def ref_type(subj, tt, prev_content):
    kind, h, ws = subj
    tt = [bare(t) for t in tt]
    if kind == "NAME":
        if any(t in ws for t in tt):
            return 0
        # "the society", "the quartet": the name's head word in lower case is a definite description
        low = {w.lower() for w in ws}
        if any(t in low for t in tt):
            return 2
    elif kind == "TERM":
        lw = [t.lower() for t in tt]
        if any(t == h or t == h + "s" or (t.endswith("s") and t[:-1] == h) for t in lw):
            return 0
    if kind != "NONE" and any(t.lower() in PRON3 for t in tt[:6]):
        return 1
    if kind != "NONE":
        for i, t in enumerate(tt[:4]):
            if t in DET_ANA or (i > 0 and t.lower() in ("the", "this", "these")):
                nxt = [x.lower() for x in tt[i + 1:i + 3]]
                if any(x in prev_content for x in nxt):
                    return 2
    return 3


# ---------------------------------------------------------------- one window

def window(prompt_sents, cont_sents):
    """The labels of one window: per continuation sentence (open, func, ref, on_prompt, on_prev, new_share, ntok,
    ended) plus the prompt's last sentence's open and func types."""
    subj = subject_of(prompt_sents)
    pw = set()
    for s in prompt_sents:
        pw.update(content(s))
    seen = set(pw)
    plt = toks(prompt_sents[-1]) if prompt_sents else []
    prev_c = set(content(prompt_sents[-1])) if prompt_sents else set()
    rows = []
    for s in cont_sents:
        tt = toks(s)
        cw = content(s)
        cs = set(cw)
        anchor = pw | prev_c
        rows.append(dict(open=open_type(tt), func=func_type(s, tt), ref=ref_type(subj, tt, anchor),
                         on_prompt=int(bool(cs & pw)), on_prev=int(bool(cs & prev_c)),
                         new=(sum(1 for w in cs if w not in seen) / len(cs)) if cs else 0.0,
                         ntok=len(tt), ended=int(ended(s))))
        seen |= cs
        prev_c = cs
    return dict(subj=subj[0], p_open=open_type(plt), p_func=func_type(prompt_sents[-1], plt) if prompt_sents else 9, rows=rows)


def cut_text(t, n):
    """blind_rate.py's cut (near n characters at a word), so every source is read as the raters read it."""
    t = " ".join(t.split())
    if len(t) <= n:
        return t
    t = t[:n]
    return t[: t.rfind(" ")].rstrip(" ,;:")


def cont_sents_of(text, n):
    """The continuation as read: cut near n characters; a last piece without an end mark (cut mid-sentence) is dropped."""
    ss = split_sents(cut_text(text, n))
    if ss and not ended(ss[-1]):
        ss = ss[:-1]
    return ss


# ---------------------------------------------------------------- aggregation (one Agg per source)

NPOS = 4  # sentence positions 1, 2, 3, 4+


class Agg:
    def __init__(self):
        self.n = 0
        self.subj = collections.Counter()
        self.ref_pos = [[0] * 4 for _ in range(NPOS)]
        self.ref_tr = [[0] * 4 for _ in range(4)]
        self.open_pos = [[0] * 6 for _ in range(NPOS)]
        self.open_tr = [[0] * 6 for _ in range(7)]  # prev (6 = the prompt's last sentence when first) -> next
        self.func_pos = [[0] * 10 for _ in range(NPOS)]
        self.func_tr = [[0] * 10 for _ in range(11)]  # prev (10 = prompt's last) ... kept as prompt's func for first
        self.on_prompt = [[0, 0] for _ in range(NPOS)]
        self.on_prev = [0, 0]
        self.new = [0.0, 0]
        self.ntok = [0, 0]
        self.run = []  # per window: leading sentences that keep the subject
        self.nsent = []
        self.same_func = [0, 0]

    def add(self, w):
        rows = w["rows"]
        if not rows:
            return
        self.n += 1
        self.subj[w["subj"]] += 1
        r = 0
        while r < len(rows) and rows[r]["ref"] != 3:
            r += 1
        self.run.append(r)
        self.nsent.append(len(rows))
        for i, x in enumerate(rows):
            k = min(i, NPOS - 1)
            self.ref_pos[k][x["ref"]] += 1
            self.open_pos[k][x["open"]] += 1
            self.func_pos[k][x["func"]] += 1
            self.on_prompt[k][0] += x["on_prompt"]
            self.on_prompt[k][1] += 1
            self.ntok[0] += x["ntok"]
            self.ntok[1] += 1
            self.new[0] += x["new"]
            self.new[1] += 1
            po = rows[i - 1]["open"] if i else 6
            self.open_tr[po][x["open"]] += 1
            pf = rows[i - 1]["func"] if i else w["p_func"]
            self.func_tr[pf][x["func"]] += 1
            if i:
                self.ref_tr[rows[i - 1]["ref"]][x["ref"]] += 1
                self.on_prev[0] += x["on_prev"]
                self.on_prev[1] += 1
                self.same_func[0] += int(rows[i - 1]["func"] == x["func"])
                self.same_func[1] += 1

    def merge(self, o):
        def m2(a, b):
            for i in range(len(a)):
                for j in range(len(a[i])):
                    a[i][j] += b[i][j]
        self.n += o.n
        self.subj.update(o.subj)
        for f in ("ref_pos", "ref_tr", "open_pos", "open_tr", "func_pos", "func_tr", "on_prompt"):
            m2(getattr(self, f), getattr(o, f))
        for f in ("on_prev", "new", "ntok", "same_func"):
            a, b = getattr(self, f), getattr(o, f)
            a[0] += b[0]
            a[1] += b[1]
        self.run += o.run
        self.nsent += o.nsent


def scalars(windows):
    """The registered scalar measures of a list of windows (each a window() dict); None where undefined."""
    a = Agg()
    for w in windows:
        a.add(w)
    return scalars_from_agg(a)


def jsd(p, q):
    sp, sq = sum(p), sum(q)
    if not sp or not sq:
        return None
    p = [x / sp for x in p]
    q = [x / sq for x in q]
    m = [(x + y) / 2 for x, y in zip(p, q)]
    kl = lambda a, b: sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------------------------------------------------------------- mining

def doc_windows(sents, every=8, max_prompt=2):
    """GEN-A-shaped windows of one document: starts at sentence 0, every, 2*every, ...; the prompt is one sentence, or
    two if the first is under 100 characters; the continuation is the text after it (cut later by the caller)."""
    for st in range(0, len(sents), every):
        p = [sents[st]]
        j = st + 1
        if len(p[0]) < 100 and j < len(sents) and max_prompt > 1:
            p.append(sents[j])
            j += 1
        if j >= len(sents):
            break
        yield st, p, sents[j:j + 24]


class Mine:
    """Counts over whole documents (lead and every window) plus window measures at the raters' cuts."""

    def __init__(self):
        self.docs = 0
        self.sents = 0
        self.open2 = collections.Counter()      # (p2, p1, t) over every sentence of a document (K5's open_ layout)
        self.func1 = collections.Counter()      # (prev func or 10 = document start, func)
        self.state1 = collections.Counter()     # ((func, ref) prev or ("START",), (func, ref)) in windows (ref to the window subject)
        self.lead_ref = collections.Counter()   # (sentence index 1..15 after the lead sentence, ref to the lead's subject)
        self.lead_func = collections.Counter()  # (sentence index 0..15, func)
        self.persist = collections.Counter()    # (d, shared) over anchor sentence i and i+d, d = 1..10
        self.prompt_persist = collections.Counter()  # (d, shared with the window prompt), d = 1..12
        self.ref_by_func = collections.Counter()  # (func, ref) in windows
        self.agg = {460: Agg(), 1500: Agg(), "inner460": Agg(), "lead460": Agg()}

    def add_doc(self, line):
        ss = split_sents(line)
        if len(ss) < 3:
            return
        self.docs += 1
        self.sents += len(ss)
        tts = [toks(s) for s in ss]
        ots = [open_type(t) for t in tts]
        fts = [func_type(s, t) for s, t in zip(ss, tts)]
        cws = [set(content(s)) for s in ss]
        p2 = p1 = 6
        for i, o in enumerate(ots):
            self.open2[(p2, p1, o)] += 1
            p2, p1 = p1, o
            self.func1[(fts[i - 1] if i else 10, fts[i])] += 1
        for i in range(len(ss)):
            for d in range(1, 11):
                if i + d < len(ss) and cws[i]:
                    self.persist[(d, int(bool(cws[i] & cws[i + d])))] += 1
        # the lead: the document's first sentence and its subject
        subj = subject_of([ss[0]])
        prev = cws[0]
        for i in range(0, min(len(ss), 16)):
            self.lead_func[(i, fts[i])] += 1
            if i:
                self.lead_ref[(i, ref_type(subj, tts[i], cws[0] | prev))] += 1
                prev = cws[i]
        # windows
        for st, p, rest in doc_windows(ss):
            pw = set()
            for s in p:
                pw |= set(content(s))
            for d, s in enumerate(rest[:12], 1):
                self.prompt_persist[(d, int(bool(set(content(s)) & pw)))] += 1
            text = " ".join(rest)
            w460 = window(p, cont_sents_of(text, 460))
            w1500 = window(p, cont_sents_of(text, 1500))
            self.agg[460].add(w460)
            self.agg[1500].add(w1500)
            self.agg["lead460" if st == 0 else "inner460"].add(w460)
            prev = ("START", w1500["p_func"])
            for x in w1500["rows"]:
                cur = (x["func"], x["ref"])
                self.state1[(prev, cur)] += 1
                self.ref_by_func[cur] += 1
                prev = cur

    def merge(self, o):
        self.docs += o.docs
        self.sents += o.sents
        for f in ("open2", "func1", "state1", "lead_ref", "lead_func", "persist", "prompt_persist", "ref_by_func"):
            getattr(self, f).update(getattr(o, f))
        for k in self.agg:
            self.agg[k].merge(o.agg[k])


def _ranges(path, n):
    size = os.path.getsize(path)
    cuts = [0]
    with open(path, "rb") as f:
        for i in range(1, n):
            f.seek(size * i // n)
            f.readline()
            cuts.append(f.tell())
    cuts.append(size)
    return list(zip(cuts[:-1], cuts[1:]))


def _work(args):
    path, a, b, limit = args
    m = Mine()
    with open(path, "rb") as f:
        f.seek(a)
        k = 0
        while f.tell() < b:
            line = f.readline()
            if not line:
                break
            m.add_doc(line.decode("utf-8", "replace").strip())
            k += 1
            if limit and k >= limit:
                break
    return m


def build_caps(path, nbytes):
    with open(path, "rb") as f:
        data = f.read(nbytes)
    for line in data.decode("utf-8", "replace").split("\n")[:-1]:
        CAPS.add_doc(line)


def write_tables(m, out):
    td = os.path.join(out, "tables")
    os.makedirs(td, exist_ok=True)
    hdr = "# J2 para_plan_mine.py, Odin Loch, 2026-11-12; counted from %s documents, %s sentences\n" % (m.docs, m.sents)
    with open(os.path.join(td, "open2.tsv"), "w") as f:
        f.write(hdr + "# K5_STRUCTURE opening types 0 conn 1 pron 2 the 3 date 4 name 5 other; 6 none. Layout of Stats::open_: p2 p1 t count\n")
        for p2 in range(7):
            for p1 in range(7):
                for t in range(6):
                    f.write(f"{p2}\t{p1}\t{t}\t{m.open2.get((p2, p1, t), 0)}\n")
    with open(os.path.join(td, "func1.tsv"), "w") as f:
        f.write(hdr + "# prev_func next_func count; prev 10 = document start. Func: " + " ".join(f"{i} {n}" for i, n in enumerate(FUNC_NAMES)) + "\n")
        for p in range(11):
            for t in range(10):
                f.write(f"{p}\t{t}\t{m.func1.get((p, t), 0)}\n")
    with open(os.path.join(td, "state1.tsv"), "w") as f:
        f.write(hdr + "# windows (1,500 chars): prev_state next_state count; a state is func:ref; START:f is the prompt's last sentence's func. Ref: 0 NAME 1 PRON 2 DEFNP 3 NONE\n")
        for (p, c), v in sorted(m.state1.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            ps = f"START:{p[1]}" if p[0] == "START" else f"{p[0]}:{p[1]}"
            f.write(f"{ps}\t{c[0]}:{c[1]}\t{v}\n")
    with open(os.path.join(td, "lead_ref.tsv"), "w") as f:
        f.write(hdr + "# sentence index after the lead (1..15), ref to the lead sentence's subject, count\n")
        for i in range(1, 16):
            for r in range(4):
                f.write(f"{i}\t{r}\t{m.lead_ref.get((i, r), 0)}\n")
    with open(os.path.join(td, "lead_func.tsv"), "w") as f:
        f.write(hdr + "# sentence index in the document (0..15), func, count\n")
        for i in range(16):
            for t in range(10):
                f.write(f"{i}\t{t}\t{m.lead_func.get((i, t), 0)}\n")
    with open(os.path.join(td, "persist.tsv"), "w") as f:
        f.write(hdr + "# d, sentences i and i+d share a content word: n_shared n_all (any sentence i); then prompt d n_shared n_all (windows)\n")
        for d in range(1, 11):
            f.write(f"sent\t{d}\t{m.persist.get((d, 1), 0)}\t{m.persist.get((d, 1), 0) + m.persist.get((d, 0), 0)}\n")
        for d in range(1, 13):
            f.write(f"prompt\t{d}\t{m.prompt_persist.get((d, 1), 0)}\t{m.prompt_persist.get((d, 1), 0) + m.prompt_persist.get((d, 0), 0)}\n")


def summarise_agg(a):
    s = {"n": a.n, "subj": dict(a.subj), "ref_pos": a.ref_pos, "ref_tr": a.ref_tr, "open_pos": a.open_pos, "open_tr": a.open_tr,
         "func_pos": a.func_pos, "func_tr": a.func_tr, "on_prompt": a.on_prompt, "on_prev": a.on_prev, "new": a.new,
         "ntok": a.ntok, "same_func": a.same_func,
         "run_mean": sum(a.run) / max(len(a.run), 1), "run_hist": dict(collections.Counter(a.run)),
         "nsent_mean": sum(a.nsent) / max(len(a.nsent), 1)}
    return s


def cmd_mine(a):
    import multiprocessing as mp
    import time
    t0 = time.time()
    build_caps(a.text, a.caps_bytes)
    print(f"caps: {len(CAPS.c)} words from {a.caps_bytes} bytes, {time.time() - t0:.0f} s", file=sys.stderr)
    rs = _ranges(a.text, a.threads * 8)
    with mp.get_context("fork").Pool(a.threads) as pool:
        total = Mine()
        for i, m in enumerate(pool.imap_unordered(_work, [(a.text, x, y, a.limit_docs) for x, y in rs])):
            total.merge(m)
            print(f"chunk {i + 1}/{len(rs)}: {total.docs} docs, {time.time() - t0:.0f} s", file=sys.stderr)
    os.makedirs(a.out, exist_ok=True)
    write_tables(total, a.out)
    js = {"docs": total.docs, "sents": total.sents, "caps_bytes": a.caps_bytes, "text": a.text,
          "agg": {str(k): summarise_agg(v) for k, v in total.agg.items()},
          "ref_by_func": {f"{k[0]}:{k[1]}": v for k, v in total.ref_by_func.items()}}
    with open(os.path.join(a.out, "mined.json"), "w") as f:
        json.dump(js, f)
    for k in (460, 1500, "lead460", "inner460"):
        sc, _ = scalars_from_agg(total.agg[k])
        with open(os.path.join(a.out, f"wiki_{k}.scalars.json"), "w") as f:
            json.dump(sc, f, indent=1)
    print(f"done: {total.docs} docs, {total.sents} sentences, {time.time() - t0:.0f} s", file=sys.stderr)


def scalars_from_agg(a):
    out = {}
    for k in range(NPOS):
        n = sum(a.ref_pos[k])
        out[f"kept_s{k + 1}"] = (1 - a.ref_pos[k][3] / n) if n else None
        out[f"name_s{k + 1}"] = a.ref_pos[k][0] / n if n else None
        out[f"pron_s{k + 1}"] = a.ref_pos[k][1] / n if n else None
        out[f"on_prompt_s{k + 1}"] = a.on_prompt[k][0] / a.on_prompt[k][1] if a.on_prompt[k][1] else None
    out["run"] = sum(a.run) / len(a.run) if a.run else None
    out["on_prev"] = a.on_prev[0] / a.on_prev[1] if a.on_prev[1] else None
    out["new_share"] = a.new[0] / a.new[1] if a.new[1] else None
    out["same_func"] = a.same_func[0] / a.same_func[1] if a.same_func[1] else None
    out["ntok"] = a.ntok[0] / a.ntok[1] if a.ntok[1] else None
    nt = sum(map(sum, a.open_pos))
    for t in range(6):
        out[f"open_{OPEN_NAMES[t]}"] = sum(a.open_pos[k][t] for k in range(NPOS)) / nt if nt else None
    nf = sum(map(sum, a.func_pos))
    for t in range(10):
        out[f"func_{FUNC_NAMES[t]}"] = sum(a.func_pos[k][t] for k in range(NPOS)) / nf if nf else None
    return out, a


# ---------------------------------------------------------------- compare (GEN-A dev human vs writers)

def decode_gen(s):
    """K5's GEN lines write non-ASCII bytes as \\xNN; turn them back into UTF-8 text."""
    if "\\x" not in s:
        return s
    b = re.sub(rb"\\x([0-9a-fA-F]{2})", lambda m: bytes([int(m.group(1), 16)]), s.encode("utf-8"))
    return b.decode("utf-8", "replace")


def read_gen(path):
    out, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("PROMPT\t"):
                cur = int(line.split("\t")[1])
            elif line.startswith("GEN\t") and cur is not None and cur not in out:
                out[cur] = decode_gen(line.split("\t", 2)[2])
    return out


SCALAR_KEYS = (["kept_s1", "kept_s2", "kept_s3", "kept_s4", "run", "pron_s2", "pron_s3", "on_prompt_s1", "on_prompt_s2",
                "on_prompt_s3", "on_prompt_s4", "on_prev", "new_share", "same_func", "ntok"]
               + [f"open_{n}" for n in OPEN_NAMES[:6]] + [f"func_{n}" for n in FUNC_NAMES])


def boot_diff(wa, wb, keys, B=1000, seed=20261112):
    """Paired bootstrap over prompts of scalars(wb) - scalars(wa): (diff, lo, hi) per key."""
    rnd = random.Random(seed)
    ids = sorted(set(wa) & set(wb))
    base_a, _ = scalars([wa[i] for i in ids])
    base_b, _ = scalars([wb[i] for i in ids])
    samples = {k: [] for k in keys}
    for _ in range(B):
        s = [rnd.choice(ids) for _ in ids]
        sa, _ = scalars([wa[i] for i in s])
        sb, _ = scalars([wb[i] for i in s])
        for k in keys:
            if sa.get(k) is not None and sb.get(k) is not None:
                samples[k].append(sb[k] - sa[k])
    out = {}
    for k in keys:
        v = sorted(samples[k])
        if base_a.get(k) is None or base_b.get(k) is None or len(v) < 20:
            out[k] = None
            continue
        d = base_b[k] - base_a[k]
        lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
        sd = (sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5
        out[k] = (base_a[k], base_b[k], d, lo, hi, d / sd if sd > 0 else 0.0)
    return out


def cmd_compare(a):
    with open(a.caps_json) as f:
        CAPS.c = json.load(f)
    rows = {}
    with open(a.prompts, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)):
            rows[i] = r
    res = {}
    for cutn, specs in ((460, a.arm), (1500, a.long)):
        if not specs:
            continue
        srcs = {}
        if cutn == 460:
            srcs["human"] = {i: r["continuation"] for i, r in rows.items()}
        for s in specs:
            n, p = s.split("=", 1)
            srcs[n] = read_gen(p)
        wins = {}
        for n, g in srcs.items():
            wins[n] = {i: window(split_sents(rows[i]["prompt"]), cont_sents_of(g[i], cutn)) for i in g if i in rows}
        res[cutn] = {}
        for n, w in wins.items():
            sc, ag = scalars(list(w.values()))
            res[cutn][n] = {"scalars": sc, "agg": summarise_agg(ag)}
        # distributions against the human text (460) and against the mined Wikipedia windows (both cuts)
        mined = {}
        mp_ = os.path.join(a.out, "mined.json")
        if os.path.exists(mp_):
            with open(mp_) as f:
                mined = json.load(f)["agg"].get(str(cutn), {})
        def dist(ag, key):
            return [sum(ag[key][k][t] for k in range(NPOS)) for t in range(len(ag[key][0]))]
        refs = {"human": res[cutn]["human"]["agg"]} if "human" in res[cutn] else {}
        if mined:
            refs["wiki"] = mined
        for n in res[cutn]:
            ag = res[cutn][n]["agg"]
            res[cutn][n]["jsd"] = {r: {key: jsd(dist(ag, key), dist(rg, key)) for key in ("open_pos", "func_pos", "ref_pos")}
                                   for r, rg in refs.items()}
        base = "human" if cutn == 460 else None
        if base:
            for n in wins:
                if n != base:
                    res[cutn][n]["vs_human"] = boot_diff(wins[base], wins[n], SCALAR_KEYS, B=a.boot)
    with open(os.path.join(a.out, "compare.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({str(k): {n: v["scalars"] for n, v in d.items()} for k, d in res.items()}, indent=1)[:4000])


def cmd_caps(a):
    build_caps(a.text, a.caps_bytes)
    with open(a.caps_json, "w") as f:
        json.dump(CAPS.c, f)
    print(f"caps: {len(CAPS.c)} words", file=sys.stderr)


def cmd_debug(a):
    build_caps(a.text, a.caps_bytes)
    with open(a.text, encoding="utf-8", errors="replace") as f:
        for k, line in enumerate(f):
            if k >= a.docs:
                break
            ss = split_sents(line)
            if len(ss) < 3:
                continue
            p, cont = ss[:1], ss[1:6]
            w = window(p, cont)
            print(f"== doc {k} subject {subject_of(p)}")
            print("  P:", p[0][:160])
            for s, x in zip(cont, w["rows"]):
                print(f"  {OPEN_NAMES[x['open']]:5} {FUNC_NAMES[x['func']]:6} {REF_NAMES[x['ref']]:5} | {s[:150]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--text", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--threads", type=int, default=4)
    m.add_argument("--caps-bytes", type=int, default=30_000_000)
    m.add_argument("--limit-docs", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    c.add_argument("--prompts", default="bench/eval/gen_prompts_dev.tsv")
    c.add_argument("--caps-json", required=True)
    c.add_argument("--arm", action="append", default=[])
    c.add_argument("--long", action="append", default=[])
    c.add_argument("--boot", type=int, default=1000)
    k = sub.add_parser("caps")
    k.add_argument("--text", required=True)
    k.add_argument("--caps-json", required=True)
    k.add_argument("--caps-bytes", type=int, default=30_000_000)
    d = sub.add_parser("debug")
    d.add_argument("--text", required=True)
    d.add_argument("--docs", type=int, default=20)
    d.add_argument("--caps-bytes", type=int, default=2_000_000)
    a = ap.parse_args()
    {"mine": cmd_mine, "compare": cmd_compare, "caps": cmd_caps, "debug": cmd_debug}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
