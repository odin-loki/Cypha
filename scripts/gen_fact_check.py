#!/usr/bin/env python3
"""CT3.4, fact precision of generated text (docs/reports/GEN_FACTS.md; GENERATION_PLAN §7, CT3.4).

Finds statements of fact in generated text with F2's kept patterns (pattern_learn's patterns.tsv and
patterns_add.tsv of the stores), and grades each against CyphaLM's own fact store as supported, contradicted
or unknown. A measure only: it changes no writer.

usage:
  gen_fact_check.py run --out DIR [--store DIR ...] [--drop LO:HI|none] ARM=FILE [ARM=FILE ...]
      FILE is a GEN file (PROMPT / GEN lines) or gen_prompts_{dev,test}.tsv (its human continuations).
      Writes DIR/ARM.facts.tsv per arm and DIR/summary.md, DIR/summary.json.
  gen_fact_check.py draw --out FILE --n 40 --seed S TSV [TSV ...]
      Draws graded statements (half contradicted, half supported) for the hand check.

The rules are fixed in GEN_FACTS.md, section "The measure". In short:
- Text: byte escapes decoded, dashes written "-", wiki markup dropped; sentences split at . ! ? before a
  capital, an opening quote or a digit; the last sentence is dropped if it does not end in . ! ? (the cut).
- Statements: every kept F2 pattern (kept = 1) of every store, compiled to a token regex. [S] is a name chunk,
  [P] a sentence-initial he / she / it, [O] typed by relation (a date, a place, a name, or a word of the
  relation's store lexicon). Two relaxations for written text: "is" also matches "was" ("are" / "were"), and
  "a" also matches "an". One added pattern family, not F2's: "[S] is/was a(n) [nationality] [is_a word]"
  with lists (", X", "and X"), for nationality and is_a (occupation or kind).
- Subjects: the longest sub-span of the name chunk that is a store subject (or an alias), rightmost on ties.
  A pronoun takes the last resolved person of the sample. Birth, death and nationality need a person: a
  resolved subject must have a birth, death, nationality, occupation or birth/death place fact.
- Grades (each against all the store's values for the subject):
  - dates: compared at the common precision (year, month, day). Agree -> supported; else contradicted.
  - birth / death place: equal, or one inside the other by the store's located_in -> supported; different
    top places (located_in walked up, continents excluded) -> contradicted; else unknown.
  - located_in: the object is the subject's store ancestor -> supported; different top places -> contradicted.
  - nationality: demonyms and country names joined by a fixed name table; British contains English,
    Scottish, Welsh. Match -> supported; the store has a nationality in the table and none match ->
    contradicted. Store nationalities outside the table ("Film", "Jewish") are not used.
  - director, class: name match -> supported; the store has a value and none match -> contradicted.
  - is_a, language: open lists. Match -> supported; else unknown.
  - a subject or relation the store does not hold -> unknown.
"""
import argparse
import collections
import json
import math
import os
import random
import re
import sys
import unicodedata

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
MON_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}
MON_NUM.update({m[:3]: i + 1 for i, m in enumerate(MONTHS)})
MON_NUM["sept"] = 9
MON_RE = r"(?:(?i:january|february|march|april|may|june|july|august|september|october|november|december)|" \
         r"(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)(?: \.)?)"
UP = "A-ZÀ-ÖØ-ÞĀ-Ž"
CAP = r"(?:[" + UP + r"][^\s]*)"
CONN = r"(?:of|de|da|di|del|della|der|den|van|von|la|le|du|des|y|bin|ibn|al|el|-|\.|'s)"
NAME = CAP + r"(?:(?: " + CONN + r")* " + CAP + r"){0,5}"
DAY = r"(?:[0-3]?\d)"
YEAR = r"(?:\d{3,4})"
ERA = r"(?: (?:BC|BCE|AD|CE))?"
DATE = r"(?:(?:(?i:c|ca|circa) (?:\. )?)?(?:" + DAY + " " + MON_RE + r"(?: ,)? " + YEAR + "|" + MON_RE + " " + DAY + \
       r"(?: ,)? " + YEAR + "|" + MON_RE + r"(?: ,)? " + YEAR + "|" + YEAR + ")" + ERA + ")"
PLACE = NAME + r"(?: , " + NAME + r")?"
DATE_RELS = {"birth_date": "birth", "birth_year": "birth", "death_date": "death", "death_year": "death"}
PLACE_RELS = {"birth_place", "death_place", "located_in"}
NAME_RELS = {"director"}
LEX_RELS = {"nationality", "is_a", "language", "class"}
OPEN_RELS = {"is_a", "language"}
CONTINENTS = {"asia", "europe", "africa", "north america", "south america", "oceania", "antarctica",
              "australasia", "americas", "eurasia", "middle east", "caribbean", "scandinavia",
              "central america", "latin america", "western europe", "eastern europe", "southeast asia"}
PRON = {"he", "she", "it"}

# Demonym and country names (a name table, not facts): every name in a row is one nationality.
NAT_TABLE = """American|United States|U.S.|USA|America
British|United Kingdom|UK|Britain|Great Britain
English|England
Scottish|Scotland|Scots
Welsh|Wales
Irish|Ireland|Republic of Ireland
Northern Irish|Northern Ireland
Canadian|Canada
Australian|Australia
New Zealand|New Zealander
French|France
German|Germany|West German|East German
Italian|Italy
Spanish|Spain
Portuguese|Portugal
Dutch|Netherlands|Holland
Belgian|Belgium
Swiss|Switzerland
Austrian|Austria
Swedish|Sweden
Norwegian|Norway
Danish|Denmark
Finnish|Finland
Icelandic|Iceland
Russian|Russia
Soviet|Soviet Union|USSR
Ukrainian|Ukraine
Polish|Poland
Czech|Czech Republic|Czechoslovak|Czechoslovakia
Slovak|Slovakia
Hungarian|Hungary
Romanian|Romania
Bulgarian|Bulgaria
Greek|Greece
Turkish|Turkey
Serbian|Serbia
Croatian|Croatia
Slovenian|Slovenia|Slovene
Bosnian|Bosnia
Albanian|Albania
Lithuanian|Lithuania
Latvian|Latvia
Estonian|Estonia
Belarusian|Belarus
Armenian|Armenia
Japanese|Japan
Chinese|China
Korean|Korea|South Korean|South Korea|North Korean|North Korea
Indian|India
Pakistani|Pakistan
Bangladeshi|Bangladesh
Sri Lankan|Sri Lanka
Nepalese|Nepal
Thai|Thailand
Vietnamese|Vietnam
Filipino|Philippines|Philippine
Indonesian|Indonesia
Malaysian|Malaysia
Singaporean|Singapore
Iranian|Iran|Persian|Persia
Iraqi|Iraq
Israeli|Israel
Egyptian|Egypt
Lebanese|Lebanon
Syrian|Syria
Saudi|Saudi Arabia|Saudi Arabian
Afghan|Afghanistan
Mexican|Mexico
Brazilian|Brazil
Argentine|Argentina|Argentinian
Chilean|Chile
Colombian|Colombia
Peruvian|Peru
Venezuelan|Venezuela
Cuban|Cuba
Jamaican|Jamaica
Puerto Rican|Puerto Rico
Uruguayan|Uruguay
South African|South Africa
Nigerian|Nigeria
Kenyan|Kenya
Ethiopian|Ethiopia
Ghanaian|Ghana
Moroccan|Morocco
Algerian|Algeria
Tunisian|Tunisia
Zimbabwean|Zimbabwe
Luxembourgish|Luxembourg
Maltese|Malta
Cypriot|Cyprus
Mongolian|Mongolia
Taiwanese|Taiwan
Flemish|Flanders
Catalan|Catalonia
Basque
Prussian|Prussia
Bavarian|Bavaria
Ottoman|Ottoman Empire
Byzantine|Byzantine Empire"""
NAT_PARENT = {"english": "british", "scottish": "british", "welsh": "british", "northern irish": "british",
              "flemish": "belgian", "catalan": "spanish", "basque": "spanish", "bavarian": "german",
              "prussian": "german"}


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm(s):
    s = strip_accents(s).lower().replace("’", "'")
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)  # "X (film)"
    s = re.sub(r"[^\w' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.startswith("the "):
        s = s[4:]
    return s


NAT_CANON = {}
for _row in NAT_TABLE.splitlines():
    _names = [n for n in _row.split("|") if n]
    for _n in _names:
        NAT_CANON[norm(_n)] = norm(_names[0])


def unescape(s):
    s = s.replace("\\n", "\n")
    b = re.sub(r"\\x([0-9a-f]{2})", lambda m: chr(int(m.group(1), 16)), s)
    return b.encode("latin-1", "replace").decode("utf-8", "replace")


def clean_text(t):
    t = t.replace("'''", "").replace("''", "").replace("[[", "").replace("]]", "")
    t = re.sub(r"[‐-―−]", " - ", t)
    t = t.replace(" ", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


ABBR = {"c", "ca", "b", "d", "fl", "st", "mr", "mrs", "dr", "jr", "sr", "no", "vol", "mt", "ft", "gen", "lt", "col",
        "capt", "sgt", "rev", "prof", "inc", "co", "corp", "ltd", "vs", "etc", "jan", "feb", "mar", "apr", "jun",
        "jul", "aug", "sep", "sept", "oct", "nov", "dec"}


def split_sentences(t):
    """Sentences split at . ! ? followed by space and a capital, quote or digit; an unterminated last one dropped."""
    out, start = [], 0
    for m in re.finditer(r"[.!?][\"')\]]?(?= +[\"'(\[]?[" + UP + r"0-9])", t):
        end = m.end()
        if t[m.start()] == ".":
            w = re.search(r"(\w+)\.$", t[start:m.start() + 1])
            if w and (w.group(1).lower() in ABBR or (len(w.group(1)) == 1 and w.group(1).isupper())):
                continue
        s = t[start:end].strip()
        if s:
            out.append(s)
        start = end
    tail = t[start:].strip()
    if tail:
        out.append(tail)
    if out and not re.search(r"[.!?][\"')\]]?$", out[-1]):
        out = out[:-1]
    return out


def tokenize(s):
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*|\S", s)


def parse_date(txt):
    """(year, month, day) with None for missing parts; BC years negative."""
    toks = txt.replace(",", " ").replace(".", " ").split()
    y = m = d = None
    bc = any(t.upper() in ("BC", "BCE") for t in toks)
    nums = []
    for t in toks:
        tl = t.lower()
        if tl in MON_NUM and m is None:
            m = MON_NUM[tl]
        elif t.isdigit():
            nums.append(int(t))
    for n in nums:
        if n >= 100 or len(nums) == 1:
            y = n
        else:
            d = n
    if y is None:
        return None
    if bc:
        y = -y
    return (y, m, d)


def parse_store_date(v):
    v = v.strip()
    mm = re.match(r"^(\d{1,4})(?:-(\d{2}))?(?:-(\d{2}))?$", v)
    if mm:
        return (int(mm.group(1)), int(mm.group(2)) if mm.group(2) else None, int(mm.group(3)) if mm.group(3) else None)
    mm = re.match(r"^(\d{1,4}) ?BC", v)
    if mm:
        return (-int(mm.group(1)), None, None)
    return None


def date_agree(a, b, era_free=False):
    if era_free:
        a, b = (abs(a[0]),) + tuple(a[1:]), (abs(b[0]),) + tuple(b[1:])
    for x, y in zip(a, b):
        if x is None or y is None:
            return True
        if x != y:
            return False
    return True


def date_valid(t):
    y, m, d = t
    if y > 2026:
        return False
    if d is not None:
        if m is None:
            return False
        lim = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
        if d < 1 or d > lim:
            return False
    return True


class Store:
    def __init__(self, dirs, drop):
        self.vals = collections.defaultdict(set)    # (norm subj, rel) -> set of objects
        self.subj_name = {}                          # norm subj -> display
        self.parent = collections.defaultdict(set)  # norm place -> norm parents
        self.children = collections.Counter()
        self.lex = collections.defaultdict(set)     # rel -> norm objects
        self.alias = {}
        self.patterns = collections.OrderedDict()   # (rel, pattern) -> (precision, wilson)
        self.n_facts = 0
        self.n_dropped = 0
        lo, hi = drop if drop else (None, None)
        seen = set()
        for d in dirs:
            for fn, P_col in (("seeds.tsv", None), ("facts_f2.tsv", 3)):
                path = os.path.join(d, fn)
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as f:
                    head = f.readline().rstrip("\n").split("\t")
                    io = head.index("byte_offset")
                    for line in f:
                        c = line.rstrip("\n").split("\t")
                        if len(c) <= io or c[1] in ("relation", ""):
                            continue
                        if P_col is not None:
                            try:
                                if float(c[P_col]) < 0.8:
                                    continue
                            except ValueError:
                                continue
                        try:
                            off = int(c[io])
                        except ValueError:
                            off = 0
                        key = (c[0], c[1], c[2])
                        if key in seen:
                            continue
                        if lo is not None and lo <= off < hi:
                            self.n_dropped += 1
                            continue
                        seen.add(key)
                        self.add(c[0], c[1], c[2])
            ap = os.path.join(d, "aliases.tsv")
            if os.path.exists(ap):
                with open(ap, encoding="utf-8") as f:
                    f.readline()
                    for line in f:
                        c = line.rstrip("\n").split("\t")
                        if len(c) >= 2:
                            a = norm(c[0])
                            if a and a not in self.alias:
                                self.alias[a] = norm(c[1])
            for pf in ("patterns.tsv", "patterns_add.tsv"):
                path = os.path.join(d, pf)
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as f:
                    head = f.readline().rstrip("\n").split("\t")
                    ik, ip, iw = head.index("kept"), head.index("precision"), head.index("wilson_lower")
                    for line in f:
                        c = line.rstrip("\n").split("\t")
                        if len(c) > ik and c[ik] == "1":
                            self.patterns.setdefault((c[0], c[1]), (float(c[ip]), float(c[iw])))

    def add(self, s, r, o):
        ns, no = norm(s), o.strip()
        if not ns or not no:
            return
        self.n_facts += 1
        self.vals[(ns, r)].add(no)
        self.subj_name.setdefault(ns, s)
        if r in ("located_in", "country"):
            p = norm(no)
            if p and p not in CONTINENTS and p != ns:
                self.parent[ns].add(p)
                self.children[p] += 1
        if r in LEX_RELS or r == "occupation":
            self.lex["is_a" if r == "occupation" else r].add(norm(no))
        if r in ("birth_place", "death_place", "located_in"):
            self.add_comma(no)
            if "," in s:
                self.add_comma(s)

    def add_comma(self, place):
        parts = [norm(x) for x in place.split(",")]
        parts = [p for p in parts if p]
        for a, b in zip(parts, parts[1:]):
            if b not in CONTINENTS and a != b:
                self.parent[a].add(b)
                self.children[b] += 1
        if len(parts) > 1 and parts[-1] not in CONTINENTS:
            self.parent[norm(place)].add(parts[-1])

    def resolve(self, chunk):
        """Longest sub-span of the name chunk that is a store subject or alias; rightmost on ties."""
        toks = chunk.split(" ")
        n = len(toks)
        for L in range(n, 0, -1):
            for i in range(n - L, -1, -1):
                span = " ".join(toks[i:i + L])
                if not span[:1].isupper():
                    continue
                k = norm(span)
                if not k:
                    continue
                if k in self.subj_name:
                    return k
                if k in self.alias and self.alias[k] in self.subj_name:
                    return self.alias[k]
        return None

    def is_person(self, k):
        return any((k, r) in self.vals for r in ("birth_date", "birth_year", "death_date", "death_year",
                                                   "nationality", "occupation", "birth_place", "death_place"))

    def ancestors(self, p):
        seen, stack = set(), [p]
        while stack:
            x = stack.pop()
            for y in self.parent.get(x, ()):
                if y not in seen and len(seen) < 200:
                    seen.add(y)
                    stack.append(y)
        seen.discard(p)
        return seen

    def tops(self, p):
        """Top places above p (or p itself if it is a top that holds at least 3 places)."""
        anc = self.ancestors(p) | {p}
        tops = {a for a in anc if not self.parent.get(a)}
        if p in tops and self.children[p] < 3:
            tops.discard(p)
        return tops

    def known_place(self, p):
        return bool(self.parent.get(p)) or self.children[p] >= 3


def nat_compatible(text_c, store_c):
    # "British" for a store "English": the coarser name holds
    return text_c == store_c or NAT_PARENT.get(store_c) == text_c


class Extractor:
    def __init__(self, store):
        self.store = store
        self.pats = []
        for (rel, pat) in store.patterns:
            rx = self.compile(rel, pat)
            if rx is not None:
                self.pats.append((rel, pat, rx))
        self.add_rx = re.compile(r"(?:^|(?<= ))(?:(?P<P>(?i:he|she|it))|(?P<S>" + NAME + r")) (?i:is|was) (?i:a|an) "
                                 r"(?P<R>.{1,160})")

    def obj_re(self, rel):
        if rel in DATE_RELS:
            return r"(?P<O>" + DATE + ")"
        if rel in PLACE_RELS:
            return r"(?:(?i:the) )?(?P<O>" + PLACE + ")"
        if rel in NAME_RELS:
            return r"(?P<O>" + NAME + ")"
        return r"(?P<O>\S+(?: \S+){0,3})"

    def compile(self, rel, pat):
        toks = pat.split(" ")
        parts, anchored, has_o, has_s = [], False, False, False
        for i, t in enumerate(toks):
            if t == "^" and i == 0:
                anchored = True
                continue
            if t == "[S]":
                parts.append(r"(?P<S>" + NAME + ")")
                has_s = True
            elif t == "[P]":
                parts.append(r"(?P<P>(?i:he|she|it))")
                has_s = True
            elif t == "[O]":
                parts.append(self.obj_re(rel))
                has_o = True
            elif t == "#":
                parts.append(r"\d+")
            elif t == "<M>":
                parts.append(MON_RE)
            else:
                lt = t.lower()
                if lt == "is":
                    parts.append(r"(?i:is|was)")
                elif lt == "are":
                    parts.append(r"(?i:are|were)")
                elif lt in ("a", "an"):
                    parts.append(r"(?i:a|an)")
                else:
                    parts.append(r"(?i:" + re.escape(t) + ")")
        if not has_o or not has_s:
            return None
        body = " ".join(parts)
        body = ("^" if anchored else r"(?:^|(?<= ))") + body + r"(?= |$)"
        try:
            return re.compile(body)
        except re.error as e:
            sys.stderr.write(f"pattern skipped ({rel} {pat}): {e}\n")
            return None

    def lex_ok(self, rel, s):
        n = norm(s)
        if not n:
            return False
        if rel == "nationality":
            return n in NAT_CANON and s[:1].isupper()
        lex = self.store.lex.get(rel, set())
        if rel == "is_a":
            return s[:1].islower() and (n in lex or (n.endswith("s") and n[:-1] in lex))
        return n in lex

    def lex_match(self, rel, o, exact):
        toks = o.split(" ")
        if exact:
            return o if self.lex_ok(rel, o) else None
        for L in range(len(toks), 0, -1):
            cand = " ".join(toks[:L])
            if self.lex_ok(rel, cand):
                return cand
        return None

    def sentence_statements(self, sent):
        """Raw statements (rel, subj chunk or None, pronoun or None, object text, pattern) in one sentence."""
        line = " ".join(tokenize(sent))
        out = []
        for rel, pat, rx in self.pats:
            for m in rx.finditer(line):
                o = m.group("O")
                if o is None:
                    continue
                if rel in LEX_RELS:
                    o = self.lex_match(rel, o, exact=not pat.endswith("[O]"))
                    if o is None:
                        continue
                gd = m.groupdict()
                pron = gd.get("P")
                if pron is not None and m.start("P") != 0:
                    continue
                out.append((rel, gd.get("S"), pron, o, pat))
        # The added family: "[S] is/was a(n) [nationality] [is_a word], [is_a word] and [is_a word]"
        lex = self.store.lex.get("is_a", set())
        for m in self.add_rx.finditer(line):
            if m.group("P") is not None and m.start("P") != 0:
                continue
            rest = m.group("R").split(" ")
            i, nats = 0, []
            while i < len(rest):
                hit = None
                for L in (3, 2, 1):
                    if i + L <= len(rest) and rest[i][:1].isupper() and norm(" ".join(rest[i:i + L])) in NAT_CANON:
                        hit = " ".join(rest[i:i + L])
                        break
                if not hit:
                    break
                nats.append(hit)
                i += L
                if i + 1 < len(rest) and rest[i] in ("-", "and", "/") and rest[i + 1][:1].isupper():
                    i += 1
                    continue
                break
            occs = []
            while i < len(rest):
                hit = None
                for L in (3, 2, 1):
                    if i + L <= len(rest):
                        c = " ".join(rest[i:i + L])
                        nc = norm(c)
                        if c[:1].islower() and nc and (nc in lex or (nc.endswith("s") and nc[:-1] in lex)):
                            hit = c
                            break
                if not hit:
                    break
                occs.append(hit)
                i += L
                if i + 1 < len(rest) and rest[i] in (",", "and"):
                    i += 1
                    if rest[i] == "and" and i + 1 < len(rest):
                        i += 1
                    continue
                break
            subj, pron = m.group("S"), m.group("P")
            for n in nats:
                out.append(("nationality", subj, pron, n, "ADD [S] is a [NAT] [OCC]"))
            for o in occs:
                out.append(("is_a", subj, pron, o, "ADD [S] is a [NAT] [OCC]"))
        return out


class Grader:
    def __init__(self, store):
        self.st = store

    def grade(self, rel, k, obj):
        """-> (verdict, store values shown, note)"""
        st = self.st
        if k is None:
            return "unknown", "", "subject not in store"
        if rel in DATE_RELS:
            fam = DATE_RELS[rel]
            t = parse_date(obj)
            vals = st.vals.get((k, fam + "_date"), set()) | st.vals.get((k, fam + "_year"), set())
            sv = [(v, parse_store_date(v)) for v in sorted(vals)]
            sv = [(v, p) for v, p in sv if p is not None]
            if not sv:
                return "unknown", "", "no " + fam + " in store"
            shown = ";".join(v for v, _ in sv)
            era_free = not re.search(r"(?:BC|BCE|AD|CE)", obj)  # "322" written for "322 BC" is no contradiction
            ok = [p for _, p in sv if date_agree(t, p, era_free)]
            if ok:
                best = max(sum(x is not None and y is not None for x, y in zip(t, p)) for p in ok)
                stated = sum(x is not None for x in t)
                return "supported", shown, "" if best >= stated else f"checked {best} of {stated} parts"
            return "contradicted", shown, ""
        if rel in PLACE_RELS:
            q = norm(obj)
            qh = norm(obj.split(",")[0])
            if rel == "located_in":
                anc = st.ancestors(k)
                if not anc:
                    return "unknown", "", "no located_in in store"
                shown = ";".join(sorted(st.parent.get(k, ())))
                if q in anc or qh in anc:
                    return "supported", shown, ""
                if q == k or k in st.ancestors(q):
                    return "unknown", shown, "object inside subject"
                tq, ts = st.tops(q) | st.tops(qh), st.tops(k)
                if tq and ts and not (tq & ts):
                    return "contradicted", shown, "different top places"
                return "unknown", shown, "places not comparable in store"
            vals = st.vals.get((k, rel), set())
            if not vals:
                return "unknown", "", "no " + rel + " in store"
            shown = ";".join(sorted(vals))
            for v in vals:
                p, ph = norm(v), norm(v.split(",")[0])
                if q in (p, ph) or qh in (p, ph):
                    return "supported", shown, ""
                if {q, qh} & (st.ancestors(p) | st.ancestors(ph)):
                    return "supported", shown, "statement coarser"
                if {p, ph} & (st.ancestors(q) | st.ancestors(qh)):
                    return "supported", shown, "statement finer, inside the store's place"
            tq = st.tops(q) | st.tops(qh)
            ts = set()
            for v in vals:
                ts |= st.tops(norm(v)) | st.tops(norm(v.split(",")[0]))
            if tq and ts and not (tq & ts):
                return "contradicted", shown, "different top places"
            return "unknown", shown, "places not comparable in store"
        if rel == "nationality":
            vals = st.vals.get((k, "nationality"), set())
            shown = ";".join(sorted(vals))
            sc = {NAT_CANON.get(norm(v)) for v in vals} - {None}
            if not sc:
                return "unknown", shown, "no store nationality in the name table"
            tc = NAT_CANON.get(norm(obj))
            if tc is None:
                return "unknown", shown, "not in the name table"
            if any(nat_compatible(tc, s) for s in sc):
                return "supported", shown, ""
            if any(NAT_PARENT.get(tc) == s for s in sc):
                return "unknown", shown, "statement finer"
            return "contradicted", shown, ""
        if rel == "is_a":
            vals = st.vals.get((k, "is_a"), set()) | st.vals.get((k, "occupation"), set())
            if not vals:
                return "unknown", "", "no is_a or occupation in store"
            shown = ";".join(sorted(vals))
            q = norm(obj)
            qs = q[:-1] if q.endswith("s") else q
            nv = {norm(v) for v in vals}
            if q in nv or qs in nv or any(q.endswith(" " + v) or v.endswith(" " + q) for v in nv):
                return "supported", shown, ""
            return "unknown", shown, "open list"
        vals = st.vals.get((k, rel), set())
        if not vals:
            return "unknown", "", "no " + rel + " in store"
        shown = ";".join(sorted(vals))
        q = norm(obj)
        nv = {norm(v) for v in vals}
        if q in nv or any(v.endswith(" " + q) or q.endswith(" " + v) for v in nv):
            return "supported", shown, ""
        if rel in OPEN_RELS:
            return "unknown", shown, "open list"
        return "contradicted", shown, ""


def read_samples(path):
    """[(sample id, text)] from a GEN file, or the human continuations of a prompts TSV."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    if lines and lines[0].startswith("id\tdoc\tprompt\tcontinuation"):
        for line in lines[1:]:
            c = line.split("\t")
            if len(c) >= 4:
                out.append((c[0], unescape(c[3])))
        return out
    pid, n = None, collections.Counter()
    for line in lines:
        if line.startswith("PROMPT\t"):
            pid = line.split("\t")[1]
        elif line.startswith("GEN\t"):
            c = line.split("\t", 2)
            sid = f"{pid}:{c[1]}"
            n[sid] += 1
            if n[sid] > 1:
                sid += f"#{n[sid]}"
            out.append((sid, unescape(c[2]) if len(c) > 2 else ""))
    return out


def check_arm(samples, ext, grader):
    st = grader.st
    rows = []
    for sid, text in samples:
        sents = split_sentences(clean_text(text))
        last = None  # the referent of a pronoun: the last resolved person
        for si, sent in enumerate(sents, 1):
            seen = {}
            sent_last = None
            for rel, subj, pron, obj, pat in ext.sentence_statements(sent):
                person_rel = rel in DATE_RELS or rel in ("birth_place", "death_place", "nationality")
                if pron is not None:
                    if pron.lower() == "it" and person_rel:
                        continue
                    k, subj_txt = last, pron
                else:
                    k, subj_txt = st.resolve(subj), subj
                if person_rel and k is not None and not st.is_person(k):
                    continue
                if rel == "located_in" and k is not None and st.is_person(k):
                    continue
                if person_rel and k is None and pron is None:
                    ws = subj.split(" ")
                    if len(ws) < 2 or len(ws) > 4 or ws[0] in ("The", "A", "An") or                             not all(w[:1].isupper() for w in ws):
                        continue
                fam = DATE_RELS.get(rel, rel)
                prec = 0
                if rel in DATE_RELS:
                    t = parse_date(obj)
                    if t is None:
                        continue
                    key = (fam, k or norm(subj_txt), t[0])
                    prec = sum(x is not None for x in t)
                else:
                    key = (fam, k or norm(subj_txt), norm(obj))
                if key in seen and seen[key][0] >= prec:
                    continue
                verdict, shown, note = grader.grade(rel, k, obj)
                valid = ""
                if rel in DATE_RELS:
                    valid = "valid" if date_valid(parse_date(obj)) else "invalid"
                rel_out = (fam + "_date") if rel in DATE_RELS else rel
                seen[key] = (prec, dict(sample=sid, sent=si, n_sents=len(sents), relation=rel_out,
                                        subject=subj_txt, entity=st.subj_name.get(k, "") if k else "",
                                        pronoun="1" if pron else ("0" if k is None or norm(subj) == k else "partial"),
                                        object=obj, verdict=verdict,
                                        store=shown, note=note, date=valid, pattern=pat, sentence=sent))
                if k is not None and pron is None and st.is_person(k):
                    sent_last = k
            rows.extend(r for _, r in seen.values())
            if sent_last is not None:
                last = sent_last
            else:
                m = re.match(NAME, " ".join(tokenize(sent)))
                if m:
                    k = st.resolve(m.group(0))
                    if k is not None and st.is_person(k):
                        last = k
    return rows


POS_BUCKETS = ["1", "2", "3", "4+"]


def pos_bucket(i):
    return str(i) if i <= 3 else "4+"


def summarise(sample_ids, rows, reps, seed):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["sample"]].append(r)
    per = []
    for sid in sample_ids:
        rs = by.get(sid, [])
        c = collections.Counter(r["verdict"] for r in rs)
        pos = collections.Counter()
        for r in rs:
            b = pos_bucket(r["sent"])
            pos[(b, "n")] += 1
            pos[(b, r["verdict"])] += 1
        inv = sum(1 for r in rs if r["date"] == "invalid")
        part = sum(1 for r in rs if r["verdict"] == "supported" and r["note"].startswith("checked"))
        per.append((len(rs), c["supported"], c["contradicted"], pos, inv, part))

    def stats(items):
        n = len(items)
        N = sum(x[0] for x in items)
        S = sum(x[1] for x in items)
        C = sum(x[2] for x in items)
        nan = float("nan")
        d = {"per_sample": N / n if n else nan,
             "contradicted": C / N if N else nan,
             "supported": S / N if N else nan,
             "precision": S / (S + C) if S + C else nan,
             "invalid_dates": sum(x[4] for x in items) / N if N else nan,
             "supported_partly": sum(x[5] for x in items) / S if S else nan}
        for b in POS_BUCKETS:
            nb = sum(x[3][(b, "n")] for x in items)
            d[f"pos{b}_n"] = nb
            d[f"pos{b}_contradicted"] = sum(x[3][(b, "contradicted")] for x in items) / nb if nb else nan
            d[f"pos{b}_supported"] = sum(x[3][(b, "supported")] for x in items) / nb if nb else nan
        return d

    point = stats(per)
    rng = random.Random(seed)
    boots = collections.defaultdict(list)
    for _ in range(reps):
        smp = [per[rng.randrange(len(per))] for _ in range(len(per))]
        for k, v in stats(smp).items():
            if not (isinstance(v, float) and math.isnan(v)):
                boots[k].append(v)
    ci = {}
    for k, vs in boots.items():
        vs.sort()
        ci[k] = (vs[int(0.025 * (len(vs) - 1))], vs[int(0.975 * (len(vs) - 1))])
    tot = collections.Counter(r["verdict"] for r in rows)
    return {"samples": len(sample_ids), "statements": len(rows), "supported_n": tot["supported"],
            "contradicted_n": tot["contradicted"], "unknown_n": tot["unknown"],
            "with_statement": sum(1 for x in per if x[0] > 0), "point": point, "ci": ci,
            "by_relation": {rel: dict(collections.Counter(r["verdict"] for r in rows if r["relation"] == rel))
                            for rel in sorted({r["relation"] for r in rows})}}


COLS = ["sample", "sent", "n_sents", "relation", "subject", "entity", "pronoun", "object", "verdict", "store", "note",
        "date", "pattern", "sentence"]


def fmt(p, ci, k, digits=3):
    v = p.get(k)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    lo, hi = ci.get(k, (float("nan"), float("nan")))
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def cmd_run(a):
    drop = None
    if a.drop:
        lo, hi = a.drop.split(":")
        drop = (int(lo), int(hi))
    store = Store(a.store, drop)
    ext = Extractor(store)
    grader = Grader(store)
    sys.stderr.write(f"store: {store.n_facts} facts, {len(store.subj_name)} subjects, {store.n_dropped} dropped by "
                     f"--drop; {len(ext.pats)} patterns compiled\n")
    os.makedirs(a.out, exist_ok=True)
    summ = collections.OrderedDict()
    for spec in a.arms:
        name, path = spec.split("=", 1)
        samples = read_samples(path)
        rows = check_arm(samples, ext, grader)
        with open(os.path.join(a.out, f"{name}.facts.tsv"), "w", encoding="utf-8") as f:
            f.write("\t".join(COLS) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]).replace("\t", " ") for c in COLS) + "\n")
        s = summarise([sid for sid, _ in samples], rows, a.reps, a.seed)
        s["file"] = path
        summ[name] = s
        sys.stderr.write(f"{name}: {len(samples)} samples, {len(rows)} statements\n")
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"store": a.store, "drop": a.drop, "facts": store.n_facts, "dropped": store.n_dropped,
                   "patterns": len(ext.pats), "arms": summ}, f, indent=1)
    lines = ["| Arm | samples | statements (S, C, U) | per sample [95%] | contradicted [95%] | supported [95%] | "
             "precision S/(S+C) [95%] | supported only in part | invalid dates |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in summ.items():
        p, ci = s["point"], s["ci"]
        lines.append(f"| {name} | {s['samples']} | {s['statements']} ({s['supported_n']}, {s['contradicted_n']}, "
                     f"{s['unknown_n']}) | {fmt(p, ci, 'per_sample', 2)} | {fmt(p, ci, 'contradicted')} | "
                     f"{fmt(p, ci, 'supported')} | {fmt(p, ci, 'precision')} | {fmt(p, ci, 'supported_partly', 2)} | "
                     f"{fmt(p, ci, 'invalid_dates')} |")
    lines += ["", "By sentence position (C = share contradicted, S = share supported, n = statements):", "",
              "| Arm | " + " | ".join(f"sentence {b}" for b in POS_BUCKETS) + " |",
              "|---|" + "---|" * len(POS_BUCKETS)]
    for name, s in summ.items():
        p, ci = s["point"], s["ci"]
        cells = [f"C {fmt(p, ci, f'pos{b}_contradicted', 2)}; S {fmt(p, ci, f'pos{b}_supported', 2)}; "
                 f"n {p[f'pos{b}_n']}" for b in POS_BUCKETS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "By relation (supported / contradicted / unknown):", ""]
    for name, s in summ.items():
        parts = [f"{rel} {d.get('supported', 0)}/{d.get('contradicted', 0)}/{d.get('unknown', 0)}"
                 for rel, d in s["by_relation"].items()]
        lines.append(f"- {name}: " + "; ".join(parts))
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_draw(a):
    rows = []
    for path in a.tsvs:
        arm = os.path.basename(path).split(".")[0]
        with open(path, encoding="utf-8") as f:
            head = f.readline().rstrip("\n").split("\t")
            for line in f:
                c = dict(zip(head, line.rstrip("\n").split("\t")))
                c["arm"] = arm
                rows.append(c)
    rng = random.Random(a.seed)
    out = []
    for v in ("contradicted", "supported"):
        pool = [r for r in rows if r["verdict"] == v]
        rng.shuffle(pool)
        out += pool[:a.n // 2]
    cols = ["arm"] + COLS + ["hand_extraction", "hand_verdict", "hand_note"]
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in out:
            f.write("\t".join(r.get(c, "") for c in cols) + "\n")
    print(f"{len(out)} drawn to {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--store", action="append", default=None)
    r.add_argument("--drop", default="75497472:100000000")
    r.add_argument("--reps", type=int, default=2000)
    r.add_argument("--seed", type=int, default=20261112)
    r.add_argument("arms", nargs="+")
    d = sub.add_parser("draw")
    d.add_argument("--out", required=True)
    d.add_argument("--n", type=int, default=40)
    d.add_argument("--seed", type=int, default=20261112)
    d.add_argument("tsvs", nargs="+")
    a = ap.parse_args()
    if a.cmd == "run":
        if not a.store:
            a.store = ["bench/fact_store/e9_text", "bench/fact_store/e9_v2"]
        if a.drop in ("", "none"):
            a.drop = None
        cmd_run(a)
    else:
        cmd_draw(a)


if __name__ == "__main__":
    main()
