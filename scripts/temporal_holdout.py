#!/usr/bin/env python3
"""S15: the temporal-holdout benchmark. enwik8 (a 2006-03-04 Wikipedia snapshot) against current Wikidata.

For the enwik8 subjects D4 mapped to Wikidata items (D4's T-QA exclusions kept), every non-deprecated statement of D4's
relations and of some dated ones (with start time, end time and point in time), each labelled:
  c  became true after the snapshot (its date is after 2006-03-04);
  u  unknown when (the date straddles the snapshot, or an undated relation that can change over time);
  a  true by 2006 and stated in the 2006 text (the v2 fact store, a 2006 text seed, or the page rule below);
  b  true by 2006 and unstated.
Page rule (labelling only): the value's surface form occurs on word boundaries in the subject's normalised 2006 page; a
year if its digits occur; a day date if its year and "day month" or "month day" occur. One false fact per true fact: the
same subject and relation, a value drawn from that relation's values over subjects of the same type (one ticket per
true fact; the whole relation if none fits), never one the subject holds. Split by subject: test if sha256("S15|" + item)[0] >= 128. docs/reports/S15_TEMPORAL_HOLDOUT.md has the rules.

Reuses scripts/wikidata_seeds.py (D4): its fetcher with the response cache, its title map (items.tsv) and its date
conversion. Python standard library only; responses are cached under RUN/raw, so a rebuild from the cache is byte for byte.

Usage:
  python3 -I scripts/temporal_holdout.py build <enwik8> <pages.tsv> <D4 OUT> <facts_f2.tsv> <seeds.tsv> <RUN> <OUT dir>
  python3 -I scripts/temporal_holdout.py check <OUT dir> FILE ...   (lines naming a subject with one of its b or c values)
Author: Odin Loch
"""
import functools, hashlib, json, os, random, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wikidata_seeds as ws  # noqa: E402

ws.UA = "CyphaS15/0.1 (https://github.com/odin-loki/Cypha; Odin Loch's Cypha project) python-urllib"
SNAP = (2006, 3, 4)
ENTITY = dict(ws.ENTITY, P166="award_received", P39="position_held", P54="member_of_sports_team",
              P102="member_of_political_party", P463="member_of", P6="head_of_government", P35="head_of_state",
              P69="educated_at", P108="employer", P127="owned_by", P749="parent_organization", P159="headquarters",
              P488="chairperson", P1037="director_manager")
TIME = dict(ws.TIME, P576="dissolved")
TIMELESS = {"birth_place", "birth_year", "birth_date", "death_year", "death_date", "death_place", "founded", "dissolved",
            "author", "composer", "director", "inventor", "creator", "founder", "genre", "language"}
COUNTRY = {"Q6256", "Q3624078", "Q3024240", "Q7275"}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december"]
# The set's name: its ids, its negatives' seeds, its split salt and its file names. S8e reuses this builder for its fresh slots.
PREFIX, OUTNAME = "S15", "temporal_holdout"
COLS = ["id", "subject", "subject_item", "subject_type", "region", "relation", "property", "value", "value_item", "date",
        "date_kind", "category", "stated_by", "named_on_page", "named_elsewhere", "label", "pair"]


def period(t, prec):
    """A Wikidata time (Gregorian, astronomical years) as (first, last) (y, m, d) of the period its precision names."""
    m = re.match(r"(-?\d+)-(\d\d)-(\d\d)", t)
    y, mo, d = (int(x) for x in m.groups())
    if prec >= 11:
        return (y, mo, d), (y, mo, d)
    if prec == 10:
        return (y, mo, 1), (y, mo, 31)
    span = {9: 1, 8: 10, 7: 100, 6: 1000}.get(prec, 10 ** 9)
    lo = y if span == 1 else (y // span) * span if prec != 7 else ((y - 1) // 100) * 100 + 1
    return (lo, 1, 1), (lo + span - 1, 12, 31)


def when(t, prec):
    """'before', 'after' or 'straddles' the snapshot day, for a time and its precision."""
    lo, hi = period(t, prec)
    return "after" if lo > SNAP else "before" if hi < SNAP else "straddles"


def show(t, prec):  # a qualifier time as text: YYYY-MM-DD, YYYY-MM or YYYY (BC years as "N BC")
    m = re.match(r"(-?\d+)-(\d\d)-(\d\d)", t)
    y, mo, d = (int(x) for x in m.groups())
    if y < 1:
        return f"{1 - y} BC"
    return f"{y:04d}-{mo:02d}-{d:02d}" if prec >= 11 else f"{y:04d}-{mo:02d}" if prec == 10 else str(y)


def fetch_all(items):
    """Subject types, entity statements with qualifiers, time statements and value labels (cached under ws.RAW)."""
    types, geo, ent, tim = defaultdict(set), set(), [], []
    for i, b in enumerate(ws.blocks(items, 100)):
        its = " ".join("wd:" + x for x in b)
        q = (f"SELECT ?item ?c WHERE {{ VALUES ?item {{ {its} }} VALUES ?c {{ wd:Q5 {' '.join('wd:' + c for c in sorted(COUNTRY))} }} "
             "?item wdt:P31 ?c . }")
        for r in ws.sparql(q, f"type{i:03d}"):
            types[r["item"]].add(r["c"])
        q = f"SELECT ?item WHERE {{ VALUES ?item {{ {its} }} FILTER EXISTS {{ ?item wdt:P625 ?x }} }}"
        geo.update(r["item"] for r in ws.sparql(q, f"geo{i:03d}"))
        q = (f"SELECT ?item ?prop ?st ?v ?s ?sp ?e ?ep ?p ?pp WHERE {{ VALUES ?item {{ {its} }} "
             f"VALUES ?prop {{ {' '.join('wd:' + p for p in ENTITY)} }} ?prop wikibase:claim ?c ; wikibase:statementProperty ?ps . "
             "?item ?c ?st . ?st ?ps ?v ; wikibase:rank ?rk . FILTER(?rk != wikibase:DeprecatedRank) FILTER(isIRI(?v)) "
             "OPTIONAL { ?st pqv:P580 ?sn . ?sn wikibase:timeValue ?s ; wikibase:timePrecision ?sp } "
             "OPTIONAL { ?st pqv:P582 ?en . ?en wikibase:timeValue ?e ; wikibase:timePrecision ?ep } "
             "OPTIONAL { ?st pqv:P585 ?pn . ?pn wikibase:timeValue ?p ; wikibase:timePrecision ?pp } }")
        ent += ws.sparql(q, f"stmt{i:03d}")
        q = (f"SELECT ?item ?prop ?st ?t ?pr ?cal WHERE {{ VALUES ?item {{ {its} }} VALUES ?prop {{ {' '.join('wd:' + p for p in TIME)} }} "
             "?prop wikibase:claim ?c ; wikibase:statementValue ?sv . ?item ?c ?st . ?st ?sv ?n ; wikibase:rank ?rk . "
             "?n wikibase:timeValue ?t ; wikibase:timePrecision ?pr ; wikibase:timeCalendarModel ?cal . FILTER(?rk != wikibase:DeprecatedRank) }")
        tim += ws.sparql(q, f"time{i:03d}")
    values = sorted({r["v"] for r in ent if re.fullmatch(r"Q\d+", r["v"])}, key=lambda x: int(x[1:]))
    surf = {}
    for i, b in enumerate(ws.blocks(values, 250)):  # D4's label query: enwiki title, English label, aliases
        q = (f"SELECT ?v ?l ?n (GROUP_CONCAT(DISTINCT ?a; separator=\"|\") AS ?as) WHERE {{ VALUES ?v {{ {' '.join('wd:' + x for x in b)} }} "
             "OPTIONAL { ?v rdfs:label ?l FILTER(LANG(?l) = \"en\") } OPTIONAL { ?v skos:altLabel ?a FILTER(LANG(?a) = \"en\") } "
             f"OPTIONAL {{ ?art schema:about ?v ; schema:isPartOf {ws.ENWIKI} ; schema:name ?n }} }} GROUP BY ?v ?l ?n")
        for r in ws.sparql(q, f"lab{i:03d}"):
            forms = [r.get("n", ""), r.get("l", "")] + sorted(a for a in r.get("as", "").split("|") if len(a) >= 3)[:6]
            surf[r["v"]] = [f.replace("\t", " ") for j, f in enumerate(forms) if f and f not in forms[:j]]
    return types, geo, ent, tim, surf


def page_texts(enwik8, pages):
    """Each article's 2006 text, from its offset to </text>."""
    data = open(enwik8, "rb").read()
    out = {}
    for t, off in pages.items():
        end = data.find(b"</text>", off)
        out[t] = data[off:end if end >= 0 else len(data)].decode("utf-8", "replace")
    return out


def build(enwik8, pages_tsv, d4, facts_tsv, seeds_tsv, run, outdir):
    ws.RAW = f"{run}/raw"
    os.makedirs(ws.RAW, exist_ok=True)
    pages = {}
    for line in open(pages_tsv, encoding="utf-8"):
        t, off = line.rstrip("\n").split("\t")[:2]
        pages[t] = int(off)
    titles_of = defaultdict(list)
    for line in list(open(f"{d4}/items.tsv", encoding="utf-8"))[1:]:
        t, q = line.rstrip("\n").split("\t")
        titles_of[q].append(t)
    items = sorted(titles_of, key=lambda x: int(x[1:]))
    types, geo, ent, tim, surf = fetch_all(items)

    # Statements -> facts (item, relation, value key) with every statement's (date, kind, when).
    stmts = defaultdict(lambda: defaultdict(list))  # (item, rel, value key) -> statement -> [rows]
    prop_of, show_of, vitem_of = {}, {}, {}
    death = defaultdict(list)  # item -> [(t, prec)]
    for r in tim:
        prec, jul = int(r["pr"]), r["cal"] == "Q1985786"
        if TIME[r["prop"]] == "death":
            death[r["item"]].append((r["t"], prec))
        name = TIME[r["prop"]]
        rels = [(name, True)] if name in ("founded", "dissolved") else [(name + "_year", True), (name + "_date", False)]
        for rel, year_only in rels:
            v = ws.date_str(r["t"], prec, year_only, jul)
            if v:
                key = (r["item"], rel, v)
                stmts[key][r["st"]].append(("value", r["t"], prec))
                prop_of[key], show_of[key], vitem_of[key] = r["prop"], v, ""
    for r in ent:
        if not re.fullmatch(r"Q\d+", r["v"]):
            continue  # "some value" (a skolem IRI)
        rel = ENTITY[r["prop"]]
        key = (r["item"], rel, r["v"])
        quals = []
        for k, kind in (("s", "start"), ("p", "point"), ("e", "end")):
            if r.get(k):
                quals.append((kind, r[k], int(r[k + "p"])))
        stmts[key][r["st"]].extend(quals or [("none", "", 0)])
        prop_of[key], vitem_of[key] = r["prop"], r["v"]
        show_of[key] = (surf.get(r["v"]) or [r["v"]])[0]

    def classify(key):
        """(category before a/b split, date, date kind): 'old' (true by 2006), 'c' or 'u'."""
        item, rel, _ = key
        verdicts, dates = [], []
        for _, rows in sorted(stmts[key].items()):
            q = {}
            for kind, t, p in sorted(rows):  # the earliest of each kind
                q.setdefault(kind, (t, p))
            if "value" in q:
                kind, tp = "value", q["value"]
            elif rel == "death_place" and death.get(item):
                kind, tp = "death", min(death[item])
            elif "start" in q:
                kind, tp = "start", q["start"]
            elif "point" in q:
                kind, tp = "point", q["point"]
            else:
                kind, tp = "none", None
            if tp:
                w = when(*tp)
                verdicts.append("old" if w == "before" else "c" if w == "after" else "u")
                dates.append((period(*tp)[0], show(*tp), kind))
            elif "end" in q and when(*q["end"]) == "before":
                verdicts.append("old")
                dates.append((period(*q["end"])[0], show(*q["end"]), "end"))
            else:
                verdicts.append("old" if rel in TIMELESS else "u")
        cat = "old" if "old" in verdicts else "c" if verdicts and all(v == "c" for v in verdicts) else "u"
        first = min(dates) if dates else None
        return cat, (first[1] if first else ""), (first[2] if first else "none")

    # Texts: normalised pages per subject; stated sources (store and 2006 text seeds).
    texts = page_texts(enwik8, {t: o for q in items for t in titles_of[q] for o in [pages[t]]})
    norm_page = {t: " " + ws.norm(x) + " " for t, x in texts.items()}
    stated = defaultdict(set)  # (title, rel) -> normalised objects
    for path, src in ((facts_tsv, "store"), (seeds_tsv, "seed")):
        for line in list(open(path, encoding="utf-8"))[1:]:
            c = line.rstrip("\n").split("\t")
            if len(c) < 4 or (src == "seed" and c[3] == "wikidata") or (src == "store" and float(c[3]) < 0.8):
                continue  # Wikidata seeds are not 2006 text; store facts below P 0.8 are candidates, not admitted
            stated[(c[0], c[1], src)].add(ws.norm(c[2]))

    @functools.lru_cache(maxsize=None)
    def forms(key_value, rel):
        if re.fullmatch(r"Q\d+", key_value):
            fs = surf.get(key_value) or []
            fs = fs + [re.sub(r" \(.*\)$", "", f) for f in fs]
            return tuple(sorted({ws.norm(f) for f in fs if ws.norm(f)}))
        return ()

    def on_text(npage, key_value, rel):
        if re.fullmatch(r"Q\d+", key_value):
            return any(f" {f} " in npage for f in forms(key_value, rel))
        m = re.fullmatch(r"(\d{4})(?:-(\d\d)-(\d\d))?", key_value)
        if not m:
            return f" {ws.norm(key_value)} " in npage
        if f" {m.group(1)} " not in npage:
            return False
        if not m.group(2):
            return True
        mo, d = MONTHS[int(m.group(2)) - 1], str(int(m.group(3)))
        return f" {d} {mo} " in npage or f" {mo} {d} " in npage

    # Value pages elsewhere: every entity surface form and every date, counted over all pages.
    allforms = set()
    for (item, rel, v) in stmts:
        allforms.update(forms(v, rel))
    first = defaultdict(int)
    for f in allforms:
        w = f.split(" ")
        first[w[0]] = max(first[w[0]], min(len(w), 12))
    form_pages, year_pages, md_pages = Counter(), defaultdict(set), defaultdict(set)
    all_texts = page_texts(enwik8, pages)
    for pi, (t, x) in enumerate(sorted(all_texts.items())):
        words = ws.norm(x).split(" ")
        found = set()
        for i, w in enumerate(words):
            n = first.get(w, 0)
            for k in range(1, n + 1):
                c = " ".join(words[i:i + k])
                if c in allforms:
                    found.add(c)
        form_pages.update(found)
        for i, w in enumerate(words):
            if len(w) == 4 and w.isdigit():
                year_pages[w].add(pi)
            if w in MONTHS and i + 1 < len(words) and words[i + 1].isdigit() and len(words[i + 1]) <= 2:
                md_pages[(w, words[i + 1])].add(pi)
            if w.isdigit() and len(w) <= 2 and i + 1 < len(words) and words[i + 1] in MONTHS:
                md_pages[(words[i + 1], w)].add(pi)
    page_index = {t: pi for pi, t in enumerate(sorted(all_texts))}
    del all_texts

    def elsewhere(own_titles, key_value, rel):
        own = [norm_page[t] for t in own_titles]
        if re.fullmatch(r"Q\d+", key_value):
            return any(form_pages[f] - sum(f" {f} " in p for p in own) > 0 for f in forms(key_value, rel))
        m = re.fullmatch(r"(\d{4})(?:-(\d\d)-(\d\d))?", key_value)
        if not m:
            return False
        mine = {page_index[t] for t in own_titles}
        ps = set(year_pages.get(m.group(1), ()))
        if m.group(2):
            mo, d = MONTHS[int(m.group(2)) - 1], str(int(m.group(3)))
            ps &= md_pages.get((mo, d), set())
        return bool(ps - mine)

    def subj_type(q):
        return "person" if "Q5" in types[q] else "country" if types[q] & COUNTRY else "place" if q in geo else "other"

    def region(q):
        off = min(pages[t] for t in titles_of[q])
        return "train" if off < 7340032 else "holdout" if off < 8388608 else "rest"

    # Rows: true facts, then one false fact each.
    held = defaultdict(set)  # (item, rel) -> value keys the subject holds (any statement)
    for (item, rel, v) in stmts:
        held[(item, rel)].add(v)
    pool, tpool = defaultdict(list), defaultdict(list)  # relation -> values; (subject type, relation) -> values
    for (item, rel, v) in sorted(stmts):
        pool[rel].append(v)
        tpool[(subj_type(item), rel)].append(v)
    rows, skipped, fallback = [], Counter(), Counter()
    for key in sorted(stmts, key=lambda k: (int(k[0][1:]), k[1], k[2])):
        item, rel, v = key
        title = sorted(titles_of[item], key=lambda t: pages[t])[0]
        cat, date, kind = classify(key)
        on = any(on_text(norm_page[t], v, rel) for t in titles_of[item])
        by = ""
        if cat == "old":
            nv = ws.norm(show_of[key]) if not vitem_of[key] else None
            for src in ("store", "seed"):
                objs = set().union(*(stated.get((t, rel, src), set()) for t in titles_of[item]))
                if (nv and nv in objs) or (not nv and objs & set(forms(v, rel))):
                    by = src
                    break
            by = by or ("page" if on else "")
            cat = "a" if by else "b"
        fid = f"{PREFIX}-" + hashlib.sha256(f"{item}|{rel}|{v}".encode()).hexdigest()[:12]
        base = [title, item, subj_type(item), region(item), rel, prop_of[key]]
        rows.append([fid] + base + [show_of[key], vitem_of[key], date, kind, cat, by, str(int(on)),
                                    str(int(elsewhere(titles_of[item], v, rel))), "1", "N" + fid[1:]])
        rng = random.Random(hashlib.sha256(f"{PREFIX}|neg|{fid}".encode()).digest())
        w = None
        for p in (tpool[(subj_type(item), rel)], pool[rel]):  # the subject's type first, then the whole relation
            for _ in range(100):
                x = p[rng.randrange(len(p))]
                if x not in held[(item, rel)]:
                    w = x
                    break
            if w is not None:
                break
            fallback[rel] += 1
        if w is None:
            skipped[rel] += 1
            rows[-1][-1] = ""
            continue
        wshow = (surf.get(w) or [w])[0] if re.fullmatch(r"Q\d+", w) else w
        won = any(on_text(norm_page[t], w, rel) for t in titles_of[item])
        rows.append(["N" + fid[1:]] + base + [wshow, w if re.fullmatch(r"Q\d+", w) else "", "", "none", cat, "",
                                             str(int(won)), str(int(elsewhere(titles_of[item], w, rel))), "0", fid])

    os.makedirs(outdir, exist_ok=True)
    for split in ("dev", "test"):
        keep = [r for r in rows if (hashlib.sha256(f"{PREFIX}|{r[2]}".encode()).digest()[0] >= 128) == (split == "test")]
        with open(f"{outdir}/{OUTNAME}_{split}.tsv", "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(COLS) + "\n")
            for r in keep:
                f.write("\t".join(x.replace("\t", " ").replace("\n", " ") for x in r) + "\n")
    stats = {"items": len(items), "statements_entity_rows": len(ent), "statements_time_rows": len(tim),
             "values_labelled": len(surf), "facts": len(stmts), "negatives_skipped": dict(skipped), "negatives_from_whole_relation": dict(fallback), "requests": ws.n_req}
    json.dump(stats, open(f"{run}/build_stats.json", "w"), indent=1, sort_keys=True)
    print(json.dumps(stats, sort_keys=True))


def check(outdir, files):
    """Lines of FILES that name a subject together with the value of one of its b or c facts (leak check)."""
    pairs = defaultdict(set)
    for split in ("dev", "test"):
        for line in list(open(f"{outdir}/temporal_holdout_{split}.tsv", encoding="utf-8"))[1:]:
            c = line.rstrip("\n").split("\t")
            if c[15] == "1" and c[11] in ("b", "c") and c[8]:  # entity values only: a bare year matches too much prose
                s, v = ws.norm(re.sub(r" \(.*\)$", "", c[1])), ws.norm(c[7])
                if s and v and s != v:
                    pairs[s].add((v, c[11]))
    first = defaultdict(int)
    for s in pairs:
        w = s.split(" ")
        first[w[0]] = max(first[w[0]], len(w))
    lines, found = 0, Counter()
    for path in files:
        for ln, line in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
            n = " " + ws.norm(line) + " "
            words = n.split()
            hit = set()
            for i, w in enumerate(words):
                for k in range(1, first.get(w, 0) + 1):
                    s = " ".join(words[i:i + k])
                    hit.update((s, v, cat) for v, cat in pairs.get(s, ()) if f" {v} " in n)
            for s, v, cat in sorted(hit):
                print(f"{path}:{ln}\t{cat}\t{s}\t{v}")
                found[(s, v, cat)] += 1
            lines += bool(hit)
    by = Counter(cat for _, _, cat in found)
    print(f"S15 check: {lines} lines name a subject with one of its b or c values; distinct pairs b {by['b']} c {by['c']}",
          file=sys.stderr)
    return 1 if lines else 0


if __name__ == "__main__":
    if len(sys.argv) >= 9 and sys.argv[1] == "build":
        build(*sys.argv[2:9])
    elif len(sys.argv) >= 4 and sys.argv[1] == "check":
        sys.exit(check(sys.argv[2], sys.argv[3:]))
    else:
        print(__doc__)
        sys.exit(2)
