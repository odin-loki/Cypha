#!/usr/bin/env python3
"""S8e: a fresh confirm set on slots S15 never held, and the Wikidata tables S8e's grader reads.

build: S15's builder (scripts/temporal_holdout.py) on relations S15 did not fetch, for the same enwik8 subjects (D4's map,
D4's T-QA exclusions kept): nationality (P27), location (P276), predecessor (P1365 replaces, P155 follows), successor (P1366
replaced by, P156 followed by), and the taxon's family, order and class (the P171 parent-taxon ancestors whose P105 rank is
family Q35409, order Q36602 or class Q37517). Direct statements only. Same columns, categories, negatives and file rules as
S15; ids "S8E-", split by subject: test if sha256("S8E|" + item)[0] >= 128. Writes s8e_slots_{dev,test}.tsv.

grader: for every Wikidata item a grade can touch (the value items of S15 and of the fresh set, D4's subject items, and
their place ancestors), the names S8e's grader matches (English label, every English alias, the enwiki title, English
demonyms P1549, the taxon name P225), the place ancestors ((P131 | P276 | P17)+), and two class flags by P31/P279*:
country-like (country Q6256, sovereign state Q3624078, historical country Q3024240, country of the UK Q3336843) and coarse
(country-like, first-level subdivision Q10864048, continent Q5107). Writes names.tsv, anc.tsv and flags.tsv.

Python standard library only; every query and response is cached under RUN/raw (D4's fetcher), so a rerun is byte for byte.
Usage:
  python3 -I scripts/fresh_slots.py build <enwik8> <pages.tsv> <D4 OUT> <facts_f2.tsv> <seeds.tsv> <RUN> <OUT dir>
  python3 -I scripts/fresh_slots.py grader <RUN> <OUT dir> <D4 OUT> <set file> ...
  python3 -I scripts/fresh_slots.py audit-draw <PREFIX.s8e_calib.tsv> <audit.tsv>
  python3 -I scripts/fresh_slots.py audit-info <RUN> <audit.tsv> <set file> ...   (prints what Wikidata says, for the hand check)
Author: Odin Loch
"""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import temporal_holdout as th  # noqa: E402
import wikidata_seeds as ws  # noqa: E402

ws.UA = "CyphaS8e/0.1 (https://github.com/odin-loki/Cypha; Odin Loch's Cypha project) python-urllib"
th.PREFIX, th.OUTNAME = "S8E", "s8e_slots"
th.ENTITY = {"P27": "nationality", "P276": "location", "P1365": "predecessor", "P155": "predecessor", "P1366": "successor",
             "P156": "successor"}
th.TIME = {}
TAXON = {"Q35409": "family", "Q36602": "order", "Q37517": "class"}
th.TIMELESS = th.TIMELESS | set(TAXON.values())
COUNTRYLIKE = ["Q6256", "Q3624078", "Q3024240", "Q3336843"]
COARSE = COUNTRYLIKE + ["Q10864048", "Q5107"]
_fetch_all = th.fetch_all


def label_rows(values, tag):
    """S15's label query (enwiki title, English label, six aliases): the display forms of the set's values."""
    surf = {}
    for i, b in enumerate(ws.blocks(values, 250)):
        q = (f"SELECT ?v ?l ?n (GROUP_CONCAT(DISTINCT ?a; separator=\"|\") AS ?as) WHERE {{ VALUES ?v {{ {' '.join('wd:' + x for x in b)} }} "
             "OPTIONAL { ?v rdfs:label ?l FILTER(LANG(?l) = \"en\") } OPTIONAL { ?v skos:altLabel ?a FILTER(LANG(?a) = \"en\") } "
             f"OPTIONAL {{ ?art schema:about ?v ; schema:isPartOf {ws.ENWIKI} ; schema:name ?n }} }} GROUP BY ?v ?l ?n")
        for r in ws.sparql(q, f"{tag}{i:03d}"):
            forms = [r.get("n", ""), r.get("l", "")] + sorted(a for a in r.get("as", "").split("|") if len(a) >= 3)[:6]
            surf[r["v"]] = [f.replace("\t", " ") for j, f in enumerate(forms) if f and f not in forms[:j]]
    return surf


def walk(start, path, tag, rank=False):
    """Up one hop a round: each item's direct (truthy) values on `path`, and its P105 rank when asked, until nothing is new.
    A transitive path in one query times out on Wikidata's endpoint for taxa, so the walk is done here."""
    key = lambda x: int(x[1:])  # noqa: E731
    parents, ranks, seen = {}, {}, set(start)
    frontier, depth = sorted(start, key=key), 0
    while frontier and depth < 40:
        for i, b in enumerate(ws.blocks(frontier, 200)):
            its = " ".join("wd:" + x for x in b)
            q = f"SELECT ?x ?p ?r WHERE {{ VALUES ?x {{ {its} }} OPTIONAL {{ ?x {path} ?p }}" + (" OPTIONAL { ?x wdt:P105 ?r }" if rank else "") + " }"
            for r in ws.sparql(q, f"{tag}{depth:02d}_{i:03d}"):
                if re.fullmatch(r"Q\d+", r.get("p", "")) and r["p"] != r["x"]:
                    parents.setdefault(r["x"], set()).add(r["p"])
                if r.get("r"):
                    ranks[r["x"]] = r["r"]
        new = {p for x in frontier for p in parents.get(x, ()) if p not in seen}
        seen |= new
        frontier, depth = sorted(new, key=key), depth + 1
    return parents, ranks


def fetch_all(items):
    """S15's fetch for the entity relations above, plus each taxon's nearest family, order and class ancestor by P171
    (truthy parents) and P105 (no qualifiers: a taxon's rank does not date)."""
    types, geo, ent, tim, surf = _fetch_all(items)
    parents, ranks = walk(items, "wdt:P171", "taxp", rank=True)
    tax = []
    for x in items:
        if x not in parents:
            continue
        found, level, seen = {}, set(parents[x]), {x}
        while level:  # breadth first: the nearest ancestor of each rank
            for a in sorted(level, key=lambda y: int(y[1:])):
                rel = TAXON.get(ranks.get(a, ""))
                if rel and rel not in found:
                    found[rel] = a
            seen |= level
            level = {p for a in level for p in parents.get(a, ()) if p not in seen}
        for rel, a in sorted(found.items()):
            tax.append({"item": x, "prop": "P171-" + rel, "st": f"tax-{x}-{a}", "v": a})
    th.ENTITY.update({"P171-" + rel: rel for rel in TAXON.values()})
    new = sorted({r["v"] for r in tax} - set(surf), key=lambda x: int(x[1:]))
    surf.update(label_rows(new, "taxlab"))
    return types, geo, ent + tax, tim, surf


def grader_tables(run, outdir, d4, sets):
    ws.RAW = f"{run}/raw"
    os.makedirs(ws.RAW, exist_ok=True)
    items = set()
    for line in list(open(f"{d4}/items.tsv", encoding="utf-8"))[1:]:
        items.add(line.rstrip("\n").split("\t")[1])
    for path in sets:
        for line in list(open(path, encoding="utf-8"))[1:]:
            c = line.rstrip("\n").split("\t")
            if len(c) > 8 and re.fullmatch(r"Q\d+", c[8]):
                items.add(c[8])
    key = lambda x: int(x[1:])  # noqa: E731
    parents, _ = walk(items, "(wdt:P131|wdt:P276|wdt:P17)", "place")
    anc = {}
    for x in parents:  # the place ancestors: everything reachable upward
        seen, stack = set(), list(parents[x])
        while stack:
            a = stack.pop()
            if a in seen or a == x:
                continue
            seen.add(a)
            stack.extend(parents.get(a, ()))
        anc[x] = seen
    more = set(parents) | set().union(*parents.values()) if parents else set()
    more -= items
    every = sorted(items | more, key=key)
    names = {}
    for i, b in enumerate(ws.blocks(every, 200)):
        its = " ".join("wd:" + x for x in b)
        q = (f"SELECT ?v ?n WHERE {{ VALUES ?v {{ {its} }} {{ ?v rdfs:label ?n FILTER(LANG(?n) = \"en\") }} UNION "
             "{ ?v skos:altLabel ?n FILTER(LANG(?n) = \"en\") } UNION { ?v wdt:P1549 ?n FILTER(LANG(?n) = \"en\") } UNION "
             f"{{ ?v wdt:P225 ?n }} UNION {{ ?art schema:about ?v ; schema:isPartOf {ws.ENWIKI} ; schema:name ?n }} }}")
        for r in ws.sparql(q, f"names{i:03d}"):
            names.setdefault(r["v"], set()).add(r["n"].replace("\t", " ").replace("|", " "))
    flags = {}
    for i, b in enumerate(ws.blocks(every, 100)):
        its = " ".join("wd:" + x for x in b)
        q = (f"SELECT DISTINCT ?x ?c WHERE {{ VALUES ?x {{ {its} }} VALUES ?c {{ {' '.join('wd:' + c for c in COARSE)} }} "
             "?x wdt:P31/wdt:P279* ?c . }")
        for r in ws.sparql(q, f"flag{i:03d}"):
            flags.setdefault(r["x"], set()).add(r["c"])
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/names.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("item\tnames\n")
        for x in every:
            if names.get(x):
                f.write(x + "\t" + "|".join(sorted(names[x])) + "\n")
    with open(f"{outdir}/anc.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("item\tancestors\n")
        for x in every:
            if anc.get(x):
                f.write(x + "\t" + "|".join(sorted(anc[x], key=key)) + "\n")
    with open(f"{outdir}/flags.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("item\tcountry_like\tcoarse\n")
        for x in every:
            if flags.get(x):
                f.write(f"{x}\t{int(bool(flags[x] & set(COUNTRYLIKE)))}\t1\n")
    print(f"S8e grader tables: items {len(every)} (named {sum(1 for x in every if names.get(x))}, with ancestors "
          f"{sum(1 for x in every if anc.get(x))}, flagged {len(flags)}), requests {ws.n_req}")


def audit_draw(calib, out, seed=20261029):
    """The grader audit's sample (Part 1): 100 graded calibration items, 60 where S8e's grade differs from S8c's grader or
    rests on rules 2-5 (rule2, rule3, rule5), 40 where both agree on another basis; random.Random(seed) on sorted rows."""
    import random
    rows = [line.rstrip("\n").split("\t") for line in list(open(calib, encoding="utf-8"))[1:]]
    head = open(calib, encoding="utf-8").readline().rstrip("\n").split("\t")
    ix = {h: i for i, h in enumerate(head)}
    graded = sorted(r for r in rows if int(r[ix["grade_s8e"]]) >= 0)
    a = [r for r in graded if r[ix["grade_s8e"]] != r[ix["grade_s8c"]] or r[ix["basis"]] in ("rule2", "rule3", "rule5")]
    b = [r for r in graded if r not in a]
    rng = random.Random(seed)
    da = rng.sample(a, min(60, len(a)))
    db = rng.sample(b, min(40, len(b)))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("n\tstratum\t" + "\t".join(head) + "\thand\tnote\n")
        for n, (st, r) in enumerate([("A", r) for r in da] + [("B", r) for r in db], 1):
            f.write(f"{n}\t{st}\t" + "\t".join(r) + "\t\t\n")
    print(f"S8e audit draw: graded {len(graded)} | stratum A {len(a)} -> {len(da)} | stratum B {len(b)} -> {len(db)}")


def audit_info(run, audit, sets):
    """For each audit line, what Wikidata says, read from the entity pages (wbgetentities, every rank) rather than the
    grader's tables: the subject's statements on the slot's properties, each value's names, classes and containing
    places, and the items Wikidata's search finds for the hypothesis. Printed for a hand check; cached under RUN/raw."""
    import hashlib, json, urllib.parse
    ws.RAW = f"{run}/raw"
    os.makedirs(ws.RAW, exist_ok=True)
    api = "https://www.wikidata.org/w/api.php?"
    ent = {}

    def get(ids):
        need = sorted({x for x in ids if x and x not in ent}, key=lambda y: int(y[1:]))
        for i in range(0, len(need), 50):
            b = need[i:i + 50]
            q = urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(b), "props": "labels|aliases|claims|descriptions",
                                        "languages": "en", "format": "json"})
            tag = "ent_" + hashlib.sha256("|".join(b).encode()).hexdigest()[:20]
            res = ws.fetch(api + q, None, f"{ws.RAW}/{tag}.json")
            for k, v in res.get("entities", {}).items():
                ent[k] = v

    def lab(x):
        e = ent.get(x, {})
        return e.get("labels", {}).get("en", {}).get("value", "?")

    def vals(x, p):
        out = []
        for c in ent.get(x, {}).get("claims", {}).get(p, []):
            dv = c.get("mainsnak", {}).get("datavalue", {})
            v = dv.get("value")
            if isinstance(v, dict) and "id" in v:
                q = {qp: [((qq.get("datavalue") or {}).get("value") or {}).get("time", "?") for qq in qs] for qp, qs in c.get("qualifiers", {}).items() if qp in ("P580", "P582", "P585")}
                out.append((v["id"], c.get("rank", "?"), q))
            elif isinstance(v, dict) and "text" in v:
                if v.get("language") == "en":
                    out.append((v["text"], c.get("rank", "?"), {}))
            elif isinstance(v, str):
                out.append((v, c.get("rank", "?"), {}))
        return out

    def chain(x, props, depth):
        lines, frontier, seen = [], [x], {x}
        for d in range(depth):
            get(frontier)
            ups_of = {y: [(p, v) for p in props for v, rk, q in vals(y, p) if rk != "deprecated"][:6] for y in frontier}
            get([v for ups in ups_of.values() for p, v in ups])
            nxt = []
            for y in frontier:
                ups = ups_of[y]
                if ups:
                    lines.append("    " * (d + 1) + f"{lab(y)} ({y}) -> " + ", ".join(f"{p}:{lab(v)}({v})" for p, v in ups))
                for p, v in ups:
                    if v not in seen:
                        seen.add(v), nxt.append(v)
            frontier = nxt
            if not frontier:
                break
        return lines

    def names(x):
        e = ent.get(x, {})
        al = [a["value"] for a in e.get("aliases", {}).get("en", [])][:10]
        dem = [v for v, rk, q in vals(x, "P1549")]
        return f"label '{lab(x)}' aliases {al} demonyms {dem[:6]} P31 {[lab(v) for v, rk, q in vals(x, 'P31')][:4]}"

    slot = {}
    for path in sets:
        for line in list(open(path, encoding="utf-8"))[1:]:
            c = line.rstrip("\n").split("\t")
            slot.setdefault((c[1], c[5]), (c[2], set()))[1].add(c[6])
    rows = [line.rstrip("\n").split("\t") for line in list(open(audit, encoding="utf-8"))[1:]]
    head = open(audit, encoding="utf-8").readline().rstrip("\n").split("\t")
    ix = {h: i for i, h in enumerate(head)}

    def search(o):
        import time
        q = urllib.parse.urlencode({"action": "wbsearchentities", "search": o, "language": "en", "limit": "3", "format": "json"})
        path = f"{ws.RAW}/search_" + "".join(ch if ch.isalnum() else "_" for ch in o)[:80] + ".json"
        for attempt in range(6):  # "search is too busy" comes back as a 200 with an error body: not kept
            res = ws.fetch(api + q, None, path)
            if "error" not in res:
                return [h["id"] for h in res.get("search", [])]
            os.remove(path)
            time.sleep(15 * (attempt + 1))
        return []

    def up(xs, props, depth):  # prefetch the containing places (or parent taxa) of many items at once, level by level
        frontier, seen = [x for x in xs if x.startswith("Q")], set(xs)
        for d in range(depth):
            get(frontier)
            nxt = sorted({v for y in frontier for p in props for v, rk, q in vals(y, p) if v.startswith("Q") and v not in seen}, key=lambda y: int(y[1:]))
            seen |= set(nxt)
            frontier = nxt
            if not frontier:
                break
        get(frontier)

    subj = [slot.get((r[ix["subject"]], r[ix["relation"]]), ("", set()))[0] for r in rows]
    get(subj)
    vs_all, taxa = [], []
    for r, it in zip(rows, subj):
        for p in slot.get((r[ix["subject"]], r[ix["relation"]]), ("", set()))[1]:
            if p.startswith("P171-"):
                taxa.append(it)
            else:
                vs_all += [v for v, rk, q in vals(it, p) if v.startswith("Q")]
    up(vs_all, ["P131", "P276", "P17"], 4)
    get([v for x in vs_all for v, rk, q in vals(x, "P31")])
    up(taxa, ["P171", "P105"], 14)
    with_search = not os.environ.get("S8E_NO_SEARCH")  # Wikidata's search is often "too busy": the entity part can be read first
    if with_search:
        hits = [h for r in rows for h in search(r[ix["object"]])]
        up(hits, ["P131", "P17"], 3)
        get([v for x in hits for v, rk, q in vals(x, "P31")])
    for r in rows:
        s, rel, o = r[ix["subject"]], r[ix["relation"]], r[ix["object"]]
        item, props = slot.get((s, rel), ("", set()))
        print(f"=== {r[0]} [{r[1]}] {s} ({item}) | {rel} | h = '{o}' | S8c {r[ix['grade_s8c']]} S8e {r[ix['grade_s8e']]} ({r[ix['basis']]}) | slot {r[ix['slot_values']]}")
        for p in sorted(props):
            if p.startswith("P171-"):
                for ln in chain(item, ["P171", "P105"], 14):
                    print(ln)
                continue
            vs = vals(item, p)
            get([v for v, rk, q in vs if v.startswith("Q")])
            print(f"  {p}: " + "; ".join(f"{lab(v)}({v},{rk}{',' + json.dumps(q) if q else ''})" for v, rk, q in vs))
            for v, rk, q in vs[:6]:
                if v.startswith("Q"):
                    get([v])
                    print(f"    {v}: {names(v)}")
                    for ln in chain(v, ["P131", "P276", "P17"], 4):
                        print("  " + ln)
        for h in search(o) if with_search else []:
            print(f"  search '{o}': {h} {lab(h)} — {ent.get(h, {}).get('descriptions', {}).get('en', {}).get('value', '')} | P31 {[lab(v) for v, rk, q in vals(h, 'P31')][:3]}")
            for ln in chain(h, ["P131", "P17"], 3):
                print("  " + ln)
    print(f"requests {ws.n_req}")


if __name__ == "__main__":
    if len(sys.argv) >= 9 and sys.argv[1] == "build":
        th.fetch_all = fetch_all
        th.build(*sys.argv[2:9])
    elif len(sys.argv) >= 6 and sys.argv[1] == "grader":
        grader_tables(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif len(sys.argv) == 4 and sys.argv[1] == "audit-draw":
        audit_draw(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 5 and sys.argv[1] == "audit-info":
        audit_info(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        print(__doc__)
        sys.exit(2)
