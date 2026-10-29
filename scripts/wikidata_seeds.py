#!/usr/bin/env python3
"""D4: Wikidata values for pattern_learn's relations. Distant supervision only: they choose and check patterns, never facts.

Maps enwik8 article titles (pattern_learn's pages.tsv) to Wikidata items: (1) enwiki sitelinks (SPARQL); (2) for titles
renamed since 2006, the enwiki query API with redirects resolved (wikibase_item page prop), within a time budget because
the API throttles shared addresses; (3) for the rest, an English label or alias held by exactly one item with an enwiki
article. Then fetches the relation values with their enwiki title, English label and aliases (the surface forms
pattern_learn looks for in the text), and dates with their precision.

Subjects that are a T-QA dev or test source_title, or whose name occurs in a dev or test question, are excluded before
any query. Only the question and source_title columns are read (string comparison); answers are never read.

Writes to OUT: wd_values.tsv (subject, relation, surfaces | separated, subject item, value), phi.tsv (share of subjects
with exactly one value), excluded.tsv, unmapped.tsv, items.tsv (title, item; S15 reads it), manifest.json, and every
query and raw response under raw/. The helpers are importable (scripts/temporal_holdout.py uses them).
Usage: python3 -I scripts/wikidata_seeds.py <pages.tsv> <dev.tsv> <test.tsv> <OUT> [API budget in minutes, default 30]
Author: Odin Loch
"""
import json, os, re, sys, time, unicodedata, urllib.error, urllib.parse, urllib.request

UA = "CyphaD4/0.1 (https://github.com/odin-loki/Cypha; Odin Loch's Cypha project) python-urllib"
SPARQL, API, ENWIKI = "https://query.wikidata.org/sparql", "https://en.wikipedia.org/w/api.php", "<https://en.wikipedia.org/>"
ENTITY = {"P36": "capital", "P50": "author", "P57": "director", "P26": "spouse", "P86": "composer", "P112": "founder",
          "P37": "language", "P17": "country", "P61": "inventor", "P170": "creator", "P84": "creator", "P136": "genre",
          "P19": "birth_place", "P20": "death_place", "P131": "located_in", "P38": "currency", "P106": "occupation",
          "P178": "developer", "P123": "publisher"}
TIME = {"P569": "birth", "P570": "death", "P571": "founded"}
n_req = 0


def norm(s):
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s.lower()).split())


def fetch(url, data=None, raw=None):
    """One request at a polite pace; backs off on 429/5xx (honouring Retry-After) and on dropped connections.
    A response already saved under raw/ is reused, so a rerun reprocesses the same data without querying again."""
    global n_req
    if os.path.exists(raw):
        return json.load(open(raw))
    for attempt in range(12):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read().decode("utf-8")
            n_req += 1
            open(raw, "w").write(body)
            time.sleep(1.0)
            return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(max(float(e.headers.get("Retry-After") or 0), 5 * 2 ** min(attempt, 4)))
        except OSError:  # connection reset, timeout, tunnel closed
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("gave up on " + url[:120])


def sparql(query, tag):
    open(f"{RAW}/{tag}.rq", "w").write(query)
    res = fetch(SPARQL, urllib.parse.urlencode({"query": query}).encode(), f"{RAW}/{tag}.json")
    return [{k: v["value"].rsplit("/", 1)[-1] if v["type"] == "uri" else v["value"] for k, v in b.items()}
            for b in res["results"]["bindings"]]


def blocks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def lits(ts):
    return " ".join('"' + t.replace("\\", "\\\\").replace('"', '\\"') + '"@en' for t in ts)


def date_str(t, prec, year_only, julian):  # "1932-04-04T00:00:00Z" -> 1932-04-04 (day precision) or 1932; "-0469-..." -> "470 BC"
    m = re.match(r"(-?\d+)-(\d\d)-(\d\d)", t)  # WDQS years are astronomical (year 0 is 1 BC) and day dates Gregorian
    if not m or prec < 9 or (not year_only and prec < 11):
        return None
    y, mo, d = (int(x) for x in m.groups())
    if julian and prec >= 11:  # back to the calendar the statement was made in (the text's calendar): JDN -> Julian date
        a = (14 - mo) // 12
        j = d + (153 * (mo + 12 * a - 3) + 2) // 5 + 365 * (y + 4800 - a) + (y + 4800 - a) // 4 - (y + 4800 - a) // 100 + (y + 4800 - a) // 400 - 32045
        c = j + 32082
        dd = (4 * c + 3) // 1461
        e = c - 1461 * dd // 4
        mm = (5 * e + 2) // 153
        y, mo, d = dd - 4800 + mm // 10, mm + 3 - 12 * (mm // 10), e - (153 * mm + 2) // 5 + 1
    return f"{1 - y} BC" if y < 1 else str(y) if year_only else f"{y:04d}-{mo:02d}-{d:02d}"


def main():
    global RAW
    pages, dev, test, OUT = sys.argv[1:5]
    budget = 60 * float(sys.argv[5]) if len(sys.argv) > 5 else 1800.0
    RAW = f"{OUT}/raw"
    os.makedirs(RAW, exist_ok=True)
    titles = [l.split("\t")[0] for l in open(pages, encoding="utf-8") if l.strip()]
    srcs, qs = set(), []  # dev/test source titles and questions (columns 2 and 0; answers are never read)
    for path in (dev, test):
        for line in list(open(path, encoding="utf-8"))[1:]:
            c = line.rstrip("\n").split("\t")
            qs.append(" " + norm(c[0]) + " ")
            if len(c) > 2 and c[2]:
                srcs.add(norm(c[2]))
    base = {t: norm(re.sub(r" \(.*\)$", "", t)) for t in titles}
    excluded = [t for t in titles if norm(t) in srcs or base[t] in srcs or any(" " + base[t] + " " in q for q in qs)]
    ex = set(excluded)
    subjects = [t for t in titles if t not in ex]
    item = {}
    for i, b in enumerate(blocks(subjects, 200)):  # (1) sitelinks
        q = f"SELECT ?t ?item WHERE {{ VALUES ?t {{ {lits(b)} }} ?art schema:name ?t ; schema:isPartOf {ENWIKI} ; schema:about ?item . }}"
        item.update({r["t"]: r["item"] for r in sparql(q, f"map{i:03d}")})
    by_sitelink, t_api = len(item), time.time()
    for i, b in enumerate(blocks([t for t in subjects if t not in item], 50)):  # (2) enwiki redirects
        if time.time() - t_api > budget:
            break
        try:
            res = fetch(API + "?" + urllib.parse.urlencode({"action": "query", "titles": "|".join(b), "redirects": 1, "prop": "pageprops",
                                                            "ppprop": "wikibase_item", "format": "json", "formatversion": 2}), raw=f"{RAW}/api{i:03d}.json")
        except RuntimeError:
            break
        qr = res.get("query", {})
        hop = {n["from"]: n["to"] for n in qr.get("normalized", []) + qr.get("redirects", [])}
        wb = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in qr.get("pages", [])}
        for t in b:
            x = t
            for _ in range(3):  # normalised title, then up to two redirects
                x = hop.get(x, x)
            if wb.get(x):
                item[t] = wb[x]
    by_api, api_min = len(item) - by_sitelink, round((time.time() - t_api) / 60, 1)
    for i, b in enumerate(blocks([t for t in subjects if t not in item], 200)):  # (3) a label or alias held by one item only
        q = (f"SELECT ?t (COUNT(DISTINCT ?item) AS ?n) (SAMPLE(?item) AS ?one) WHERE {{ VALUES ?t {{ {lits(b)} }} {{ ?item rdfs:label ?t }} "
             f"UNION {{ ?item skos:altLabel ?t }} ?art schema:about ?item ; schema:isPartOf {ENWIKI} . }} GROUP BY ?t")
        item.update({r["t"]: r["one"] for r in sparql(q, f"label{i:03d}") if r["n"] == "1"})
    by_label = len(item) - by_sitelink - by_api
    inv = {}
    for t, q in item.items():
        inv.setdefault(q, []).append(t)
    rows, values = [], set()
    for i, b in enumerate(blocks(sorted(inv), 150)):  # values: entity-valued (truthy) and dates (non-deprecated, with precision)
        its = " ".join("wd:" + x for x in b)
        q = (f"SELECT ?item ?prop ?v WHERE {{ VALUES ?item {{ {its} }} VALUES ?prop {{ {' '.join('wd:' + p for p in ENTITY)} }} "
             "?prop wikibase:directClaim ?d . ?item ?d ?v . FILTER(isIRI(?v)) }")
        rows += [(r["item"], r["prop"], r["v"], None) for r in sparql(q, f"ent{i:03d}")]
        q = (f"SELECT ?item ?prop ?t ?pr ?cal WHERE {{ VALUES ?item {{ {its} }} VALUES ?prop {{ {' '.join('wd:' + p for p in TIME)} }} "
             "?prop wikibase:claim ?c ; wikibase:statementValue ?sv . ?item ?c ?st . ?st ?sv ?n ; wikibase:rank ?rk . "
             "?n wikibase:timeValue ?t ; wikibase:timePrecision ?pr ; wikibase:timeCalendarModel ?cal . FILTER(?rk != wikibase:DeprecatedRank) }")
        rows += [(r["item"], r["prop"], r["t"], (int(r["pr"]), r["cal"] == "Q1985786")) for r in sparql(q, f"time{i:03d}")]
    values = sorted({v for _, _, v, pr in rows if pr is None})
    surf = {}
    for i, b in enumerate(blocks(values, 250)):  # value surface forms: enwiki title, English label, up to 6 aliases
        q = (f"SELECT ?v ?l ?n (GROUP_CONCAT(DISTINCT ?a; separator=\"|\") AS ?as) WHERE {{ VALUES ?v {{ {' '.join('wd:' + x for x in b)} }} "
             "OPTIONAL { ?v rdfs:label ?l FILTER(LANG(?l) = \"en\") } OPTIONAL { ?v skos:altLabel ?a FILTER(LANG(?a) = \"en\") } "
             f"OPTIONAL {{ ?art schema:about ?v ; schema:isPartOf {ENWIKI} ; schema:name ?n }} }} GROUP BY ?v ?l ?n")
        for r in sparql(q, f"lab{i:03d}"):
            forms = [r.get("n", ""), r.get("l", "")] + [a for a in r.get("as", "").split("|") if len(a) >= 3][:6]
            surf[r["v"]] = [f.replace("\t", " ") for j, f in enumerate(forms) if f and f not in forms[:j]]
    out, per = set(), {}
    for s, p, v, pr in rows:
        if pr is None:
            rels = [(ENTITY[p], surf.get(v))]
        elif TIME[p] == "founded":
            rels = [("founded", [date_str(v, pr[0], True, pr[1])])]
        else:
            rels = [(TIME[p] + "_year", [date_str(v, pr[0], True, pr[1])]), (TIME[p] + "_date", [date_str(v, pr[0], False, pr[1])])]
        for rel, forms in rels:
            for t in inv[s] if forms and forms[0] else []:
                out.add((t, rel, "|".join(forms), s, v if pr is None else forms[0]))
                per.setdefault((rel, t), set()).add(forms[0])
    open(f"{OUT}/wd_values.tsv", "w").write("subject\trelation\tsurfaces\tsubject_item\tvalue\n" + "".join("\t".join(r) + "\n" for r in sorted(out)))
    phi = {}
    for (rel, _), vs in per.items():
        phi[rel] = [phi.get(rel, [0, 0])[0] + 1, phi.get(rel, [0, 0])[1] + (len(vs) == 1)]
    open(f"{OUT}/phi.tsv", "w").write("relation\tsubjects\texactly_one\tphi\n" + "".join(f"{r}\t{a}\t{b}\t{b / a:.3f}\n" for r, (a, b) in sorted(phi.items())))
    open(f"{OUT}/excluded.tsv", "w").write("".join(t + "\n" for t in excluded))
    open(f"{OUT}/unmapped.tsv", "w").write("".join(t + "\n" for t in subjects if t not in item))
    open(f"{OUT}/items.tsv", "w").write("title\titem\n" + "".join(f"{t}\t{item[t]}\n" for t in subjects if t in item))
    man = {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "endpoints": [SPARQL, API], "user_agent": UA,
           "properties": {**ENTITY, **TIME}, "titles": len(titles), "excluded": len(excluded), "subjects": len(subjects),
           "mapped_by_sitelink": by_sitelink, "mapped_by_api_redirect": by_api, "api_minutes": api_min, "mapped_by_unique_label": by_label,
           "mapped_total": len(item), "items": len(inv), "statements": len(rows), "values_labelled": len(surf), "rows": len(out), "requests": n_req}
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
    print(json.dumps(man))


if __name__ == "__main__":
    main()
