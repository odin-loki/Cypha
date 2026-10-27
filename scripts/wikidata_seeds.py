#!/usr/bin/env python3
"""D4: Wikidata values for pattern_learn's relations (distant supervision only; they never become facts).

Maps enwik8 article titles (pattern_learn's pages.tsv) to Wikidata items through enwiki sitelinks (SPARQL), then the
action API with redirects=yes for titles renamed since 2006. Fetches the relation values with their enwiki title, English
label and aliases, so pattern_learn can find the object's surface form in the text.

Subjects that are a T-QA dev or test source_title, or whose name occurs in a dev or test question, are excluded before any
query. Only the question and source_title columns are read (string comparison); answers are never read.

Writes to OUT: wd_values.tsv (subject, relation, surfaces | separated, subject item, value), phi.tsv (share of subjects
with exactly one value), excluded.tsv, unmapped.tsv, manifest.json, and every query and raw response under raw/.
Usage: python3 -I scripts/wikidata_seeds.py <pages.tsv> <dev.tsv> <test.tsv> <OUT>
Author: Odin Loch
"""
import json, os, re, sys, time, unicodedata, urllib.error, urllib.parse, urllib.request

UA = "CyphaD4/0.1 (https://github.com/odin-loki/Cypha; Odin Loch's Cypha project) python-urllib"
SPARQL, API = "https://query.wikidata.org/sparql", "https://www.wikidata.org/w/api.php"
ENTITY = {"P36": "capital", "P50": "author", "P57": "director", "P26": "spouse", "P86": "composer", "P112": "founder",
          "P37": "language", "P17": "country", "P61": "inventor", "P170": "creator", "P84": "creator", "P136": "genre",
          "P19": "birth_place", "P20": "death_place", "P131": "located_in", "P38": "currency", "P106": "occupation",
          "P178": "developer", "P123": "publisher"}
TIME = {"P569": "birth", "P570": "death", "P571": "founded"}
MONTHS = "January February March April May June July August September October November December".split()
n_req = 0


def norm(s):
    s = unicodedata.normalize("NFKD", s)
    return " ".join(re.sub(r"[^0-9a-z]+", " ", "".join(c for c in s if not unicodedata.combining(c)).lower()).split())


def fetch(url, data=None, raw=None):
    """GET/POST with a polite pace; back off on 429/5xx, honouring Retry-After."""
    global n_req
    for attempt in range(12):
        req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read().decode("utf-8")
            n_req += 1
            if raw:
                with open(raw, "w") as f:
                    f.write(body)
            time.sleep(1.0)
            return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(max(float(e.headers.get("Retry-After") or 0), 5 * 2 ** min(attempt, 4)))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("gave up on " + url[:120])


def sparql(query, tag):
    with open(f"{RAW}/{tag}.rq", "w") as f:
        f.write(query)
    res = fetch(SPARQL, urllib.parse.urlencode({"query": query}).encode(), f"{RAW}/{tag}.json")
    return [{k: v["value"] for k, v in b.items()} for b in res["results"]["bindings"]]


def qid(uri):
    return uri.rsplit("/", 1)[-1]


def lit(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"@en'


def blocks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def date_str(t, prec, year_only):
    m = re.match(r"([+-])(\d+)-(\d\d)-(\d\d)", t)
    if not m or int(prec) < 9:
        return None
    y = int(m.group(2))
    if m.group(1) == "-":
        return f"{y} BC"
    if year_only or int(prec) == 9:
        return str(y)
    return f"{y:04d}-{m.group(3)}-{m.group(4)}" if int(prec) >= 11 else f"{MONTHS[int(m.group(3)) - 1]} {y}"


pages, dev, test, OUT = sys.argv[1:5]
RAW = f"{OUT}/raw"
os.makedirs(RAW, exist_ok=True)
titles = [l.split("\t")[0] for l in open(pages, encoding="utf-8") if l.strip()]
# Exclusion: dev/test source titles and question entities (columns 0 and 2 only; answers are never read).
srcs, qs = set(), []
for path in (dev, test):
    for line in list(open(path, encoding="utf-8"))[1:]:
        c = line.rstrip("\n").split("\t")
        qs.append(" " + norm(c[0]) + " ")
        if len(c) > 2 and c[2]:
            srcs.add(norm(c[2]))
excluded = [t for t in titles if norm(t) in srcs or norm(re.sub(r" \(.*\)$", "", t)) in srcs
            or any(" " + norm(re.sub(r" \(.*\)$", "", t)) + " " in q for q in qs)]
ex = set(excluded)
subjects = [t for t in titles if t not in ex]
# Titles -> items: sitelinks (exact title), then the action API with redirects for the rest.
item = {}
for i, b in enumerate(blocks(subjects, 200)):
    q = ("SELECT ?t ?item WHERE { VALUES ?t { " + " ".join(lit(t) for t in b) + " } ?art schema:name ?t ; "
         "schema:isPartOf <https://en.wikipedia.org/> ; schema:about ?item . }")
    for r in sparql(q, f"map{i:03d}"):
        item[r["t"]] = qid(r["item"])
by_sitelink = len(item)
for i, b in enumerate(blocks([t for t in subjects if t not in item], 50)):
    url = API + "?" + urllib.parse.urlencode({"action": "wbgetentities", "sites": "enwiki", "titles": "|".join(b),
                                              "redirects": "yes", "props": "sitelinks", "sitefilter": "enwiki", "format": "json"})
    res = fetch(url, raw=f"{RAW}/api{i:03d}.json")
    for e in res.get("entities", {}).values():
        if "missing" in e:
            continue
        name = e.get("sitelinks", {}).get("enwiki", {}).get("title", "")
        for t in b:  # the requested title whose (normalised) form or redirect target this item answers
            if t not in item and norm(t) == norm(name):
                item[t] = e["id"]
    for n in res.get("normalized", []) + res.get("redirects", []):
        pass  # the API answers per resolved title; unmatched redirected titles are matched below by their target
    for e in res.get("entities", {}).values():
        if "missing" not in e and e["id"] not in item.values():
            name = e.get("sitelinks", {}).get("enwiki", {}).get("title", "")
            cand = [t for t in b if t not in item and norm(name).startswith(norm(re.sub(r" \(.*\)$", "", t)).split(" ")[0])]
            if len(cand) == 1:
                item[cand[0]] = e["id"]
# Values: entity-valued (truthy) and dates (non-deprecated, with precision).
inv = {}
for t, q in item.items():
    inv.setdefault(q, []).append(t)
items = sorted(inv)
rows, values = [], set()
for i, b in enumerate(blocks(items, 150)):
    vals = " ".join("wd:" + x for x in b)
    q = ("SELECT ?item ?prop ?v WHERE { VALUES ?item { " + vals + " } VALUES ?prop { " + " ".join("wd:" + p for p in ENTITY)
         + " } ?prop wikibase:directClaim ?d . ?item ?d ?v . FILTER(isIRI(?v)) }")
    for r in sparql(q, f"ent{i:03d}"):
        rows.append((qid(r["item"]), qid(r["prop"]), qid(r["v"]), None))
        values.add(qid(r["v"]))
    q = ("SELECT ?item ?prop ?t ?pr WHERE { VALUES ?item { " + vals + " } VALUES ?prop { " + " ".join("wd:" + p for p in TIME)
         + " } ?prop wikibase:claim ?c ; wikibase:statementValue ?sv . ?item ?c ?st . ?st ?sv ?n ; wikibase:rank ?rk . "
         "?n wikibase:timeValue ?t ; wikibase:timePrecision ?pr . FILTER(?rk != wikibase:DeprecatedRank) }")
    for r in sparql(q, f"time{i:03d}"):
        rows.append((qid(r["item"]), qid(r["prop"]), r["t"], r["pr"]))
surf = {}
for i, b in enumerate(blocks(sorted(values), 250)):
    q = ("SELECT ?v ?l ?n (GROUP_CONCAT(DISTINCT ?a; separator=\"|\") AS ?as) WHERE { VALUES ?v { " + " ".join("wd:" + x for x in b)
         + " } OPTIONAL { ?v rdfs:label ?l FILTER(LANG(?l) = \"en\") } OPTIONAL { ?v skos:altLabel ?a FILTER(LANG(?a) = \"en\") } "
         "OPTIONAL { ?art schema:about ?v ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?n } } GROUP BY ?v ?l ?n")
    for r in sparql(q, f"lab{i:03d}"):
        forms = [r.get("n", ""), r.get("l", "")] + [a for a in r.get("as", "").split("|") if len(a) >= 3][:6]
        surf[qid(r["v"])] = [f.replace("\t", " ") for j, f in enumerate(forms) if f and f not in forms[:j]]
# wd_values.tsv and phi (share of subjects with exactly one distinct value).
out, per = [], {}
for s, p, v, pr in rows:
    if pr is None:
        rels = [(ENTITY[p], surf.get(v))]
    else:
        base = TIME[p]
        rels = [(base, [date_str(v, pr, True)])] if base == "founded" else \
            [(base + "_year", [date_str(v, pr, True)]), (base + "_date", [date_str(v, pr, False)] if int(pr) >= 11 else None)]
    for rel, forms in rels:
        if not forms or not forms[0]:
            continue
        for t in inv[s]:
            out.append((t, rel, "|".join(forms), s, v if pr is None else forms[0]))
            per.setdefault((rel, t), set()).add(forms[0])
out = sorted(set(out))
with open(f"{OUT}/wd_values.tsv", "w", encoding="utf-8") as f:
    f.write("subject\trelation\tsurfaces\tsubject_item\tvalue\n" + "".join("\t".join(r) + "\n" for r in out))
phi = {}
for (rel, _), vs in per.items():
    a = phi.setdefault(rel, [0, 0])
    a[0] += 1
    a[1] += len(vs) == 1
with open(f"{OUT}/phi.tsv", "w") as f:
    f.write("relation\tsubjects\texactly_one\tphi\n" + "".join(f"{r}\t{a[0]}\t{a[1]}\t{a[1] / a[0]:.3f}\n" for r, a in sorted(phi.items())))
with open(f"{OUT}/excluded.tsv", "w", encoding="utf-8") as f:
    f.write("".join(t + "\n" for t in excluded))
with open(f"{OUT}/unmapped.tsv", "w", encoding="utf-8") as f:
    f.write("".join(t + "\n" for t in subjects if t not in item))
man = {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "endpoints": [SPARQL, API], "user_agent": UA,
       "properties": {**ENTITY, **{p: TIME[p] for p in TIME}}, "titles": len(titles), "excluded": len(excluded),
       "subjects": len(subjects), "mapped_by_sitelink": by_sitelink, "mapped_total": len(item), "items": len(items),
       "statements": len(rows), "values_labelled": len(surf), "rows": len(out), "requests": n_req}
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
print(json.dumps(man))
