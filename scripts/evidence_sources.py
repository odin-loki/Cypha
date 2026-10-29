#!/usr/bin/env python3
"""S8c: evidence sources other than Wikidata, as flat fact tables for pattern_learn --sources.

  subjects S15_DEV S8A_R1 S8A_R2 S8B_PROMO OUT.txt
                                  The subjects to look up: S15's dev subjects, S8a's two dev hypothesis files' subjects
                                  and S8b's dev rows' subjects, sorted, one a line.
  dbpedia  SUBJECTS.txt RAW OUT   DBpedia's mapping-based properties (and the raw infobox occupation) of each enwik8
                                  subject, from the public SPARQL endpoint. Redirects from the 2006 titles are followed.
                                  Writes OUT/dbpedia_ib.tsv (subject, relation, value, value_qid, property) and
                                  OUT/dbpedia_subjects.tsv (subject, resource, its Wikidata items). The endpoint no longer
                                  serves abstracts (2026-10-29), so the lead paragraph comes from wplead.
  wplead   SUBJECTS.txt RAW OUT   The current lead paragraph of each subject's article (Wikipedia action API extracts),
                                  in place of DBpedia's abstracts, which the endpoint no longer serves. Writes
                                  OUT/wp_lead.tsv (subject, extract, revision).
  wpdump   SUBJECTS.txt DBPEDIA_SUBJECTS.tsv RAW OUT
                                  The lead of each subject's article from the Wikipedia dump of 2023-11-01 as Hugging
                                  Face publishes it (wikimedia/wikipedia, 20231101.en, 41 parquet shards; needs pyarrow).
                                  Used because the action API answers 429 to this container's shared address. Shards are
                                  read one at a time and deleted; the matched leads stay under RAW (hits-NNNNN.tsv, with
                                  each shard's SHA-256 in RAW/manifest.tsv). Writes OUT/wp_lead.tsv like wplead.
  geonames RAW OUT                GeoNames' cities with 15,000+ people, country info and first-level divisions. Writes
                                  OUT/geonames.tsv (subject, relation, value) and OUT/geonames_alias.tsv (alias, subject).

Responses are cached under RAW (one file per request, named by the SHA-1 of the request), so a rerun from the cache is
byte for byte and makes no request. Python standard library only (run with python3 -I), except wpdump, which
needs pyarrow. docs/reports/S8C_EVIDENCE_SOURCES.md
has the rules. The tables are evidence for Cypha's own store, never answers for CyphaLM (PLAN rule 10).
Author: Odin Loch
"""
import hashlib, io, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request, zipfile

UA = "CyphaS8c/0.1 (https://github.com/odin-loki/Cypha; Odin Loch's Cypha project) python-urllib"
ENDPOINT = "https://dbpedia.org/sparql"
DBO = "http://dbpedia.org/ontology/"
DBP = "http://dbpedia.org/property/"
DBR = "http://dbpedia.org/resource/"
WD = "http://www.wikidata.org/entity/"
REQUESTS = 0

# D4's relations (and S8's) -> DBpedia ontology properties. occupation also reads the raw infobox field (dbp:occupation).
PROPS = {
    "birth_place": ["birthPlace"], "death_place": ["deathPlace"],
    "birth_date": ["birthDate"], "birth_year": ["birthDate", "birthYear"],
    "death_date": ["deathDate"], "death_year": ["deathDate", "deathYear"],
    "occupation": ["occupation"], "spouse": ["spouse"], "nationality": ["nationality", "citizenship"],
    "country": ["country"], "capital": ["capital"], "currency": ["currency"], "largest_city": ["largestCity"],
    "language": ["language", "officialLanguage"], "located_in": ["isPartOf", "location", "state", "region"],
    "founded": ["foundingDate", "foundingYear", "formationDate", "formationYear"],
    "dissolved": ["dissolutionDate", "dissolutionYear"], "founder": ["founder", "foundedBy"],
    "author": ["author", "writer"], "director": ["director"], "composer": ["composer", "musicBy"],
    "genre": ["genre"], "developer": ["developer"], "publisher": ["publisher"], "creator": ["creator"],
    "predecessor": ["predecessor"], "successor": ["successor"], "party": ["party"],
    "member_of_political_party": ["party"], "award_received": ["award"], "educated_at": ["almaMater", "education"],
    "employer": ["employer"], "headquarters": ["headquarter", "headquarters"], "owned_by": ["owner", "owningCompany"],
    "owner": ["owner", "owningCompany"], "parent_organization": ["parentCompany", "parentOrganisation"],
    "chairperson": ["chairman", "chairperson"], "head_of_government": ["leader", "primeMinister", "mayor"],
    "head_of_state": ["president", "monarch"], "member_of_sports_team": ["team"], "director_manager": ["manager"],
    "class": ["class"], "family": ["family"], "order": ["order"], "location": ["location"],
}
BY_PROP = {}
for rel, ps in PROPS.items():
    for p in ps:
        BY_PROP.setdefault(p, []).append(rel)


def cached(raw, key, fetch):
    """The cached response for key, or fetch() stored under RAW/<sha1(key)>."""
    global REQUESTS
    os.makedirs(raw, exist_ok=True)
    path = os.path.join(raw, hashlib.sha1(key.encode("utf-8")).hexdigest())
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    for attempt in range(6):
        try:
            data = fetch()
            break
        except Exception as e:  # network or server error: back off and retry
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
            print(f"retry {attempt + 1}: {e}", file=sys.stderr)
    REQUESTS += 1
    with open(path + ".tmp", "wb") as f:
        f.write(data)
    os.replace(path + ".tmp", path)
    time.sleep(1.0)  # polite: one request a second
    return data


def sparql(raw, query):
    def fetch():
        url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query, "format": "application/sparql-results+json"})
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    return json.loads(cached(raw, "sparql\n" + query, fetch).decode("utf-8"))["results"]["bindings"]


def iri(title):
    """DBpedia resource IRI of a Wikipedia title (spaces as underscores; IRI-unsafe characters escaped)."""
    t = title.replace(" ", "_")
    return "<" + DBR + "".join(c if c not in '"<>{}|\\^`' else urllib.parse.quote(c) for c in t) + ">"


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def label_of(uri):
    return urllib.parse.unquote(uri[len(DBR):]).replace("_", " ") if uri.startswith(DBR) else uri


def dbpedia(subjects_txt, raw, out):
    subjects = [l.rstrip("\n") for l in open(subjects_txt, encoding="utf-8") if l.strip()]
    os.makedirs(out, exist_ok=True)
    # 1. 2006 title -> current resource (one redirect hop).
    final = {}
    for batch in chunks(subjects, 40):
        q = ("SELECT ?s ?r WHERE { VALUES ?s { " + " ".join(iri(t) for t in batch) + " } OPTIONAL { ?s <" + DBO +
             "wikiPageRedirects> ?r } }")
        res = {}
        for b in sparql(raw, q):
            res[b["s"]["value"]] = b.get("r", {}).get("value", b["s"]["value"])
        for t in batch:
            u = iri(t)[1:-1]
            final[t] = res.get(u, u)
    # 2. Properties, value labels and Wikidata items (identity links only), and the subject's own Wikidata item.
    rev = {}
    for t, u in final.items():
        rev.setdefault(u, []).append(t)
    uris = sorted(rev)
    props = " ".join("<" + DBO + p + ">" for p in sorted(BY_PROP)) + " <" + DBP + "occupation>"
    rows, sub_qid = set(), {}
    for batch in chunks(uris, 25):
        vals = " ".join("<" + u + ">" for u in batch)
        q = ("SELECT ?s ?p ?o ?l ?q WHERE { VALUES ?s { " + vals + " } VALUES ?p { " + props + " } ?s ?p ?o . "
             "OPTIONAL { ?o <http://www.w3.org/2000/01/rdf-schema#label> ?l FILTER(lang(?l) = 'en') } "
             "OPTIONAL { ?o <http://www.w3.org/2002/07/owl#sameAs> ?q FILTER(STRSTARTS(STR(?q), '" + WD + "')) } }")
        for b in sparql(raw, q):
            s, p, o = b["s"]["value"], b["p"]["value"], b["o"]
            if p == DBP + "occupation":
                if o["type"] != "literal":
                    continue
                vs = [v.strip() for v in re.split(r",|;| and |/|\n", o["value"]) if v.strip()]
                for t in rev[s]:
                    for v in vs:
                        rows.add((t, "occupation", v, "", "dbp:occupation"))
                continue
            name = p[len(DBO):]
            if o["type"] == "uri":
                if "__" in o["value"].rsplit("/", 1)[-1]:  # PersonFunction and similar intermediate nodes
                    continue
                v = b.get("l", {}).get("value") or label_of(o["value"])
                qid = b.get("q", {}).get("value", "")[len(WD):] if "q" in b else ""
            else:
                v, qid = o["value"], ""
            for rel in BY_PROP.get(name, []):
                for t in rev[s]:
                    rows.add((t, rel, v, qid, "dbo:" + name))
        q = ("SELECT ?s ?q WHERE { VALUES ?s { " + vals + " } ?s <http://www.w3.org/2002/07/owl#sameAs> ?q "
             "FILTER(STRSTARTS(STR(?q), '" + WD + "')) }")
        for b in sparql(raw, q):
            for t in rev[b["s"]["value"]]:
                sub_qid.setdefault(t, set()).add(b["q"]["value"][len(WD):])
    with open(os.path.join(out, "dbpedia_ib.tsv"), "w", encoding="utf-8") as f:
        f.write("subject\trelation\tvalue\tvalue_qid\tproperty\n")
        for r in sorted(rows):
            f.write("\t".join(x.replace("\t", " ").replace("\n", " ") for x in r) + "\n")
    with open(os.path.join(out, "dbpedia_subjects.tsv"), "w", encoding="utf-8") as f:
        f.write("subject\tresource\tqids\n")
        for t in subjects:
            f.write(f"{t}\t{final[t]}\t{'|'.join(sorted(sub_qid.get(t, [])))}\n")
    print(json.dumps({"subjects": len(subjects), "resolved_redirects": sum(1 for t in subjects if final[t] != iri(t)[1:-1]),
                      "ib_rows": len(rows), "ib_subjects": len({r[0] for r in rows}),
                      "subject_qids": len(sub_qid), "requests": REQUESTS}))


def subjects(s15_dev, r1, r2, s8b, out_txt):
    def rows(path):
        with open(path, encoding="utf-8") as f:
            h = f.readline().rstrip("\n").split("\t")
            for line in f:
                yield dict(zip(h, line.rstrip("\n").split("\t")))
    u = {r["subject"] for r in rows(s15_dev)}
    for p in (r1, r2):
        u |= {r["subject"] for r in rows(p)}
    u |= {r["subject"] for r in rows(s8b) if r.get("split") == "dev"}
    with open(out_txt, "w", encoding="utf-8") as f:
        for t in sorted(x for x in u if x):
            f.write(t + "\n")
    print(json.dumps({"subjects": len(u)}))


def wplead(subjects_txt, raw, out):
    """The current lead paragraph of each subject's English Wikipedia article (action API, prop=extracts with exintro and
    explaintext, 20 titles a request, redirects followed). Replaces DBpedia's abstracts, which the public endpoint no
    longer serves. Writes OUT/wp_lead.tsv (subject, extract, revision)."""
    subjects = [l.rstrip("\n") for l in open(subjects_txt, encoding="utf-8") if l.strip()]
    os.makedirs(out, exist_ok=True)
    got = {}
    for batch in chunks(subjects, 20):
        params = {"action": "query", "format": "json", "formatversion": "2", "prop": "extracts|revisions", "rvprop": "ids",
                  "exintro": "1", "explaintext": "1", "exlimit": "20", "redirects": "1", "titles": "|".join(batch)}
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)

        def fetch(url=url):
            for wait in (0, 30, 60, 120, 240, 480):
                time.sleep(wait)
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        return r.read()
                except urllib.error.HTTPError as e:
                    if e.code != 429:
                        raise
                    print(f"429, waiting {wait * 2 or 30} s", file=sys.stderr)
            raise RuntimeError("still throttled")
        d = json.loads(cached(raw, "wplead-batch\n" + url, fetch).decode("utf-8"))
        q = d.get("query", {})
        to = {t: t for t in batch}
        for n in q.get("normalized", []):
            for t, v in list(to.items()):
                if v == n["from"]:
                    to[t] = n["to"]
        for rd in q.get("redirects", []):
            for t, v in list(to.items()):
                if v == rd["from"]:
                    to[t] = rd["to"]
        pages = {p.get("title"): p for p in q.get("pages", [])}
        for t in batch:
            p = pages.get(to[t])
            if p and p.get("extract"):
                rev = str((p.get("revisions") or [{}])[0].get("revid", ""))
                got[t] = (p["extract"].replace("\t", " ").replace("\n", " "), rev)
        time.sleep(2.0)
    with open(os.path.join(out, "wp_lead.tsv"), "w", encoding="utf-8") as f:
        f.write("subject\textract\trevision\n")
        for t in subjects:
            if t in got:
                f.write(f"{t}\t{got[t][0]}\t{got[t][1]}\n")
    print(json.dumps({"subjects": len(subjects), "extracts": len(got), "requests": REQUESTS}))


HF_WP = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-{:05d}-of-00041.parquet"


def lead_of(text):
    """The paragraphs before the first section heading (in this dump a heading is a short paragraph of its own that ends
    with a space and no sentence mark), at most 4,000 characters."""
    out = []
    for para in text.split("\n\n"):
        st = para.strip()
        if out and len(st) < 100 and para.endswith(" ") and not st.endswith((".", "!", "?", ":")):
            break
        out.append(st)
        if sum(len(x) for x in out) >= 4000:
            break
    return " ".join(x for x in out if x)[:4000].replace("\t", " ").replace("\n", " ")


def wpdump(subjects_txt, dbpedia_subjects, raw, out):
    import pyarrow.parquet as pq  # not standard library; only this command needs it
    subjects = [l.rstrip("\n") for l in open(subjects_txt, encoding="utf-8") if l.strip()]
    final = {}
    for i, line in enumerate(open(dbpedia_subjects, encoding="utf-8")):
        c = line.rstrip("\n").split("\t")
        if i and len(c) >= 2 and c[1].startswith(DBR):
            final[c[0]] = urllib.parse.unquote(c[1][len(DBR):]).replace("_", " ")
    want = {}
    for t in subjects:
        for title in (final.get(t), t):
            if title:
                want.setdefault(title, []).append(t)
    os.makedirs(raw, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    found = {}
    for k in range(41):
        hits = os.path.join(raw, f"hits-{k:05d}.tsv")
        if not os.path.exists(hits):
            global REQUESTS
            pq_path = os.path.join(raw, "shard.parquet")
            h = hashlib.sha256()
            req = urllib.request.Request(HF_WP.format(k), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r, open(pq_path, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
                    f.write(b)
            REQUESTS += 1
            pf = pq.ParquetFile(pq_path)
            rows = []
            for g in range(pf.num_row_groups):
                tb = pf.read_row_group(g, columns=["id", "title", "text"])
                titles = tb.column("title").to_pylist()
                idx = [i for i, x in enumerate(titles) if x in want]
                if idx:
                    ids, texts = tb.column("id").to_pylist(), tb.column("text").to_pylist()
                    rows += [(titles[i], ids[i], lead_of(texts[i])) for i in idx]
            with open(hits + ".tmp", "w", encoding="utf-8") as f:
                for r_ in sorted(rows):
                    f.write("\t".join(r_) + "\n")
            os.replace(hits + ".tmp", hits)
            with open(os.path.join(raw, "manifest.tsv"), "a", encoding="utf-8") as f:
                f.write(f"{k}\t{h.hexdigest()}\t{len(rows)}\n")
            os.remove(pq_path)
            print(f"shard {k}: {len(rows)} leads", file=sys.stderr)
            time.sleep(1.0)
        for line in open(hits, encoding="utf-8"):
            title, pid, lead = line.rstrip("\n").split("\t")
            for t in want.get(title, []):
                if t not in found or title == final.get(t):  # the redirect target wins over the 2006 title
                    found[t] = (lead, "enwiki-20231101:" + pid)
    with open(os.path.join(out, "wp_lead.tsv"), "w", encoding="utf-8") as f:
        f.write("subject\textract\trevision\n")
        for t in subjects:
            if t in found and found[t][0]:
                f.write(f"{t}\t{found[t][0]}\t{found[t][1]}\n")
    print(json.dumps({"subjects": len(subjects), "leads": sum(1 for t in subjects if t in found and found[t][0]),
                      "requests": REQUESTS}))


GN = "https://download.geonames.org/export/dump/"


def geonames(raw, out):
    def get(name):
        def fetch():
            req = urllib.request.Request(GN + name, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        return cached(raw, "geonames\n" + name, fetch)
    os.makedirs(out, exist_ok=True)
    country, cur, cap = {}, {}, {}
    for l in get("countryInfo.txt").decode("utf-8").split("\n"):
        if not l or l.startswith("#"):
            continue
        c = l.split("\t")
        country[c[0]] = c[4]
        cap[c[0]] = c[5]
        cur[c[0]] = c[11]
    adm1 = {}
    for l in get("admin1CodesASCII.txt").decode("utf-8").split("\n"):
        c = l.split("\t")
        if len(c) >= 2:
            adm1[c[0]] = c[1]
    z = zipfile.ZipFile(io.BytesIO(get("cities15000.zip")))
    cities = []
    for l in z.read("cities15000.txt").decode("utf-8").split("\n"):
        c = l.split("\t")
        if len(c) < 15:
            continue
        cities.append((c[1], c[3], c[7], c[8], c[10], int(c[14] or 0)))  # name, alternates, feature, country, admin1, pop
    facts, alias = set(), set()
    largest_c, largest_a = {}, {}
    for name, alts, feat, cc, a1, pop in cities:
        cn = country.get(cc)
        if not cn:
            continue
        facts.add((name, "country", cn))
        facts.add((name, "located_in", cn))
        an = adm1.get(cc + "." + a1)
        if an:
            facts.add((name, "located_in", an))
            if feat == "PPLA":
                facts.add((an, "capital", name))
            if pop > largest_a.get(an, ("", -1))[1]:
                largest_a[an] = (name, pop)
        if pop > largest_c.get(cn, ("", -1))[1]:
            largest_c[cn] = (name, pop)
        for a in alts.split(","):
            a = a.strip()
            if a and a != name and re.fullmatch(r"[\w .,'()\-]+", a):
                alias.add((a, name))
    for cc, cn in country.items():
        if cap.get(cc):
            facts.add((cn, "capital", cap[cc]))
        if cur.get(cc):
            facts.add((cn, "currency", cur[cc]))
    for k, (n, _) in largest_c.items():
        facts.add((k, "largest_city", n))
    for k, (n, _) in largest_a.items():
        facts.add((k, "largest_city", n))
    with open(os.path.join(out, "geonames.tsv"), "w", encoding="utf-8") as f:
        f.write("subject\trelation\tvalue\n")
        for r in sorted(facts):
            f.write("\t".join(r) + "\n")
    with open(os.path.join(out, "geonames_alias.tsv"), "w", encoding="utf-8") as f:
        f.write("alias\tsubject\n")
        for r in sorted(alias):
            f.write("\t".join(r) + "\n")
    print(json.dumps({"countries": len(country), "admin1": len(adm1), "cities": len(cities), "facts": len(facts),
                      "aliases": len(alias), "requests": REQUESTS}))


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "subjects":
        subjects(*sys.argv[2:7])
    elif len(sys.argv) == 5 and sys.argv[1] == "dbpedia":
        dbpedia(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 5 and sys.argv[1] == "wplead":
        wplead(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 6 and sys.argv[1] == "wpdump":
        wpdump(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif len(sys.argv) == 4 and sys.argv[1] == "geonames":
        geonames(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
