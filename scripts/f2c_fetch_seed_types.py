# F2c: fetch P31 classes and P625 presence for D4's seed values, throttled and cached.
import json, sys, time, urllib.request, urllib.parse, os
ids = sorted(set(l.strip() for f in sys.argv[1:-1] for l in open(f) if l.strip().startswith('Q')))
out = sys.argv[-1]; cache = os.environ.get('F2C_CACHE', 'wdcache')
UA = "CyphaF2c/0.1 (https://github.com/odin-loki/Cypha; Odin Loch's Cypha project) python-urllib"
B = 500
rows = {}
os.makedirs(cache, exist_ok=True)
for k in range(0, len(ids), B):
    batch = ids[k:k + B]; fn = f"{cache}/t500_{k // B:03d}.json"
    q = ("SELECT ?item ?c ?geo WHERE { VALUES ?item { " + " ".join("wd:" + i for i in batch) +
         " } OPTIONAL { ?item wdt:P31 ?c } BIND(EXISTS { ?item wdt:P625 ?x } AS ?geo) }")
    open(fn[:-5] + ".rq", "w").write(q)
    if not os.path.exists(fn):
        for attempt in range(6):
            try:
                req = urllib.request.Request("https://query.wikidata.org/sparql", data=urllib.parse.urlencode({"query": q, "format": "json"}).encode(),
                                             headers={"User-Agent": UA, "Accept": "application/sparql-results+json",
                                                      "Content-Type": "application/x-www-form-urlencoded"})
                data = urllib.request.urlopen(req, timeout=90).read()
                json.loads(data); open(fn, "wb").write(data); break
            except Exception as e:
                time.sleep(65 * (attempt + 1)); print("retry", k, e, file=sys.stderr)
        else:
            sys.exit("failed batch %d" % k)
        time.sleep(65)
    for b in json.load(open(fn))["results"]["bindings"]:
        i = b["item"]["value"].rsplit("/", 1)[1]
        r = rows.setdefault(i, [set(), 0])
        if "c" in b: r[0].add(b["c"]["value"].rsplit("/", 1)[1])
        if b.get("geo", {}).get("value") == "true": r[1] = 1
with open(out, "w") as f:
    f.write("# F2c seed-value types: item, P31 classes (comma-separated), has P625. Fetched %s from https://query.wikidata.org/sparql\n"
            % time.strftime("%Y-%m-%d", time.gmtime()))
    for i in ids:
        r = rows.get(i, [set(), 0])
        f.write("%s\t%s\t%d\n" % (i, ",".join(sorted(r[0])), r[1]))
print("items", len(ids), "with a row", len(rows))
