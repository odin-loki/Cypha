# data/s8c/

S8c's DBpedia tables, kept because DBpedia's public endpoint changes and a refetch would not rebuild them byte for byte ([report](../../docs/reports/S8C_EVIDENCE_SOURCES.md)). Evidence for Cypha's own store and hypotheses only: never an answer for CyphaLM, never training text (PLAN rule 10).

| File | What it is | Rows | sha256 (unpacked) |
|---|---|---:|---|
| `dbpedia_ib.tsv.gz` | DBpedia's mapping-based properties (and the raw infobox occupation) of S8c's 2,647 subjects: subject, relation, value, value's Wikidata item (an identity link), DBpedia property | 21,485 | `679d0da185bd3290e2143560b5bdb89676f42594f45b1584fd368c055177140c` |
| `dbpedia_subjects.tsv.gz` | Each subject's DBpedia resource after redirects, and its Wikidata items | 2,647 | `43e5fef4e01023e61fe9bcd8dc8c471747e8f2c9e1df8f7ab600602c5d4fe3fd` |

Fetched from https://dbpedia.org/sparql on 2026-10-29 by `scripts/evidence_sources.py dbpedia`. Packed with `gzip -9n`, so packing again gives the same bytes.

**Licence.** DBpedia's data is CC BY-SA 3.0 (and GFDL), derived from Wikipedia. These two files keep that licence, with attribution to DBpedia and to Wikipedia's contributors. The rest of the repository's licence does not apply to them.

The other S8c tables are not kept here. Each is rebuilt by `scripts/evidence_sources.py`:
- GeoNames (CC BY 4.0) comes from the daily dump. A refetch differs from the run's copy (sha256 `b18fe58a…cc76e3c`, alias table `57d2f0df…7c02db`).
- The Wikipedia leads come from the fixed 2023-11-01 dump. Its shards' SHA-256s are in the run's manifest. The lead table's sha256 is `106fd44f…7cc80`.
