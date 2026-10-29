# F2c: Wikidata types of D4's seed values

**Odin Loch** · 2026-10-29 · [report](../../docs/reports/F2_PATTERN_LEARNING.md) ("F2c")

`wd_seed_types.tsv`: for each value item of a person or place relation in D4's `wd_values.tsv` (4,592 items), its Wikidata "instance of" classes (P31) and whether it has a coordinate location (P625). Tab-separated: item, classes (comma-separated), has P625 (0 or 1). One comment line gives the fetch date.

- **Source:** the Wikidata Query Service (`https://query.wikidata.org/sparql`), fetched 2026-10-29. Wikidata's data is CC0.
- **Used by:** `pattern_learn --lexicon-type-check data/f2c/wd_seed_types.tsv [--lexicon-type-arm P|PL]`. A Wikidata seed value widens the person lexicon only if Wikidata types it as human (Q5); with arm PL, the place lexicon only if it has a coordinate location.
- **Rebuild:** `python3 -I scripts/f2c_fetch_seed_types.py qid_person.txt qid_place.txt data/f2c/wd_seed_types.tsv`, where the two lists are the value items (column 5) of `wd_values.tsv`'s person-relation and place-relation rows. The endpoint may rate-limit; the script waits a minute between batches and caches every response (`F2C_CACHE`, default `wdcache`).
