# e9_text: the planner's text-only fact store

**Odin Loch** · 2026-11-06 · GENERATION_PLAN §13.46

`e9_aug` without its 17,907 Wikidata seeds (`source` = `wikidata`), so the planner plans only from facts CyphaLM read in the text:
- seeds from infoboxes, categories and tables;
- prose facts admitted by F2.

The Wikidata check file is left out. Everything else is a copy of `e9_aug`. Build:

```
awk -F'\t' 'NR==1 || $4!="wikidata"' bench/fact_store/e9_aug/seeds.tsv > bench/fact_store/e9_text/seeds.tsv
```
