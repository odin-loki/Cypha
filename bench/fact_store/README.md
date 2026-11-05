# CyphaLM's fact store (copied from the cloud, 2026-11-05)

**Odin Loch** · 2026-11-05

CyphaLM's own fact store, as `pattern_learn` (F2, `native/tools/knowledge/pattern_learn.cpp`) wrote it from enwik8. It is copied here so Odin's PC can use it for the plan → skeleton → fill → gate generator (GENERATION_PLAN §13.37). Nothing in it is fetched from outside the text, apart from the D4 Wikidata seeds that F2's learning used.

| Path | What | Used by |
|---|---|---|
| `verify_f2r_aug/` | the F2R store: `seeds.tsv` (25,392 facts from infoboxes, tables and categories), `facts_f2.tsv` (1,394 facts F2 admitted from prose), `pages.tsv`, `patterns.tsv`, evidence and checks | the content plan (`lm/content_plan.hpp`, `sample --plan DIR`) |
| `v2_aug/` | the v2 store GEN-E used (no `seeds.tsv`) | GEN-E's runs (`--plan bench/fact_store/v2_aug,...`) |
| `patterns_raw.tsv` | F2R's raw-byte frames (`--plan DIR,patterns_raw.tsv`) | the frames that turn a fact into a phrase |

Every fact carries a `byte_offset` into enwik8. The content plan drops facts whose evidence lies in the bytes a prompt comes from (`--plan-drop LO:HI`), so a plan cannot copy the continuation it is judged against.
