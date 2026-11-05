# The gibberish filter's labelled sample

**Odin Loch** · 2026-11-05 · GENERATION_PLAN §13.42 Q1 · [PLAN_FACTS](../../docs/reports/PLAN_FACTS.md)

These files were written and committed before any label was read.

| File | What |
|---|---|
| `labels_packet.txt` | 300 store facts, numbered, as "subject — relation — object" lines, nothing else. This is what the raters see. |
| `LABEL_PROMPT.md` | The raters' instructions. |
| `key.tsv` | For each number: the fact, where it came from (source, template or F2 pattern, P, evidence count, rows behind it, byte offset) and its half. Not shown to the raters. |

**The sample.** `fs_filter sample bench/fact_store/e9_v2 bench/fact_filter --n 300 --seed 20261105`: 300 facts drawn uniformly (seeded) from the store as the planner loads it (`facts.hpp` `read_rows`: the holdout [7340032, 8388608) and the GEN-A prompts' bytes [75497472, 100000000) dropped, canonical relations and values, one row per fact). A second seeded order (seed 20261106) puts 150 facts in the **choose** half and 150 in the **measure** half.

**The labels.** Four blind Claude raters label each fact "sensible" or "gibberish" (`LABEL_PROMPT.md`). A fact's label is the majority of the four. A 2–2 tie is left out of the choice and of precision and recall, and counted.

**The choice** (`fs_filter choose`, choose half only). Each rule family of `fact_filter.hpp` is tried alone, at each of its thresholds. A family is turned on, at the threshold with the largest (gibberish flagged − sensible flagged), only if that number is above 0, so its flags are mostly gibberish. Families: shape, self, year, place (support 1, 2, 3, 5), person (1, 2), name, lex (support 1, 2, 3, 5, 10), taxon, agree, xrel, p (min P 0.6, 0.8, 0.9).

**The store** `bench/fact_store/e9_v2` is fixed from here on: the filter's code may not change once a label is read, and the store may not change at all, or the sample would no longer be a sample of it.

**The measure** (`fs_filter measure`, measure half, read once with the chosen parameters). Gibberish is the positive class:
- precision = flagged facts that are gibberish / flagged facts;
- recall = gibberish facts flagged / gibberish facts;
- sensible kept = sensible facts not flagged / sensible facts.
All with Wilson 95% intervals.

**Predictions** (§13.42, registered before any run): the filter removes at least 70% of gibberish facts and keeps at least 90% of sensible ones; about 30% of store facts are gibberish.
