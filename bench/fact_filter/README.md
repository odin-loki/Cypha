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

**The label files** (2026-11-06, run by the lead session): `labels_R1.tsv` … `labels_R4.tsv`, 300 lines each, "N<TAB>sensible|gibberish". **Label rule, stated before the labels were used:** gibberish if 3 or 4 of the 4 raters say gibberish; sensible if 3 or 4 say sensible; a 2–2 tie is reported on its own and left out of the choice and the measure (as above; `fs_filter`'s `label_of`).

**The choice** (`fs_filter choose STORE KEY OUT LABELS... --stats ~/k5s_data/S10.stats`, choose half only). Each rule family of `fact_filter.hpp` is tried alone, at each of its thresholds. A family is turned on, at the threshold with the largest (gibberish flagged − sensible flagged), only if that number is above 0, so its flags are mostly gibberish. Families: shape, self, year, place (support 1, 2, 3, 5), person (1, 2), name, lex (support 1, 2, 3, 5, 10), taxon, agree, xrel, spread (2, 3, 5, 10 occupations), p (min P 0.6, 0.8, 0.9). `--stats` is the planner model's own statistics file: the place rule reads the relations expert's name types from it, as the planner does.

**Rule revisions before any label** (2026-11-06, after listing what each rule flags over the whole store, `fs_filter stats … --list`; the packet and key are unchanged, re-drawn byte for byte):
- xrel flagged 442 "nationality English" facts because "English" is more often a language; nationality and language now are not compared, and an infobox or table field (which names its relation) is not flagged;
- self no longer applies to developer, publisher and owner ("Microsoft Word" developer "Microsoft");
- taxon accepted only one capitalised word, so "class bird" was flagged; it now flags a digit, more than 2 words or a non-letter start ("Windows Me" family "Windows 9x");
- place also counts a value as a known place if the relations expert types it as a place (it mostly follows "in", "at", ...);
- year no longer flags ancient lives that run backwards because "BC" is missing (Empedocles 490–430);
- a new family, spread: a nationality read with fewer than N distinct occupations over the store (the occupation read in the same category: "Baroque" composers, "Film" actors, "Batman": category words read as nationalities).

**The store** `bench/fact_store/e9_v2` is fixed from here on: the filter's code may not change once a label is read, and the store may not change at all, or the sample would no longer be a sample of it.

**The measure** (`fs_filter measure`, measure half, read once with the chosen parameters). Gibberish is the positive class:
- precision = flagged facts that are gibberish / flagged facts;
- recall = gibberish facts flagged / gibberish facts;
- sensible kept = sensible facts not flagged / sensible facts.
All with Wilson 95% intervals.

**Predictions** (§13.42, registered before any run): the filter removes at least 70% of gibberish facts and keeps at least 90% of sensible ones; about 30% of store facts are gibberish.
