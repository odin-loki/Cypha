# More facts for the planner, and a gibberish filter

**Odin Loch** · 2026-11-06 · branch `wip/plan-facts` · GENERATION_PLAN §13.42 Q1

**Outcome first.**
- The planner now finds facts for its main entity on 62 of the 250 dev prompts. P1 manages 50, but 26 of those 50 come from Wikidata facts that the leak guard cannot drop.
- A relations-pattern fallback plans 78 more prompts. Plain K5 now writes 105 prompts, down from P1's 157.
- The gibberish filter is built from CyphaLM's own statistics. On the held-out half of a blind-labelled sample it keeps 99.3% of sensible facts but catches only 36% of the gibberish.
- Only 8% of store facts are gibberish, not the 30% predicted.
- On dev, SCRIBE v2 is level with P1 (0.520 vs 0.532), and v3.1 is lower (0.499 vs 0.537).
- The blind pair packets against P1 are committed and not yet rated.
- With none of the new flags, the planner's output is byte-identical to main's.

## Why the planner had few facts

- **The prompt's own page is dropped, by design.** The GEN-A prompts come from enwik8 [75497472, 1e8). `--plan-drop` removes every fact read there. The prompt page's title is a store subject for 72 prompts before the drop and for none after it (`scripts/fs_diag.cpp`).
- **Wikidata seeds get past the drop.** The P1 store `e9_aug` holds 17,907 Wikidata seeds. All of them have byte_offset 0, so `--plan-drop` never drops them. In P1's plans they are mostly facts about the prompt page's own subject (Japan, Hong Kong, Isaac Newton, Hadrian, Galba). They are modern values, for example ISIS with "capital Mayadin".
  - Without them, `e9_aug` plans the main entity on 24 prompts, not 50. So **26 of P1's 50 main-entity plans rest on Wikidata facts.**
  - This breaks two rules at once. The facts come from an outside source, not CyphaLM's reading. And they carry the prompt page's topic past the guard.
  - The new store `e9_v2` is text only, so it starts from 24. **Odin decides what the default store should be.**
- **The overlap guard skips about half of enwik9's prose** (§13.38). Big topics are missing whole: "New Zealand" and "Wellington" are not in the store at all.
- 183 of the 250 prompts have no candidate (from the content plan's entity finder) that is a store subject, even before the drops.

## What was added

All of it is behind flags. Without them, every output is as before.

1. **`is_a` from page leads** (`pattern_learn --lead-class`). Each page's lead ("'''Loki''' is the [[god]] of mischief") gives a text seed (Loki, is_a, god). F2 then learns and applies `is_a` prose patterns like any other relation.
   - Seeds: 17,608 (15,755 from enwik9, 1,853 from enwik8).
   - Kept patterns: 115, for example "[S] is a [O] in", with checked precision 0.985 over 1,102 checks.
   - Prose facts: 7,452.
   - The other relations' facts are unchanged: the enwik9 prose facts besides `is_a` number 9,735, as in `e9_aug`.
   - Without the flag, every enwik8 `--cv` output file is byte-identical, and `stats.txt` differs only in timings.
2. **Aliases** (`scripts/fs_aliases.cpp`). Other names of a page come from the dump's own markup.
   - Redirects: 64,150. Link anchors used at least twice and for at least half of that display's links: 36,242.
   - Overlap pages and the dropped ranges give nothing.
   - 17,569 aliases name a store subject (`e9_v2/aliases.tsv`). 14,620 of them load, because their subject still has facts after the drops.
3. **Entity finding** (`--k5-plan-find 2`, `facts.hpp`).
   - The main entity is also found through an alias, or without a possessive or plural ending ("Holdens" → Holden).
   - Month and day names are dropped as candidates.
   - So is a lone capitalised word at a sentence's start that is mostly lowercase in running text, by the structure statistics ("Black powder …").
   - The prompt's lowercase multi-word store subjects are added as candidates ("aircraft carrier").
4. **The relations-pattern fallback** (`--k5-plan-fallback pattern`, `fallback.hpp`). It runs when no prompt entity has facts of its own, before the co-occurring fallback.
   - Tier 1: facts whose value is a prompt name of the right type, typed by the relations expert ("X is located in New Zealand").
   - Tier 2: the store subject with the highest document co-occurrence (PPMI) with the prompt's names, and up to 2 of its facts. Examples: Trajan for a Hadrian prompt, Euclid for a greatest-common-divisor prompt.
5. **The gibberish filter** (`--k5-plan-filter PARAMS`, `fact_filter.hpp`). Every rule reads CyphaLM's own store and structure statistics: shape, self, year, place, person, name, lex, taxon, agree, xrel, spread and P. The section below gives the details.
6. **The store `bench/fact_store/e9_v2`**. It is text only, built from enwik8 plus enwik9 [1e8, end) with `--lead-class`, with the §13.38 guards and the aliases.
   - Size: 80,390 facts after the drops (2,814 dropped), 35,389 subjects, 31 relations.
   - Its skeleton bank `skel_q1.txt` is `skel_e9.txt` plus 300 `is_a` skeletons, mined by `k5_planner mine` on `e9_v2`.
   - Its model `plan_q1.model` is `plan_e9.model`'s mixer, unchanged, pointed at `skel_q1.txt`.

## Commands

The build is in WSL. The tree is copied to `~/q1_src` and built into `~/build_q1` (`cmake -G Ninja`, Release). Runs are on 4 threads.

```
pattern_learn enwik9 e9_lead --cv --page-range 100000000:1000000000 --skip-pages skip_offsets.txt --lead-class
pattern_learn enwik8 e8_lead --cv --lead-class
python3 -I scripts/fs_merge_store.py e8_lead e9_lead bench/fact_store/e9_v2 enwik8 enwik9
fs_aliases enwik9 aliases_all.tsv --skip skip_offsets.txt --drop 75497472:100000000 --drop 7340032:8388608
python3 -I scripts/fs_alias_filter.py aliases_all.tsv bench/fact_store/e9_v2
k5_planner mine --k5 fuzzy/train.txt --store bench/fact_store/e9_v2 --text k5scale/clean/train_all.txt --lo 0 --hi 100000000 \
    --drop 75497472:100000000 --out skel_v2_full.txt
python3 -I scripts/fs_skel_add.py bench/k5_planner/model/skel_e9.txt skel_v2_full.txt bench/fact_store/e9_v2/skel_q1.txt is_a \
    bench/k5_planner/model/plan_e9.model bench/fact_store/e9_v2/plan_q1.model
fs_coverage bench/fact_store/e9_v2 bench/eval/gen_prompts_dev.tsv --planner bench/fact_store/e9_v2/skel_q1.txt \
    --stats ~/k5s_data/S10.stats --find 2 --fallback pattern --filter bench/fact_filter/chosen.params --plan-drop 75497472:100000000
fs_filter sample bench/fact_store/e9_v2 bench/fact_filter --n 300 --seed 20261105
fs_filter choose  bench/fact_store/e9_v2 bench/fact_filter/key.tsv bench/fact_filter/chosen.params labels_R{1..4}.tsv --stats ~/k5s_data/S10.stats
fs_filter measure bench/fact_store/e9_v2 bench/fact_filter/key.tsv bench/fact_filter/chosen.params labels_R{1..4}.tsv --stats ~/k5s_data/S10.stats
scripts/gen_eval.sh dev Q1 OUT --prompts 250 --k5 fuzzy/train.txt --k5-plan bench/fact_store/e9_v2 \
    --k5-plan-model bench/fact_store/e9_v2/plan_q1.model --k5-plan-gate full --k5-plan-find 2 --k5-plan-fallback pattern \
    --k5-plan-filter bench/fact_filter/chosen.params
```

Costs:
- `pattern_learn` on enwik9: 2:42 and 1.06 GB.
- `fs_aliases`: 1:57 and 1.95 GB.
- `mine`: 1:50 and 732 MB.
- Q1's `gen_eval` (sampling plus both SCRIBE runs): 4:11 and 645 MB.

## Guards

- **Overlap pages skipped whole**: the same list as §13.38 (`skip_offsets.txt`), for `pattern_learn` and for the aliases.
- **`--plan-drop 75497472:100000000` and the holdout**: dropped for facts and for aliases, at load.
- No GEN-A test prompt, continuation or entity list was read.
- The label raters saw only the packet and its instructions.

## Coverage (250 dev prompts, Wilson 95%)

These are the planner's own paths, without K5 (`fs_coverage --planner`). The dev runs give the same counts.

| Store and flags | Main entity | Pattern fallback | Co-occurring | Plain K5 |
|---|---:|---:|---:|---:|
| `e9_aug` (P1) | 50 (0.200 [0.155, 0.254]) | – | 43 | 157 |
| `e9_aug` without its Wikidata seeds | 24 (0.096 [0.065, 0.139]) | – | 53 | 173 |
| `e9_v2`, old skeleton bank (no `is_a`) | 24 (0.096) | – | 52 | 174 |
| `e9_v2`, `skel_q1` | 51 (0.204 [0.159, 0.258]) | – | 53 | 146 |
| + find 2 | 62 (0.248 [0.199, 0.305]) | – | 39 | 149 |
| + pattern fallback | 62 | 78 (0.312 [0.258, 0.372]) | 5 | 105 |
| + chosen filter (**Q1**) | 62 | 78 | 5 | 105 |
| `e9_aug` + find 2 (side) | 49 | – | 42 | 159 |

- `is_a` is in 43 of the 62 main plans.
- Find 2 brings prompts through aliases ("Göteborgs" → Gothenburg, "ACD" → Automatic call distributor) and lowercase subjects.
- Find 2 also maps adjectives to places ("French" → France, "German" → Germany). That keeps the plan near the topic but not on it.
- Without the structure statistics, find 2 would reach 74. Most of the extra 12 are sentence-initial words ("Black is a color", "We is a nominative case"), so the statistics are on.

## The gibberish filter

**The sample** (`bench/fact_filter/`, committed before any label was read). 300 facts were drawn uniformly (seeded) from `e9_v2` as the planner loads it. A second seeded order puts 150 in a choose half and 150 in a measure half.

**Rule changes before any label.** I listed what each rule flagged over the whole store and fixed the plain false positives:
- xrel had flagged 442 "nationality English" facts. It no longer compares nationality with language, and it skips infobox fields.
- self no longer applies to developer, publisher and owner.
- taxon now accepts "class bird".
- place also accepts a value the relations expert types as a place.
- year no longer flags ancient lives that lack "BC".
- spread is new: it flags a nationality read with few distinct occupations in the same category ("Baroque" composers, "Film" actors).

The packet re-draws byte for byte.

**The labels.** Four blind Claude raters each labelled all 300 facts. The label rule was stated before the labels were used: gibberish or sensible by 3 or 4 of 4, and a 2–2 tie is reported apart and left out.

| Gibberish votes | 0 | 1 | 2 (tie) | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Facts | 272 | 2 | 2 | 8 | 16 |

So 24 facts are gibberish, 274 sensible and 2 tied.

**The choice** (choose half, each rule family alone at each threshold, on only if gibberish flagged minus sensible flagged is above 0):
- **shape** is on (net +1).
- **spread at 5 occupations** is on (net +4).
- Every other family is off: its net was 0 or below. Place at support 2 and above flagged 6 sensible capitals and 0 gibberish. Lex flagged sensible unique `is_a` values.
- The parameters are in `bench/fact_filter/chosen.params`.

**The measure** (measure half, read once). Gibberish is the positive class.

| | Measure half | Choose half (seen) | All 298 labelled |
|---|---|---|---|
| precision (flagged that are gibberish) | 4 / 5 = 0.800 [0.376, 0.964] | 4 / 4 | 8 / 9 = 0.889 [0.565, 0.980] |
| recall (gibberish flagged) | 4 / 11 = 0.364 [0.152, 0.646] | 4 / 13 = 0.308 | 8 / 24 = 0.333 [0.180, 0.533] |
| sensible kept | 137 / 138 = 0.993 [0.960, 0.999] | 136 / 136 | 273 / 274 = 0.996 [0.980, 0.999] |
| gibberish share | 11 / 149 = 0.074 [0.042, 0.127] | 13 / 149 = 0.087 | 24 / 298 = 0.081 [0.055, 0.117] |

**Where the gibberish is.**
- **Nationality:** 9 gibberish facts (Blues, Baroque, Historical, Modern, Renaissance, Film, Fictional, Catholic, Character), all category words read as nationalities. The filter catches 8.
- **`is_a`:** 14 gibberish facts. The filter catches none.
  - In 7 of them the value repeats a word of the subject ("Var River — is a — river", "Lake Como — lake", "Dekker's algorithm — algorithm"). The label instructions call that gibberish. The self rule lets a lexicon value sit inside the subject on purpose ("Horned owl — owl"), so the rule and the raters disagree here.
  - The other 7 have a wrong head word read from prose ("Lost in Space — science", "Clinical neurophysiology — medical", "Ada — structured").
- **Publisher:** 1 gibberish fact ("Canada United States VIZ Media"), not caught.

**Over the store**, the chosen filter leaves out 2,732 facts (3.4%): 2,443 by spread and 320 by shape. Spread takes 40% of the nationality facts.

## Dev (GEN-A dev, seed 0, 250 prompts)

| Arm | SCRIBE v2 [95%] | v3.1 [95%] | copy32 | 13-word runs | Paths main / pattern / cooc / K5 | Facts stated | Relation cue |
|---|---|---|---:|---:|---|---:|---|
| P1 (today's default) | 0.532 [0.502, 0.564] | 0.537 [0.514, 0.558] | 0.0376 | 0 | 50 / – / 43 / 157 | 246 | 7 / 48 (0.146) |
| Q1a: e9_v2, find 2, fallback, no filter | 0.522 [0.494, 0.555] | 0.498 [0.479, 0.518] | 0.0407 | 0 | 62 / 78 / 5 / 105 | 246 | 10 / 51 (0.196) |
| **Q1**: Q1a + chosen filter | 0.520 [0.492, 0.552] | 0.499 [0.480, 0.519] | 0.0411 | 0 | 62 / 78 / 5 / 105 | 245 | 10 / 52 (0.192) |

- Q1 states 90 `is_a` facts. P1 stated 61 nationality facts; Q1 states 1.
- 144 of the 250 continuations differ from P1, and 23 of the 50 prompts the raters will see.
- The fill around the facts is still K5's, and it often says something false or meaningless ("it is named in honour of the god city of Loki, California"). That is Q2's job.
- Tier 2 of the fallback sometimes picks an off-topic subject: "Hare" for a hydrogen peroxide prompt.
- SCRIBE v3.1 is 0.038 lower than P1. The intervals touch.
- copy32 rises a little (0.041 against 0.038). There are no 13-word copies.

## Byte identity

- Default planner (`sample --k5`, no Q1 flag), Q1 build against main's build (c10935b), all 250 dev prompts: the GEN file and stdout are identical under `cmp` (sha256 c05220e2…).
- That output also equals the checked-in `bench/k5_planner/P1_e9_s0.gen.txt` after stripping CRs, apart from the MODEL line.
- Selftests pass: cyphalm_trie, scribe 120/120, k5_planner, cypha_report_card 29/29.

## Blind pair packets

`bench/blind/q1_pair`: Q1 against P1, the 50 prompts of `bench/blind/llm_rating/prompts.json`, `--mode pair --baseline P1 --seed 42`. The packets `R1.txt` to `R4.txt`, `RATER_PROMPT.md` and `key.b64` were committed before any rating. They are not rated here. The deciding rule is §13.38's.

## Predictions (§13.42 Q1), scored

| Prediction | Result | Score |
|---|---|---|
| Main-entity coverage from 50 to at least 80 of 250 | 62 (0.248 [0.199, 0.305]); text only, the honest start was 24 | **miss** |
| The filter removes at least 70% of gibberish facts | measure half 0.364 [0.152, 0.646] | **miss** |
| …and keeps at least 90% of sensible ones | measure half 0.993 [0.960, 0.999] | **hit** |
| About 30% of store facts are gibberish | 0.081 [0.055, 0.117] (24 of 298) | **miss** (far fewer) |
| Raters prefer Q1 to P1 | packets committed, not rated | open |

## What is weak, and next

- **The default store.** P1's `e9_aug` plans half of its main-entity prompts from Wikidata facts that `--plan-drop` cannot drop. If those go, P1's main path falls to 24. Odin decides.
- **The filter misses `is_a` junk.** Two rules would likely catch most of it: "the value repeats a word of the subject" for `is_a`, and a check that the head word is a noun. Both need a fresh sample, because this filter's code is now fixed for this measurement.
- **The fill**, as above (Q2).
- **Tier 2 of the fallback** can write about a subject the reader does not expect. The raters decide whether that reads better than plain K5.

## Files

- Code:
  - `native/include/cypha/lm/planner/{facts,fact_filter,fallback}.hpp`
  - `writer.hpp` (three marked lines)
  - `cyphalm_trie.cpp` (the three flags)
  - `native/tools/knowledge/pattern_learn.cpp` (`--lead-class`)
  - `scripts/fs_{aliases,alias_filter.py,coverage,diag,filter,skel_add.py}`
- Store: `bench/fact_store/e9_v2/`. Its non-TSV files are kept LF by `.gitattributes`, so WSL reads them as written.
- Filter: `bench/fact_filter/`: packet, key, prompt, labels, `chosen.params`, `choose.out`, `measure.out`, `stats_chosen.out`, `flagged_chosen.tsv`.
- Dev: `bench/plan_facts/`: `Q1`/`Q1a` GEN files, summaries, plan logs and measures; the default run's plan log; the coverage outputs.
- Blind: `bench/blind/q1_pair/`.
- Raw runs: `/mnt/d/cypha_llm/plan/q1/`.
