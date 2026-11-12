# GEN_FACTS: fact precision of generated text (CT3.4)

**Odin Loch** · 2026-11-12 · GENERATION_PLAN §7 (CT3.4), §13.38, §13.48

(The measure, the arms, the hand check and the predictions below were written and committed before the measure ran on any of the arms' GEN files. Results are added under them.)

## Why

- §13.38 scored the planner's fact precision as 1.0. That number was hollow: it only counted the facts the plan put in, and those are store facts by construction.
- §13.48 found that the planned sentences still invent details: "Barbara Anderson (May 2, 1926 - March 20, 1568) is an American playwright."
- SCRIBE cannot see a wrong date or a wrong nationality. G4 (§1.1) asks for fact precision of at least 0.9 on statements the store can check, and no contradictions.
- So we need a measure that reads every statement of fact in the written text, not only the planned ones.

This is a measure only. It changes no writer.

## The measure

`scripts/gen_fact_check.py run` (output in `bench/gen_facts/`).

**The store.** CyphaLM's own fact store: `bench/fact_store/e9_text` and `bench/fact_store/e9_v2`, together. Seeds from infoboxes, categories, tables and leads, plus F2's prose facts with P at least 0.8 (F2's admission rule). No Wikidata seeds: neither store holds any. Facts whose evidence lies in GEN-A's prompt bytes (enwik8 [75,497,472, 100,000,000)) are dropped, as the writers' `--plan-drop` drops them. 92,460 facts on 36,094 subjects; 4,601 dropped.

**The text.**
- Byte escapes decoded, dashes written as "-", wiki markup dropped.
- Sentences split at . ! ? before a capital, a quote or a digit (not after "c.", "b.", "St." or an initial).
- The last sentence is dropped if it does not end in . ! ? (the 460-byte cut).

**Statements.**
- Every pattern F2 kept (`kept` = 1 in `patterns.tsv` and `patterns_add.tsv` of both stores): 359 patterns over birth and death dates and years, birth and death place, located in, nationality, director, class, language and is_a.
- Each is compiled to a token pattern. [S] is a name (capitalised words), [P] a sentence-initial he, she or it. [O] is typed by relation: a date, a place (a name, maybe ", name"), a name, or a word of the relation's store lexicon (F2's own type check).
- Two relaxations for written text: "is" also matches "was", and "a" also matches "an".
- One added pattern family, not F2's: "[S] is/was a(n) [nationality] [is_a word]", with lists (", X", "and X"). It reads nationality and occupation from sentences such as "X was a New Zealand politician and writer".
- Birth year and birth date are one statement (the more precise form is kept), as are death year and death date. One statement per sentence, relation, subject and value.

**Subjects.**
- The longest part of the name that is a store subject or a store alias, the rightmost on ties. "Ben Barbara Anderson" is graded as Barbara Anderson; such matches are marked `partial`.
- A pronoun takes the last store person named in the sample.
- Birth, death and nationality need a person: a subject in the store must have a birth, death, nationality, occupation or birth/death place fact. A name not in the store must be 2 to 4 capitalised words, not starting with "The".

**Grades.** Each statement is graded against all the store's values for its subject and relation.

| Relation | Supported | Contradicted | Else |
|---|---|---|---|
| birth, death dates | agrees at the common precision (year, month, day) | the store has a date and none agrees | unknown |
| birth, death place | equal, or one inside the other by the store's `located_in` | the two lie under different top places (`located_in` walked up, continents left out) | unknown |
| located in | the place is above the subject in the store's `located_in` | different top places | unknown |
| nationality | the same nationality by a fixed name table (demonyms and country names; British holds English, Scottish, Welsh) | the store has a nationality in the table and none matches | unknown |
| director, class | the same name | the store has a value and none matches | unknown |
| is_a (occupation, kind), language | the same word | never: open lists | unknown |

- A subject the store does not hold is unknown.
- A year written without BC is compared without the era ("322" for "322 BC" is not a contradiction).
- Two side counts: dates that cannot exist (34 March, a day past the month), and supported dates checked only in part (the store has the year, the text gives a day).

**Reported per arm, with 95% intervals** (bootstrap over samples, 2,000 draws, seed 20261112):
- statements per sample;
- share contradicted and share supported (of all statements);
- precision, supported / (supported + contradicted);
- the same two shares by sentence position (1, 2, 3, 4 and later);
- counts by relation.

**Beside it, not the measure:** the same run with the full store (`--drop none`).

## The arms

All files are already committed. Dev, seed 0, 250 prompts each:
- **D0** (`bench/gen_quality/runs/D0.gen.txt`): the discourse planner on 7 MB (§13.46);
- **W1b** (`…/W1b.gen.txt`): K5 counts from 337 MB (§13.48);
- **H1b** (`…/H1b.gen.txt`): the honest fill (§13.48);
- **G16** (`bench/tune_write/runs/G16.gen.txt`): a gate of 16 drafts (§13.51);
- **L1** (`bench/k5_coherence/runs/L1.gen.txt`): the LDA topic term, today's default (§13.53);
- **HUMAN** (`bench/eval/gen_prompts_dev.tsv`, its continuations): the reference.

Test reads already made (one each, read here as files; no new test read is made):
- **W1b_TEST**, **G16_TEST**, **L1_TEST**, **P1_e9_TEST** (the planner with the Wikidata seeds, §13.38), from `bench/*/test_read/`;
- **HUMAN_TEST** (`bench/eval/gen_prompts_test.tsv`).

## The hand check of the grader

- After the run, `gen_fact_check.py draw` draws 40 graded statements from the five dev writers pooled (D0, W1b, H1b, G16, L1), seed 20261112: 20 contradicted and 20 supported.
- I read each one and mark:
  - **extraction:** right if the sentence states this relation of this subject with this value;
  - **verdict:** right if the extraction is right, the subject is the store's entity, and the verdict is true of the world: a contradicted statement is false, a supported one is true at the store's precision.
- **Grader precision** = verdicts right / 40, with a Wilson 95% interval; also by stratum.
- The marks go in `bench/gen_facts/hand_check.tsv`.

## How it was built

- The code was written and tried only on GEN files that are not arms here: `bench/tune_write/runs/N7.gen.txt` and `O6.gen.txt` and `bench/k5_planner/P1t_s0.gen.txt` (D0 equals `Q3t_s0`, so that file was not used). While learning the GEN format I read the first sample of W1b's file (it is quoted in §13.48).
- On those three files: N7 1.48 statements a sample, 0.24 contradicted, precision 0.59; O6 1.38, 0.22, 0.63; P1t 0.34, 0.21, 0.47. Two fixes came from reading them: the BC rule above, and the rule on unknown names ("The University of Miami (1941)" is not a birth).

## Predictions (registered before the run)

| # | Prediction |
|---|---|
| F1 | W1b, G16, L1: 1.0 to 2.0 statements a sample; share contradicted 0.15 to 0.30; precision 0.50 to 0.75. G4's 0.9 is not met |
| F2 | D0: 0.5 to 1.5 statements a sample; precision 0.40 to 0.75 |
| F3 | H1b contradicts less than D0: its share contradicted at most half of D0's, and its precision at least 0.8 |
| F4 | HUMAN: at most 0.3 statements a sample; share contradicted at most 0.05; precision at least 0.9 if it has 10 or more graded statements |
| F5 | In every writer, sentence 1 has a lower share contradicted than sentences 2 and later together |
| F6 | Of the supported dates of W1b, G16 and L1, at least half are checked only in part (the store has the year; the writer added month and day) |
| F7 | Each test read lies within its dev arm's interval for share contradicted (W1b, G16, L1) |
| F8 | Grader precision on the 40 hand-checked statements at least 0.85 |

## Results of the registered measure (v1, 2026-11-12)

Run after b174b9ce. `bench/gen_facts/` (dev), `bench/gen_facts/test/` (test reads), `bench/gen_facts/full_store/` (`--drop none`).

| Arm | statements a sample | contradicted | supported | precision S/(S+C) | supported only in part |
|---|---|---|---|---|---|
| D0 | 1.12 [0.84, 1.43] | 0.228 [0.166, 0.290] | 0.349 [0.302, 0.395] | 0.605 [0.531, 0.682] | 0.60 |
| W1b | 1.47 [1.15, 1.82] | 0.240 [0.187, 0.294] | 0.338 [0.295, 0.379] | 0.585 [0.530, 0.642] | 0.66 |
| H1b | 0.56 [0.40, 0.72] | 0.007 [0.000, 0.022] | 0.165 [0.099, 0.227] | 0.958 [0.880, 1.000] | 0.00 |
| G16 | 1.49 [1.17, 1.83] | 0.236 [0.184, 0.290] | 0.332 [0.289, 0.374] | 0.585 [0.530, 0.642] | 0.66 |
| L1 | 1.48 [1.15, 1.82] | 0.238 [0.186, 0.294] | 0.335 [0.291, 0.378] | 0.585 [0.530, 0.642] | 0.66 |
| HUMAN | 0.04 [0.01, 0.09] | 0 of 11 | 0 of 11 | none graded | – |

The full store (`--drop none`) gives the same numbers for every writer (HUMAN: 12 statements, none graded).

**The hand check (v1).** 40 statements drawn with seed 20261112 (`hand_check.tsv`). Every extraction is right. **Verdicts right: 36 of 40, 0.90 [0.77, 0.96].** Contradicted: 20 of 20 [0.84, 1.00]. Supported: 16 of 20, 0.80 [0.58, 0.92]. The four wrong ones are two faults:
- **A coarse store date hid a precise one** (3 of 4). The store holds both "1931" and "1931-02-01" for Boris Yeltsin. "November 23, 1931" agrees with "1931", so it was marked supported, though the store's own full date contradicts it. The same for Bono and Nigella Lawson.
- **Only the head of "A, B" was checked** (1 of 4). "Amakusa is a city in Japan, China" was supported on "Japan", though it puts Amakusa in China.

So v1 makes no false contradictions in the sample, but it calls some wrong dates supported. Its share contradicted is a lower bound.

Also seen, not in the sample: in HUMAN_TEST, "in Paris (1892-1895)" was read as the birth of Paris (rapper), a one-word name that matched a person's page. This is one false contradiction. It is not fixed here.

## Amendment A1 (registered 2026-11-12, after the v1 hand check, before the v2 run)

`gen_fact_check.py run --v2` fixes the two faults the hand check found. Nothing else changes; without `--v2` the output is the same as v1's (`cmp` on W1b's file).
- **Dates:** only the store's finest dates for the subject count. If the store has a full date, a statement must agree with it.
- **Places "A, B":** each part is graded alone. One contradicted part contradicts the statement; else one supported part supports it.

**Run:** the same arms, into `bench/gen_facts/v2/` and `bench/gen_facts/v2/test/`.

**A fresh hand check:** 40 more statements (20 contradicted, 20 supported) drawn from v2's five dev writers with seed 20261113, read as before (`hand_check_v2.tsv`).

**Predictions:**
- A1.1: the share contradicted of W1b, G16 and L1 rises by 0.05 to 0.12 (on N7, a file outside the arms, it rose from 0.24 to 0.33); precision falls to 0.40–0.55.
- A1.2: H1b stays at or below 0.03 contradicted.
- A1.3: grader precision on the fresh 40 at least 0.9.

## Results under A1 (v2, 2026-11-12; the numbers to use)

Run after d4515696. `bench/gen_facts/v2/` and `bench/gen_facts/v2/test/`.

| Arm | statements a sample | contradicted | supported | precision S/(S+C) | supported only in part |
|---|---|---|---|---|---|
| D0 | 1.12 [0.84, 1.43] | 0.324 [0.250, 0.395] | 0.260 [0.215, 0.299] | 0.445 [0.374, 0.519] | 0.48 |
| W1b | 1.47 [1.15, 1.82] | 0.327 [0.259, 0.397] | 0.253 [0.209, 0.297] | 0.437 [0.369, 0.507] | 0.56 |
| H1b | 0.56 [0.40, 0.72] | 0.007 [0.000, 0.022] | 0.165 [0.099, 0.227] | 0.958 [0.880, 1.000] | 0.00 |
| G16 | 1.49 [1.17, 1.83] | 0.322 [0.255, 0.392] | 0.249 [0.205, 0.293] | 0.437 [0.369, 0.507] | 0.56 |
| L1 | 1.48 [1.15, 1.82] | 0.324 [0.260, 0.396] | 0.251 [0.207, 0.294] | 0.437 [0.369, 0.507] | 0.56 |
| HUMAN | 0.04 [0.01, 0.09] | 0 of 11 | 0 of 11 | none graded | – |
| W1b_TEST | 1.54 [1.20, 1.91] | 0.358 [0.287, 0.425] | 0.259 [0.221, 0.295] | 0.420 [0.360, 0.487] | 0.45 |
| G16_TEST | 1.57 [1.23, 1.94] | 0.346 [0.277, 0.414] | 0.254 [0.217, 0.290] | 0.424 [0.361, 0.491] | 0.45 |
| L1_TEST | 1.56 [1.22, 1.92] | 0.347 [0.277, 0.414] | 0.257 [0.220, 0.293] | 0.426 [0.364, 0.494] | 0.45 |
| P1_e9_TEST | 0.32 [0.22, 0.44] | 0.275 [0.138, 0.398] | 0.062 [0.016, 0.118] | 0.185 [0.057, 0.375] | 0.20 |
| HUMAN_TEST | 0.02 [0.00, 0.05] | 1 of 6 (the Paris fault) | 0 of 6 | – | – |

**By sentence position** (v2, share contradicted; 95% intervals in `v2/summary.md`):

| Arm | sentence 1 | 2 | 3 | 4 and later |
|---|---|---|---|---|
| D0 | 0.15 (n 34) | 0.39 (36) | 0.29 (28) | 0.35 (183) |
| W1b | 0.20 (41) | 0.29 (58) | 0.24 (37) | 0.37 (231) |
| H1b | 0.00 (33) | 0.00 (32) | 0.00 (34) | 0.03 (40) |
| G16 | 0.19 (42) | 0.30 (56) | 0.24 (38) | 0.36 (237) |
| L1 | 0.20 (41) | 0.29 (58) | 0.24 (38) | 0.37 (233) |

**By relation** (W1b, v2; supported / contradicted / unknown): birth date 70 / 55 / 6, death date 6 / 53 / 35, nationality 6 / 8 / 9, birth place 3 / 2 / 33, located in 4 / 2 / 1, is_a 4 / 0 / 66.

**The fresh hand check (v2).** 40 statements, seed 20261113 (`hand_check_v2.tsv`). **Verdicts right: 38 of 40, 0.95 [0.83, 0.99].** Contradicted 20 of 20 [0.84, 1.00]; supported 18 of 20, 0.90 [0.70, 0.97]. The two wrong ones:
- "Aristotle (January 17, 1949 - April 6, 322)": the registered era rule lets "322" match "322 BC".
- "Natacha Atlas was an Egyptian era": the added pattern read a nationality where the sentence states none.

## What it says about the writers

- **The planned sentences state wrong facts as often as right ones.** Under v2, a third of all statements contradict CyphaLM's own store, and only 0.44 of the checkable ones agree with it (G4 asks for 0.9). Today's default (L1) is at 0.437 [0.369, 0.507] on dev and 0.426 on test.
- **The second date is invented.** Of W1b's 59 graded death dates, 53 are wrong ("Averroes (January 1, 1126 - July 13, 1807)"). Birth dates come from the plan, so the year is mostly right, but the fill adds a month and day: 55 of 125 graded birth dates are wrong at the store's full date.
- **W1b, G16 and L1 give the same facts.** Their numbers agree to the third decimal. The fact sentences come from the planner, and T2's and T3's settings do not touch them. Better free text did not change the facts.
- **The honest fill works for facts.** H1b contradicts the store once in 139 statements (0.007), with precision 0.958 [0.880, 1.000]. It states almost no dates it does not hold. The raters did not prefer it (0.560, §13.48), so today's default trades facts for fluency.
- **Errors grow down the paragraph.** Sentence 1 is contradicted least (0.15–0.20); from sentence 4 on, 0.35–0.37. The later sentences are the discourse writer's linked people ("He was the son of Louis Braille (1809 - 1810), Italian painter.").
- **Human text gives the measure almost nothing to check:** 0.04 statements a sample, none graded. GEN-A's continuations are mid-article prose, not lead sentences. So this measure grades the planner's own kind of sentence, and it cannot say how a human would score.
- **P1_e9 (with the Wikidata seeds, §13.38) is worse still** against the text store: 0.185 [0.057, 0.375].

## Predictions, scored

Scored on v1, the measure as registered; A1's on v2.

| # | Result |
|---|---|
| F1 | statements 1.47–1.49 **hit**; contradicted 0.236–0.240 **hit**; precision 0.585 **hit**; G4 not met **hit** |
| F2 | D0 1.12 **hit**; precision 0.605 **hit** |
| F3 | H1b contradicted 0.007 against D0's 0.228 **hit**; precision 0.958 **hit** |
| F4 | HUMAN 0.04 a sample **hit**; contradicted 0 **hit**; precision not scorable (no graded statement) |
| F5 | sentence 1 below sentences 2 and later, in every writer **hit** |
| F6 | supported only in part 0.66 **hit** |
| F7 | test contradicted within dev intervals (0.238, 0.229, 0.229) **hit** |
| F8 | grader precision 0.90 [0.77, 0.96] **hit** |
| A1.1 | contradicted +0.086 to +0.087 **hit**; precision 0.437 **hit** |
| A1.2 | H1b 0.007 **hit** |
| A1.3 | fresh hand check 0.95 [0.83, 0.99] **hit** |

## Known limits

- Only lead-style statements are found (F2's patterns): dates in brackets, "born in", "is a", "is a town in". Statements such as "was President of France from 1887 to 1892" are not read.
- The store is coarse: most supported dates are checked to the year only, and is_a and language never contradict.
- Faults seen and not fixed: a one-word name can match a person's page ("Paris (1892-1895)"); a year without BC passes for a BC year; "British" passes for a Scot before 1707.
