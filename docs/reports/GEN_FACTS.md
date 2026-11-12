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
