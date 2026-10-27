# A2 and A3: question-form fact text, and answering by scoring statements

**Odin Loch** · 2026-10-28 · steps A2 (route R1) and A3 (route R3) of [CYPHALM_ANSWERING.md](../CYPHALM_ANSWERING.md) §4, approved by Odin on 2026-10-28

**Questions.**
1. **A3.** Does CyphaLM answer more questions when it scores its own candidate answers inside statements built from F2's learned patterns?
2. **A2.** Does question-form fact text from the fact store (R1) let CyphaLM answer the facts it covers, at a 2% wrong rate?

**Answers.**
1. **No.** R3 runs on 25–32 dev questions and gets 1 right raw (Markov → Ryazan); no right answer survives the out-of-fold gate. F2's frames are cleaned text, and CyphaLM can tie a subject to its object only through verbatim bytes. The registered 5–20% misses.
2. **Yes, where the facts are, and only there.** The shard covers 11 dev questions; 5 are right raw, and the gate keeps **2 right, 0 wrong** out of fold (coverage 1.5%). Holdout BPC +0.0014. On the frozen test it covers only 3 questions and adds nothing; R3 on the shard model keeps **1 right, 0 wrong** (Warhol → Pennsylvania, coarse).

PLAN rule 10 holds. CyphaLM writes every answer byte. The shard is training text; the frames and the parse tell CyphaLM what to score (Odin's ruling); candidates are CyphaLM's own bytes. Strings are compared only to grade, to build training text, and for leakage checks.

---

## Predictions (written before any run)

**A3 (R3 statement scoring).**
- **S1.** R3 runs on 20–30 of the 130 dev questions: only those whose relation has an F2 pattern (birth and death dates and places, capital, language, director). About two thirds of them are in-slice.
- **S2.** Coverage at 2% wrong (5-fold CV in dev): 0–3 of 130. On in-slice questions whose fact is in the training prose: 0–5%. This is **below** the registered CYPHALM_ANSWERING prediction (5–20%). F2's frames are cleaned text, while CyphaLM read raw markup (`'''Name'''`, `[[link]]`). Only InfiniGram's long match can tie a subject to its object, and only where the frame's words occur verbatim.
- **S3.** Raw (ungated) R3 right answers: 2–6 of the questions it runs on.

**A2 (R1 fact shards).**
- **S4 (registered, CYPHALM_ANSWERING §4).** A2 raises coverage only on covered questions: 0 to +6 per 100 dev, almost all dates and places.
- **S5.** The F2 shard covers 3–8 dev questions (entity and relation). Raw right on covered questions: 30–70%. Coverage at 2% wrong: +0 to +4 per 100 dev. No change on questions it does not cover.
- **S6.** Holdout BPC changes by ≤ 0.005.
- **S7.** A2 + A3 together: no better than the better of the two alone, within one question.

---

## Setup

As A0/A1 ([report](A0_A1_CYPHALM_QA_AND_DONT_KNOW.md) "Setup"): enwik8's first 8 MiB, Dolly 15,015 records, `cyphalm_trie qa`, dev 130 (100 in-slice, 30 not-in-slice), the frozen test 200 once at the end. The base model B is A0's model plus the copy expert of [A1C_ENTITY_FAMILIARITY_AND_COPY.md](A1C_ENTITY_FAMILIARITY_AND_COPY.md) (holdout BPC 2.114 trie, 1.926 trie + InfiniGram). 4 shared cores, `nice -n 5`, one run at a time.

**Fact in the prose (grading only).** An in-slice question counts as "fact in the prose" when one of its gold answers (over 3 bytes, normalised) occurs in its source article's text in wiki train. 98 of the 100 in-slice dev questions pass. This is loose: a year or a city name can occur in the article for another reason.

---

## A3: R3 statement scoring

### Method (`qa --r3 patterns.tsv`)

1. **Relation and entity, from the parse.** The question's entity is E1's (`--parse`). Its relation is the dktext template that, filled with that entity, equals the question after normalisation (`When was {E} born?` gives `birth_date`). No other parser.
2. **Frames, from F2.** For that relation, F2's patterns (`patterns.tsv`, the D4 run, with F2's seed fix): kept patterns first, then candidates by training subjects, at most 3 distinct frames. A date question also takes the year patterns. Patterns with slots that cannot be filled (`<M>`, `#`) are skipped, so `death_date`'s kept patterns (which need the birth date) drop out and its candidates remain. `[P]` (a pronoun subject) is filled with the entity. **A relation with no pattern gets no frame, and R3 does not run.** No frame is hand-written.
3. **Candidates, all CyphaLM's own bytes:**
   - its greedy chat answer and 5 seeded samples (T = 0.8, nucleus 0.9);
   - for each frame, its greedy and 2 seeded continuations of the frame prefix in the chat form (`User: <q>\nAssistant: Andrey Markov was born in `), up to 40 bytes.
   - From each, the candidate objects: years (3–4 digits) for a date relation; otherwise runs of capitalised words, and, for a continuation, its bytes before the first `, . ; ( ) |` or newline. Each candidate keeps the bytes CyphaLM wrote (markup included) for scoring, and a value with the markup dropped for grading. A candidate inside the question is dropped.
4. **Score.** Each candidate fills each frame; the statement is read after the chat prompt, and its bits per byte (the whole statement) is its score, the best frame counting. The lowest wins.
5. **Confidence, CyphaLM's own:** the margin to the second candidate; agreement (the share of the 6 + 3 × frames sources whose candidates include the winner); the winner's −bpb; its minimum log2 p over the object's bytes; InfiniGram's match length when it predicts the object's first byte (how much of the frame and subject it found).
6. **Gate.** As `qa`: answer when one feature clears a threshold, chosen for ≤ 2% wrong by 5-fold CV inside dev (fold = row mod 5); the test gets the gate chosen on all of dev.

### First attempt (run C1): candidates scored without their markup

R3 ran on **25** of 130 dev questions: `birth_place` 8, `capital` 7, `birth_date` 6, `director` 2, `death_date` 1, `language` 1. 16 are in-slice. It did not run on 83 questions with no template relation, nor on 22 whose relation has no F2 pattern (`definition` 9, `composer` 4, `founder` 3, `author` 2, `inventor` 2, `painter` 2).

- **Raw: 0 right, 25 wrong.** Coverage at 2% wrong: 0.
- **Why.** The first version stripped the markup from every candidate before scoring. The raw text has "Markov was born in [[Ryazan]]" and "Johnston was born in [[Washington, Kentucky…". Scored as plain "Andrey Markov was born in Ryazan", InfiniGram cannot follow the object (the text has `[[` before it), so a common word ("London") scores better. Its own fluent continuations also won by construction: "What is the capital of Angola?" gave "met with in the 13th century".
- **Fix (a dev choice):** score each candidate in the bytes CyphaLM wrote, and take InfiniGram's match length at the object's first byte as the grounding feature, instead of the longest match anywhere in the object.

### Second attempt (run E): candidates scored in the bytes CyphaLM wrote

- **Raw: 1 right, 24 wrong** (in-slice 1 / 15; not-in-slice 0 / 9). The one: "Where was Andrey Markov born?" → **Ryazan**. CyphaLM continued "Andrey Markov was born in " with "[[Ryazan]]", because InfiniGram matched "Markov was born in " (19 bytes) in the article.
- **Gate:** chosen on all of dev, `min_lp ≥ −1.088` keeps Markov alone. Out of fold it keeps 1 wrong answer and no right one: coverage 0.8%, wrong 0.8% (Wilson upper 4.2%).
- **What still wins.** "Albert Sidney Johnston was born in Genghis Khan": its best frame was `[O], [S]` (object first). There the subject comes after the candidate, the copy expert copies it from the question almost free, and that drags any candidate's bits per byte down. Junk continuations still pass as objects ("met with in the 13th century" for the capital of Angola).
- **Third revision (a dev choice, made once):** skip frames whose object comes before the subject; count bits per byte from the object onward (the bytes a candidate changes); a continuation's head is a candidate only if it is a name (at most 5 words, one capitalised), as F2's own type checks require.

### Third revision (run E2): the A3 result on dev

Run E2: model B, F2's patterns (D4's seed-fixed run), the third revision.

- **R3 runs on 25 questions** (16 in-slice, 9 not-in-slice). **S1: hit** (20–30; about two thirds in-slice).
- **Raw: 1 right, 24 wrong** (selective accuracy 0.04, Wilson 0.007–0.195). Still only Markov → Ryazan. **S3: miss** (predicted 2–6).
- **Coverage at 2% wrong (5-fold CV): 0 right, 2 wrong** (1.5% answered, all wrong; Wilson upper of the wrong rate 5.4%). On in-slice questions with the fact in the prose: 0 of 98. **S2: hit** (0–3 of 130; 0–5%). **The registered CYPHALM_ANSWERING prediction (5–20%) misses.**
- The gate chosen on all of dev (`min_lp ≥ −1.088`) keeps the one right answer; out of fold it keeps two wrong ones instead. With one right answer in 130, no gate can generalise.
- What the revision changed: Johnston now gets "North Carolina" (it was "Genghis Khan"), Allan Dwan "Aldershot", the capital of Angola "Southeast". These are names now, but not the right ones.

**Why R3 cannot work much better here.** Where the article says "X was born in [[Y]]", CyphaLM continues the frame with Y and wins (Markov). That form is rare. "Andrei Tarkovsky was born" does not occur at all; the lead says `'''Andrei Arsenyevich Tarkovsky''' (… [[April 4]], [[1932]] …`, and the date frames (`[S] ([O]`) never match a bold name followed by a Russian gloss. Of the 6 in-slice birth-place questions, the frame's prefix with the subject occurs verbatim in the training text for 2 ("Markov was born in [[Ryazan]]", "Johnston was born in [[Washington, Kentucky…"); De Morgan's article says "De Morgan was born June 27, 1806 … in Madura", and the training bytes have no "Koestler was born", "Dwan was born" or "Schweitzer was born" at all (their articles may say "He was born…") (checked with grep on the training bytes, for diagnosis only). An order-4 trie and a suffix-match expert can connect subject and object only through verbatim text.

---

## A2: R1 question-form fact shards

### The shard and the leakage check (`qa --shard F.tsv`, `--shard2 F.tsv`)

- **Shard:** D4's `pattern_learn --shard` output ([report](D4_WIKIDATA_SEEDS_AND_FACT_SHARD.md)), in question form (`question entity relation source response`), facts from the seeds and extractions of F2. Two versions were run on dev: the base shard (`shard_facts_base.tsv`, 1,915 rows) and the final one with Wikidata-assisted seeds (`shard_facts.tsv`, 2,001 rows).
- **Rows from the BPC holdout are dropped at load** (source ending in `_holdout`): 486 of the base rows, 513 of the final ones. They come from pages inside `[7340032, 8388608)`, so training on them would leak holdout text and teach facts CyphaLM never read. Kept: 1,429 and 1,488 rows.
- **Leakage check at load:** no shard question may equal a dev or test question after normalisation (test questions from `--test`, or `--leak` on dev-only runs, which reads only the questions). Otherwise `qa` exits with status 1. Both shards: **0**. Tested on a synthetic file: 2 leaking rows give exit 1, and a missing `--leak` gives exit 1.
- **Turns:** `User: <question>\nAssistant: <response>\n\n`, spread evenly through both Dolly passes, as the don't-know turns are.
- **Coverage (analysis only).** A dev question is covered when a shard row has its E1 entity and its relation (the dktext template relation; a year relation counts for its date). **11 dev questions are covered**, all in-slice: 5 birth dates (Meighen, Avicenna, Tarkovsky, Al Gore, Alan Kay), 3 capitals (Angola, Anguilla, Alabama), 2 birth places (Johnston, Schweitzer), 1 death date (Alcuin). 30 more have a shard row on their entity but for another relation; 87 have none; 2 have no E1 entity. D4 counted 16 of 100 in-slice before dropping holdout rows, with its own matching rule, so the counts differ.

### Results on dev

| Run | Shard | Holdout BPC trie / + IG (Δ vs A0) | Raw right, trie + IG (in-slice) | CV at 2% wrong: right / wrong (coverage) | Covered: CV right | Not covered: CV right |
|---|---|---|---|---|---|---|
| A0 / E2 (B) | none | 2.344571 / 2.071574 | 2 (luck) | 0 / 0 (0.0%) | 0 / 11 | 0 / 117 |
| F | base, 1,429 rows | 2.345865 (+0.0013) / 2.072343 (+0.0008) | **7** | **2 / 0 (1.5%)** | **2 / 11** | 0 / 117 |
| F2 | final, 1,488 rows | 2.345941 (+0.0014) / 2.072393 (+0.0008) | **7** | **2 / 0 (1.5%)** | **2 / 11** | 0 / 117 |

- **What it answers now** (trie + InfiniGram, chat form): "When was Avicenna born?" → *Avicenna was born in 980.*; Al Gore → *1948*; Alan Kay → *1940*; "When did Alcuin die?" → *Alcuin died in 804.*; "In which Mexican state is Acapulco?" → *Acapulco is in Guerrero.* (the shard's `located_in` row; the question's relation has no template, so it counts as not covered). The other two right answers are A0's luck (Alexis Carrel, Montgomery).
- **Raw right on covered questions:** 5 of 11 (0.45, Wilson 0.21–0.72), one of them A0's lucky Montgomery.
- **The gate keeps 2 out of fold** (`min_lp`, chosen in all five folds), both right: coverage 1.5% (Wilson 0.4–5.4%), selective accuracy 2/2 (Wilson lower 0.34), wrong rate 0 (Wilson upper 2.9%). On all of dev the gate (`min_lp ≥ −0.895`) keeps 3, all right.
- **The trie alone gains nothing:** its 4-byte context cannot reach the question. **With copy** (trie + InfiniGram + copy), raw right is 6 but the gate keeps none out of fold: the CV picks `ig_match0` in four folds and no gate in one.
- **The final shard adds nothing over the base one on dev.**
- **Predictions.** S4 (registered): **hit**, +1.5 per 100, only on covered questions, all dates. S5: **miss** on the count (11 covered, predicted 3–8); **hit** on raw right (0.45 in 0.30–0.70) and on coverage (+1.5 in 0–4 per 100); not-covered questions gain one raw right answer (Acapulco, an entity-only match) and nothing gated. S6: **hit** (+0.0014).

### A2 + A3 together (run F2)

R3 on the shard model runs on 33 questions (the final patterns add `author`, `founder` and `inventor` frames; one question gets no candidate). **Raw: 3 right**, all from the shard's facts: Anguilla → *The Valley*, Schweitzer → *Kaysersberg* (InfiniGram match 71 bytes into the shard sentence), Alcuin → *804*. Out of fold: **0 right, 1 wrong**. The gate on all of dev (`min_lp ≥ −0.632`) keeps 2 right. So A2 + A3 is worse than A2 alone at 2% wrong (0 against 2): **S7 hit** (no better than the better of the two).

### A3 alone with the final patterns (run E3)

Model B, D4's final `patterns.tsv`. R3 runs on 32 questions (`author`, `founder` and `inventor` now have frames). **Raw: 1 right** (Markov again), 31 wrong. **Out of fold: 0 right, 4 wrong** (3.1% answered, all wrong; Wilson upper 7.6%): the 5-fold gate breaks its 2% target, because the folds have no right answer to anchor a threshold. The gate on all of dev is "answer nothing".

### The teacher shard (`--shard2`)

Built and tested on a small synthetic file only: rows load with the same schema, a `_holdout` row is dropped, 2 rows equal to dev questions make `qa` exit with status 1, and the turns enter the chat text (`chat_turns … shard2 2`). Odin has ruled that the Qwen models do not run in this container, so **the real teacher-shard runs (B + teacher, B + both) are for Odin's PC**:

```bash
./cyphalm_trie qa $E $D --dev $DEV --leak $TEST --parse parse_dev.tsv --shard2 shard_teacher.tsv              # B + teacher
./cyphalm_trie qa $E $D --dev $DEV --leak $TEST --parse parse_dev.tsv --shard shard_facts.tsv --shard2 shard_teacher.tsv   # B + both
```

`qa` drops the teacher shard's `_holdout` rows and refuses any row whose question equals a dev or test question. Its effect is measured against the fact shard alone (run F2).

---

## Frozen test (200 questions, run once): A2 + A3

The final configuration is run F2: model B with copy, the final fact shard (1,488 rows; 513 holdout rows dropped; leakage 0 against dev and test), R3 with the final patterns. The `--test` run re-prints every dev line; they are byte-identical to run F2.

| System | Raw R / A / W | Gated (dev gate) R / A / W | Coverage | Selective acc. (Wilson lo) | Wrong rate (Wilson hi) |
|---|---|---|---:|---|---|
| chat / trie | 0 / 0 / 200 | 0 / 200 / 0 | 0.0% | n/a | 0.0% (1.9%) |
| chat / trie + InfiniGram (A2) | 2 / 0 / 198 | 0 / 200 / 0 | 0.0% | n/a | 0.0% (1.9%) |
| chat / trie + InfiniGram + copy | 2 / 0 / 198 | 0 / 200 / 0 | 0.0% | n/a | 0.0% (1.9%) |
| R3 on the shard model (A2 + A3) | 2 / 178 / 20 | **1 / 199 / 0** | 0.5% | 1/1 (0.21) | 0.0% (1.9%) |

- **The shard covers only 3 test questions** (entity and relation), against 11 on dev. The test's in-slice subjects are other "A" pages, and few of them are in F2's store for the relation asked.
- **A2 adds nothing on the test.** Raw right is A0's two (Alaska, a stored Dolly answer; Apollo 8, luck). The dev gate (`min_lp ≥ −0.895`) keeps nothing.
- **A2 + A3 keeps one test answer, and it is right:** "Where was Andy Warhol born?" → *Pennsylvania*. It counts under the T-QA containment rule (gold "Forest City, Pennsylvania"): a coarse answer, the state rather than the town. R3's other right answer, Kurosawa → *1910*, was not kept by the gate.
- R3 ran on 22 test questions (`capital` 7, `author` 6, `inventor` 3, `birth_place` 2, `founder` 2, `birth_date` 1, `director` 1). Its most frequent wrong answer, on 6 of the `author`, `founder` and `inventor` questions, is "American-Canadian writer William Gibson": a Dolly answer that those frames' continuations reach.
- **BPC** (this model): trie mixture 2.345941, trie + InfiniGram 2.072393; with copy 2.114645 and 1.926773.

---

## Predictions, scored

| Prediction | Result | Score |
|---|---|---|
| S1. R3 runs on 20–30 dev questions, about two thirds in-slice | 25 (16 in-slice) with the seed-fixed patterns; 32 with the final ones | **Hit** (on the patterns the prediction was made for) |
| S2. A3 coverage at 2% wrong 0–3 of 130; 0–5% of in-slice in-prose | 0 right (E2: 2 wrong out of fold; E3: 4 wrong) | **Hit** |
| CYPHALM_ANSWERING §4. A3 answers 5–20% of in-slice questions whose fact is in the prose, at 2% wrong | 0 of 98 | **Miss** |
| S3. R3 raw right 2–6 | 1 (E2, E3); 3 on the shard model | **Miss** (alone) |
| S4 (registered). A2 raises coverage only on covered questions, 0 to +6 per 100 dev, mostly dates and places | dev +1.5 per 100 (2 of 130), both covered birth dates; test +0 | **Hit** |
| S5. Shard covers 3–8 dev questions; raw right on covered 30–70%; coverage +0 to +4 per 100; no change elsewhere | 11 covered; 5 of 11 raw right (0.45); +1.5; not-covered +0 gated (+1 raw, an entity-only match) | **Miss** on the count, **hit** on the rest |
| S6. BPC change ≤ 0.005 | +0.0014 (trie), +0.0008 (+ IG) | **Hit** |
| S7. A2 + A3 no better than the better alone | dev: 0 against A2's 2; test: 1 against A2's 0 | **Hit** on dev; on test A2 + A3 is better by one answer |

---

## Flags for Odin

1. **R3's frames and CyphaLM's text do not meet.** F2 learned its patterns on cleaned text; CyphaLM reads raw markup, so a frame's subject and object rarely sit next to each other in its bytes. R3 works only where a sentence says "X was born in [[Y]]" verbatim (Markov), or where the shard wrote such a sentence (Warhol, Schweitzer, Alcuin). Two ways on: learn the patterns on the raw bytes CyphaLM reads (F2's next step), or let R3 score frames inside the article's own lead (that needs a retrieval step, which rule 10 forbids).
2. **The gates choose on two or three right answers.** Every dev gate here rests on 1–3 right answers; out of fold it either answers nothing or breaks the 2% target (E3: 4 wrong). Until a route produces tens of right answers on dev, coverage at 2% wrong cannot be estimated, only bounded (Wilson upper 2.9%).
3. **The T-QA containment rule is lenient.** "Pennsylvania" for "Forest City, Pennsylvania" and "1910" for "23 March 1910" count as right; so do echoes of the question when the gold is inside it ("The Apple Newton?" for "Apple", "the alan parsons project" for "Alan Parsons", both with the copy expert on). I left the rule as defined.
4. **The final shard is no better than the base one on dev** (both 2 right out of fold). Its Wikidata-assisted seeds mainly gave R3 more frames (`author`, `founder`, `inventor`), which raised R3's wrong answers.

---

## Verdict

- **A3 (R3 alone) does not answer.** 1 right in 25–32 attempts on dev, and no right answer survives the out-of-fold gate. The registered 5–20% misses: the frames are cleaned text and CyphaLM can tie a subject to its object only through verbatim bytes.
- **A2 (R1) answers where its facts are.** On dev, 5 of 11 covered questions are right raw, and 2 survive the 2% gate with no wrong answer (coverage 1.5%, selective accuracy 2/2). On the test it covers only 3 questions and adds nothing. BPC +0.0014.
- **A2 + A3 kept the first right gated test answer of any CyphaLM configuration** (Warhol, coarse). One answer, Wilson 0.21–1.00.
- **Defaults:** A2's shard text is safe to keep (BPC +0.0014, no wrong answers gated on dev or test, holdout rows dropped, leakage 0). R3 should stay an option. Odin decides.

---

## Costs

| Item | Value |
|---|---|
| Code | shared with A1c: `cyphalm_trie.cpp` 1,765 → 2,542 lines. A2/A3's part: shard loading with the holdout drop and leakage check, coverage, frames from F2, candidates, statement scoring, the R3 gate and report, seeded sampling in `decode` |
| `qa` run with `--r3` and `--shard` | 7.3–8.7 min wall time on a shared core (A0: 7.5 min); R3 is not timed separately |
| Shard | 1,488 rows (about 0.2 MB) after dropping 513 holdout rows |
| External data | enwik8, Dolly, F2's `patterns.tsv` and `shard_facts.tsv` (D4), E1's parse |

---

## Reproduce

```bash
# build, selftest, dktext and the E1 parse as in A1C_ENTITY_FAMILIARITY_AND_COPY.md "Reproduce"
E=/root/cdata/enwik8; D=/root/cdata/dolly.jsonl; DEV=bench/eval/t_qa_wiki_dev.tsv; TEST=bench/eval/t_qa_wiki.tsv
P0=f2/basefix/patterns.tsv; P=f2/patterns.tsv; S0=f2/shard_facts_base.tsv; S=f2/shard_facts.tsv   # D4's files
./cyphalm_trie qa $E $D --dev $DEV --leak $TEST --parse parse_dev.tsv --copy --r3 $P0              # runs C1, E, E2 (R3 versions)
./cyphalm_trie qa $E $D --dev $DEV --leak $TEST --parse parse_dev.tsv --copy --r3 $P0 --shard $S0  # run F
./cyphalm_trie qa $E $D --dev $DEV --leak $TEST --parse parse_dev.tsv --copy --r3 $P --shard $S    # run F2
./cyphalm_trie qa $E $D --dev $DEV --leak $TEST --parse parse_dev.tsv --copy --r3 $P               # run E3
# the frozen test, once, at the end:
./cyphalm_trie qa $E $D --dev $DEV --test $TEST --parse parse_all.tsv --copy --r3 $P --shard $S
```

The runs used these files, sha256 (equal to D4's final outputs): `shard_facts.tsv` 438aa81c…3d3e7, `patterns.tsv` 57e7b4f3…73f0, `shard_facts_base.tsv` ea34124c…8e28e97 (run F). Runs C1, E and E2 used the first, second and third R3 versions; only the third is in the code. stdout lines: `R3_FRAME`, `R3_RUNS`, `R3_EX`, `RAW R3` / `CV R3` (A3), `shard …` and `SHARD` (A2 load and coverage), `CV … shard_covered` / `in_slice_fact_in_prose` (splits), `TEST_*` (the frozen test). Run outputs are in `/root/work/answer/` (`c1.out`, `e.out`, `e2.out`, `e3.out`, `f.out`, `f2.out`, `t2.out`). In `t2_rows.tsv`, written before a last relabelling, R3 rows have set `test_r3`; the code now writes them as set `test` with model `r3`.
