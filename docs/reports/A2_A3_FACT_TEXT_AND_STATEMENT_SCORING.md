# A2 and A3: question-form fact text, and answering by scoring statements

**Odin Loch** · 2026-10-28 · steps A2 (route R1) and A3 (route R3) of [CYPHALM_ANSWERING.md](../CYPHALM_ANSWERING.md) §4, approved by Odin on 2026-10-28

**Status: in progress.** Predictions are written first; results follow as each run ends.

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
2. **Frames, from F2.** For that relation, F2's patterns (`patterns.tsv`, the f2 agent's run with the seed fix): kept patterns first, then candidates by training subjects, at most 3 distinct frames. A date question also takes the year patterns. Patterns with slots that cannot be filled (`<M>`, `#`) are skipped, so `death_date`'s kept patterns (which need the birth date) drop out and its candidates remain. `[P]` (a pronoun subject) is filled with the entity. **A relation with no pattern gets no frame, and R3 does not run.** No frame is hand-written.
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
