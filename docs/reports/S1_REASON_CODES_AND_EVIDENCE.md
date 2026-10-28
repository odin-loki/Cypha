# S1: reason codes and evidence cards on CyphaLM's answers

**Odin Loch** · 2026-10-28 · DISCOVERY §13 S1 · follows [A1D_A2B_ANSWERING_VARIANTS.md](A1D_A2B_ANSWERING_VARIANTS.md) and [S3_S5_GAP_MAP_AND_HYPOTHESES.md](S3_S5_GAP_MAP_AND_HYPOTHESES.md)

(The rules and predictions below were written before the runs. Results and decisions are added under them.)

**Question.** Every "I don't know" should say why, and every answer should show where it came from. How often is the reason right today? How far can it go within rule 10? What does an evidence card cost?

**Gate (DISCOVERY §13).** Reason-code accuracy ≥ 90% on 100 hand-checked rows, and the I3 CTest.

## Rules (fixed before the runs)

### The system that is read

- C1L, A1d's recommended configuration in A4b's form: the `low`-only familiarity token (`ent_title`), the don't-know rows (`dk_mention.tsv`, share 0.20), D4's fact shard, the copy models, `--cls shape:6:2:f`, the similar-context expert (default, gated off in answers).
- Parses come from `parse_all_span.tsv` in every run, so dev numbers compare across runs.
- The audited output is the chat form with trie + InfiniGram (`chat/trie_ig`): the model and form the earlier tag counts used.
- On dev, the gate for a question is the one chosen on the other four folds (fold = row % 5), as qa's CV lines do. On test it is the gate chosen on all of dev.
- Choices are made on dev. The frozen test is read once, at the end.

### The reference reason (slice meaning)

Which knowledge a code describes is Odin's decision (HANDOFF §5.7). The working definition here is **what CyphaLM has read**: its wiki slice (enwik8 train) and its chat text. The reference code for a question is:

1. **Not in the slice** (its source article is not in the wiki slice): `UNKNOWN_ENTITY`.
2. **In the slice**, and a gold answer occurs in the source article's text, or the fact shard has the question's entity and relation: `LOW_CERTAINTY`. CyphaLM has read the fact and is not sure enough.
3. **In the slice** otherwise: `NO_EVIDENCE`. It has read about the entity, but not this fact.

The rule reads grading-only data: T-QA's `in_slice` and gold answers, and the shard's coverage. It never feeds an answer, a gate or a tag.

A looser count is also reported: qa's own `reason_ok`, which accepts either `NO_EVIDENCE` or `LOW_CERTAINTY` on in-slice questions.

### The audit set

- 100 dev questions are drawn once, before any tag is read: the first 100 of the 130 after a shuffle with `mt19937_64(20261028)`.
- Each row is the system's final output for that question.
- **Accuracy** is right codes divided by don't-know outputs among the 100, with a Wilson 95% interval. Kept answers carry no code and are left out.
- A rows file is written for Odin's hand-check. Its columns:
  - the question;
  - the output;
  - its code and who wrote it;
  - the reference code and the rule's evidence;
  - a blank `odin` column.

### Where a code comes from (rule 10)

- **CyphaLM's own message with a tag:** the tag's code. Nothing replaces it.
- **CyphaLM's own message without a code** (it stopped before the tag): the gate supplies the code, marked `by=gate`. DISCOVERY §3.5 lets the gate write the message and tag on CyphaLM's behalf.
- **An answer the gate withholds:** the gate writes the message and the tag (`by=gate`), with the code from its rule.

Gate rules tried, chosen on dev:

- **G0.** `LOW_CERTAINTY` always: the gate withheld the answer for low confidence.
- **G1.** `UNKNOWN_ENTITY` when the familiarity token is `low`, else `LOW_CERTAINTY`.
- **G2.** `UNKNOWN_ENTITY` when the entity has no page title in the training bytes and fewer than c mentions, else `LOW_CERTAINTY`. c is chosen by 5-fold CV on dev.
- **Reported only, not a candidate without Odin:** G1x, where the gate also replaces CyphaLM's own code when its statistic disagrees.

### Evidence cards

A card is made after the answer, never before.

**For a kept answer**, the card gives:

- the longest suffix of prompt + answer that occurs in CyphaLM's training text (wiki train + chat);
- how many of the answer's bytes that suffix covers;
- its count;
- up to three places: a page title and offset, or a chat turn.

**For a don't-know**, the card gives:

- the entity's page-title count and mention count in the training bytes;
- up to three places where it is mentioned.

Cards come from `locate` on the FM-index: sampled suffix-array entries, built only when cards are asked for.

**Cards never change an answer, a gate or a tag.** This is checked: the rows file is byte-identical with and without `--cards`.

## Predictions (registered before the runs)

| # | Prediction |
|---|---|
| P1 | CyphaLM's own codes (its tags only) on the audit set are right on 55–70% of the tags it writes (A1d on test: 45 of 72, 0.63) |
| P2 | Final cards with G1 are right on 78–88% of the audit set's don't-know outputs. The ceiling is the familiarity token: on dev it calls 20 of 100 in-slice questions `low`, and there CyphaLM's own tag says `UNKNOWN_ENTITY` |
| P3 | G2 beats G1 by at most 5 points on dev. The 90% gate is missed on dev under every rule, because CyphaLM's own tags on in-slice `low` questions cannot be replaced |
| P4 | The sampled suffix array costs about 0.28 bytes a text byte (n/32 × 4 bytes of samples plus a mark bit a row): about 5.5 MB for qa's training text. Its build, one walk over the BWT, takes under 15 s. A card takes under 5 ms |
| P5 | Each kept right answer on dev (two, both from the shard) has its whole answer inside the longest match, and that match is found in a shard turn |
| P6 | (I3) Every withheld answer becomes a don't-know card with a code. This holds by construction, and the CTest checks it |

---

## Results (runs 2026-10-28)

**The runs:**

- **B0:** the tool before S1 (`22ef43b`), with C1L's options.
- **C1:** with `--cards`, rule G1.
- **P0:** the final tool without `--cards`.
- **F1:** the final tool with `--cards` (rule G0) and `--test`. This was the one read of the frozen test.

All runs are on dev (130 questions). The machine was shared with two other agents, so wall-clock numbers are rough.

### Cards change nothing

- **Rows file.** With `--cards` (C1, and F1's dev rows) and without them (P0), the rows file is byte-identical to B0's (sha256 `e8b4d816…`).
- **Stdout.** It is identical to B0's apart from the new `REASON` and `CARDS` lines and the wall-clock fields: the similar-context build seconds and the decode latencies.
- **FM-index CTest.** All 178,635 query checks pass unchanged. `locate` adds 13,250 checks against a brute-force search on the same random texts:
  - the suffix-array order;
  - every occurrence of a pattern;
  - `longest_suffix`;
  - stored files of version 2 (with locate) and version 1 (without).
- **Planted bugs.** The FM-index CTest catches a planted bug in the LF step. The `cyphalm_trie` selftest catches a planted I3 bug (a withheld answer shown as an answer).

### Reason codes on the audit set (dev, chat form, trie + InfiniGram)

All 100 audit rows end in a don't-know: the CV gate kept no answer among them.

| Who wrote the code | Cards | Right (G0) |
|---|---:|---:|
| CyphaLM, in its own tag | 42 | 25 (0.595) |
| CyphaLM's message, with the gate's code (its message had none) | 15 | 15 |
| The gate, on CyphaLM's behalf (it withheld the answer) | 43 | 42 |
| **All** | **100** | **82** |

| Gate rule | Right of 100 | Wilson 95% | Looser count (`reason_ok`) |
|---|---:|---|---:|
| **G0** (chosen) | **82** | 0.733–0.883 | 83 |
| G1 | 81 | 0.722–0.875 | 82 |
| G2, cut 1,024 mentions (chosen on all 100) | 82 | 0.733–0.883 | 83 |
| G2, 5-fold CV estimate | 81 | 0.722–0.875 | 82 |
| G1x, reported only | 81 | | |
| G2x, reported only | 83 | 0.745–0.891 | 84 |

**Where the 18 wrong codes are (G0):**

- **17** are CyphaLM's own `UNKNOWN_ENTITY` on in-slice questions.
  - Every one is on a turn where the familiarity token said `low`.
  - The token comes from E1's entity having no page title in the slice. Yet most of these entities are mentioned often: Ireland 243 times, Canada 673, "the world" 5,238.
  - Example: "Who was the second Prime Minister of Canada?". Its source article is in the slice. CyphaLM says it has nothing about the entity.
- **1** is the gate's `LOW_CERTAINTY` where the reference is `NO_EVIDENCE`: "What is an android?" (the gold answer is not in the source article's text).

**Why 90% is out of reach within rule 10:**

- CyphaLM's own code follows its input, and nothing may replace it.
- Mention counts do not separate the cases:
  - not-in-slice entities on the audit set are mentioned 0 to 301 times (one at 5,238);
  - in-slice entities with no title are mentioned 0 to 5,238 times.
- Even a gate allowed to replace CyphaLM's own codes (G2x) reaches 83 at G2's cut. Its best cut, fitted on these same 100 rows, gives 85.

**A training-text variant was not run.** Every wrong own code comes from the `low` token. The don't-know rows already teach the code the token means (`UNKNOWN_ENTITY`). A token that also counts mentions would trade about as many right codes as it fixes: at a cut of 100 mentions, by count on the audit rows, +10 in-slice and −8 not-in-slice. It would also change answering, so rule 3 would have to be read again.

**On dev, CyphaLM sometimes writes the token itself.** On 14 audit questions with no token in the prompt it wrote `<|fam=low|>I don't know.` and stopped at the full stop, before any tag. A 15th message had a garbled tag. These 15 cards take the gate's code. Decoding was not changed.

### The frozen test, read once (F1, rule G0)

| | Cards | Right | Rate (Wilson 95%) |
|---|---:|---:|---|
| All don't-know cards, chat, trie + IG | 200 | 169 | **0.845** (0.788–0.889) |
| CyphaLM's own codes | 71 | 45 | 0.634 |
| The gate's codes (with and without CyphaLM's message) | 129 | 124 | 0.961 |

- The other rules on test: G1, G2, G1x and G2x are each right on 170. The choice was fixed on dev before this read.
- The gate chosen on all of dev kept no test answer, so the test has no answer cards.

### Evidence cards

**An answer card (dev, row 43, kept by the gate):**

```
answer  cyphalm  min_lp -1.334 >= -1.426  Avicenna  birth_date  "Avicenna was born in 980."
match 56 answer_bytes 25/25 count 2 | shard:User: In what year was Avicenna born?+18 | shard:User: What year was Avicenna born?+15
```

- The whole answer lies inside a 56-byte match.
- That match occurs twice in CyphaLM's training text, both times in D4's shard turns.
- The other kept answer (Al Gore, 1948) is the same: a 55-byte match, found twice, both in shard turns.

**A don't-know card (dev, row 52, the gate's):**

```
dont_know  gate  LOW_CERTAINTY  min_lp -1.799 < -1.593  "an android"  definition
I don't know. I am not sure enough to answer. <|unknown reason=LOW_CERTAINTY entity="an android" relation=definition gap=G-A032|>
entity="an android" titles 1 mentions 105 | wiki:Android+9401 | wiki:Android+9416 | dolly:User: What is WestWorld?+747
```

**Format.** The cards file has one line per question, form and model: set, row, form, model, kind, who wrote it, code, gate decision, entity, relation, text, evidence.

**The audit rows file** is `bench/eval/s1_reason_audit_dev.tsv`. It has 100 rows with the reference code, the rule's evidence, the system's output and code, and a blank `odin` column for the hand-check.

### Cost

| | Value |
|---|---|
| Locate samples, qa's training text T (about 20.5 M bytes) | 5.28 MB (0.258 bytes a byte) |
| Locate samples, A1c's lowered text | 5.04 MB |
| Source tables (page titles, chat turns) | 3.74 MB |
| Locate build, both indexes, one walk each | 28.1 s (C1), 39.8 s (F1), on shared cores |
| A card | 0.016 ms (dev), 0.026 ms (test) |
| qa peak RSS | B0 284.7 MB, P0 277.2 MB; with cards C1 288.5 MB, F1 292.7 MB (with the test) |

The structures are built after the answers, below qa's peak (the class and similar-context builds). Run-to-run noise in the peak is about ±8 MB.

### Predictions, scored

| # | Result |
|---|---|
| P1 | **Hit.** CyphaLM's own codes are right on 25 of 42 audit tags (0.595); on test 45 of 71 (0.634) |
| P2 | **Hit.** G1 gives 81 of 100 (78–88 predicted). The ceiling is the `low` token, as predicted |
| P3 | **Hit.** G2 beats G1 by 1 point. 90% is missed under every rule, and even G2x reaches only 83 at its chosen cut (85 at a cut fitted on dev) |
| P4 | **Mostly hit.** 0.258 bytes a byte (5.3 MB); a card takes 0.016 ms. The build is 14–20 s an index on shared cores, against under 15 s predicted |
| P5 | **Hit.** Both kept answers lie whole inside a match found in shard turns |
| P6 | **Hit.** The I3 CTest: 600 random gate decisions × 3 codes; every withheld answer is a don't-know card with a known code |

## Decisions

1. **The cards carry G0.** When the gate withholds an answer, its code is `LOW_CERTAINTY`. That is the gate's own reason, and on dev it ties G2 without G2's parameter.
2. **S1's gate is not met.** The rate is 82% on dev (Wilson 73–88%) and 84.5% on test, not 90%. The I3 CTest passes.
   - What remains is the familiarity token's entity, not the reason codes.
   - It cannot be fixed within rule 10 without changing answering.
3. **Cards stay an option (`qa --cards FILE`).** They change no answer. They cost about 14 MB and under a minute a run.

## For Odin

- **Which knowledge a code describes (HANDOFF §5.7).** The slice meaning is used here.
  - Under it, `UNKNOWN_ENTITY` for an entity CyphaLM has read about hundreds of times (Ireland, Canada) is wrong.
- **The 17 wrong codes all come from the `low` token.** It is set by the entity's page-title count, so entities that are named but have no page get `low`. There are two ways forward, and both need Odin:
  - change the token's statistic or E1's entity choice (A1d). That changes answering, so rule 3 is read again;
  - let the gate replace CyphaLM's own code when its statistic disagrees (G2x). On dev that gives 83, at most 85, so it is not enough alone.
- **The audit file is ready for the hand-check:** `bench/eval/s1_reason_audit_dev.tsv`, with a blank `odin` column.

## How to reproduce

```
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target cyphalm_trie fm_infinigram_test -j2
build/fm_infinigram_test && build/cyphalm_trie selftest
# C1L as in A4b, parses from parse_all_span.tsv (A1d's files under /root/work; see A1D_A2B_ANSWERING_VARIANTS.md)
build/cyphalm_trie qa enwik8 dolly.jsonl --dev bench/eval/t_qa_wiki_dev.tsv --leak bench/eval/t_qa_wiki.tsv \
  --parse parse_all_span.tsv --dk dk_mention.tsv --dk-ratio 0.20 --fam after --fam-stat ent_title --fam-train low \
  --shard shard_facts.tsv --copy --base a0_rows.tsv --cls shape:6:2:f --out rows.tsv \
  --cards cards.tsv [--test bench/eval/t_qa_wiki.tsv]
```

The cards file, `cards.tsv.audit.tsv`, and the `REASON` / `CARDS` lines on stdout come from that run. The `REASON` lines give every rule per form and model.
