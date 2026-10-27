# S1 and S2: a fast chat index with reasons, and the epistemic library

> **Removed 2026-10-27.** The question-answering tool measured here did not use CyphaLM, so it broke [PLAN rule 10](../PLAN.md#2-rules-of-the-work) and its code was deleted. The numbers below are not CyphaLM results. See [REMOVED_NON_CYPHALM_QA.md](REMOVED_NON_CYPHALM_QA.md).

**Odin Loch** · 2026-10-27

**Questions.**
1. Can `chat_index --ask` answer in seconds instead of re-running the 52 s sweep, with the same decisions?
2. What does the chat index get right, abstain on and get wrong on T-QA-wiki once Dolly counts as its knowledge?
3. Can the epistemic layer of DISCOVERY (states, conflict, reason codes, unseen estimators, cards) be a tested header-only library?

**Answers.**
1. Yes. One question now takes 1.3 s (was 53.1 s). 20 of 20 test questions give the same decision, score, margin and record as the old path. The first run on new data spends a one-off 28 s on calibration.
2. Strict (slice-only test): 2 right / 194 abstained / 4 wrong, as in `T_QA_WIKI_200.md`. Knowledge-source (Dolly is the evidence): 5 right / 194 abstained / 1 wrong. Only the Don Quixote answer is truly wrong.
3. Yes. `epistemic.hpp` is header-only, reuses the `Opinion` and `Grade` types from `ecs.hpp`, and has 195 known-answer checks that pass under g++ and clang++. One number in the brief did not match its formula (see Ambiguities).

Files: [`chat_index.cpp`](../../native/tools/chat_index.cpp), [`epistemic.hpp`](../../native/include/cypha/eval/epistemic.hpp), [`epistemic_test.cpp`](../../native/tools/epistemic_test.cpp), the S2 block at the end of [`native/CMakeLists.txt`](../../native/CMakeLists.txt). Spec: [`DISCOVERY.md`](../DISCOVERY.md) sections 2.1-2.3, 3.1, 3.3, 10 and PLAN rule 9. Earlier results: [`B7_CHAT_INDEX.md`](B7_CHAT_INDEX.md), [`T_QA_WIKI_200.md`](T_QA_WIKI_200.md).

---

## S1. The fast path

### What changed

- `--ask "q"` and the new `--ask-file questions.tsv` (first tab-separated column is the question; a first line reading `question` is a header) build one index and answer. They no longer run the evaluation sweep.
- The index needs the headline operating point: configuration, scorer, and the gate point G (`ts`, `tm`). It is kept in a small file next to the data, `<dolly.jsonl>.chat_index_op` (override with `--thresholds PATH`).
  - The file stores `ts` and `tm` with 17 significant digits, so they read back bit-identical to the sweep's.
  - It also stores a fingerprint: file size, record count, indexed count, `--irep` and the `--configs` list. A file with a different fingerprint, or a missing or malformed one, is not used.
  - The sweep writes the file at the end of every run.
  - With no valid file, the fast path runs only the dev stage (the five perturbed forms of each D record and the leave-one-out queries) for the four configurations, picks the headline by the same rule as the sweep, writes the file, then answers. This is the "recompute only what is needed" option, cached.
- The dev stage and the headline rule were factored out of the sweep (`dev_operating_points`, `pick_headline`), so both paths run the same code.
- `--sweep` with `--ask` or `--ask-file` restores the old behaviour (sweep, then answer). `--recalibrate` forces the dev stage. `--quiet` prints only the machine lines.

### Timing

2 cores, g++ 13.3 `-O2`, Dolly 15,015 records, other agents running.

| Path | Wall time |
|---|---|
| Old `--ask` (whole sweep, then answer) | 53.1 s |
| New, thresholds file present, one question | 1.26-1.37 s (5 runs). Load 0.20 s, index build 1.0-1.1 s, search 2 ms |
| New, 200 questions in one call | 1.28 s (search 54 ms in total) |
| New, 1,502 questions in one call | 2.7 s |
| New, first run on new data (one-off calibration) | 28.7 s (dev stage 27.4 s) |

Peak RSS of a fast-path run is 156 MB. The target of under 3 s for one question, including the build, is met with a factor of two to spare.

### Same decisions as the old path

- **Thresholds.** The file written by the dev-only calibration and the file written by the full sweep are byte-identical (`plain`, scorer `sum`, `ts 0.72999999999999998`, `tm 0.10000000000000001`). A truncated file triggers recalibration and reproduces the same file.
- **Sweep.** The new binary's sweep output is identical to the old binary's, line for line, apart from timing lines. The refactor changed nothing the sweep prints.
- **20 questions.** I took a scratch copy of the unmodified old source and replaced only its final `--ask` block by a loop over a question file, so the old sweep runs once and the old gate is applied to each question. The 20 questions are: the 6 that `chat_index` answered on T-QA-wiki, the 1 ambiguous one, the 5 highest-scoring abstentions, and 8 questions made from Dolly records (verbatim, first letter lowercased, " Thanks.", " please", all lowercase). Result: decision, score and margin to 3 decimals, and record index agree on 20 of 20. 9 answered / 11 abstained on both sides.
- **Real old binary.** The unmodified old `--ask`, run separately on three of those questions (Brazil, Don Quixote, Amazon.com), gave the same score and margin as the new path.
- **Sweep path versus fast path.** `--sweep --ask-file` and the fast path print identical `RESULT_JSON` lines for the 20 questions.

### Reason codes and cards

Rule: answer only if `score >= ts` and `(best - runner-up) >= tm`, as before. Otherwise abstain with one of two codes:

| Code | When |
|---|---|
| `NO_EVIDENCE` | The best stored question scores below `ts`. Nothing stored is close. |
| `AMBIGUOUS_QUESTION` | The best scores at least `ts`, but a stored question with a different text is within `tm` of it. |

If both tests fail, the code is `NO_EVIDENCE` (see Ambiguities for why).

- **Answer card.** The stored response, then `[RETRIEVED · score .. (needs >= ..) · margin .. (needs >= ..)]`, the matched stored question with its record index and category, and a line saying this is retrieval.
- **"I don't know" card.** `[REASON]`, a "Why" line with the numbers, the nearest stored question (record, category, score), the runner-up for `AMBIGUOUS_QUESTION`, and what would let it answer.
- **Machine line.** One `RESULT_JSON {...}` line per question: `n`, `question`, `decision`, `reason`, `score`, `margin`, `runner_up_score`, `ts`, `tm`, `score_below_ts`, `margin_below_tm`, `scorer`, `config`, `nearest_question`, `nearest_record`, `nearest_category`, `nearest_has_context`, the same four for `runner_up_`, and `answer`. The record index is 0-based in load order (after dropping records with an empty instruction or response).

Reason-code distribution:

| Set | Questions | Answered | `NO_EVIDENCE` | `AMBIGUOUS_QUESTION` |
|---|---|---|---|---|
| T-QA-wiki, in-slice | 150 | 2 | 148 | 0 |
| T-QA-wiki, not in slice | 50 | 4 | 45 | 1 |
| T-QA-wiki, all | 200 | 6 | 193 | 1 |
| Dolly held-out candidates, instruction only | 1,502 | 98 | 1,350 | 54 |

- Of the 193 `NO_EVIDENCE` abstentions on T-QA-wiki, 178 also fail the margin test. Of the 1,350 on the Dolly set, 1,171 do.
- The Dolly row uses all 1,502 records with index mod 10 equal to 3, instruction only, so it includes the 49 known near-duplicates (30 of the answered ones score at least 0.999) and drops the context. It is not the B7 held-out number (92.6% abstain, 107 answered). Here 93.5% abstain.
- The one `AMBIGUOUS_QUESTION` on T-QA-wiki is "Who painted Girl with a Pearl Earring?" (0.853 against 0.827). Both stored questions are paraphrases of the same question, so this abstention is a lost answer, not a real ambiguity. The index does not check whether the two stored responses agree.

### Rescoring T-QA-wiki with the fast path

`--ask-file bench/eval/t_qa_wiki.tsv` runs in 1.3 s. Scored with the T-QA-wiki match rule (case, accents and punctuation ignored; equal, or one contains the other when at least one is multi-word). It confirms the earlier result exactly: 6 answered, the same six questions, 2 right / 194 abstained / 4 wrong.

The six answered questions, each read by hand:

| Question | Matched stored question | Given | Verdict |
|---|---|---|---|
| capital of Alberta? | What city is the capital of Alberta Canada? | Edmonton | True, answers the question. In slice. |
| When did Alaska become part of the United States as a state? | When did Alaska become a state? | January 3, 1959 | True, answers the question. In slice. |
| What is the capital of Brazil? | What is the capital of Brazil? | Brasília | True, answers the question. Not in slice. |
| Who founded Amazon.com? | Who founded Amazon? | Jeff Bezos | True, answers the question. Not in slice. |
| Who was the first woman to win a Nobel Prize? | Who was the first woman to win a Nobel Prize, and which prizes did she win? | Marie Curie, Physics 1903, Chemistry 1911 | True, answers the question. Not in slice. |
| Who wrote Don Quixote? | Tell me whether each of these refers to the book called "Don Quixote": ... | A yes/no list of title variants | Wrong. It never names an author. It answers a different question. Score 0.738 against `ts` 0.730. |

Both readings, same decisions:

| Reading | Right | Abstained | Wrong | Wrong rate (Wilson 95% upper) | Selective accuracy (Wilson lower) |
|---|---|---|---|---|---|
| Strict: slice-only test, any answer to a not-in-slice question is wrong | 2 | 194 | 4 | 2.0% (5.0%) | 2/6 = 33% (9.7%) |
| Knowledge-source: Dolly is the evidence, so a true answer to the question asked is not wrong | 5 | 194 | 1 | 0.5% (2.8%) | 5/6 = 83% (43.6%) |

- Under PLAN rule 9 the number to drive down is the wrong count. For a chat index whose knowledge is Dolly, the knowledge-source reading is the one that applies: that count is 1, and it is the Don Quixote answer.
- The strict reading stays the right one for the Wikipedia fact store (`qa_infobox`), whose knowledge is the slice. It is kept here so the two tools stay comparable.
- The 3 true answers go from wrong to right only for `chat_index`. The abstentions do not change, so coverage is the same under both readings: 6 of 200 answered (3.0%, Wilson 1.4% to 6.4%).
- The card exposes the Don Quixote failure: the matched stored question asks for a different task. For the other five, the matched question asks the same thing as the asked one. There is no automatic check for that yet (see Next steps). The thresholds were left alone, as the brief required.

---

## S2. The epistemic library

`native/include/cypha/eval/epistemic.hpp`, namespace `cypha::eval`, header-only, includes `ecs.hpp`. 195 checks in `native/tools/epistemic_test.cpp`; it returns nonzero on failure. Built and run with the brief's commands: `cmake -S native -B /root/cb_ecs`, `cmake --build /root/cb_ecs --target epistemic_test`, `ctest --test-dir /root/cb_ecs -R native_epistemic_test` (passes in 0.00 s). It is warning-free under `-Wall -Wextra -Wpedantic` with g++ 13 and with clang++ 18. A deliberately wrong expected value (185.45 for Chapman) makes it fail, so the checks do bite.

| Part | Spec | What it provides |
|---|---|---|
| States | 2.1 | `EpistemicState` (Known, Believed, Contested, Ignorant, Vacuous, Hypothesis, Refuted); `classify(opinion, optional rival, ClassifyConfig)`; `hypothesis_state(refuted)`; `state_use()` for the "may be used for" column |
| Conflict | 2.3 | `degree_of_conflict(a, b)` = `|P_A - P_B| (1 - u_A)(1 - u_B)`, on opinions or on raw P and u |
| Two kinds of uncertainty | 2.2 | `split_uncertainty()` returns a kind (`LackOfEvidence`, `MixedEvidence`, `OneSided`, `Unclear`) plus two continuous scores; `uncertainty_action()` gives the P1 or P4 action |
| Reason codes | 3.1 | `ReasonCode` with all 11 codes in priority order; `ReasonInputs` (what the caller measured); `ReasonBreakdown` (statistic, threshold, measured and applies for every code); `explain_unknown()`, `first_reason()`, `pick_reason()`; `entropy_bits()`, `ambiguity_of()`; `describe_breakdown()` |
| Unseen estimators | 3.3 | `good_turing_missing_mass`; `chao1` and `chao1_from_counts` (including n2 = 0); `chapman` (estimate, Seber variance, known, coverage); `fit_heaps` (OLS in log space, with R^2), `heaps_vocab`, `heaps_new_words` (per million tokens by default) |
| Cards | 10 | `AnswerCard`, `UnknownCard`, `HypothesisCard` and `HypotheticalCard` structs with `format_*` functions, plus `fill_card_opinion()`. The answer and "I don't know" cards reproduce the section 10 examples character for character (the bracket sits at column 33). The hypothesis card starts with `SPECULATION — not knowledge.`; the hypothetical card with `HYPOTHETICAL — assuming: ...` and ends with `Nothing here is stored.` |

What the tests pin down, besides the section 10 example cards:
- DC of two confident, disagreeing opinions: (18, 0) against (0, 8) gives 0.85 x 0.9 x 0.8 = 0.612. Symmetry, zero for agreement, zero against a vacuous opinion.
- Classify thresholds on both sides of each cut: d = 0.5 exactly is Contested; u = 0.8 exactly is Ignorant; DC 0.47 and 0.37 are Contested, DC 0.27 is not; a Known item with a confident rival is Contested.
- Reason-code priority: all 11 names and their order, then a chain in which each added trigger takes over as the first code, from `HYPOTHESIS_ONLY` up to `OUT_OF_SCOPE`.
- Good-Turing 4/27 and Chao1 = 14 on the frequency list {1,1,1,1,2,2,3,4,5,7}; Chao1 = 9 and Good-Turing 3/16 on {1,1,1,3,4,6} (n2 = 0).
- Chapman on n_A = 50, n_B = 40, m = 10: 2091/11 - 1 = 189.0909...
- Heaps: exact points from K = 10, beta = 0.6 are recovered; a hand-computed three-point fit; 1,000 new words per million tokens for K = 1, beta = 0.5.

---

## Ambiguities in the spec, and the choices made

1. **Chapman value in the brief.** The brief says n_A = 50, n_B = 40, m = 10 gives 185.45. The formula in DISCOVERY 3.3, `(n_A + 1)(n_B + 1)/(m + 1) - 1`, gives 51 x 41 / 11 - 1 = **189.09**. The number 185.45 is 51 x 40 / 11, which drops the +1 on n_B and is not the Chapman estimator. I implemented the DISCOVERY formula and the test asserts 189.0909. If 185.45 was meant, the spec formula is what needs changing, not the code.
2. **Grade Very low has no state.** With P < 0.5, u < 0.8 and d < 0.5 (for example r = 1, s = 2), the 2.1 table has no row. `classify` returns `Ignorant` for it (a gap, never an answer). It is a field, `ClassifyConfig::very_low_state`, and a test pins the default.
3. **Order of the state checks.** Vacuous, Contested, Ignorant, Known, Believed. The first three cannot overlap: u >= 0.8 forces d <= 0.2 and DC <= 0.2. Contested does overlap Known, and wins, since the table says Contested is never an answer. A hand-built opinion with `Grade::Contradicted` is also Contested.
4. **DC and rival values.** The 2.3 formula compares opinions on the same proposition. Two opinions that each say "I am sure about a different capital" both have high P, so DC is near 0 although the slot is contested. That case is carried by the d >= 0.5 route, because a competing value adds phi x mass to s in `fuse()`. `classify` takes the rival's opinion as given and does not remap values.
5. **2.2 has no numbers.** Choices: lack of evidence if u >= 0.5; mixed evidence if u <= 0.3 and b >= 0.2 and d >= 0.2 and |P - 0.5| <= 0.2; one-sided for u <= 0.3 otherwise; `Unclear` for 0.3 < u < 0.5. The u <= 0.3 cut is the Moderate grade's own. All four numbers are in `UncertaintyConfig`.
6. **Two codes could never be reported.** In 3.1, `NO_EVIDENCE` is "the slot is empty", and it sits before `BEYOND_DEPTH` and `HYPOTHESIS_ONLY`. But those two always have an empty slot in the fact store, so they would never win. I defined `NO_EVIDENCE` as no evidence of any kind: it does not apply if the only candidate is in the hypothesis store, or if a derivation exists but is deeper than the cap. `LOW_CERTAINTY` and `STALE` need a stored value, so an empty slot never triggers them.
7. **Unspecified thresholds for the reason codes.** OOD score above 0.5; best link P below 0.5 (the parse cut of 0.5, P(parse) < 0.5, is from 3.1); "close in probability" means the second-best is at least 0.5 times the best and at least 0.15 after normalising. A statistic the caller does not supply is "not measured" and never makes a code apply. All are in `ReasonThresholds`.
8. **`chat_index` with both tests failing.** Strict 3.1 order puts `AMBIGUOUS_QUESTION` before `NO_EVIDENCE`, so a low score with a small margin would read as ambiguous. I report `NO_EVIDENCE`: with nothing close, a small margin between two weak matches is not ambiguity (3.1 defines the code as "two or more fit the question"). It matters: 178 of 193 `NO_EVIDENCE` on T-QA-wiki and 1,171 of 1,350 on the Dolly set would otherwise flip. Both flags are in the machine line, so the strict order can be recomputed.
9. **Chao1 and Good-Turing edge cases.** With n2 = 0, Chao1 uses `n1 (n1 - 1) / 2` as the spec says. That is the bias-corrected form `n1 (n1 - 1) / (2 (n2 + 1))` evaluated at n2 = 0. Good-Turing with N = 0 returns 1 (everything is unseen). Chapman clamps m to min(n_A, n_B) and the coverage to 1.
10. **Thresholds file freshness.** The fingerprint is file size plus counts, not a content hash. A Dolly file edited in place to the same size and record count would not be noticed; `--recalibrate` fixes that.
11. **File location.** The thresholds file sits next to the data, `/root/cdata/dolly.jsonl.chat_index_op`, outside the repo. It is not a deliverable and it can be deleted at any time.

---

## Next steps

1. Add a same-question check to the chat index: compare the question type and the stored question's shape (for example "Who wrote X" against "Tell me whether each of these ..."). It would catch the one remaining wrong answer. It needs the dev and T-perturb gates re-run, which is why the thresholds were not touched here.
2. Treat two paraphrases whose stored responses agree as one candidate when computing the margin. In the Pearl Earring case the two responses are "Vermeer" and "...painted by Johannes Vermeer", so agreement needs a short-answer match, not string equality. It would turn that abstention into an answer.
3. Wire `epistemic.hpp` into the fact-store tool and use `ReasonCode` there, so both tools report the same codes. Add the code-audit statistic of section 12.1.
