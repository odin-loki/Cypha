# F1: fixing where the writing breaks

**Odin Loch** · registered 2026-11-13, before any arm runs · branch `wip/k5-fixes` (from main ea3a784c) · GENERATION_PLAN §13.76, §13.75 · [WHERE_IT_BREAKS](WHERE_IT_BREAKS.md)

(The arms, the screen, the deciding rule and the predictions below were written and committed before any arm ran. Results are added under them.)

## Why

- X1 (§13.75) marked 90 broken spots in T8's text, today's default (temperature 0.8, the K1 retrieval stack). It ranked the causes:
  1. **K5's 4-word window.** It rates splices as likely: a seen 4-word context and a word never seen after it. 45 of the 63 breaks in K5's own sentences are words K5 ranks in its top 10. Its own beam output is stitched training text.
  2. **The planner's fact sentences:** 27 of 90 breaks from 21% of the words.
  3. **A wide nucleus where K5 is unsure:** 18 tail draws. A min-p cut at 0.1 would remove 14 of the 63.
  4. **The gate picks topic over form:** the kept draft is the most fluent of the 16 only 19% of the time (−0.85 bits a word).
- This job tries one fix per cause and puts the ones that pass a screen before blind raters.

## The baseline

**T8**, today's default: `scripts/gen_eval.sh dev T8 OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, no other flag. `cyphalm_trie` and `scribe` are built with `g++ -O2` as `scripts/build_1tb_tools.sh` builds them; `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`.

**Check first** (no arm runs until it holds): my build with every new flag off gives a GEN file equal to `bench/k5_decode/runs/T8.gen.txt` under `cmp`, MODEL line aside.

## The new code (behind flags, off by default)

All in a new header, `lm/k5_fixes.hpp`. With every flag off, nothing new runs and the output is unchanged.

- **`--k5-minp P`** (min-p). In each draw, after the repeat ban and the copy cap, a word whose probability under K5 is below P times the most probable word's is dropped before the nucleus. The probabilities are K5's before temperature, as in X1's map (at temperature T this is P^(1/T) after temperature: the same words). The most probable word left is never dropped.
- **`--k5-ff X`** (fluency floor). At each sentence, a draft whose mean log2 P a word is more than X bits below the most fluent draft's is dropped. The gate then chooses among the rest as it does today (prompt association, topic, markup). It is a floor, not a weight.
- **`--k5-cj J`** (clause judge). For each word of a draft, L is the length in words of the longest span ending at that word (the prompt, the text so far and the draft) that occurs in K5's own 337 MB id stream. It is read from the suffix array K5 already holds (`count_index.hpp`), capped at 16 words, never across <unk>.
  - A **join** is a word where the match restarts: L is no longer than at the word before, so the span before does not go on.
  - A **weak join** is a join carried by at most J words: the new span shares at most J words with what came before. With J = 2, the 4-word span across the join was never seen in 337 MB.
  - A draft's score is its weak joins per word. The gate keeps the drafts at or below the lower median score (at least half of them, ties kept), then chooses among them as today.
  - With FF on too, the fluency floor goes first, and the median is taken over what is left.
  - The 12-word copy cap stays in force. No draft can complete 13 training words, so the judge cannot reward longer copies. verbatim13 must stay 0; copy32 is reported.
- FF and CJ act in today's gate (the z-scored gate) and in the discourse writer's free sentences. They do not act in the plain gate (used only with `--k5-lda 0`) or the single-step planner's free sentences (not the default). The planner's fact sentences are filled by the planner's own search; no flag here touches them.
- **Lines changed outside the new header** (other jobs edit `word_kn.hpp` too; these are all):
  - `word_kn.hpp` (+20 −2): the include; the `fix` field in `Options`; its MODEL-line entry; its parse in `apply_flag`; in `gate_z` two vectors, the raw fluency and join score per draft, and the skip in the choice loop; the min-p cut in `full_pick` (7 lines, before the nucleus); `fast_applies` declines with min-p on (today's T 0.8 never uses `fast_pick`); one selftest line.
  - `planner/discourse.hpp` (+10 −2): the same in `free_sentence`. The choice loop starts from `best = -1` instead of `k == 0`; with no draft skipped this is the same choice.

## The arms

Dev, seed 0, 250 prompts, the baseline's command plus:

| Arm | Flags | Cause (X1) |
|---|---|---|
| MP | `--k5-minp 0.1` | 3, the wide nucleus |
| FF | `--k5-ff 1` | 4, the gate ignores form |
| CJ | `--k5-cj 2` | 1, splices |
| NP | `--no-k5-plan` | 2, the planner's fact sentences |
| C | the passing single arms together | runs only if 2 or more pass |

**Why these values:**
- **MP, P = 0.1:** X1's own figure. On X1's map, 0.1 of the top word's probability removes 14 of the 63 breaks in K5's sentences and 9.5% of K5's sampled words.
- **FF, X = 1 bit:** from X1's gate redraw (207 sentences). The most fluent draft is a median 1.05 bits a word above the mean draft, so a 1-bit floor keeps about the more fluent half. The kept draft was more than 1 bit below the most fluent in 38% of sentences, so FF changes about that share of choices.
- **CJ, J = 2:** at X1's broken spots, 44% of the words were held only by order 3 or lower (all T8 words: 24%). J = 2 counts exactly those joins, but over the 337 MB and across any length of span before.
- No value was tried on dev before this registration. A smoke run on dev prompts 248 and 249 (outside X1's 30 and the rated 50) checked that the flags run; it is not reported.

**Machine limits:** one arm at a time, at most 6 threads, no start while `free -g` shows under 16 GB available.

## Measures (dev)

- **verbatim13** (Odin's 12-word cap) and **copy32**;
- **fact precision:** `python3 -I scripts/gen_fact_check.py run --v2`, with its 95% interval and statements per sample;
- SCRIBE v2 with its 95% interval, v3.1 beside it;
- samples unlike T8 (all 250, and the 50 rated); wall time, peak RAM;
- **breaks per 100 words** (below).

**Breaks, marked as X1 marked them.** The 30 prompts of WHERE_IT_BREAKS (`bench/where_it_breaks/items.tsv`: 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 35, 36), each text's first 600 bytes.
- T8 is marked again in the same pass as the arms, so all texts are read the same way.
- The reader is one fresh Claude reader, blind. It gets one file: every arm's text and T8's for each prompt, in a random order (seed 20261213), with no labels. The key stays with me.
- The rule is X1's: in each sentence that stops working, mark the first word where it does, **grammar** (the clause is abandoned or cannot be parsed) or **sense** (grammatical, but the word makes the sentence false or absurd). One mark per broken sentence. X1 stopped at three a text; here every broken sentence is marked, and the count capped at three a text is reported beside it.
- The reader also counts each text's sentences, so the share of whole sentences (sentences with no mark) is reported too.
- Reported: breaks per 100 words (words as X1 counts them), with a 95% bootstrap interval over prompts paired against T8; whole-sentence share; breaks by kind. The marks are committed as `bench/k5_fixes/breaks/marks.tsv`, with the key.

**My strict reading** (§13.71's): the 12 prompts read before (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13), first 460 characters, each arm beside T8. I count whole sentences (one English clause or more, with a subject and a finite verb, that does not break off or run words together) and note any new failure (repeated phrases, lists of names, boilerplate, leaving the subject).

## The screen

**An arm passes if all hold:**
- verbatim13 is 0;
- fact precision at least 0.874 (NP: reported, not screened; without the planner the text makes few checkable statements, and the honest fill's true dates go with it);
- **breaks per 100 words below T8's** in the blind marking;
- my strict reading: whole sentences no fewer than T8's, and no new failure.

**Reported, not screened:** copy32 (Odin's rule is the 12-word cap), SCRIBE v2 (§13.57: it rewards topic words).

**Candidates:** at most 2 single arms that pass, by fewest breaks per 100 words. If 2 or more single arms pass, the combination C (all the passing single arms' flags) runs and is screened the same way; if it passes, it is a third candidate. If none pass, nothing goes to the raters and I say so.

## The deciding judge

- `python3 scripts/blind_rate.py make --out bench/blind/fix_pair --ids bench/blind/llm_rating/prompts.json --arm T8=bench/k5_decode/runs/T8.gen.txt --arm X=... --mode pair --baseline T8 --seed 20261220`: the 50 rated prompts, 460 characters, four fresh blind Claude raters.
- The packets, `key.b64`, `RATER_PROMPT.md` and the GEN files are committed and pushed before the raters run. `KEY.json` stays out of the repository. The lead runs the raters.
- **The §13.38 rule:** an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- The best arm that reads better (higher lower bound first, then lower copy32) is proposed as the default; Odin can veto. Then GEN-A test is read once, for it only, with the fact check.

## Predictions

| # | Prediction |
|---|---|
| P1 | With every flag off, my build equals T8's GEN file (`cmp`, MODEL line aside). |
| P2 | verbatim13 is 0 in every arm. |
| P3 | MP: breaks per 100 words 10–25% below T8's; copy32 within ±0.05 of T8's 0.487; fact precision within ±0.02 of 0.924. |
| P4 | FF: breaks per 100 words 5–20% below T8's; copy32 up by 0.02–0.08 (fluent drafts copy more). |
| P5 | CJ: breaks per 100 words 5–20% below T8's; copy32 up by 0.02–0.10. |
| P6 | NP: breaks per 100 words 15–35% below T8's (no fact sentences); statements per sample below 0.3. |
| P7 | MP and CJ pass the screen; FF does not (its gain is in what the raters already reward least). |
| P8 | Raters: the best arm wins 0.55–0.65 against T8. At most one arm reads better by the rule. |
| P9 | NP against T8: 0.45–0.60 (readers lose the facts but also the fact sentences' breaks). |
## Screen results (2026-11-13)

**Check.** My build with every new flag off gives `bench/k5_decode/runs/T8.gen.txt` under `cmp`, the MODEL line too. Selftest passes. **P1 hit.**

**The run.** One arm at a time, 6 threads, each started with at least 16 GB available. WSL restarted at about 02:53 (load near 170 from all jobs on the PC) and killed CJ while it was building K5. CJ and NP were rerun from the start under `setsid`; T8, MP and FF had finished before. Load was 140–190 throughout, so wall times are not comparable between arms.

**Dev, 250 prompts** (GEN files and summaries in `bench/k5_fixes/runs/`, the fact check in `bench/k5_fixes/screen/facts/`):

| Arm | SCRIBE v2 [95%] | v3.1 | copy32 | verbatim13 | fact precision [95%] | statements a sample | samples unlike T8 (all / rated 50) | wall | peak RAM |
|---|---|---|---|---|---|---|---|---|---|
| T8 | 0.811 [0.785, 0.838] | 0.691 | 0.487 | 0 | 0.924 [0.876, 0.966] | 1.18 | — | 21:55 | 11.3 GB |
| MP | 0.846 [0.818, 0.875] | 0.698 | 0.562 | 0 | 0.935 [0.888, 0.974] | 1.19 | 213 / 44 | 15:38 | 11.3 GB |
| FF | 0.805 [0.776, 0.832] | 0.678 | 0.540 | 0 | 0.924 [0.874, 0.964] | 1.18 | 193 / 40 | 17:46 | 11.3 GB |
| CJ | 0.800 [0.772, 0.830] | 0.613 | 0.520 | 0 | 0.929 [0.883, 0.970] | 1.21 | 211 / 44 | 11:34 | 11.3 GB |
| NP | 0.785 [0.755, 0.811] | 0.617 | 0.570 | 0 | none graded right (0 S, 1 C) | 0.19 | 77 / 11 | 15:27 | 3.9 GB |
| C (MP + FF + NP) | 0.810 [0.782, 0.836] | — | 0.694 | 0 | none graded right (0 S, 1 C) | 0.13 | 250 / 50 | 9:16 | 3.9 GB |

**Breaks, marked blind** (`bench/k5_fixes/breaks/`: the reader's file, its marks, the key and `summary.md`). Two fresh readers each read half of the 150 texts (T8 and the four single arms on X1's 30 prompts, shuffled, no labels). C ran later, so it was marked in a second pass beside T8 again (`bench/k5_fixes/breaks_c/`, seed 20261214, one fresh reader). T8 read 5.43 and 5.45 breaks per 100 words in the two passes, so the marking is stable.

| Arm | breaks per 100 words | arm − T8, paired [95%] | whole sentences | grammar / sense |
|---|---|---|---|---|
| T8 | 5.43 | — | 0.190 | 118 / 87 |
| MP | 4.73 | −0.71 [−1.20, −0.23] | 0.231 | 96 / 84 |
| FF | 4.68 | −0.71 [−1.28, −0.15] | 0.254 | 89 / 87 |
| CJ | 4.73 | −0.70 [−1.14, −0.26] | 0.220 | 96 / 85 |
| NP | 5.03 | −0.42 [−0.81, +0.01] | 0.175 | 127 / 61 |
| T8 (second pass) | 5.45 | — | 0.186 | 104 / 102 |
| C | 4.49 | −0.98 [−1.49, −0.48] | 0.231 | 103 / 67 |

- The readers marked far more than X1's three a text (6.2 a text on average), so the count capped at three is 90 for every arm and says nothing.
- All 930 + 376 marks were found in the texts as quoted.

**My strict reading** (12 prompts, 460 characters; whole sentences): T8 9, **MP 16**, FF 10, CJ 7, NP 9, C 12.
- MP's free sentences hold together more often ("The bill is black, the legs and feet are black."; "The term "Messianic" is not new."). The planner's fact sentences are the same as T8's.
- FF is close to T8; it sometimes keeps a cleaner opening ("Ground and pound is also used as a tactic to win their fights.").
- CJ reads worse: its drafts change subject more often ("Most Jewish groups consider the movement to be an anti-war demonstration"). It fails the reading.
- NP keeps T8's free text on 173 of 250 prompts (no plan there). Where it changes, it trades false fact sentences for free K5 text that breaks as often.

**By the screen:**
- MP passes; FF passes; NP passes the registered conditions (its breaks are below T8's by the point estimate, its reading equals T8's; fact precision is reported only).
- CJ fails (my reading: 7 against 9).
- C ran, since 2 or more single arms passed. It holds the flags of all three passing arms. It fails the fact screen as registered: the 0.874 bound applies to every arm but NP, and C writes no fact sentences (0 supported, 1 contradicted). It would otherwise pass (−0.98 breaks per 100 words, reading 12). Its copy32 is 0.694.
- **Candidates:** FF (4.68) and MP (4.73), the two single arms with the fewest breaks. NP is third and not sent.

**Predictions, scored so far:**
- P1 byte-identical when off: **hit**.
- P2 verbatim13 0 everywhere: **hit**.
- P3 MP: breaks −13% (**hit**, 10–25%); copy32 +0.075 (**miss**, ±0.05); fact precision +0.011 (**hit**).
- P4 FF: breaks −14% (**hit**); copy32 +0.053 (**hit**).
- P5 CJ: breaks −13% (**hit**); copy32 +0.033 (**hit**).
- P6 NP: breaks −7% (**miss**, 15–35%); statements a sample 0.19 (**hit**).
- P7 MP and CJ pass, FF does not: **miss** (MP and FF pass, CJ fails).
- P8, P9: wait for the raters (NP is not rated, so P9 cannot be scored).

**Packets** (committed before the raters run): `bench/blind/fix_pair/`, seed 20261220, arms MP and FF against T8 (`bench/k5_decode/runs/T8.gen.txt`) on the 50 rated prompts at 460 characters; R1–R4, `key.b64`, `RATER_PROMPT.md`. `KEY.json` is kept outside the repository. MP differs from T8 on 44 of the 50 rated prompts, FF on 40.
