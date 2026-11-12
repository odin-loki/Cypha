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