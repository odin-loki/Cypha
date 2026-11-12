# D5: D4's decoder on today's default (T8)

**Odin Loch** · registered 2026-11-13, before any arm runs · branch `wip/mixdec-t8` (from main deda93c3, with `wip/k5-mixdec` e33e19c2 merged) · GENERATION_PLAN §13.38, §13.69, §13.71, §13.74 · [K5_MIXDEC](K5_MIXDEC.md), [K5_DECODE](K5_DECODE.md), [../DECODER.md](../DECODER.md)

(The arms, the screen, the deciding rule and the predictions below were written and committed before any arm ran. Results are added under them.)

## Why

- Today's default writer is **T8** (§13.71): K1's stack at temperature 0.8. Blind raters gave it 0.855 against K1, and 2.50 of 10 in the absolute round (§13.74).
- D4's decoder (`--mixdec`, K5_MIXDEC) was judged against K1, not T8. Against K1 the raters gave **MR 0.750 [0.650, 0.840]** and **MRJ 0.760 [0.660, 0.860]**.
- Both levers won on their own. This job asks whether they add up: D4's decoder on top of T8, judged against T8.

## The merge

`wip/k5-mixdec` merged into main. Two files conflicted: `lm/word_kn.hpp` (4 blocks) and `lm/planner/discourse.hpp` (3 blocks). In every block D2's gate terms (`wf` in the free sentences, `wg`) and D4's judge term were added at the same lines. They are independent, so each block keeps both, D2's first. Both stay off by default.

With T8, the temperature is applied to K5's distribution before the nucleus. The decoder then reranks and decodes inside that nucleus, as it does at temperature 1.

## The baseline and the checks

**T8**, today's default: `scripts/gen_eval.sh dev LABEL OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, with `cyphalm_trie` and `scribe` built by `g++ -O2` as D2 and D4 built them, `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`.

**Checks** (the merged build, 250 GEN-A dev prompts, `cmp` of the GEN lines, MODEL line aside):
- the default equals `bench/k5_decode/runs/T8.gen.txt`;
- `--temps 1.0` equals `bench/k5_retrieve/runs/K1.gen.txt`;
- `--temps 1.0 --mixdec ratio=0.3,copy=0` equals D4's MR file (`bench/k5_mixdec/runs/MR.gen.txt`);
- `--temps 1.0 --mixdec ratio=0.3,judge=1,copy=0` equals D4's MRJ file;
- the selftest passes.

**Done before this was written** (merge b463645c, files in `bench/mixdec_t8/checks/`): all four are **equal**. The selftest passes.

| Check | SCRIBE v2 | copy32 | verbatim13 | wall | peak RAM | `cmp` |
|---|---|---|---|---|---|---|
| default (T8) | 0.811 | 0.487 | 0 | 8:43 | 11.3 GB | equal |
| `--temps 1.0` (K1) | 0.704 | 0.275 | 0 | 7:19 | 11.1 GB | equal |
| `--temps 1.0` + MR | 0.810 | 0.277 | 0 | 23:07 | 12.6 GB | equal |
| `--temps 1.0` + MRJ | 0.802 | 0.268 | 0 | 37:01 | 12.6 GB | equal |

The PC was shared with other jobs (load average 55–190), so wall times are loose. The GEN files equal those already on the branch, so only the summaries and `cmp` log are kept.

## The arms

Dev, seed 0, 250 prompts, the baseline's command plus:

| Arm | Flags | What it tests |
|---|---|---|
| T8MR | `--mixdec ratio=0.3,copy=0` | T8 with MR's decoder: the context ratio inside the nucleus (top 16), arithmetic decoding |
| T8MRJ | `--mixdec ratio=0.3,judge=1,copy=0` | T8 with MRJ's decoder: the ratio and the judge in the gate |

One arm at a time, at most 6 threads, no start while `free -g` shows under 16 GB available.

## The screen

An arm passes if all hold:
- **verbatim13 is 0.** This is Odin's rule, the 12-word cap.
- **Fact precision at least 0.874** (`python3 -I scripts/gen_fact_check.py run --v2`).
- **My strict reading.** I read the 12 unrated dev prompts D2 read (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13), first 460 characters, each arm beside T8. The cut last sentence is not counted. For each sentence I mark whether it is whole (a subject and a finite verb, not broken off), a verbless fragment ("Monetary stability."), or neither. Keyword repetition and verbless fragments were D4's weak points, so they are checked first. The arm passes only if:
  - its share of whole sentences is at least T8's;
  - its share of verbless fragments is at most T8's + 0.05;
  - no keyword patchwork: the subject's word is not repeated in three sentences running in more than 2 of the 12 prompts;
  - and it is no worse than T8 in any other way I can see (broken clauses, lists of names, phrases pasted from the article, leaving the subject).

**Reported, not screened:** SCRIBE v2 (with its interval) and v3.1; copy32; mean surprise, the link to the sentence before and stuff3 (`bench/para_plan/hook_screen.py`, 460 characters, each arm minus T8 with a 95% paired bootstrap); sentences in 460 characters; samples unlike T8; wall time and peak RAM.

At most 2 candidates. If neither arm passes, nothing goes to the raters and I say so.

## The deciding judge

- `python3 scripts/blind_rate.py make --out bench/blind/mixt8_pair --ids bench/blind/llm_rating/prompts.json --arm T8=bench/k5_decode/runs/T8.gen.txt --arm X=... [--arm Y=...] --mode pair --baseline T8 --seed 20261205`: the 50 prompts, 460 characters, four fresh blind raters.
- Packets, `key.b64`, `RATER_PROMPT.md` and the GEN files are committed and pushed before the raters run. `KEY.json` stays off the branch. The lead runs the raters.
- **§13.38's rule:** an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- The best arm that reads better (higher lower bound first, then lower copy32) is a winner. For a winner the lead also runs the 1–10 round of §13.69 and §13.74, beside T8 and human text, before any default changes. Any default is Odin's call.

## Predictions

| # | Prediction |
|---|---|
| P1 | All four checks equal (`cmp`); the selftest passes |
| P2 | verbatim13 0 in both arms |
| P3 | Fact precision within ±0.03 of T8's 0.924; both pass 0.874 |
| P4 | copy32 within ±0.04 of T8's 0.487 (MR left K1's copy32 unchanged) |
| P5 | SCRIBE v2 above T8's 0.811, by 0.01–0.06 (less than MR's +0.106 over K1: T8 already stays on topic words) |
| P6 | More sentences in 460 characters than T8, and stuff3 up by 0.005–0.020 against T8 |
| P7 | Mean surprise falls by 0.1–0.3 bits a word against T8 |
| P8 | By my reading, at least one arm fails on verbless fragments or keyword repetition |
| P9 | Raters: the best arm that reaches them 0.50–0.65 against T8; at most one adopted by the rule |
| P10 | Wall time 15–30 minutes an arm; peak RAM about 12.6 GB |

## Results

(Added after the runs, under this line.)
