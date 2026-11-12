# D2: decoding that favours grammar

**Odin Loch** · registered 2026-11-12, before any arm runs · branch `wip/k5-decode` (from main 4629f68c) · GENERATION_PLAN §13.51, §13.57, §13.58, §13.66, §13.69

(The arms, the screen, the deciding rule and the predictions below were written and committed before any arm ran. Results are added under them.)

## Why

- Today's writer is K1 (§13.66). In the absolute round (§13.69) blind readers gave it 2.28 out of 10; human text got 8.54.
- The readers mark down broken clauses first. K1's free sentences still break after a few words (K5_RETRIEVE, "The grammar is no better").
- Two decoding levers were never put before readers:
  - **Lower temperature.** T2 (§13.51) dropped T8 and T9 on our own copy32 screen (0.385 and 0.312, the bound 0.30). Neither broke Odin's actual rule, the 12-word copy cap (verbatim13 was 0 for both).
  - **A fluency gate.** Today's sentence gate picks among 16 drafts by prompt association and the LDA topic term. It gives the draft's own fluency no weight (`wf=0`). The discourse writer's free sentences have no fluency term at all.

## The baseline

**K1**, today's default: `scripts/gen_eval.sh dev K1 OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, with `cyphalm_trie` and `scribe` built by `g++ -O2` as `scripts/build_1tb_tools.sh` builds them (`libsais.o` linked into `scribe`), `CYPHALM_TRIE`, `SCRIBE`, `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`.

**Checks first** (no arm runs until both hold):
- main's build (4629f68c): the GEN file equals `bench/k5_retrieve/runs/K1.gen.txt` under `cmp`, MODEL line aside;
- my build with the new field off: the same.

## The new code (small, behind a flag, off by default)

- `--k5-gate K,wg=W` (`lm/word_kn.hpp`): a **grammar term**. For each draft, the mean over its words of log2 P(class of the word | classes of the two words before), under K5's own class model (its 256 exchange classes, Kneser–Ney of order 3 over class ids). It ignores which words they are, so it scores the shape of the sentence, not its topic. It is z-scored over the drafts like the other terms.
- The discourse writer's free sentences (`planner/discourse.hpp`) now read `wf` and `wg` too, z-scored. Before, they ignored `wf`. No default sets `wf`, so this changes no default.
- With `wg=0` and `wf=0` (the defaults), nothing new is computed and the output is unchanged.
- Lines changed: `word_kn.hpp` +24 −1 (the `class_lp` method in `ClassModel`, the `gate_wg` option, its parse, its MODEL-line entry, the z-gate's term, one selftest line); `discourse.hpp` +20.
- **Not built:** typical (locally typical) sampling and a temperature schedule. Neither has a flag. Both would change `fast_pick`, the exact nucleus draw that three other jobs' writer code shares. The E1 tagger ("has a main verb") is not cheap here: its model file is not on the PC and needs a download.

## The arms

Dev, seed 0, 250 prompts, the baseline's command plus:

| Arm | Flags | Lever |
|---|---|---|
| T9 | `--temps 0.9` | temperature 0.9 |
| T8 | `--temps 0.8` | temperature 0.8 |
| F1 | `--k5-gate 16,wf=1` | fluency weighed as much as the prompt term |
| F2 | `--k5-gate 16,wf=2` | fluency weighed twice |
| GR | `--k5-gate 16,wg=1` | the class-sequence grammar term |
| TF | the best passing temperature arm plus the best passing gate arm | both levers |

- `--k5-gate 16,...` keeps `wp=1`, `wm=5` and the LDA term (`lda_w` 1), as today.
- `--temps 0.8` keeps the seed of position 0 (TUNE_WRITE).
- **TF runs only if** at least one temperature arm and at least one gate arm (F1, F2, GR) pass the screen. "Best" is by the candidate order below. Otherwise TF does not run. At most 6 arms in all.
- **Machine limits:** one arm at a time, at most 6 threads, no start while `free -g` shows under 16 GB available.

## Measures (dev)

- verbatim13 (Odin's 12-word cap) and copy32;
- fact precision: `python3 -I scripts/gen_fact_check.py run --v2` (GEN_FACTS amendment A1), with its 95% interval;
- SCRIBE v2 with its 95% interval, v3.1 beside it;
- mean surprise per word (WRITING_SIGNAL's word trigram model; `bench/para_plan/hook_screen.py`, 460 characters, each arm minus K1 with a 95% paired bootstrap), and the link to the sentence before;
- samples unlike K1 (all 250, and the 50 rated); wall time, peak RAM.

## The screen

**An arm passes if all hold:**
- **verbatim13 is 0.** This is Odin's rule, the 12-word cap.
- **Fact precision at least 0.874.**
- **My strict reading of grammar.** I read the 12 unrated dev prompts read before (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13), first 460 characters, each arm beside K1. I count the sentences that are whole: one English clause or more, with a subject and a finite verb, that does not break off or run words together. The arm passes if its count of whole sentences is above K1's and it is no worse in any other way I can see (repeated words or phrases, lists of names, boilerplate pasted from the training text, staying on the subject).

**Reported, not screened, and why:**
- **copy32.** Our own bound (at most 0.30) is not Odin's rule; his rule is the 12-word cap. A colder sampler copies more short spans by design (K5_UPGRADES: "copying is the price of a sharper sampler"). This job asks whether that price buys grammar readers notice.
- **SCRIBE v2.** It rewards topic words and has disagreed with the raters on settings like these (§13.34, §13.57: it tracks the raters 0.547 overall; T8 raised it while v3.1 fell).
- **Mean surprise.** WRITING_SIGNAL found lower surprise tracks the raters between settings of one writer (0.738), so it is the tie-break below. It is not a gate: a colder sampler lowers it whatever it does to grammar.

**Candidates:** at most 2 that pass, one per lever (temperature, gate) where both have one; TF counts as both levers and can replace either. Chosen by my count of whole sentences first, then lower mean surprise. If none pass, nothing goes to the raters and I say so.

## The deciding judge

- `python3 scripts/blind_rate.py make --out bench/blind/dec_pair --ids bench/blind/llm_rating/prompts.json --arm K1=bench/k5_retrieve/runs/K1.gen.txt --arm X=... [--arm Y=...] --mode pair --baseline K1 --seed 20261201`: the 50 prompts, 460 characters, four fresh blind Claude raters.
- The packets, `key.b64`, `RATER_PROMPT.md` and the GEN files are committed and pushed before the raters run. `KEY.json` stays out of the repository. The lead runs the raters.
- **The §13.38 rule:** an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- The best arm that reads better (higher lower bound first, then lower copy32) is proposed as the default; Odin can veto. Then GEN-A test is read once, for it only.

## Predictions

| # | Prediction |
|---|---|
| D1 | main's build and my build with the field off equal K1 (`cmp`) |
| D2 | verbatim13 0 in every arm (the cap holds whatever the sampler) |
| D3 | copy32 (K1 0.275): T9 0.32–0.36, T8 0.38–0.45; F1 0.29–0.33, F2 0.30–0.36; GR 0.27–0.30 |
| D4 | Mean surprise falls: T8 by 0.4–0.8 bits a word, T9 by 0.2–0.5, F1 by 0.2–0.4, F2 by 0.3–0.6, GR by 0.0–0.2 |
| D5 | Fact precision within ±0.03 of K1 (0.918) in every arm; all pass 0.874 |
| D6 | SCRIBE v2 within ±0.04 of K1 for the gate arms; T8 up (as in T2) |
| D7 | By reading, T8 and F2 have more whole sentences than K1 and pass; GR changes too little to see and fails |
| D8 | Raters: the best candidate against K1 0.55–0.65; at most one arm is adopted by the rule |
| D9 | Wall time 5–9 minutes per arm; peak RAM 11.3–11.6 GB |

## Results

(Added after the runs, under this line.)

### Checks (2026-11-12)

- main's build (4629f68c) and my build with `wg` and `wf` off: both equal `bench/k5_retrieve/runs/K1.gen.txt` under `cmp` (MODEL line aside). **Hit.** 10:03 and 9:53 wall, 11.3 GB. The selftest passes.

### The screen (dev, seed 0, 250 prompts; files in `bench/k5_decode/runs/`, measures in `bench/k5_decode/screen/`)

One arm at a time, 6 threads, on a shared PC.

| Arm | verbatim13 | copy32 | fact precision [95%] | mean surprise, minus K1 [95%] | SCRIBE v2 [95%] | v3.1 | whole sentences (12 prompts) | samples unlike K1 (of 50 rated) | wall | screen |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 0 | 0.275 | 0.918 [0.867, 0.962] | 7.806 bits (level) | 0.704 [0.673, 0.735] | 0.665 | 24 of 60 (0.40) | – | 9:53 | baseline |
| T9 | 0 | 0.394 | 0.918 [0.868, 0.961] | −0.190 [−0.281, −0.105] | 0.759 [0.730, 0.788] | 0.672 | 20 of 62 (0.32) | 214 (44) | 12:48 | fail (reading) |
| **T8** | 0 | 0.487 | 0.924 [0.876, 0.966] | −0.227 [−0.320, −0.135] | 0.811 [0.785, 0.838] | 0.691 | **31 of 68 (0.46)** | 214 (44) | 12:42 | **pass, candidate** |
| **F1** | 0 | 0.339 | 0.924 [0.874, 0.966] | −0.249 [−0.321, −0.179] | 0.719 [0.690, 0.747] | 0.650 | **29 of 65 (0.45)** | 198 (41) | 8:46 | **pass, candidate** |
| F2 | 0 | 0.377 | 0.929 [0.881, 0.970] | −0.393 [−0.478, −0.313] | 0.707 [0.675, 0.739] | 0.631 | 29 of 67 (0.43) | 209 (43) | 9:31 | fail (leaves the subject) |
| GR | 0 | 0.275 | 0.906 [0.853, 0.953] | −0.467 [−0.554, −0.382] | 0.705 [0.673, 0.735] | 0.622 | 32 of 74 (0.43) | 203 (41) | 10:47 | fail (leaves the subject) |
| TF (T8 + F1) | **1 of 250** | 0.530 | 0.935 [0.888, 0.974] | −0.244 [−0.334, −0.147] | 0.829 [0.804, 0.856] | 0.687 | not counted | 215 (45) | 14:42 | **fail (verbatim13)** |

Peak RAM 11.3 GB in every arm. The 36 samples that no arm changes are planned prompts whose text is all fact sentences.

**Staying on the subject** (`coh_measures.py`, 460 characters; share of sentences with a prompt word, and the link to the sentence before):

| Arm | s1 | s2 | s3 | s4+ | link | stuff3 |
|---|---:|---:|---:|---:|---:|---:|
| K1 | 0.472 | 0.422 | 0.348 | 0.202 | 0.179 | 0.021 |
| T9 | 0.556 | 0.436 | 0.332 | 0.210 | 0.219 | 0.025 |
| T8 | 0.516 | 0.462 | 0.391 | 0.239 | 0.247 | 0.030 |
| F1 | 0.432 | 0.408 | 0.324 | 0.199 | 0.177 | 0.020 |
| F2 | 0.412 | 0.420 | 0.294 | 0.167 | 0.162 | 0.020 |
| GR | 0.416 | 0.386 | 0.331 | 0.184 | 0.156 | 0.020 |
| TF | 0.544 | 0.420 | 0.434 | 0.228 | 0.238 | 0.028 |

J2's link (`hook_screen.py`, minus K1): T8 +0.078 [+0.048, +0.108], T9 +0.068, F1 +0.006 [−0.014, +0.026], F2 −0.005, GR −0.018 [−0.037, +0.001].

**The reading** (dev 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13; first 460 characters; the cut last sentence not counted). A sentence counts as whole if it is a grammatical English clause with a subject and a finite verb, whatever it means. "Brian Henderson was one of the most popular and successful New Zealand and attack ships in the world" counts; "The oxides of nitrogen and hydrogen." and "They are usually use more restrictive rules" do not. Dev 0, 5 and 7 open with the same fact sentences in every arm.
- **Every arm is still far from readable text.** The best has fewer than half its sentences whole. The meaning breaks even where the grammar holds ("Auckland is a town located throughout the animal kingdom of New Zealand").
- **T8** has the most whole sentences and stays nearer the subject. Dev 1: "The acid mixture was also changed to two parts sulfuric acid. … Color negative film was never manufactured with a deep lustre." Dev 10: "As the astronauts begin to die, the warning changes to the original plans". Dev 3: "For example, peroxide is an aqueous solution of hydrogen and helium." It also writes more noun phrases that end in a full stop ("The oxides of nitrogen and hydrogen.", "Children's book author."). It copies more short spans (copy32 0.487), within the cap. No boilerplate, lists or repeated names beyond K1's.
- **T9** reads worse than K1: long run-ons ("As an example, say the Chicago School is a highly stylized way: unless the projection of the Earth's magnetic field …"). It fails.
- **F1** trades long broken clauses for short fragments. It has fewer broken long sentences (14 against K1's 22) but more fragments (22 against 14). Dev 4 opens "In the United States, France and Spain. In the 18 th century. Some religious groups (see below). The main population." Dev 9 is better than K1: "This article focuses on the availability of oracles … These concepts are also privacy concerns." Its link to the sentence before is K1's. By the registered rule it passes: more whole sentences, no repetition, and no measurable loss of the subject. It passes narrowly.
- **F2** has F1's fragments and leaves the subject more. Dev 12 (Alia, Dune) turns to "The fall of Kabul marked with the murder of her brother". It fails.
- **GR** has the most whole sentences (32), but it writes longer texts of more, shorter sentences, and it leaves the subject. Dev 12: "He was the son of Poseidon. In his inaugural address"; dev 2 turns to Sussex. Its link falls (−0.018) and its prompt words fall at s1–s2. It fails.
- **TF** (run because T8 and F1 passed, as registered) reads about as T8 by a glance (dev 13: "Messianic Jews consider the movement to be "too Jewish" for holding on entirely different grounds."). But one of its 250 samples has a 13-word copy. It fails Odin's cap, so it was not read further.

**Candidates:** **T8** (temperature) and **F1** (gate), one per lever.

### Packets (built before any rating)

`bench/blind/dec_pair/`: K1 against T8 and F1, the 50 prompts of `bench/blind/llm_rating/prompts.json`, 460 characters, seed 20261201. T8 differs from K1 on 44 of the 50, F1 on 41. `KEY.json` is kept outside the repository; `key.b64` and `RATER_PROMPT.md` are committed. The lead runs the raters.

### Predictions so far

- D1 the field off equals K1: **hit**.
- D2 verbatim13 0 in every arm: **hit** for the five single arms; **miss** for TF (1 of 250). The cap held for T8 alone and F1 alone, but not for both together.
- D3 copy32: GR **hit** (0.275); T9 **miss** (0.394), T8 **miss** (0.487), F1 **miss** (0.339), F2 **miss** (0.377). Each copies more than predicted, by about 0.03–0.04 for the temperature arms. Retrieval adds to it (K1 started at 0.275, not T2's 0.239).
- D4 mean surprise: F1 **hit** (−0.25), F2 **hit** (−0.39); T8 **miss** (−0.23, predicted 0.4–0.8), T9 **miss** (−0.19, just under 0.2), GR **miss** (−0.47, predicted at most 0.2).
- D5 fact precision within ±0.03 of K1 and all at least 0.874: **hit** (0.906–0.935).
- D6 SCRIBE v2 within ±0.04 for the gate arms, T8 up: **hit** (+0.001 to +0.015; T8 +0.107).
- D7 T8 and F2 pass, GR fails: T8 **hit**; F2 **miss** (fails); GR fails **hit**, but not for the reason predicted: it changes 203 samples and leaves the subject.
- D8 raters: wait for the raters.
- D9 wall 5–9 minutes: **miss** (8:46–14:42, one arm at a time); peak RAM 11.3–11.6 GB: **hit**.
