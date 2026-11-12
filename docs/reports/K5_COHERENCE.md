# T3: keeping CyphaLM's paragraphs on topic

**Odin Loch** · registered 2026-11-12, before any arm runs · branch `wip/k5-coherence` (from main 5abf1afc) · GENERATION_PLAN §13.31–13.36, §13.48, §13.51

## Why now

- Every way of choosing K5's sentences read at about 2 of 10 when K5 counted from 7 MB (§13.31–13.36): the gate, MBR, the entity grid, LDA topic steering, paragraph MCMC and the structure meta-mixer. §13.36 found the reason: **K5's drafts held no good sentence to choose.**
- Since §13.48 K5 counts from 337 MB, and its sentences are fluent. The raters preferred that 0.94 to the old default, and a 16-draft gate a further 0.67 (§13.51, G16, today's default).
- What is left is **drift**: every arm leaves the subject after the first one or two sentences (TUNE_WRITE.md, reading).
- **Hypothesis:** the old choosers failed because there was nothing good to choose. Now there is. So a chooser that rewards staying on the prompt's subject and the previous sentence's subject may help now.

## What drifts, measured on today's files

`bench/k5_coherence/coh_measures.py` (new; it decides nothing). Each continuation is cut near 460 characters, as the raters see it. A sentence is "on the prompt" if it holds a content word of the prompt (4+ letters, not a function word):

| | first sentence on the prompt | 2nd | 3rd | 4th and later | shares a word with the sentence before | 3rd+ use of a word (stuffing) |
|---|---:|---:|---:|---:|---:|---:|
| G16 (default) | 0.376 | 0.316 | 0.253 | 0.086 | 0.150 | 0.015 |
| human | 0.632 | 0.488 | 0.510 | 0.310 | 0.460 | 0.032 |

The gap grows with each sentence, and K5's sentences rarely share a word with the one before (0.15 against 0.46).

## One small new term: the previous sentence

Today's gate scores each of its 16 drafts by its mean PPMI association with the **prompt** (minus markup). It never looks at the sentence just written. New, behind a flag, off by default:

- `--k5-gate K,wr=W` adds W × the draft's mean association with the **previous sentence** (for the first draft, the prompt's last sentence). It is the same PPMI as the prompt term (`prompt_assoc`), on the previous sentence's words.
- It works in the plain gate (`word_kn.hpp`, prompts with no plan) and in the discourse writer's free sentences (`discourse.hpp`, z-scored there like its other terms).
- With `wr` 0 (the default) nothing is computed and the score is unchanged. Checked by `cmp` on all 250 dev prompts before any arm is read (below). The selftest checks the flag parses and is off by default.

## The baseline

**G16**, today's default: `scripts/gen_eval.sh dev G16 OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, with `CYPHALM_TRIE` and `SCRIBE` built by `g++ -O2` as `scripts/build_1tb_tools.sh` builds them, `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`.

**Checks first** (no arm is read until both hold):
- main's build (5abf1afc) gives a GEN file equal to `bench/tune_write/runs/G16.gen.txt` under `cmp`, MODEL line aside;
- my build with no new flag gives the same file.

## The arms (at most 8; each is G16 plus the flags shown)

| Arm | Flags | Family | What it rewards |
|---|---|---|---|
| R1 | `--k5-gate 16,wr=1` | previous sentence (new) | drafts that share the previous sentence's subject, as much as the prompt's |
| R2 | `--k5-gate 16,wr=2` | previous sentence (new) | the same, twice as hard |
| E1 | `--k5-eg 1 --k5-eg-mode cont` | entity grid | keeping the paragraph's entities in their roles |
| E2 | `--k5-eg 1` | entity grid | the grid's whole log-probability change (few new entities) |
| L1 | `--k5-lda 100 --k5-lda-w 1` | LDA topic | drafts whose words fit the prompt's topic mixture (a gate term; no rescaling of K5's words) |
| M1 | `--k5-mbr-sent 1` | MBR | the draft most like the other 15 (consensus) |
| D1 | `--k5-plan-discourse g_topic=2,g_grid=2` | discourse writer | the planned paragraphs' free sentences lean twice as hard on the paragraph's topic and grid |

- E1, E2 and L1 learn the grid and LDA from the same 330 MB K5 now counts from: `--k5-disc-text /mnt/d/cypha_llm/k5scale/clean/train_all.txt --k5-disc-cache ~/coh_cache` (K5_DISCOURSE.md's cached models, E4 and T6c, copied there).
- E1, E2, L1 and M1 use the z-scored gate (K5_DISCOURSE.md): z(pmi) plus the term, minus markup.
- D1 changes only the prompts with a plan (about 77 of 250).
- **Not run:** paragraph MCMC (`--k5-mcmc`). It edits words, not sentence choice, so it does not test the hypothesis; at 337 MB each paragraph costs several times the gate's time; and it read the same as K5 before (E1b, §13.34). The structure meta-mixer (`--k5-struct`) was trained on 7 MB drafts, and §13.36 showed it picks the oddest draft.
- **Combination:** if two candidates from different families pass the screen, their flags together (arm C) run on dev and are screened by the same rule. C is the eighth arm. If more than two families pass, C joins the two that read best.

## The screen (dev, seed 0, 250 prompts)

**Measures:** SCRIBE v2 (95% interval) and v3.1, copy32, verbatim13, wall time, peak RAM, samples unlike G16, and `coh_measures.py` (on the prompt by sentence, on the previous sentence, stuffing).

**Reading:** I read 12 dev prompts the raters never see (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13), each arm beside G16. I look for two things: does it keep to the prompt's or the previous sentence's subject past sentence 2, and does it repeat keywords. SCRIBE rewards topic stuffing (§13.34), so I am strict about repeated topic words: an arm that stays "on topic" by repeating the prompt's words fails the reading.

**An arm passes the screen if all hold:**
- verbatim13 is 0;
- copy32 at most 0.30;
- SCRIBE v2 at least G16's minus 0.03;
- by my reading, it reads at least as well as G16 on the 12 prompts, with no more keyword repetition.

**Candidates:** at most 3 that pass, at most one per family (R, E, L, M, D, or C), chosen by my reading first, then by "shares a word with the sentence before", then SCRIBE v2.

## The long reading

Drift shows most in long text (§13.42 Q3). For G16 and each candidate: `cyphalm_trie sample` on all 250 dev prompts with `--bytes 1800 --k5-max-words 600` (the same seeds; Q3's long setup), then `blind_rate.py --cut 1500` on the 25 prompts of `bench/blind/q3_long/prompts.json`.

## The deciding judge

- `python scripts/blind_rate.py make --out bench/blind/coh_pair --ids bench/blind/llm_rating/prompts.json --arm G16=... --arm X=... --mode pair --baseline G16 --seed 20261115`: the 50 prompts, four fresh blind Claude raters.
- Long: `--out bench/blind/coh_long --ids bench/blind/q3_long/prompts.json --seed 20261116 --cut 1500`, four more fresh raters.
- Packets, `key.b64` and `RATER_PROMPT.md` are committed and pushed before the raters run. `KEY.json` stays out of the repository.
- **The §13.38 rule:** an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- **Adopted** if it reads better by the rule on either reading (460 or long) and its win share on the other reading is at least 0.5. The best such arm (higher lower bound first, then lower copy32) becomes the default; Odin can veto. Then GEN-A test is read once, for it only.

## Predictions

- Both checks (main's build and mine with no new flag equal G16): **hit expected**.
- **R1:** "shares a word with the sentence before" rises from 0.15 to 0.22–0.30; sentences 2–3 on the prompt up 0.02–0.05; stuffing up at most 0.01; SCRIBE v2 within ±0.02; copy32 within ±0.02. It reads a little better. **Raters prefer R1 to G16 at about 0.60 on 460 characters and 0.60–0.65 on the long reading.** It is the most likely candidate.
- **R2:** previous-sentence share 0.25–0.35, stuffing up about 0.01–0.02; reads about as R1, with more repeated words. It may fail the reading for repetition.
- **E1, E2:** previous-sentence share within ±0.03 of G16; SCRIBE v2 within ±0.02; E2 writes more fragments. Neither passes the reading (as at 7 MB).
- **L1:** on the prompt up 0.01–0.03 per position; SCRIBE v2 +0.01 to +0.03; reads about as G16.
- **M1:** copy32 up 0.01–0.03; reads about as G16; previous-sentence share up a little (consensus favours common words).
- **D1:** fewer than 80 samples change; within ±0.01 on every measure.
- Peak RAM 11.3 GB ± 0.5 for R, M and D; E and L about +0.4 GB (the 330 MB helper text is read whole).
- If C runs: at the raters, about its best member's share.

## Results

(Added after the runs, under this line.)

### The screen (2026-11-12; dev, seed 0, 250 prompts; files in `bench/k5_coherence/runs/`)

**Checks:** main's build (5abf1afc) gives a GEN file equal to `bench/tune_write/runs/G16.gen.txt` under `cmp`, MODEL line aside. My build with no new flag (G16n) equals it too. **Hit.** The selftest passes.

| Arm | SCRIBE v2 [95%] | v3.1 | copy32 | verbatim13 | wall | peak RAM | samples unlike G16 | screen |
|---|---|---|---|---|---|---|---|---|
| G16 | 0.630 [0.597, 0.659] | 0.547 | 0.241 | 0 | 5:02 | 11.3 GB | – | baseline |
| R1 wr=1 | 0.691 [0.659, 0.721] | 0.570 | 0.248 | 0 | 5:12 | 11.3 GB | 201 | pass, not chosen (R2 is the same family) |
| R2 wr=2 | 0.693 [0.663, 0.722] | 0.573 | 0.255 | 0 | 5:18 | 11.3 GB | 209 | **pass, candidate** |
| E1 grid cont | 0.609 [0.575, 0.638] | 0.544 | 0.237 | 0 | 5:19 | 11.3 GB | 63 | pass, not chosen |
| E2 grid full | 0.602 [0.568, 0.632] | 0.542 | 0.225 | 0 | 5:20 | 11.3 GB | 168 | fail (reads worse: fragments) |
| L1 LDA gate term | 0.610 [0.577, 0.645] | 0.546 | 0.242 | 0 | 5:24 | 11.4 GB | 166 | **pass, candidate** |
| M1 MBR | 0.643 [0.605, 0.679] | 0.570 | 0.252 | 0 | 4:56 | 11.3 GB | 171 | fail (reads worse) |
| D1 discourse g×2 | 0.626 [0.592, 0.657] | 0.543 | 0.241 | 0 | 4:53 | 11.3 GB | 26 | pass, not chosen (26 samples change) |
| C = R2 + D1 | 0.692 [0.662, 0.722] | 0.573 | 0.252 | 0 | 5:26 | 11.3 GB | 208 (26 unlike R2) | pass, not chosen (it is R2 but for 26 samples) |

Two arms at once on a shared PC. M1's SCRIBE row holds 238 samples (12 too short for SCRIBE: consensus picks short drafts).

**`coh_measures.py`** (460 characters, 250 prompts):

| Arm | 1st sentence on the prompt | 2nd | 3rd | 4th+ | shares a word with the sentence before | 3rd+ use of a word | top word's share |
|---|---:|---:|---:|---:|---:|---:|---:|
| G16 | 0.376 | 0.316 | 0.253 | 0.086 | 0.150 | 0.015 | 0.057 |
| R1 | 0.376 | 0.292 | 0.228 | 0.084 | 0.177 | 0.015 | 0.059 |
| R2 | 0.376 | 0.276 | 0.228 | 0.078 | 0.183 | 0.016 | 0.059 |
| C | 0.376 | 0.276 | 0.224 | 0.075 | 0.180 | 0.016 | 0.059 |
| E1 | 0.352 | 0.308 | 0.244 | 0.079 | 0.139 | 0.015 | 0.055 |
| E2 | 0.364 | 0.284 | 0.248 | 0.067 | 0.121 | 0.016 | 0.058 |
| L1 | 0.360 | 0.276 | 0.241 | 0.090 | 0.150 | 0.015 | 0.056 |
| M1 | 0.392 | 0.342 | 0.259 | 0.113 | 0.233 | 0.015 | 0.054 |
| D1 | 0.376 | 0.316 | 0.253 | 0.089 | 0.153 | 0.016 | 0.058 |
| human | 0.632 | 0.488 | 0.510 | 0.310 | 0.460 | 0.032 | 0.076 |

**Reading** (dev 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, each arm beside G16):
- **The drift starts in the first sentence, not the third.** In half of the 12 prompts G16's first sentence is already off the subject ("Black powder arms …" → "Various Jews in Eastern European democracies"). Every arm keeps the same first sentence: the first gate has only the prompt to go on. So no chooser here can fix the start.
- **R1 and R2** chain neighbouring sentences a little more often. Dev 4: "In 2005, the first of a series of battles … The Russian Revolution of 1917 … During World War I, the Austro-Hungarian and Russian." Dev 12 (R2): "He had two brothers … he grew up in a household … his younger brother." That is a thread, but a new one, not the prompt's. They do not repeat keywords (3rd+ use 0.015–0.016, as G16). About as G16 to read, sometimes a little more connected; R2 is the same as R1 with slightly more links. **SCRIBE v2 rises +0.06**, more than reading suggests. The stuffing measures do not move, but treat the rise with care (§13.34).
- **E1** changes 63 samples and reads as G16. **E2** writes more fragments ("When his mother. After some weeks later."): worse.
- **L1** reads about as G16, sometimes a little worse (dev 4). It does not keep closer to the prompt.
- **M1** reads worse: more broken clauses and shorter drafts. Its "shares a word with the sentence before" (0.233) comes from short fragments sharing common words.
- **D1** and **C** change only the planned prompts' free sentences, mostly past the 460-character cut.
- **On the measures, only R1, R2 and C move the link to the sentence before up (0.150 → 0.18). No arm moves the prompt share up after sentence 1;** R and L lower it a little at sentences 2–3.

**Candidates for the raters:** R2 (previous sentence) and L1 (LDA topic term). By the registered order (reading, then the link to the sentence before), C and D1 rank above L1. They are not sent because they change too few samples to rate (26 each; TUNE_WRITE's DH4 precedent). E1 changes 63 and has the lowest link share. So only two candidates go out, not three.
- **The combination**, as registered, would join R with E or L. The `wr` term is not read by the z-scored gate that E and L use, so R + E and R + L would drop `wr`. The only combination that works is R + D (C), which ran and passed, but is R2 for all but 26 samples.

### The long reading (1,800 bytes, cut 1,500; 25 prompts of `bench/blind/q3_long/prompts.json`)

`--bytes 1800 --k5-max-words 600`, the same seeds. G16_long 6:28 and 11.3 GB; R2_long 6:57; L1_long 6:36, 11.4 GB.

| Arm | 1st on the prompt | 2nd | 3rd | 4th+ | link to the sentence before | 3rd+ use | top word |
|---|---:|---:|---:|---:|---:|---:|---:|
| G16 | 0.360 | 0.400 | 0.280 | 0.090 | 0.095 | 0.017 | 0.027 |
| R2 | 0.360 | 0.320 | 0.200 | 0.072 | 0.139 | 0.016 | 0.025 |
| L1 | 0.360 | 0.320 | 0.200 | 0.095 | 0.115 | 0.019 | 0.028 |
| human | 0.560 | 0.400 | 0.400 | 0.364 | 0.482 | 0.037 | 0.070 |

By reading (dev 4 and 12 at 1,500 characters), all three wander through unrelated subjects by the fourth sentence. R2 keeps short local threads (battles → revolutions → the war; a man's brothers → his household → his funeral) a little more often.

### Packets (built before any rating)

- `bench/blind/coh_pair/`: G16 against R2 and L1, the 50 prompts of `bench/blind/llm_rating/prompts.json`, seed 20261115. R2 differs from G16 on 45 of them, L1 on 39.
- `bench/blind/coh_long/`: G16_long against R2_long and L1_long, the 25 long prompts, seed 20261116, `--cut 1500`. R2 differs on 25, L1 on 18.
- `KEY.json` of each is kept outside the repository; `key.b64` and `RATER_PROMPT.md` are committed.

### Predictions so far

- Both checks equal G16: **hit**.
- R1 link to the sentence before 0.22–0.30: **miss** (0.177). Sentences 2–3 on the prompt up 0.02–0.05: **miss** (down 0.02). Stuffing up at most 0.01: **hit** (0.000). SCRIBE v2 within ±0.02: **miss** (+0.061). copy32 within ±0.02: **hit** (+0.007).
- R2 link 0.25–0.35: **miss** (0.183); stuffing up 0.01–0.02: **miss** (+0.001); may fail the reading for repetition: it did not.
- E1, E2 link within ±0.03: **hit** (−0.011, −0.029). SCRIBE v2 within ±0.02: **miss** (−0.021, −0.028). E2 more fragments: **hit**. Neither passes the reading: **hit** for E2, **miss** for E1 (it reads as G16, so it passes).
- L1 on the prompt up 0.01–0.03: **miss** (down); SCRIBE v2 +0.01 to +0.03: **miss** (−0.020); reads about as G16: **hit**.
- M1 copy32 up 0.01–0.03: **hit** (+0.011); reads about as G16: **miss** (worse).
- D1 fewer than 80 samples, within ±0.01: **hit** (26; SCRIBE v2 −0.004).
- Peak RAM 11.3 ± 0.5 GB: **hit**; E and L +0.4 GB: **miss** (+0.0 to +0.04 GB).
- The raters' predictions (R1/R2 about 0.60 on 460 characters, 0.60–0.65 long) wait for the raters.
