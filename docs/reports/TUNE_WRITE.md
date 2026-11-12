# T2: tuning CyphaLM's writing

**Odin Loch** · registered 2026-11-12, before any arm runs · branch `wip/tune-write` · GENERATION_PLAN §13.42 (T2), §13.43 ("For writing")

## Why now

§13.48 made K5's counts come from 337 MB (W1b, the default since 9124a00). The raters preferred it to the old default 0.940. Every K5 and planner setting was chosen when K5 counted from 7 MB. With 50 times more text, the best nucleus, temperature, order and gate size may have moved. T2 tunes them, one at a time, on dev.

## The baseline

**W1b**: today's default. `scripts/gen_eval.sh dev W1b OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, which means:
- the discourse planner on `bench/fact_store/e9_text`;
- K5 order 5, counts from `/mnt/d/cypha_llm/k5scale/clean/train_all.txt` appended (`--k5-more`);
- nucleus 0.8, temperature 1.0, sentence gate of 8 drafts.

**Check first:** my build's W1b GEN file must equal `bench/gen_quality/runs/W1b.gen.txt` under `cmp`, MODEL line aside. If it does not, no arm runs until the cause is found.

## The arms

Each arm moves one setting from the default. Dev, seed 0, 250 prompts, `scripts/gen_eval.sh` with the baseline's command plus the flag.

| Arm | Flag | What it moves |
|---|---|---|
| N7 | `--k5-nucleus 0.7` | K5's nucleus, narrower |
| N9 | `--k5-nucleus 0.9` | K5's nucleus, wider |
| T8 | `--temps 0.8` | temperature 0.8 |
| T9 | `--temps 0.9` | temperature 0.9 |
| O6 | `--k5-order 6` | K5 reads five words back |
| O7 | `--k5-order 7` | K5 reads six words back (the most the code allows) |
| G4 | `--k5-gate 4` | the sentence gate picks among 4 drafts |
| G16 | `--k5-gate 16` | the sentence gate picks among 16 drafts |
| DG2 | `--k5-plan-discourse gap=2` | the discourse writer puts one free K5 sentence after every 2 fact sentences (K5's free text now reads well) |
| DH4 | `--k5-plan-discourse w_hop=4` | the discourse writer stays closer to the main entity (focus cost per link 4, default 2) |

- `--temps 0.8` keeps the seed of position 0, so T8 and T9 draw with the same seeds as the baseline.
- Only about 77 of 250 dev prompts get a plan, so DG2 and DH4 change few samples. They are cheap, so they run.
- **The §13.43 idea for writing** (a setting that wanders around its optimum while it writes) has no flag in `cyphalm_trie` or `word_kn.hpp`. **It is not built**, and T2 does not test it.

**Machine limits:** at most 2 arms at once, at most 6 threads each (`K5_THREADS=6`), no start while `free -g` shows under 14 GB available. Each arm peaks about 11.3 GB.

## The screen (dev)

**Measures:** SCRIBE v2 (with its 95% interval) and v3.1, copy32, verbatim13, wall time, peak RAM.

**Reading:** I read 12 dev prompts that the raters will not see (dev prompts 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13), each arm beside W1b. SCRIBE rewards topic words and has misled this project before (§13.34, §13.48), so the reading counts as much as SCRIBE.

**An arm passes the screen if all hold:**
- verbatim13 is 0;
- copy32 at most 0.30 (W1b 0.239);
- SCRIBE v2 at least W1b's minus 0.03;
- by my reading, it reads at least as well as W1b on the 12 prompts.

**Candidates:** at most 3 that pass, at most one per setting (nucleus, temperature, order, gate, discourse), chosen by the reading first and SCRIBE v2 second. Nucleus and temperature both narrow the same distribution, so they count as one setting for "independent". If two or more candidates move independent settings, their combination (**TC**) is run on dev and screened by the same rule, and goes to the raters too if it passes.

## The deciding judge

`python scripts/blind_rate.py make --out bench/blind/tw_pair --ids bench/blind/llm_rating/prompts.json --arm W1b=... --arm X=... --mode pair --baseline W1b --seed 20261113`: the 50 prompts of `bench/blind/llm_rating/prompts.json`, four fresh blind Claude raters. The packets, `key.b64` and `RATER_PROMPT.md` are committed and pushed before the raters run; `KEY.json` stays out of the repo.

**The §13.38 rule:** an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign. The arm that reads best of those that pass becomes the default (Odin can veto). Then GEN-A test is read once, for that arm only.

## Predictions

- The baseline reproduces W1b under `cmp`: hit expected.
- **Nucleus and temperature:** N7 and T8 read a little cleaner, with copy32 up 0.02 to 0.05; SCRIBE v2 within ±0.02 of W1b. N9 and T9 within ±0.02 too, N9 a little worse to read. At the raters, the best of them about 0.55, interval touching 0.5.
- **Order:** O6 and O7 raise copy32 to 0.30–0.40 (more of each sentence comes from one place in the text), verbatim13 stays 0 (the 12-word cap). O6 reads better than W1b; it is the most likely candidate. **Raters prefer O6 to W1b at about 0.60.** O7 copies so much that it fails the copy32 screen.
- **Gate:** G16 raises SCRIBE v2 by about 0.01 (more topic words), reads about the same; G4 about the same. Neither reaches the raters unless it reads better.
- **Discourse:** DG2 and DH4 change fewer than 80 samples; within ±0.02 on SCRIBE v2; raters about 0.5 if rated.
- **The combination**, if run: at the raters about the best single arm's share, not above it by more than 0.05.
- Peak RAM within 11.3 GB ± 1 GB for all but O6 and O7 (unchanged index, longer context keys: +0 to 2 GB).

## Results

### The screen (2026-11-12; dev, seed 0, 250 prompts; files in `bench/tune_write/runs/`)

**Check:** my build's W1b GEN file equals `bench/gen_quality/runs/W1b.gen.txt` under `cmp`, MODEL line aside. **Hit.**

| Arm | SCRIBE v2 [95%] | v3.1 | copy32 | verbatim13 | wall | peak RAM | samples unlike W1b | screen |
|---|---|---|---|---|---|---|---|---|
| W1b | 0.621 [0.588, 0.653] | 0.556 | 0.239 | 0 | 7:21 | 11.3 GB | – | baseline |
| N7 | 0.617 [0.584, 0.648] | 0.522 | 0.289 | 0 | 8:59 | 11.3 GB | 214 | **pass** |
| N9 | 0.576 [0.539, 0.614] | 0.566 | 0.193 | 0 | 6:23 | 11.3 GB | 214 | fail (SCRIBE; reads worse) |
| T8 | 0.655 [0.624, 0.687] | 0.443 | 0.385 | 0 | 6:05 | 11.3 GB | 214 | fail (copy32) |
| T9 | 0.624 [0.592, 0.657] | 0.487 | 0.312 | 0 | 6:16 | 11.3 GB | 214 | fail (copy32) |
| O6 | 0.599 [0.565, 0.633] | 0.543 | 0.248 | 0 | 7:21 | 11.3 GB | 250 | **pass** |
| O7 | 0.600 [0.565, 0.637] | 0.545 | 0.249 | 0 | 9:37 | 11.3 GB | 250 | pass, not chosen (O6 reads better) |
| G4 | 0.574 [0.538, 0.609] | 0.532 | 0.237 | 0 | 5:09 | 11.3 GB | 204 | fail (SCRIBE) |
| G16 | 0.630 [0.597, 0.659] | 0.547 | 0.241 | 0 | 7:21 | 11.3 GB | 210 | **pass** |
| DG2 | 0.599 [0.569, 0.628] | 0.545 | 0.243 | 0 | 6:15 | 11.3 GB | 61 | fail (reads worse) |
| DH4 | 0.621 [0.587, 0.653] | 0.556 | 0.239 | 0 | 6:00 | 11.3 GB | 9 | pass, not chosen (9 samples change) |
| TC = N7 + O6 + G16 | 0.636 [0.604, 0.669] | 0.516 | 0.309 | 0 | 7:41 | 11.3 GB | 250 | fail (copy32) |

Wall times are on a shared PC, two arms at once.

**Reading** (dev prompts 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, each beside W1b):
- Every arm still drifts after the first one or two sentences. No setting fixes that.
- **N7** reads a little cleaner than W1b: more whole clauses ("The simplest (and probably best) rule is to use a card - punch"). **N9** breaks more sentences.
- **T8 and T9** write longer run-on sentences. They copy more (copy32 0.385 and 0.312), so they fail the screen. SCRIBE v2 rises for T8 while v3.1 falls (0.443): v3.1's MATCH judge sees the copying.
- **O6** changes every sample, as the seed path moves. Its planned sentences hang together a little better ("After his return to England Brian Henderson was elected to the New Zealand House in 2001."). Free text about as W1b. **O7** invents more broken dates ("Brian Henderson (3 August 1931 - 34 March 2005)"). Longer reach did not raise copy32 (0.248): the 12-word cap holds it.
- **G16** gives shorter, tidier sentences with more fragments ("He spent most of his life."); about as W1b. **G4** about as W1b, and SCRIBE falls.
- **DG2** puts free fragments between fact sentences ("His funeral in early June.", "By late afternoon and evening."). It reads worse. **DH4** changes 9 samples only.
- **TC** reads about as N7 and O6, but copies more (0.309) and fails the copy32 screen.

**Candidates for the raters:** N7 (nucleus), O6 (order), G16 (gate). The combination failed the screen, so it is not rated. The §13.43 setting that wanders is not built (no flag).

**Packets** (`bench/blind/tw_pair/`): W1b against N7, O6 and G16, the 50 prompts of `bench/blind/llm_rating/prompts.json`, seed 20261113. On those 50 prompts, N7 differs from W1b on 45, O6 on 50, G16 on 44. `KEY.json` is kept outside the repository; `key.b64` is committed.

**Predictions so far:**
- baseline reproduces: **hit**;
- N7 and T8 cleaner, copy32 up 0.02–0.05: N7 **hit** (+0.050); T8 **miss** (+0.146);
- nucleus and temperature SCRIBE v2 within ±0.02: N7 **hit**, T9 **hit**, N9 **miss** (−0.045), T8 **miss** (+0.034);
- O6 and O7 copy32 0.30–0.40: **miss** (0.248, 0.249); O7 fails the copy32 screen: **miss** (it passes);
- G16 SCRIBE v2 about +0.01: **hit** (+0.009); G4 about the same: **miss** (−0.047);
- DG2 and DH4 within ±0.02, fewer than 80 samples changed: DH4 **hit**; DG2 **miss** on SCRIBE (−0.022), **hit** on samples (61);
- peak RAM within 11.3 ± 1 GB: **hit** for all.
- The raters' predictions (O6 about 0.60; the best of nucleus and temperature about 0.55) wait for the raters.

## Raters (2026-11-12)

Four fresh blind raters, 50 prompts, packets and key committed first (7e401c7a). `blind_rate.py score --dir bench/blind/tw_pair --baseline W1b`:

| Arm | Win share against W1b [95%] | Rater pairs | Rule |
|---|---|---|---|
| G16 | **0.670 [0.570, 0.770]** | 0.66, 0.68 | reads better |
| N7 | **0.670 [0.560, 0.770]** | 0.64, 0.70 | reads better |
| O6 | 0.580 [0.460, 0.690] | 0.58, 0.58 | no difference |

- **Predictions:** O6 about 0.60: **hit** (0.58, not significant); the nucleus arm about 0.55 with the interval touching 0.5: **miss** in the good direction (0.670); G16 about even: **miss** (0.670).
- **A tie the rule did not foresee.** G16 and N7 have the same share. Registered now, before any further run, so the choice is not made after seeing more data:
  - **Tie-break:** the higher lower bound of the interval, then the lower copy32. G16 wins both (0.570 against 0.560; copy32 0.241 against 0.289), and it does not reshape K5's distribution. **G16 becomes the default** (`--k5-gate 16`), and GEN-A test is read once for it.
  - **NG**, a new arm: N7 + G16 (`--k5-nucleus 0.7 --k5-gate 16`), not tried before (TC held O6 too and failed on copy32). It is screened on dev by the same rule. If it passes, raters compare it against G16, the new default, on the same 50 prompts with four fresh raters. It becomes the default only if it reads better by the §13.38 rule; then GEN-A test is read once for it.
  - **Prediction for NG:** copy32 about 0.29 (N7's); against G16, a win share of 0.55–0.60, interval touching 0.5.

**NG on dev** (2026-11-12): SCRIBE v2 0.652 [0.623, 0.682], v3.1 0.525, copy32 0.280, verbatim13 0. By reading (dev 1, 2, 4, 9, 10, not rated), it is about as good as G16: slightly cleaner clauses, with as many fragments. **It passes the screen.** Packets NG against G16: `bench/blind/ng_pair/`, seed 20261114, committed before the raters run.

**NG against G16** (four fresh raters, 50 prompts, packets committed first, e1d69808): **0.510 [0.400, 0.620]**, rater pairs 0.52 and 0.50. No difference. **NG is not adopted; G16 stays the default.** Prediction 0.55–0.60 with the interval touching 0.5: **miss** (0.51; the interval holds 0.5 as predicted).
