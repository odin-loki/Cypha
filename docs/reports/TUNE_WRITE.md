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

(After the runs.)
