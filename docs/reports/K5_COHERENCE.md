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
