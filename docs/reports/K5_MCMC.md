# Whole-paragraph sampling for K5: Metropolis–Hastings by small word edits

**Odin Loch** · 2026-11-05 · GENERATION_PLAN §13.32, session 3 (branch `wip/k5-mcmc`, Odin's PC)

**Odin:** "Whole-paragraph sampling (MCMC): improves a whole paragraph by small edits, scored by K5 plus the topic and entity terms. … I want to see that."

The arms, rules and predictions below were written and committed before any arm was read. Results are added under them.

## Why

- Four blind Claude raters gave the gated K5 2.09 of 10 and human text 8.50 (§13.31). K5 writes sentence-shaped text that does not connect.
- K5 writes left to right. It cannot fix an earlier word once a later one shows it was wrong.
- MCMC treats the paragraph as one object. It edits any word, at any point, under a score of the whole paragraph. This is CGMH (Miao et al., 2019, "Constrained sentence generation by Metropolis-Hastings sampling"), with K5 in place of the neural language models.

## What was built

`native/include/cypha/lm/k5_mcmc.hpp`, behind `cyphalm_trie sample --k5 TEXT --k5-mcmc STEPS`.

**The start.** K5's own continuation, as `sample --k5` writes it today (nucleus 0.8, 256 classes, the 8-candidate pmi gate). The chain then edits it.

**The moves.** Each step proposes one edit at a random place:
- replace the word at position i (half the steps);
- insert a word at a gap (a quarter);
- delete the word at position i (a quarter).

The last token is held fixed, because the 600-byte cut may have split it.

**The proposal: K5 on both sides.**
- A new word w is drawn from the K = 64 most probable words of K5's distribution after the left context.
- Among those, q(w) ∝ 2^(L(w)/T). L(w) is the log2 probability under K5 of w and of the 4 words after it, with w in place. Those are all the probabilities the edit changes.
- A tenth of draws (ε 0.1) come from K5's whole left distribution instead. So every word can be proposed, and every move can be undone.
- The acceptance is the Metropolis–Hastings ratio with both proposal probabilities. The chain's target is exactly π(x) ∝ 2^(E(x)/T).

**The energy E** (bits) of a paragraph x after the prompt:
- K5's log2 probability of the paragraph given the prompt: K5 as it writes (the class floor), at temperature 1.
- \+ λ_topic × the topic term.
- \+ λ_entity × the entity term.
- − λ_rep × the number of repeated 4-grams of tokens. A repeat is a position whose last 4 tokens occurred earlier, prompt included. λ_rep is 3 bits.
- \+ a length term: β·n − ((n − n0)/σ)²/2. Here n is the number of words and n0 the start's. β is the start's mean surprisal per word, so a word of average surprisal costs nothing. σ is 1.5 words. Without β, deletions always pay. Without the square, the chain pads the text with cheap words: a 16-prompt smoke run with σ 3 added 27 words a paragraph.
- **The copy cap.** A paragraph with 13 consecutive training tokens (prompt included) is never accepted. It costs 10^6 bits per window, so no chain moves into one.

**The temperature.** It runs linearly from t0 to t1 over the steps (annealing). The last state is written, cut to 600 bytes as K5 cuts it.

**The terms.**
- **Topic, stand-in (cache).** Each content word of the paragraph (outside the 100 most frequent) that also occurs in the prompt gains log2(1 + 0.1·c(w)/p1(w)) bits. Here c is the word's share of the prompt's content words, and p1 is K5's order-1 probability. The gain halves for each earlier use of the same word in the paragraph.
- **Topic, LDA** (`--k5-mcmc-topic lda`). This is wip/k5-discourse's `lda.hpp`, merged in, not edited: 100 topics, 200 sweeps, exact, learned from K5's 7 MB text. The term is `log_ratio(infer(prompt), paragraph)`: the sum over the paragraph's content words of log2(p_topic(w | the prompt's topics) / p(w)). The model is their cached file (`lda_k5train.txt_K100_it200_t1.bin`, the same text).
- **Entity grid** (`--k5-mcmc-le W`, `--k5-mcmc-entity full|cont`). This is wip/k5-discourse's `entity_grid.hpp`, learned from K5's text (their cached file). The term is `EntityGrid::score(prompt, paragraph, cont)`, with two forms:
  - full: the change in the grid's log2 probability when the paragraph's sentences follow the prompt's;
  - cont: only the entities kept from earlier sentences.

**Small hooks in `word_kn.hpp`:**
- `WordKN::prob`, `ClassModel::prob` and `K5::word_prob`: one word's probability, equal to `model_dist`'s entry (checked by the selftest);
- `Options::post` and `post_desc`: a function that rewrites each continuation after `generate`, and its MODEL-line text.

Empty, they change nothing.

**Flags** (all `--k5-mcmc…`):
- `--k5-mcmc STEPS`;
- `-t0 T`, `-t1 T` (defaults 1.0 and 0.6);
- `-lt W` (topic weight), `-topic cache|lda`, `-lda PATH`;
- `-le W` (entity weight), `-entity full|cont`, `-eg PATH`;
- `-lr W` (repeats, 3);
- `-len S` (σ, 1.5);
- `-k K` (64);
- `-eps E` (0.1);
- `-moves R,I,D` (2,1,1).

**Not built:** the optional sentence move (swap a sentence, or redraw it with K5). A redraw by K5's sampler has no proposal probability that the MH ratio could use (its nucleus and bans cut the reverse move to zero), and plain K5 at T 1 writes worse sentences than the start.

**Checks before any read:**
- `cyphalm_trie selftest` passes. It checks:
  - `word_prob` against `model_dist` at every word, with and without classes;
  - the chain's tracked energy against a full recomputation, after hundreds of moves, with every term on;
  - the LDA and grid terms against wip/k5-discourse's own calls;
  - the cap, and the flags.
- With the MCMC flags off, `sample`'s GEN file and stdout on GEN-A dev are byte-identical to main's (`cmp`). This was checked before and after merging wip/k5-discourse.
- **Reproduction:** K5's default on this PC gives SCRIBE v2 **0.498** [0.469, 0.527], v3.1 **0.483** [0.466, 0.500], surface S 0.946, copy32 0.0380 and verbatim13 0 of 250. Those are K5_UPGRADES.md's and §13.28's numbers.
- **Smoke runs:** 16 dev prompts, to check speed and length only. They were not scored, and no text from them is used.
  - Speed: about 1.4–1.9 ms a proposal on 8 threads.
  - 1,000 steps take 1.4–1.9 s a paragraph, against about 0.37 s a paragraph-thread for K5's own gated writing.
  - With σ 1.5, paragraphs gain about 8 words (7%) and K5's log2 probability per paragraph rises by about 17 bits.

## Setup

- Worktree `~/k5mcmc_wt`, binaries in `~/cypha_build/mcmc` (g++ -O2). Main's binaries are in `~/cypha_build/mcmc_base`.
- Each read is `bench/k5_mcmc/arm.sh NAME [flags]`. It runs `scripts/gen_eval.sh dev NAME /mnt/d/cypha_llm/k5mcmc --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt [flags]` with `K5_THREADS=8`.
- Each arm reports:
  - SCRIBE v2 and v3.1 (pooled row, 95% intervals);
  - surface S, copy32 and verbatim13;
  - the chain's acceptance rates, and its mean change in energy, K5 bits and words;
  - the sampling time, the wall time per 250 prompts, and peak memory.
- Every arm's dev GEN file and summary go to `bench/k5_mcmc/` (for the cloud session's blind raters).
- Seeds: offset 0 to choose (K5's seeds, 20261101 + the prompt's index). Then offsets 1000, 2000, 3000 and 4000 (`--k5-seed`) for the final arms. The chain's random stream is seeded from the continuation's seed.

## Arms (GEN-A dev, 250 prompts, seed offset 0)

All MCMC arms use 1,000 steps (about 9 proposals per word), T 1.0 → 0.6, K 64, ε 0.1, moves 2:1:1, λ_rep 3 and σ 1.5, unless stated.

| Arm | What | Flags beyond `--k5-mcmc 1000` |
|---|---|---|
| M0 | K5's default, no MCMC (the baseline) | (no `--k5-mcmc`) |
| R1 | K5 + repeats + length only | — |
| R2 | R1 with no annealing (T 1.0 throughout) | `-t1 1.0` |
| R3 | R1 with 300 steps (cost) | `--k5-mcmc 300` |
| C1 | R1 + the stand-in topic cache, λ 1 | `-lt 1` |
| L1 | R1 + LDA topic, λ 0.5 | `-lt 0.5 -topic lda` |
| L2 | R1 + LDA topic, λ 1 | `-lt 1 -topic lda` |
| E1 | R1 + entity grid, full, λ 0.5 | `-le 0.5 -entity full` |
| E2 | R1 + entity grid, cont, λ 0.5 | `-le 0.5 -entity cont` |
| LE | R1 + the better L arm's topic + the better E arm's entity term (the energy the task describes) | as chosen |
| S1 | R1 + wip/k5-structure's learned mixed score (cloud request, 2026-11-05). Run only once that branch exposes its scorer. Its weight and form are registered in an addendum here, before the run. | — |

"Better" L or E arm: the higher v3.1, among arms that keep the conditions below.

## Choosing and confirming

- **Conditions** for any arm at seed 0 (as K5_DISCOURSE.md):
  - v2 at least M0's − 0.02;
  - copy32 at most M0's + 0.02;
  - verbatim13 = 0.
- **The pick:** the arm with the highest v3.1 among the arms that keep the conditions, if it is above M0's v3.1.
- **Final arms**, run at 4 fresh seed offsets (1000–4000): M0, R1, LE and the pick (if it is not R1 or LE). That is at most 4 arms × 4 seeds.
- **The rule** (the task's). An arm becomes a K5 option worth recommending only if all of these hold:
  1. its 4-seed mean v3.1 is above M0's 4-seed mean, with intervals that do not overlap. A 4-seed mean's interval is the mean of the 4 per-seed lower bounds to the mean of the 4 upper bounds. That is conservative: it is wider than the true interval of a mean over 1,000 samples.
  2. its 4-seed mean v2 is at least M0's − 0.02;
  3. verbatim13 is 0 at every seed (the copy cap holds), and copy32's mean rises by at most 0.02;
  4. the cloud session's 4 blind Claude raters (50 dev prompts) confirm it.
- No MCMC flag becomes a default here. The GEN-A test read is made once, only for an arm adopted after the raters confirm it. That is the cloud session's call, so this session makes no test read.

## Predictions

- **P1 (R1).** The chain raises K5's log2 probability per paragraph by 10–40 bits. Surface S rises (+0.005 to +0.02), and copy32 rises by 0.005–0.03. v2 is within ±0.02 of M0. v3.1 falls by 0.01–0.04: the text gets more predictable, which SURPRISE 3.1 reads as bland.
- **P2 (R2).** With no annealing, the target is K5's raw distribution, with no nucleus. So v2 falls by 0.03–0.08 against R1.
- **P3 (R3).** 300 steps land between M0 and R1 on every measure.
- **P4 (C1).** The cache moves v3.1 by less than 0.02. If it moves it down, MATCH's prompt statistics (`m_pcoh`, `w_pgain`) are the reason.
- **P5 (L1, L2).** The LDA term raises v2 by 0.01–0.03: SCRIBE's coherence axis reads co-occurrence, as LDA does. v3.1 moves by less than 0.02. L2 is not above L1 on v3.1 (stuffing).
- **P6 (E1, E2).** E1 (full) lowers v3.1: it pays to drop entities, so the text gets blander. E2 moves v2 and v3.1 by less than 0.01.
- **P7 (LE).** LE is no better than the best single arm.
- **P8 (the rule).** About a 25% chance that any arm passes conditions 1–3 at 4 seeds. Non-overlapping intervals need about +0.03 v3.1. The raters' verdict is open. My guess: MCMC will not lift K5 from about 2 of 10 to 4.
- **P9 (cost).** 1,000 steps add 1.4–1.9 s a paragraph-thread, about 50–70 s per 250 prompts on 8 threads, on top of about 90 s for K5's gated writing. Memory rises by less than 50 MB (the LDA model is 12 MB).

## Addendum 1 (2026-11-05, after reading M0, R1, R2, R3 and C1; before R4)

**What prompted it.** Every MCMC arm read so far is below M0 on v2 (0.421–0.468 against 0.498). Every paragraph grows by 7.5–11 words, even at 300 steps (R3). The insert and delete moves, with their length term, may cause much of the damage. They pad the text with cheap words and cut it at odd places. Replace-only moves keep the length and the sentence boundaries fixed.

**New arms** (seed 0, all else as R1):
- **R4:** replace only (`--k5-mcmc-moves 1,0,0`; the length term is then constant).
- **R4X:** replace only, with the terms of the best of C1, L1, L2, E1, E2 and LE by the choosing conditions. If none keeps the conditions, it uses the one with the highest v3.1.

They join the choice as any other arm. The final arms and the rule are unchanged.

**Predictions:**
- R4 is closer to M0 than R1 on both judges: v2 within 0.03 of M0, and v3.1 0.005–0.03 below M0.
- R4X is within ±0.02 of R4.
- Neither is above M0 on v3.1 by more than 0.01.

## Results

(Added after the runs.)
