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

## Addendum 2 (2026-11-05, after reading the seed-0 arms through R4; before LE's and R4X's results and before E1b, E1c)

**What prompted it.**
- R4 (replace only) is worse than R1 on v3.1 (0.420): the insert and delete moves were not the cause. The addendum-1 prediction for R4 missed.
- In every arm the loss on v3.1 sits in SURPRISE 3.1's bland group D_B (M0 7.4; R1 11.3; R4 and the others 9–11) and MATCH 3.1's predictability group D_P (M0 1.7; R1 3.7).
- Annealing to T 0.6 raises K5's probability of the paragraph, which makes it blander and more copied.
- The entity grid (E1) is the only term that raised v2 above M0 (0.524). Its v3.1 loss (0.469) is the smallest among the annealed arms.

**New arms** (seed 0, as E1 otherwise: entity full, λ_entity 0.5, 1,000 steps, all moves):
- **E1b:** T 1.0 → 0.8 (`--k5-mcmc-t1 0.8`).
- **E1c:** T 1.0 throughout (`--k5-mcmc-t1 1.0`).

They join the choice as any other arm. The final arms and the rule are unchanged.

**Predictions:**
- E1b: v2 0.49–0.52, v3.1 0.465–0.485.
- E1c: v2 0.45–0.49 (as R2, the target has no nucleus), v3.1 0.47–0.49.
- About a 20% chance that either is above M0's v3.1 (0.483) at seed 0.

## Results

GEN-A dev, 250 prompts. SCRIBE v2 and v3.1 are the pooled rows, with 95% intervals. The time is the sampling time (K5's gated writing plus the chain) and the wall time of `gen_eval.sh`, per 250 prompts on 8 threads, on a PC shared with other sessions (its load varies by about ±30%). Peak memory is 490–500 MB in every arm, so the chain adds nothing measurable. Every row: `bench/k5_mcmc/results.tsv`. Every GEN file: `bench/k5_mcmc/NAME.gen.txt`.

### Seed 0 (choosing)

Conditions: v2 ≥ 0.478, copy32 ≤ 0.058, verbatim13 0.

| Arm | v2 [95%] | v3.1 [95%] | S | copy32 | v13 | K5 bits Δ | words Δ | sample s | conditions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M0 K5 | 0.498 [0.469, 0.527] | 0.483 [0.466, 0.500] | 0.946 | 0.0380 | 0 | — | — | 92 | baseline |
| R1 rep + length | 0.453 [0.426, 0.483] | 0.435 [0.417, 0.453] | 0.948 | 0.0599 | 0 | +12.2 | +7.7 | 260 | fail |
| R2 no annealing | 0.421 [0.391, 0.450] | 0.472 [0.458, 0.487] | 0.931 | 0.0341 | 0 | −142.6 | +10.8 | 351 | fail (v2) |
| R3 300 steps | 0.456 [0.429, 0.484] | 0.463 [0.448, 0.479] | 0.945 | 0.0484 | 0 | −18.0 | +7.5 | 133 | fail (v2) |
| C1 cache 1 | 0.468 [0.438, 0.498] | 0.455 [0.435, 0.474] | 0.950 | 0.0593 | 0 | +8.6 | +7.8 | 137 | fail |
| L1 LDA 0.5 | 0.469 [0.441, 0.500] | 0.449 [0.430, 0.466] | 0.949 | 0.0544 | 0 | +10.9 | +7.6 | 131 | fail (v2) |
| L2 LDA 1 | 0.485 [0.457, 0.517] | 0.454 [0.436, 0.472] | 0.949 | 0.0535 | 0 | +10.5 | +7.7 | 123 | keep |
| E1 grid full 0.5 | 0.524 [0.493, 0.554] | 0.469 [0.453, 0.486] | 0.945 | 0.0518 | 0 | +0.0 | +7.3 | 147 | keep |
| E2 grid cont 0.5 | 0.451 [0.424, 0.480] | 0.433 [0.416, 0.450] | 0.946 | 0.0550 | 0 | +9.5 | +7.6 | 151 | fail (v2) |
| LE LDA 1 + grid full 0.5 | 0.538 [0.508, 0.571] | 0.486 [0.469, 0.504] | 0.948 | 0.0541 | 0 | +1.6 | +7.2 | 119 | keep |
| R4 replace only | 0.461 [0.431, 0.492] | 0.420 [0.401, 0.440] | 0.950 | 0.0640 | 0 | +61.7 | 0 | 150 | fail |
| R4X replace only + LE's terms | **0.550** [0.521, 0.581] | 0.463 [0.444, 0.483] | 0.949 | 0.0569 | 0 | +54.4 | 0 | 117 | keep |
| **E1b grid full 0.5, T → 0.8** | 0.490 [0.462, 0.526] | **0.494** [0.480, 0.511] | 0.940 | 0.0376 | 0 | −68.3 | +8.7 | 115 | **keep: the pick** |
| E1c grid full 0.5, T 1 | 0.445 [0.414, 0.476] | 0.490 [0.473, 0.505] | 0.928 | 0.0323 | 0 | −151.4 | +10.3 | 120 | fail (v2) |

- Acceptance: replace 23–34%, insert 14–19%, delete 10–15%. No paragraph was skipped (every K5 text re-rendered to itself).
- **The pick is E1b** (the highest v3.1 that keeps the conditions, above M0's 0.483). The better L arm was L2 (L1 fails v2), and the better E arm E1, so LE = LDA 1 + grid full 0.5. By addendum 1, R4X took LE's terms (LE had the highest v3.1 among C1–LE that keep the conditions).
- S1 (wip/k5-structure's mixed score) was not run. Its branch exposes `structure::Scorer`, but no trained mixer was on the PC by the end of this session, and its branch conflicts with this one in `word_kn.hpp` and `cyphalm_trie.cpp` (hooks in the same places; resolvable).

**Where the losses come from** (SCRIBE's own groups, seed 0):

| Arm | v2 D_S (form) | v2 D_C (coherence) | SURPRISE 3.1 D_B (bland) | MATCH 3.1 D_P (predictable) |
|---|---:|---:|---:|---:|
| M0 | 0.66 | 3.41 | 7.36 | 1.71 |
| R1 | 0.83 | 3.72 | 11.33 | 3.68 |
| E1 | 0.83 | **3.05** | 9.03 | 2.47 |
| L2 | 0.84 | 3.38 | 10.94 | 3.91 |

- Every chain makes sentence form a little worse (D_S 0.66 → 0.83). Local edits judged by a 5-gram model join sentences in new places.
- Raising K5's probability (R1, R4) makes the text blander and more predictable to SCRIBE v3.1's judges (D_B, D_P), and more copied (copy32 +0.02 to +0.03). That is the sharp-sampler finding of K5_UPGRADES.md step 1 again.
- The entity grid is the only term that improves coherence (D_C 3.41 → 3.05). Annealing less (E1b) gives back the blandness the K5 term costs.

### Four fresh seeds (offsets 1000–4000)

Means over the 4 seeds. Each interval is the mean of the per-seed bounds (as registered).

| Arm | v2 | v3.1 | S | copy32 | v13 (of 1,000) | sample s | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.499 [0.471, 0.526] | 0.475 [0.458, 0.492] | 0.947 | 0.0444 | 0 | 83 | 162 |
| R1 | 0.461 [0.434, 0.489] | 0.433 [0.416, 0.450] | 0.946 | 0.0641 | 0 | 124 | 197 |
| LE | **0.541** [0.511, 0.573] | 0.479 [0.460, 0.497] | 0.948 | 0.0571 | 0 | 154 | 235 |
| **E1b** | 0.510 [0.480, 0.542] | **0.495** [0.480, 0.512] | 0.939 | 0.0446 | 0 | 120 | 191 |

Per seed:

| Seed offset | M0 v2 / v3.1 | R1 v2 / v3.1 | LE v2 / v3.1 | E1b v2 / v3.1 |
|---|---|---|---|---|
| 1000 | 0.496 / 0.470 | 0.464 / 0.430 | 0.536 / 0.473 | 0.506 / 0.490 |
| 2000 | 0.501 / 0.474 | 0.448 / 0.428 | 0.545 / 0.480 | 0.510 / 0.497 |
| 3000 | 0.505 / 0.476 | 0.461 / 0.432 | 0.530 / 0.472 | 0.516 / 0.503 |
| 4000 | 0.493 / 0.480 | 0.470 / 0.442 | 0.553 / 0.491 | 0.508 / 0.492 |

**The rule.**
- **E1b:** v3.1 +0.020 over M0, higher at 4 of 4 seeds (+0.012 to +0.027). But the registered intervals overlap ([0.480, 0.512] against [0.458, 0.492]), so condition 1 **fails**. Conditions 2 and 3 hold: v2 +0.011, verbatim13 0 at every seed, copy32 +0.000. Because the registered interval is conservative, E1b is the arm to send to the raters.
- **LE:** v2 +0.042 (4 of 4 seeds, the largest v2 gain), v3.1 +0.004 (overlapping, 3 of 4 seeds). Condition 1 **fails**; copy32 +0.013 is inside its bound.
- **R1:** worse on both judges at every seed. **Fails.**
- **No arm passes the rule as registered.** No MCMC flag becomes a default, and no test read was made. Condition 4, the cloud session's 4 blind raters, is still to come. M0, E1b and LE's dev GEN files at seed 0 and at the 4 seeds are in `bench/k5_mcmc/`.

### Cost

- The chain takes 1.2–1.5 s a paragraph-thread for 1,000 proposals (about 1.2–1.5 ms a proposal), and 0.5 s for 300.
- Per 250 prompts on 8 threads: K5's gated writing is 75–95 s. 1,000 steps add about 35–70 s of sampling: E1b 120 s against M0's 83 s (4-seed means), and about 30–75 s of wall time.
- The topic and grid terms add little: E1b and LE cost the same as R1.
- Where the time went, and what was cut:
  - before: 2.2 ms a proposal;
  - selecting the 64 candidates by `nth_element` took 44% (many tied tail probabilities): a one-pass min-heap now;
  - the copy-cap check searched every window of the paragraph at every step: it now checks only the windows an edit creates or destroys (exact);
  - one-word probabilities (`WordKN::prob`) are the rest, 21%;
  - after: 1.2–1.5 ms.
- Fewer steps cost less but do not help: R3 (300 steps) is no better than R1.

### Predictions

- **P1 (R1):** v3.1 down by 0.01–0.04. Missed in size: −0.048. v2 down 0.045, not within ±0.02. Missed. K5's bits rose by 12 per paragraph (inside 10–40: hit). Copy32 +0.022: hit.
- **P2 (R2):** v2 falls 0.03–0.08 against R1. Hit (−0.032). Not foreseen: v3.1 rises against R1 (0.472).
- **P3 (R3):** between M0 and R1 on v3.1 (0.463): hit. On v2, no better than R1: missed.
- **P4 (C1):** v3.1 within 0.02 of R1 (+0.020, on the edge). Hit only just.
- **P5 (L1, L2):** v2 up 0.01–0.03 over R1 (+0.016, +0.032): hit. v3.1 within 0.02: hit. L2 not above L1 on v3.1: missed (0.454 against 0.449, noise).
- **P6 (E1, E2):** E1 lowers v3.1: missed against R1 (it rises, +0.034), though it is below M0. E2 within 0.01 of R1: hit.
- **P7 (LE no better than the best single):** missed. LE is above both L2 and E1 on both judges.
- **P8 (about 25% that an arm passes conditions 1–3):** none passed. E1b came close (4 of 4 seeds, +0.020).
- **P9 (cost):** hit. +35–70 s per 250 prompts and no measurable memory.
- **Addendum 1 (R4 closer to M0 than R1):** missed. R4 is the worst on v3.1 (0.420). R4X within ±0.02 of R4: missed (+0.043 v3.1, +0.089 v2).
- **Addendum 2:** E1b v2 0.49–0.52 and v3.1 0.465–0.485: v2 hit (0.490), v3.1 missed upward (0.494). E1c v2 0.45–0.49: missed downward (0.445); v3.1 0.47–0.49: hit (0.490). Above M0's v3.1 at seed 0 (about 20%): both were (0.494, 0.490).

## What we learned

1. **Raising K5's own probability of the paragraph makes it worse.** Every arm whose chain gained K5 bits lost on both judges (R1, R4). A sharper K5 is blander and copies more. Whole-paragraph search finds the same failure as a sharper sampler. This is prediction-is-not-generation again (§13.11).
2. **The entity grid is the term that helps.** As a whole-paragraph energy, the grid's full log probability improves SCRIBE's coherence axis (D_C 3.41 → 3.05) and raises v2. With a milder temperature (E1b), it raises v3.1 at every seed. In K5_DISCOURSE.md the grid only chooses among K5's sentences. Here it edits words inside them.
3. **The two judges pull apart.** The LDA and grid together (LE, R4X) raise v2 most (+0.04 to +0.05), but not v3.1. Annealing less raises v3.1 and lowers v2 (R2, E1c). Only E1b sits above M0 on both.
4. **Small edits break sentence form a little.** D_S rose in every chain (0.66 → 0.83). The sentence-level move (not built) may be needed to edit at the level where K5 fails.

## Requests for wip/k5-discourse and wip/k5-structure (their headers were not edited)

- `lda.hpp`: a `log_ratio` over ids (or a cached per-word ratio table for a theta) would save the string lookups. `k5_mcmc.hpp` builds its own table at each chain's start.
- `entity_grid.hpp`: an incremental `append_logp` for one changed sentence would let the chain rescore only the sentences an edit touches. Today it rescans the whole paragraph each step (cheap at about 6 sentences).
- `structure::Scorer`: it needs a trained MODEL file on the PC to run S1. Its `score` rebuilds `Experts` and tokenises the whole context per call, so a paragraph energy would call it once per sentence per step: about 6 calls a step. A version that takes K5 ids and reuses a prepared context would make S1 affordable.
