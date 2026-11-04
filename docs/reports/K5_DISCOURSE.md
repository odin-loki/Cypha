# K5's discourse helpers: minimum Bayes risk, an entity grid, an LDA topic model, and meta mixing

**Odin Loch** · 2026-11-05 · GENERATION_PLAN §13.32, session 2 (branch `wip/k5-discourse`, Odin's PC)

**Odin:** "More math like K5 that is synergistic with what we are doing … but is completely different maths." On the four writers: "I want to see that."

The arms, rules and predictions below were written and committed before any arm was read. Results are added under them.

## Why these helpers

- Four blind Claude raters gave the gated K5 2.09 of 10 and human text 8.50 (§13.31). K5 writes sentence-shaped text that does not connect.
- Every model mixed into K5's next-word choice so far made the text blander (K5_UPGRADES.md step 2, §13.24).
- SCRIBE v3's judges find K5's sentences share few content words with each other, and that K5 is too predictable.
- So these helpers mostly **choose and plan**: they pick among sentences or drafts K5 has already written. Only one arm (LDA rescaling) changes K5's next-word distribution.

## What was built

All in `cyphalm_trie sample --k5`. New code is in new headers under `native/include/cypha/lm/`, with small hooks in `word_kn.hpp`.

**`mbr.hpp`: minimum Bayes risk (decision theory).**
- K5 writes N drafts. The other drafts stand in for the distribution of texts K5 would write. The draft with the least expected loss is kept: the highest mean similarity to the others.
- The similarity is chrF: the F-score of character n-gram overlap, n = 1 to 6, spaces removed, recall weighted twice (beta 2). It uses no word vectors and no co-occurrence counts.
- Sentence level (`--k5-mbr-sent W`): a term in the sentence gate, over the gate's K candidates.
- Whole continuation (`--k5-mbr-doc N`): N whole drafts, one kept.

**`entity_grid.hpp`: an entity grid (Barzilay and Lapata, 2008).**
- A paragraph is a grid: a row per sentence, a column per entity. Each cell is the entity's role: S (subject), O (object), X (other) or − (absent).
- Each column is a Markov chain of order 2 over the four roles. The transition probabilities are counted on the training text, add-0.5 smoothed.
- No parser. Entities are runs of capitalised words, and the last word of a run of content words after a determiner. Roles go by position: after a preposition, X; else the sentence's first mention is S and later ones are O.
- In the gate (`--k5-eg W`), the history is the prompt's sentences plus the sentences kept so far. Two forms of the term (`--k5-eg-mode`):
  - **full:** the change in the grid's log probability when the candidate is appended. New entities cost their entry; dropped ones cost their absence.
  - **cont:** for each entity already in the grid that the candidate mentions, log P(role | history) − log P(− | history). New and dropped entities count 0.

**`lda.hpp`: an LDA topic model (Bayesian).**
- Collapsed Gibbs sampling, one document per line: 100 topics, alpha 0.1, beta 0.01, 200 sweeps, exact (one thread).
- Content words only: lower case, alphabetic, at least 3 letters, not a function word, seen at least 5 times, at most 50,000.
- The prompt's topic mixture theta is inferred by Gibbs sampling with the topics fixed: 50 sweeps, the last 30 averaged.
- **(a) Rescaling** (`--k5-lda-beta B`). For each content word, p ∝ p_K5 · (p_topic(w | theta) / p(w))^B, with the ratio clipped to [1/16, 16], renormalised. This is the only helper that changes K5's distribution.
- **(b) Sentence scoring** (`--k5-lda-w W`): in the gate, the mean of log2 (p_topic(w | theta) / p(w)) over the candidate's words.

**`--k5-meta SPEC`: meta mixing for writing (a count-based LLM-Blender).**
- Drafts from several K5 variants are pooled, and MBR picks one (the highest mean chrF to the pool).
- The variants are `base` (no gate), `gate` (today's K5), `eg` (the gate with the grid term), `topic` (the gate with the topic terms) and `all`.

**The gate with new terms.** With any new gate term on, each term is z-scored over the K candidates, then weighed:
score = wf·z(fluency) + wp·z(pmi) + mbr·z(chrF consensus) + eg·z(grid) + lda·z(topic) − wm·markup.
- The candidates come from the same random streams as today's gate, so only the choice differs.
- With a null term the choice equals today's gate's (checked by the selftest).

**Checks before any read:**
- `cyphalm_trie selftest` passes. It covers chrF, the grid's roles and sums, LDA's normalisation and determinism, the flags, the topic rescaling summing to one, and the gate and pool hooks.
- With the new flags off, `sample`'s GEN file and stdout on GEN-A dev are byte-identical to main's (`cmp`).
- Two smoke runs of 1 to 12 dev prompts checked speed only. They were not scored.

**Baseline on this PC** (main e9b9823, GEN-A dev, seed offset 0). It reproduces K5_UPGRADES.md and §13.28: SCRIBE v2 **0.498** [0.469, 0.527], v3.1 **0.483** [0.466, 0.500], copy32 0.0380, verbatim13 0 of 250. It took 2:48 wall for 250 prompts on 8 threads (sampling 83 s), peak 0.50 GB.

## Training text by size (cloud request, 2026-11-05)

- **K5 stays on its 7 MB `train.txt`.**
- The grid and LDA can learn from more text (`--k5-disc-text`):
  - K5's own 7 MB;
  - `train_30M.txt`, `train_100M.txt` and `train_all.txt` (330 MB clean enwik9; GEN-A and SCRIBE overlap removed, `docs/reports/K5_SCALE.md`).
- Trained models are cached (`--k5-disc-cache`).

## Setup

- Worktree `~/disc_wt`, binaries `~/cypha_build/disc` (g++ -O2). Arm files are in `/mnt/d/cypha_llm/disc/out/`, and every arm's GEN file is copied to `bench/k5_discourse/`.
- Every read is `scripts/gen_eval.sh dev LABEL OUT --k5 /mnt/d/cypha_llm/fuzzy/train.txt [flags]`, with `K5_THREADS=8`. It gives SCRIBE v2 and v3.1 (pooled row), copy32, verbatim13 and k5_cont_bpb.
- Each arm reports v2, v3.1, SURPRISE 3.1, MATCH 3.1, copy32, verbatim13, wall time per 250 prompts, and peak RSS.

## Arms (GEN-A dev, seed offset 0; all on K5's defaults: nucleus 0.8, 256 classes, gate 8 by pmi, unless stated)

**B: baselines.**
- B0: K5's default (above).
- B1: the gate off (`--k5-gate 1`), for reference.

**M: minimum Bayes risk.**
- M1: gate 8, pmi off, chrF consensus alone (`--k5-gate 8,wp=0 --k5-mbr-sent 1`).
- M2: gate 8, pmi + consensus (`--k5-mbr-sent 1`).
- M3: gate 16, pmi + consensus (`--k5-gate 16 --k5-mbr-sent 1`).
- M4: 8 whole drafts of today's K5, MBR choice (`--k5-mbr-doc 8`).
- M5: 8 whole drafts without the gate (`--k5-gate 1 --k5-mbr-doc 8`).

**E: entity grid** (learned from K5's 7 MB text).
- E1: pmi + grid, full (`--k5-eg 1`).
- E2: pmi + grid, cont (`--k5-eg 1 --k5-eg-mode cont`).
- E3: the better of E1 and E2's modes, at weight 2.
- E4: the best E arm, with the grid learned from 330 MB.

**T: LDA topics** (100 topics, learned from K5's 7 MB text).
- T1: rescaling B 0.5 (`--k5-lda 100 --k5-lda-beta 0.5`).
- T2: rescaling B 1.
- T3: pmi + topic term (`--k5-lda 100 --k5-lda-w 1`).
- T4: topic term alone (`--k5-gate 8,wp=0 --k5-lda-w 1`).
- T5: the best of T1–T4 with 200 topics.
- T6a/b/c: the best of T1–T5, with LDA learned from 30 MB, 100 MB and 330 MB.
- T7: the best rescaling and the best topic term together, if both beat B0 on v3.1.

**X: meta mixing and the combination.** These use the E and T settings chosen above.
- X1: `--k5-meta base:2,gate:2,eg:2,topic:2` (8 drafts).
- X2: `--k5-meta gate:4,eg:4,topic:4` (12 drafts).
- C: the chosen M, E and T gate terms together in one gate (z-scored), if each beat B0 on v3.1 alone.

## Choosing and confirming

- **Choice (seed 0):** in each family (M, E, T, X with C), the pick is the arm with the highest v3.1 among arms that keep three conditions:
  - v2 at least B0's − 0.02;
  - copy32 at most B0's + 0.02;
  - verbatim13 = 0.
- **Confirmation:** each family's pick and B0 run at 4 fresh seed offsets: 1000, 2000, 3000, 4000 (`--k5-seed`).
- **Rule.** An arm becomes K5's default only if all of these hold:
  1. its 4-seed mean v3.1 is above B0's, and its 4-seed range lies wholly above B0's range (intervals not overlapping);
  2. its 4-seed mean v2 is not more than 0.02 below B0's;
  3. verbatim13 is 0 at every seed, and mean copy32 rises by at most 0.02;
  4. the cloud session's 4 blind Claude raters, on 50 dev prompts, confirm it (their rule is the cloud's to register).
- **GEN-A test** is read once, and only for an adopted arm.
- Bits (k5_cont_bpb) are reported but do not decide.

## Predictions

- **P1 (MBR).** Consensus picks the most typical candidate. That is blander text.
  - M1/M2 raise copy32 (+0.005 to +0.02).
  - They move v2 by less than 0.02, and lower SURPRISE 3.1 (bland group B) and v3.1 by 0.00–0.03.
  - Whole-draft MBR (M4) moves v2 and v3.1 by less than 0.015. Whole drafts share little, so consensus mostly rewards common words.
  - M5 (MBR without the gate) stays below B0 on v2: MBR does not replace the gate.
- **P2 (entity grid).**
  - The full form favours sentences with few new entities. It lowers content-word variety: v3.1 within ±0.01, v2 +0.00–0.02.
  - The cont form rewards keeping the paragraph's entities. It raises SCRIBE's cohesion (D_C falls) and v3.1 by +0.005 to +0.02. This is the most likely helper to pass.
  - Learning the grid from 330 MB changes little: its 64 transition probabilities are already well estimated from 56,000 sentences. The change is |Δ v3.1| < 0.005.
- **P3 (LDA).**
  - Rescaling B 0.5 raises v2 (+0.01 to +0.03; word association, like the pmi gate) and changes v3.1 by less than 0.01. B 1 lowers v3.1 (SURPRISE's foreign-word group F catches words that fit the topic but not the sentence).
  - The topic term in the gate behaves like pmi: T3 within ±0.01 of B0, and T4 a little below B0.
  - 200 topics: within ±0.01 of 100. More text for LDA (30 MB to 330 MB): within ±0.01. GEN-A's prompts come from enwik8, whose topics the 7 MB text already covers.
- **P4 (meta).** The pool's MBR choice is within ±0.01 of its best single variant on v3.1. It does not beat all of them.
- **P5 (overall).**
  - No arm passes rule 1 (about 70%). The helpers move v3.1 by less than 0.02.
  - Whatever passes is a gate term (E cont, or T), not MBR.
  - None of them takes a reader from 2 to 6 of 10. Choosing among K5's own sentences cannot add meaning K5's sentences do not have.
- **P6 (cost).**
  - The gate terms add under 20% to the gate's time.
  - The LDA build adds about 60 s at 7 MB. The cache removes it after the first run.
  - Whole-draft MBR and meta cost N times the time of their drafts.
  - Peak RAM stays under 1 GB with the 7 MB helpers, and under 2 GB with the 330 MB ones.

## Amendment 1 (2026-11-05, after T1 and T2 were read, before T2b and T2c ran)

- T1 (B 0.5) and T2 (B 1) both raised v3.1, and T2 by far more. The best B may lie above 1, so two arms are added to the T family:
  - T2b: rescaling B 1.5;
  - T2c: rescaling B 2.
- They follow the same choice and confirmation rules. T5 (200 topics), T6 (training size) and T7 then use the best B among T1, T2, T2b and T2c.
- **Prediction:** v3.1 peaks at B 1–1.5. At B 2, copy32 passes the +0.02 bound or SURPRISE 3.1's foreign-word group pulls v3.1 down.
- The PC's load average was about 90 on 64 threads during these reads, from other sessions. Wall times are measured under that load.

## Results

(Added after the reads.)
