# K5 upgrades: sampler, online learning, longer contexts, classes, a sentence gate

**Odin Loch** · 2026-11-04 · after GENERATION_PLAN §13.11, §13.19 and §13.24 ("Let's just go with K5")

**Odin:** "Let's just go with K5. Do all of it and make sure you document everything we have tried and the results. Do it all on my PC."

The rules and predictions below were written and committed before any run of the arms they cover. Results are added under them.

## Setup (Odin's PC, WSL)

- Worktree `~/k5_wt`, branch `wip/k5-upgrades` from main 57bb1a3. Binaries in `~/cypha_build/k5`, built as `scripts/build_1tb_tools.sh` builds them (g++ -O2).
- K5's training text: the bake-off's clean prose, `/mnt/d/cypha_llm/fuzzy/train.txt` (7,053,155 bytes, 1,297 documents).
- SCRIBE v2: `scribe ladder --pack enwik:ENWIK8:16:48 --ref enwik:ENWIK8:48:56 --test enwik:ENWIK8:56:64 --vectors /mnt/d/cypha_llm/scribe/v32 --profile v2 --seed 20261027 --gen-trie FILE --label NAME`. The row read is "trie samples T=…".
- Prompts: GEN-A dev, `bench/eval/gen_prompts_dev.tsv`, all 250. 600 bytes per prompt.
- Each arm reports SCRIBE v2 [95%], surface S, copy32 (the share of bytes in a 32-byte span of the training text), verbatim13 (samples with 13 consecutive training tokens; must be 0 with the 12-word cap) and k5_cont_bpb (K5's bits per byte on the human continuations, word-level).

**Reproduction.** K5 on this PC: SCRIBE v2 **0.353 [0.318, 0.384]**, surface S 0.928, copy32 0.0351, verbatim13 0 of 250, k5_cont_bpb 1.549997. These are §13.19's numbers exactly.

## Seeds and the confirmation rule (for every step)

- Seed offset 0 is K5's own seeds (20261101 + prompt index). `--k5-seed N` adds N.
- **Choice:** each step picks its best arm on dev at seed offset 0.
- **Confirmation:** the pick and the step's baseline then run at 4 fresh seed offsets: 1000, 2000, 3000, 4000. The pick is adopted only if:
  1. its mean SCRIBE over the 4 seeds is at least 0.01 above the baseline's mean, and
  2. it beats the baseline at 3 or 4 of the 4 seeds, and
  3. verbatim13 is 0 at every seed, copy32 rises by at most 0.02, and surface S falls by at most 0.01 (means).
- The seed spread is reported for every confirmed pick.
- The 4 fresh seeds guard against picking noise: K5's own spread over seeds is about 0.31–0.35 (§13.19).

## Step 1: sampler tuning (registered before its runs)

**Flags:** `--temps T` (exists), `--k5-nucleus P`, `--k5-cand-mass P`, `--k5-cand-max N`, `--k5-seed N`.

**Arms** (seed offset 0):
- the grid of temperature {0.8, 0.9, 1.0, 1.1} × nucleus {0.8, 0.9, 0.95}, candidates cut at 0.995 (12 arms; T 1.0 / 0.9 is K5);
- then, at the grid's best T and nucleus, the candidate cut {0.99, 0.999, 1.0} (1.0: the 3,000 most probable, no mass cut).

**Rule:** the confirmation rule above, against K5.

**Predictions:**
- P1.1: lower temperature (0.8–0.9) or a lower nucleus raises SCRIBE by 0.01–0.04. Fewer tail words make the sentences more grammatical.
- P1.2: T 1.1 and nucleus 0.95 lower SCRIBE.
- P1.3: the candidate cut moves SCRIBE by less than 0.01 (the nucleus cuts first).
- P1.4: lower temperature raises copy32 (towards 0.05), still with 0 verbatim13.

## Step 2: K5 learns online while it writes (registered before its runs)

Both parts are per continuation. The shared model is never changed.

**(a) A word cache** (`--k5-cache W`, `--k5-cache-decay R`, `--k5-cache2 W2`).
- Unigram: every word of the prompt and of the text written so far, weighted R^(age in words), normalised. `<unk>` and `<s>` are left out.
- Bigram (W2): the same over (previous word, word) pairs; it counts only after a previous word that the cache has seen.
- Mix: p = (1 − W − W2′) p_K5 + W p_cache1 + W2′ p_cache2, before temperature. W2′ is W2 when the previous word has a cached bigram, else 0.

**(b) Dynamic counts** (`--k5-dyn L`).
- The prompt's n-grams (orders 2–5) and those of the text written so far are added to K5's counts, times L, for this continuation only.
- In each context, a dynamic count d is discounted by 0.5 and the discounted mass goes to the lower order, as in K5's own counts: P_k(w | c) = [(n(cw) − D) + L (d(cw) − 0.5)] / (n(c) + L d(c)) + backoff. Every distribution still sums to one (checked by the selftest).
- Where the main model has not seen a context but the dynamic counts have, the dynamic counts alone fill that order.

**Arms** (seed offset 0, with step 1's sampler):
- C1 unigram cache W 0.05, R 0.99; C2 W 0.1, R 0.99; C3 W 0.2, R 0.99; C4 W 0.1, R 0.95 (shorter memory); C5 W 0.1 + bigram W2 0.1, R 0.99;
- D1 dynamic counts L 1; D2 L 3; D3 L 10;
- CD: the best C arm with the best D arm.

**Rule:** the confirmation rule, against step 1's result.

**Predictions:**
- P2.1: the cache lowers k5_cont_bpb (−0.02 to −0.08). Human continuations reuse the prompt's words.
- P2.2: a small cache (W 0.05–0.1) raises SCRIBE by 0.01–0.03: it keeps the topic, which SCRIBE's coherence axis reads. W 0.2 lowers it (repetition).
- P2.3: dynamic counts lower k5_cont_bpb less than the cache (the prompts are short) and move SCRIBE by less than 0.02.
- P2.4: copy32 does not rise by more than 0.01 (the cache copies the prompt's words, not the training text).

## Step 3: order 6 and 7, and word classes for backoff (registered before its runs)

**Flags:** `--k5-order N` (5–7; the context keys widen from 64 to 128 bits, and the default answers are unchanged), `--k5-classes K`, `--k5-class-mode backoff|mix`, `--k5-class-w W`.

**Classes.** Martin, Liermann and Ney's exchange algorithm, as `bench/fuzzy/so_bench.py` arm SO1 runs it: words seen at least 3 times move to the class that most raises the class-bigram likelihood, 6 passes, most frequent first. Start: the words in frequency order dealt to the classes in turn. `<unk>` and `<s>` have their own classes. The class model: interpolated Kneser–Ney of order 3 over class ids, then P(w | class) by count. P_class(w | h) = P(c(w) | c(h−2), c(h−1)) P(w | c(w)).
- **backoff:** P_class replaces K5's order-1 distribution, the floor every backoff reaches. K5's discounts and backoff weights are unchanged, so where K5 has seen the context the class model only gets the backoff mass.
- **mix:** p = (1 − W) p_K5 + W p_class (SO1's form).

**Arms** (seed offset 0, with the sampler and online parts adopted so far):
- O6: order 6; O7: order 7;
- X1: 512 classes, backoff; X2: 256 classes, backoff; X3: 512 classes, mix W 0.1.

**Rule:** the confirmation rule, against the best so far. Bits are reported but do not decide (§13.11: prediction is not generation).

**Predictions:**
- P3.1: order 6 and 7 lower k5_cont_bpb by less than 0.005 (7 MB is too little for 6-grams) and move SCRIBE by less than 0.01. They raise copy32 a little; the cap holds verbatim13 at 0.
- P3.2: class backoff lowers k5_cont_bpb by 0.005–0.02 and moves SCRIBE by less than 0.02.
- P3.3: the class mix lowers bits more than backoff and lowers SCRIBE (as SO1: 0.294).

## Step 4: a sentence gate with K5 (GEN-B's idea; registered before its runs)

**Flag:** `--k5-gate K[,wf=..][,wp=..][,wm=..]`.
- For each sentence, K5 writes K candidates, each with its own random stream, from the same state (context, repeat ban, cache, dynamic counts).
- A candidate ends after ". ", "!", "?" or a line break once it holds at least 24 bytes, at 240 bytes, or at the 600-byte budget. The copy cap applies inside every candidate.
- The gate keeps the candidate with the highest wf × fluency + wp × pmi − wm × markup. Ties keep the lower k.
- Its features are CyphaLM's own. It reads nothing of SCRIBE (no lexicon, no vectors, no SCRIBE trigram):
  - **fluency:** the mean log2 probability of the candidate's words under K5 (before temperature, with the online parts in use);
  - **pmi:** the mean, over the candidate's words, of their association with the prompt's words. Association is positive PMI of co-occurrence within 10 words in K5's training text: PPMI(v, w) = max(0, log2 (c(v, w) N / (20 c(v) c(w)))), counted only if c(v, w) ≥ 2, averaged over the prompt's words v outside the 100 most frequent;
  - **markup:** the share of the candidate's bytes in `[]{}|=<>&*#`.
- **Arms** (seed offset 0): GB1 K 4, fluency (wf 1); GB2 K 4, pmi (wp 1); GB3 K 4, fluency + pmi (wf 1, wp 1); GB4 K 8 with the best weights. wm 5 in all.

**Rule:** the confirmation rule, against the best so far. If K 8 is within 0.005 of K 4 with the same weights, K 4 is chosen.

**Predictions:**
- P4.1: GB1 (fluency) lowers SCRIBE by 0.01–0.05: it prefers common, bland sentences and lowers word variety.
- P4.2: GB2 (pmi) raises SCRIBE by 0.01–0.03: it keeps the topic.
- P4.3: k5_cont_bpb is unchanged (the gate only chooses); copy32 rises by less than 0.01.

## Step 5: a content plan with K5 (GEN-E)

GEN-E's fact store (`seeds.tsv`, `facts_f2.tsv`) and its frames (`patterns.tsv`, `patterns_raw.tsv`) are not on this PC. They were built in the cloud (`/root/work/f2/v2/aug`, `/root/work/f2raw`). Searched: `/mnt/d/cypha_llm` and `/home/odin`, and the repository. **Step 5 is skipped.**

## Step 6: the combination, and the test read (registered before its runs)

- The adopted parts of steps 1–4 together, one dev read at seed offset 0, and the 4 confirmation seeds.
- **The test read** (`bench/eval/gen_prompts_test.tsv`, once) is made only if the combination's dev point at seed offset 0 is at least 0.373 (K5's 0.353 + 0.02), and it passed the confirmation rule. It is reported against K5's test 0.305 [0.276, 0.338].
- The new parts become K5's defaults only if adopted by their step's rule. A changed default is checked to reproduce the adopted arm exactly.
- Prediction P6.1: the combination reaches 0.36–0.39 on dev; whether it clears 0.373 is open.

## Results

All on GEN-A dev, 250 prompts. SCRIBE v2 is the pooled row (with one temperature it is that temperature's row; its bootstrap interval can differ from the T= row's in the third digit). k5_cont_bpb is K5's word-level bits per byte on the human continuations.

### Step 1: sampler tuning

**The grid** (seed offset 0, candidates cut at 0.995):

| Arm | SCRIBE v2 [95%] | surface S | copy32 | verbatim13 |
|---|---:|---:|---:|---:|
| T 0.8, nucleus 0.8 | 0.460 [0.433, 0.487] | 0.964 | 0.1658 | 0 |
| T 0.8, nucleus 0.9 | 0.416 [0.390, 0.445] | 0.962 | 0.1243 | 0 |
| T 0.8, nucleus 0.95 | 0.394 [0.368, 0.421] | 0.958 | 0.1070 | 0 |
| T 0.9, nucleus 0.8 | 0.430 [0.405, 0.459] | 0.960 | 0.0962 | 0 |
| T 0.9, nucleus 0.9 | 0.395 [0.366, 0.428] | 0.947 | 0.0667 | 0 |
| T 0.9, nucleus 0.95 | 0.351 [0.324, 0.378] | 0.940 | 0.0549 | 0 |
| **T 1.0, nucleus 0.8** | **0.395 [0.366, 0.424]** | 0.946 | 0.0481 | 0 |
| T 1.0, nucleus 0.9 (K5) | 0.353 [0.318, 0.384] | 0.928 | 0.0351 | 0 |
| T 1.0, nucleus 0.95 | 0.284 [0.253, 0.314] | 0.920 | 0.0289 | 0 |
| T 1.1, nucleus 0.8 | 0.314 [0.283, 0.345] | 0.920 | 0.0282 | 0 |
| T 1.1, nucleus 0.9 | 0.224 [0.194, 0.253] | 0.914 | 0.0177 | 0 |
| T 1.1, nucleus 0.95 | 0.172 [0.144, 0.201] | 0.903 | 0.0146 | 0 |

k5_cont_bpb is 1.549997 in every arm (the sampler does not change the model).

- **SCRIBE rises with copying.** Every step towards a sharper sampler raises SCRIBE, surface S and copy32 together. The cap holds verbatim13 at 0 everywhere, but at T 0.8 / nucleus 0.8 a sixth of the bytes lie in 32-byte spans of the training text (0.166 against K5's 0.035).
- **The copy bound decides.** The rule allows copy32 to rise by at most 0.02 (to 0.055). Only T 1.0 / nucleus 0.8, T 0.9 / nucleus 0.95 and the softer arms stay inside it. Arms that break a bound cannot be adopted, so the choice is the best arm inside the bounds: **T 1.0, nucleus 0.8** (0.395). This reading of the rule is stated here because the registration did not say it in so many words.

**The candidate cut** (T 1.0, nucleus 0.8):

| Cut | SCRIBE v2 [95%] | surface S | copy32 |
|---|---:|---:|---:|
| 0.99 | 0.395 [0.367, 0.426] | 0.946 | 0.0481 |
| 0.995 (K5's) | 0.395 [0.366, 0.424] | 0.946 | 0.0481 |
| 0.999 | 0.392 [0.363, 0.423] | 0.946 | 0.0481 |
| 1.0 (top 3,000 only) | 0.392 [0.363, 0.423] | 0.946 | 0.0481 |

The cut does not matter: 0.995 stays.

**Confirmation** (seed offsets 1000–4000):

| Seed offset | K5 | nucleus 0.8 |
|---|---:|---:|
| 1000 | 0.362 | 0.372 |
| 2000 | 0.337 | 0.396 |
| 3000 | 0.323 | 0.399 |
| 4000 | 0.312 | 0.397 |
| mean | 0.334 | **0.391** |
| surface S (mean) | 0.928 | 0.942 |
| copy32 (mean) | 0.0358 | 0.0531 |

- +0.057, 4 of 4 seeds, copy32 +0.017 (inside 0.02), surface S +0.014, verbatim13 0 at every seed. **Adopted: nucleus 0.8.** Spread over the 5 seeds: 0.372–0.399.
- Predictions: P1.1 hit (a lower nucleus +0.04; T 0.8–0.9 +0.04–0.11, more than predicted). P1.2 hit. P1.3 hit (≤ 0.003). P1.4 hit, and it is the binding constraint: copy32 reaches 0.10–0.17 at T 0.8.
- Steps 2–4 run with nucleus 0.8.

### Step 2: K5 learns online while it writes

Seed offset 0, nucleus 0.8 (step 1). The baseline is T 1.0 / nucleus 0.8: 0.395 [0.366, 0.424], k5_cont_bpb 1.549997.

| Arm | SCRIBE v2 [95%] | surface S | copy32 | verbatim13 | k5_cont_bpb |
|---|---:|---:|---:|---:|---:|
| nucleus 0.8 (baseline) | **0.395** [0.366, 0.424] | 0.946 | 0.0481 | 0 | 1.5500 |
| C1 cache W 0.05, R 0.99 | 0.359 [0.325, 0.393] | 0.949 | 0.0411 | 0 | 1.5020 |
| C2 cache W 0.1, R 0.99 | 0.284 [0.248, 0.323] | 0.961 | 0.0362 | 0 | 1.4993 |
| C3 cache W 0.2, R 0.99 | 0.114 [0.089, 0.140] | 0.959 | 0.0216 | 0 | 1.5073 |
| C4 cache W 0.1, R 0.95 | 0.261 [0.227, 0.298] | 0.966 | 0.0344 | 0 | 1.5055 |
| C5 cache W 0.1 + bigram 0.1 | 0.279 [0.242, 0.313] | 0.961 | 0.0286 | 0 | **1.4885** |
| D1 dynamic counts L 1 | 0.394 [0.367, 0.422] | 0.942 | 0.0511 | 0 | 1.5337 |
| D2 dynamic counts L 3 | 0.384 [0.356, 0.413] | 0.942 | 0.0532 | 0 | 1.5308 |
| D3 dynamic counts L 10 | 0.375 [0.345, 0.406] | 0.940 | 0.0504 | 0 | 1.5279 |
| CD cache 0.05 + dynamic L 1 | 0.358 [0.322, 0.391] | 0.951 | 0.0401 | 0 | 1.4936 |

- **Every online part lowers bits and none raises SCRIBE.** The cache gains up to 0.061 bits per byte (C5), the largest gain in this report. SCRIBE falls as the cache weight rises: 0.359, 0.284, 0.114.
- **Why the cache hurts.** A unigram cache puts mass on the prompt's words in every slot, whatever the syntax. Its most frequent entries are function words and punctuation, so the text gets ",,", ".." and "Some of the in - group of the other" (C2, prompt 0). Bigram pairs (C5) help bits but not text.
- **Dynamic counts are nearly neutral.** D1 ties the baseline at seed 0 (0.394). Over the 4 confirmation seeds it loses: 0.382, 0.382, 0.387, 0.370 (mean 0.380) against 0.391; −0.011, 1 of 4 seeds. Not adopted.
- **Nothing from step 2 is adopted.** The flags stay.
- Predictions: P2.1 hit (−0.048 to −0.061). P2.2 miss (W 0.05 already −0.036). P2.3 hit (dynamic −0.016 to −0.022, SCRIBE within 0.02). P2.4 hit (copy32 falls with the cache).
- This is §13.11's finding again: what lowers bits does not raise SCRIBE.

### Step 3: order 6 and 7, and exchange classes

Seed offset 0, nucleus 0.8. The baseline is nucleus 0.8 alone (0.395).

| Arm | SCRIBE v2 [95%] | surface S | copy32 | verbatim13 | k5_cont_bpb |
|---|---:|---:|---:|---:|---:|
| nucleus 0.8 (baseline, order 5) | 0.395 [0.366, 0.424] | 0.946 | 0.0481 | 0 | 1.549997 |
| O6 order 6 | 0.349 [0.323, 0.378] | 0.937 | 0.0527 | 0 | 1.549850 |
| O7 order 7 | 0.409 [0.380, 0.441] | 0.941 | 0.0521 | 0 | 1.549843 |
| X1 512 classes, backoff | 0.402 [0.377, 0.429] | 0.943 | 0.0531 | 0 | 1.540610 |
| **X2 256 classes, backoff** | **0.444 [0.413, 0.477]** | 0.944 | 0.0557 | 0 | 1.532327 |
| X3 512 classes, mix W 0.1 | 0.378 [0.351, 0.405] | 0.942 | 0.0473 | 0 | 1.526833 |

- **Orders 6 and 7 add nothing.** They gain 0.00015 bits per byte: 7 MB of text has almost no repeated 6-grams. Their SCRIBE (0.349, 0.409) differs from order 5 by draw noise only: once a draw differs, the rest of the text differs. That 0.06 gap between two models with the same bits is a direct look at the noise of one seed.
- **Classes.** The exchange algorithm runs 6 passes in C++ (33 s for 256 classes, 116 s for 512). The class-bigram log-likelihood rises from −18.68 M to −17.63 M (256) and from −18.36 M to −17.39 M (512).
- **The class model as K5's floor helps; as a mix it does not.** X2 gains 0.018 bits and the most SCRIBE. The mix (X3) gains more bits (0.023) and loses SCRIBE, as SO1 did (§13.11). Backoff only changes the words K5 has not seen in a context, where the unigram floor gave them by frequency alone. The classes give them by what kind of word fits after the last two words' classes.

**Confirmation of X2** (seed offsets 1000–4000):

| Seed offset | nucleus 0.8 | + 256 classes, backoff |
|---|---:|---:|
| 1000 | 0.372 | 0.406 |
| 2000 | 0.396 | 0.428 |
| 3000 | 0.399 | 0.422 |
| 4000 | 0.397 | 0.423 |
| mean | 0.391 | **0.420** |
| surface S (mean) | 0.942 | 0.943 |
| copy32 (mean) | 0.0531 | 0.0518 |

- +0.029, 4 of 4 seeds, copy32 −0.001, surface S +0.002, verbatim13 0. **Adopted: 256 exchange classes as the backoff floor.** Spread over the 5 seeds: 0.406–0.444.
- Predictions: P3.1 hit (−0.00015 bits; SCRIBE moved by noise, both ways). P3.2 hit on bits (−0.018, −0.009) but missed on SCRIBE: +0.049 at seed 0 and +0.029 over 4 seeds, more than 0.02. P3.3 hit (the mix gains the most bits, −0.023, and loses SCRIBE).
- Step 4 runs with nucleus 0.8 and the 256 classes.

### Step 4: a sentence gate with K5

Seed offset 0, nucleus 0.8, 256 classes as the backoff floor (steps 1 and 3). The baseline is X2 (0.444). wm 5 in every arm.

| Arm | SCRIBE v2 [95%] | surface S | copy32 | verbatim13 | k5_cont_bpb |
|---|---:|---:|---:|---:|---:|
| X2 (baseline, no gate) | 0.444 [0.413, 0.477] | 0.944 | 0.0557 | 0 | 1.532327 |
| GB1 K 4, fluency | 0.431 [0.405, 0.459] | 0.950 | 0.0902 | 0 | 1.532327 |
| GB2 K 4, pmi | 0.489 [0.458, 0.517] | 0.948 | 0.0419 | 0 | 1.532327 |
| GB3 K 4, fluency + pmi | 0.432 [0.406, 0.460] | 0.951 | 0.0892 | 0 | 1.532327 |
| **GB4 K 8, pmi** | **0.498 [0.469, 0.527]** | 0.946 | 0.0380 | 0 | 1.532327 |

SCRIBE's distances (lower is better; human 0.54 in all; sentences shuffled across documents: D_S 0.02, D_C 5.58):

| Arm | mean D | D_S (sentence form) | D_C (coherence) | share D_S > 0 |
|---|---:|---:|---:|---:|
| K5 | 7.41 | 3.15 | 4.26 | 43% |
| nucleus 0.8 | 5.75 | 1.85 | 3.90 | 34% |
| X2 (+ classes) | 5.03 | 0.88 | 4.15 | 21% |
| GB1 (fluency gate) | 4.82 | 0.78 | 4.04 | 23% |
| GB2 (pmi gate, K 4) | 4.31 | 0.75 | 3.52 | 21% |
| GB4 (pmi gate, K 8) | 4.08 | 0.66 | 3.41 | 20% |

- **The fluency gate copies.** Keeping K5's most probable sentence picks phrases it has seen: copy32 rises to 0.09 (+0.035, past the 0.02 bound), and SCRIBE does not rise. GB3 (fluency + pmi) behaves like GB1: the fluency term dominates the scale.
- **The pmi gate keeps the topic, and copies less.** It cuts SCRIBE's coherence distance (D_C 4.15 → 3.41) and copy32 (0.056 → 0.038). K 8 is 0.009 above K 4, more than the 0.005 margin, so K 8 is the pick.
- The gate chooses only: k5_cont_bpb is unchanged. It costs about 4× (K 4) and 8× (K 8) the CPU per word.

**Confirmation of GB4** (seed offsets 1000–4000):

| Seed offset | X2 | GB4 |
|---|---:|---:|
| 1000 | 0.406 | 0.496 |
| 2000 | 0.428 | 0.501 |
| 3000 | 0.422 | 0.505 |
| 4000 | 0.423 | 0.493 |
| mean | 0.420 | **0.499** |
| surface S (mean) | 0.943 | 0.947 |
| copy32 (mean) | 0.0518 | 0.0444 |

- +0.079, 4 of 4 seeds, copy32 −0.007, surface S +0.004, verbatim13 0. **Adopted: the 8-candidate gate by prompt association** (`--k5-gate 8,wf=0,wp=1,wm=5`). Spread over the 5 seeds: 0.493–0.505.
- Predictions: P4.1 hit in direction (GB1 −0.013), and its copying was not foreseen. P4.2 miss on size: +0.045 (K 4), more than the 0.01–0.03 predicted. P4.3 half: bits unchanged; copy32 fell with pmi and rose 0.035 with fluency.
- **A caution on the judge.** The pmi gate reads co-occurrence in K5's training text. SCRIBE's coherence axis reads word vectors, which also come from co-occurrence. The gate reads nothing of SCRIBE, but both measure the same kind of thing, so part of this gain may be agreement between two co-occurrence measures rather than better text. A judge of another kind (a human read, or Q0's neural yardstick) should check it.

### Step 5: a content plan with K5 (GEN-E)

Skipped: GEN-E's fact store and frames are not on this PC (see the registration above).

### Step 6: the combination and the test read

**The combination** is steps 1, 3 and 4 together: nucleus 0.8, 256 exchange classes as the backoff floor, and the 8-candidate pmi gate. That is GB4 exactly. Its dev point is **0.498 [0.469, 0.527]** at seed offset 0, and 0.493–0.505 over 4 more seeds (mean 0.499). It clears the 0.373 bar and passed every confirmation, so the test read was made.

**The defaults.** These are now K5's defaults in `word_kn.hpp`. Checks:
- `sample --k5 train.txt` (no other flag) writes GB4's text line for line (all 250 GEN lines identical).
- `--k5-base k5` restores §13.19's K5: its GEN lines are identical to K5's dev file (0.353).
- Only the MODEL line differs, because it now records any setting that differs from the defaults.

**The test read** (`bench/eval/gen_prompts_test.tsv`, 250 prompts, read once):

| | SCRIBE v2 [95%] | surface S | copy32 | verbatim13 | k5_cont_bpb |
|---|---:|---:|---:|---:|---:|
| K5 (§13.19, test) | 0.305 [0.276, 0.338] | — | — | 0 | — |
| **K5 upgraded (test)** | **0.524 [0.493, 0.552]** | 0.950 | 0.0456 | 0 (0 of 250) | 1.537654 |
| byte sampler (test) | 0.147 | — | — | — | — |
| human text | 0.923 [0.910, 0.935] | 0.912 | — | — | — |

- +0.219 over K5 on test, with disjoint intervals. Distances on test: mean D 3.76, D_S 0.70, D_C 3.06.
- SCRIBE's "sentences shuffled across documents" rung is 0.344, so the upgraded K5 now scores above shuffled human sentences, but far below human text (0.923).
- Prediction P6.1 missed: the combination reached 0.498 on dev, above the 0.36–0.39 predicted.
- Cost: building takes 26–33 s (the exchange classes), and writing 250 samples takes 90–160 s on 4 threads (the gate writes 8 candidates per sentence), against 1 s and 12 s for K5.

## Summary: every arm

GEN-A dev, 250 prompts, seed offset 0 unless stated. "+" adds to the arm above it in its step.

| Step | Arm | SCRIBE v2 [95%] | surface S | copy32 | verbatim13 | k5_cont_bpb | Result |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | K5 (reproduced) | 0.353 [0.318, 0.384] | 0.928 | 0.0351 | 0 | 1.549997 | baseline |
| 1 | T 0.8, nucleus 0.8 | 0.460 [0.433, 0.487] | 0.964 | 0.1658 | 0 | 1.549997 | copy bound broken |
| 1 | T 0.8, nucleus 0.9 | 0.416 [0.390, 0.445] | 0.962 | 0.1243 | 0 | 1.549997 | copy bound broken |
| 1 | T 0.8, nucleus 0.95 | 0.394 [0.368, 0.421] | 0.958 | 0.1070 | 0 | 1.549997 | copy bound broken |
| 1 | T 0.9, nucleus 0.8 | 0.430 [0.405, 0.459] | 0.960 | 0.0962 | 0 | 1.549997 | copy bound broken |
| 1 | T 0.9, nucleus 0.9 | 0.395 [0.366, 0.428] | 0.947 | 0.0667 | 0 | 1.549997 | copy bound broken |
| 1 | T 0.9, nucleus 0.95 | 0.351 [0.324, 0.378] | 0.940 | 0.0549 | 0 | 1.549997 | |
| 1 | **T 1.0, nucleus 0.8** | 0.395 [0.366, 0.424] | 0.946 | 0.0481 | 0 | 1.549997 | **adopted** (4 seeds +0.057) |
| 1 | T 1.0, nucleus 0.95 | 0.284 [0.253, 0.314] | 0.920 | 0.0289 | 0 | 1.549997 | |
| 1 | T 1.1, nucleus 0.8 | 0.314 [0.283, 0.345] | 0.920 | 0.0282 | 0 | 1.549997 | |
| 1 | T 1.1, nucleus 0.9 | 0.224 [0.194, 0.253] | 0.914 | 0.0177 | 0 | 1.549997 | |
| 1 | T 1.1, nucleus 0.95 | 0.172 [0.144, 0.201] | 0.903 | 0.0146 | 0 | 1.549997 | |
| 1 | nucleus 0.8, cut 0.99 | 0.395 [0.367, 0.426] | 0.946 | 0.0481 | 0 | 1.549997 | no effect |
| 1 | nucleus 0.8, cut 0.999 | 0.392 [0.363, 0.423] | 0.946 | 0.0481 | 0 | 1.549997 | no effect |
| 1 | nucleus 0.8, cut 1.0 | 0.392 [0.363, 0.423] | 0.946 | 0.0481 | 0 | 1.549997 | no effect |
| 2 | C1 cache 0.05 | 0.359 [0.325, 0.393] | 0.949 | 0.0411 | 0 | 1.501961 | worse |
| 2 | C2 cache 0.1 | 0.284 [0.248, 0.323] | 0.961 | 0.0362 | 0 | 1.499266 | worse |
| 2 | C3 cache 0.2 | 0.114 [0.089, 0.140] | 0.959 | 0.0216 | 0 | 1.507327 | worse |
| 2 | C4 cache 0.1, R 0.95 | 0.261 [0.227, 0.298] | 0.966 | 0.0344 | 0 | 1.505526 | worse |
| 2 | C5 cache 0.1 + bigram 0.1 | 0.279 [0.242, 0.313] | 0.961 | 0.0286 | 0 | 1.488458 | worse |
| 2 | D1 dynamic L 1 | 0.394 [0.367, 0.422] | 0.942 | 0.0511 | 0 | 1.533687 | 4 seeds −0.011 |
| 2 | D2 dynamic L 3 | 0.384 [0.356, 0.413] | 0.942 | 0.0532 | 0 | 1.530831 | worse |
| 2 | D3 dynamic L 10 | 0.375 [0.345, 0.406] | 0.940 | 0.0504 | 0 | 1.527920 | worse |
| 2 | CD cache 0.05 + dynamic 1 | 0.358 [0.322, 0.391] | 0.951 | 0.0401 | 0 | 1.493595 | worse |
| 3 | O6 order 6 | 0.349 [0.323, 0.378] | 0.937 | 0.0527 | 0 | 1.549850 | noise |
| 3 | O7 order 7 | 0.409 [0.380, 0.441] | 0.941 | 0.0521 | 0 | 1.549843 | noise |
| 3 | X1 512 classes, backoff | 0.402 [0.377, 0.429] | 0.943 | 0.0531 | 0 | 1.540610 | |
| 3 | **X2 256 classes, backoff** | 0.444 [0.413, 0.477] | 0.944 | 0.0557 | 0 | 1.532327 | **adopted** (4 seeds +0.029) |
| 3 | X3 512 classes, mix 0.1 | 0.378 [0.351, 0.405] | 0.942 | 0.0473 | 0 | 1.526833 | worse |
| 4 | GB1 gate K 4, fluency | 0.431 [0.405, 0.459] | 0.950 | 0.0902 | 0 | 1.532327 | copy bound broken |
| 4 | GB2 gate K 4, pmi | 0.489 [0.458, 0.517] | 0.948 | 0.0419 | 0 | 1.532327 | |
| 4 | GB3 gate K 4, fluency + pmi | 0.432 [0.406, 0.460] | 0.951 | 0.0892 | 0 | 1.532327 | copy bound broken |
| 4 | **GB4 gate K 8, pmi** | **0.498 [0.469, 0.527]** | 0.946 | 0.0380 | 0 | 1.532327 | **adopted** (4 seeds +0.079) |
| 6 | the new default (= GB4), **test** | **0.524 [0.493, 0.552]** | 0.950 | 0.0456 | 0 | 1.537654 | K5 test 0.305 |

Every arm kept the 12-word cap: 0 samples over it, in every arm and seed (verbatim13 0).

## What we learned

1. **Prediction is not generation, again.** The cache gains the most bits (−0.061 bits per byte) and loses the most SCRIBE. The class mix gains more bits than class backoff and loses SCRIBE. The dynamic counts gain bits and lose a little.
2. **What helped changes how K5 chooses, or where its mass goes when it knows nothing.** A tighter nucleus (fewer tail words) mostly fixes sentence form. The class floor gives unseen continuations a word of the right kind instead of a frequent word. The gate keeps the subject.
3. **Copying is the price of a sharper sampler.** Every sharper sampler raised copy32. The copy bound (+0.02) is what stopped T 0.8. The fluency gate copies too. The pmi gate copies less than no gate.
4. **One seed is noisy.** Two models with equal bits (orders 6 and 7) differed by 0.06 SCRIBE. Every adoption here needed 4 fresh seeds, and every adopted arm won at all 4.

## For the merge (notes for the cloud session)

- Branch `wip/k5-upgrades`. Code: `native/include/cypha/lm/word_kn.hpp` (most of it), a few lines in `native/tools/lm/cyphalm_trie.cpp` (every other `--k5-…` flag goes to `wkn::apply_flag`; a comment), `bench/k5up/arm.sh` and `bench/k5up/confirm.py`. `cyphalm_trie selftest` passes.
- **K5's default output changed** (adopted by the rules above). `--k5-base k5` (put it first) restores §13.19's K5 exactly. Anything that compares against K5's old default file should add `--k5-base k5`.
- `sample --k5` now takes about 30 s to build (the exchange classes; `--k5-classes 0` skips them) and about 8× the CPU to write. `K5_THREADS=N` sets its threads (default 4).
- WordKN's context keys are now 128 bits (for orders 6 and 7). The order-5 model is unchanged: the same entries, and identical text.
- `--k5-lstm` with the new default: the LSTM rerank does not combine with the gate. Use `--k5-gate 1` (or `--k5-base k5`) with `--k5-lstm`.
- The pmi gate and SCRIBE's coherence axis both measure co-occurrence (see the caution in step 4). A judge of another kind should check the gain.
- GENERATION_PLAN.md, PLAN.md and CHANGELOG.md were not edited.
- Data on Odin's PC: every arm's files (`NAME.gen.txt`, `.log`, `.eval`, `.scribe.md`) and `results.tsv` are in `/mnt/d/cypha_llm/k5up/`.
