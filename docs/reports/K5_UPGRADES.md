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

(Added below as the runs finish.)
