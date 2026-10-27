# SCRIBE v2

**Odin Loch** · 2026-10-28 · draft, filled in as the study runs (plan: [`../SCRIBE_PAPER_PLAN.md`](../SCRIBE_PAPER_PLAN.md))

## Freeze record (written before the fresh region was read)

v2's open-continuation profile was designed on development ladders A (enwik8 [56, 64) MiB, seed 20261027, v0 and v1's development ladder) and B ([64, 72) MiB, seed 31415926, v1's fresh ladder, now development data). Pack [16, 48) MiB, reference [48, 56) MiB, PPMI-SVD vectors on the pack. It was frozen at this commit, before any v2 run on [72, 95.37) MiB.

**Frozen design** (`ScribeV2Scorer`, `V2Params` defaults, in `native/include/cypha/eval/scribe.hpp`):

| Part | Setting |
|---|---|
| z-score | Mid-rank ECDF of the human reference samples of the length bucket inside the 2nd–98th percentile (`z = Φ⁻¹(F)`), linear tails outside it with the scale of the outer quantiles; a statistic that ≥ 95% of human samples have and the sample lacks counts as z = 4 on its bad side |
| Evidence | `e = max(0, u − 1)`, `u` = bad-side `|z|` |
| L (lexical validity) | s1 |
| S (syntax) | s2g, s3g (context gain of the skeleton and word trigrams over unigram frequency), s4 (well-formedness, tail floor 0.15) |
| C (coherence, one-sided) | c0 word cohesion, c1 all-pairs and c1a adjacent sentence cosine, c2 lexical cohesion (IDF-weighted shared content words), c3 topic drift |
| I (integrity) | loop8, repeated sentences, distinct-1..4, i1 (word 8-grams found in the pack) |
| Reported, not scored | s2, s3, s3min, sentence lengths, word rarity |
| D | `30·sat(L − thL) + 30·sat(S − thS) + 3·max(0, I − thI) + C`, `sat(x) = 1 − exp(−max(0, x)/4)`, thresholds = 99th percentiles of L, S and I over the reference samples |
| Score | `SCRIBE = exp(−D/5)` |

**Development results** (adjacent-pair ordering accuracy; all intervals disjoint):

| Pair | dev A | dev B |
|---|---|---|
| human > sentences shuffled | 0.974 | 0.983 |
| sentences shuffled > words shuffled | 0.998 | 0.994 |
| words shuffled > random lexicon words | 0.996 | 1.000 |
| random lexicon words > random letters | 1.000 | 1.000 |
| random letters > one phrase repeated | 1.000 | 1.000 |

Of 2,592 settings of the combination's free parameters tried on both development ladders, 1,729 pass all five pairs on both; every one-step change of a frozen parameter passes. The design steps that led here (and the failures on the way) are recorded below.

## Design log (development data only)

1. **v2 as first written** (sum of squared excess z over all statistics, coherence two-sided): 2 of 5 pairs on dev A. One weak sub-score on a human sample still dragged it down, and one failed well-formedness check (an unbalanced quote) put a human sample at z ≈ −5.
2. **Coherence one-sided.** The all-pairs sentence cosine alone separates human from shuffled sentences at 0.989 (dev A) when low cohesion is the bad side, against 0.972 two-sided. Too much cohesion from repetition is the I axis's job.
3. **No additive weighting passes.** Raising form's weight to order "shuffled sentences > shuffled words" also raises the form noise of human samples, which breaks "human > shuffled sentences". Across 2,592 additive settings the best minimum pair accuracy was 0.60.
4. **Categorical form and integrity.** Counting form and integrity only beyond a high human quantile, and coherence from the dead zone on, gave 4 of 5; the remaining conflict was between the size of the form penalty and the coherence signal inside word salad.
5. **Context gain.** The word trigram's log-probability minus the word's unigram log-probability (and the same for the skeleton model) measures how much the context predicts each word. Rare-topic human text keeps that gain, word salad loses it. Shuffled words moved from a median bad-side z of 3.1–3.5 (raw log-probability) to 6.8–8.8, with none inside the human 99.5th percentile.
6. **Two form failures.** Once syntax saturates, real words and non-words got the same penalty. Lexical validity (s1) and syntax (s2g, s3g, s4) became separate saturating failures. That passes all five pairs on both ladders, robustly.

## Fresh verdict (read once, 2026-10-28)

enwik8 [72, 95.37) MiB, seed 1123581321, 500 standard human samples from 2,409 candidates, the frozen v2 above, same pack and reference. Command: `scribe ladder --pack enwik:enwik8:16:48 --ref enwik:enwik8:48:56 --test enwik:enwik8:72:95.37 --vectors v32 --profile v2 --seed 1123581321 --gen-trie gen_fresh.txt --label wiki_fresh --export wiki_fresh` (20 s, 415 MB peak).

| Upper > lower | SCRIBE v2 | SCRIBE v1 | surface S |
|---|---|---|---|
| human > sentences shuffled | **0.975 PASS** | 0.888 | 0.627 |
| sentences shuffled > words shuffled | **0.999 PASS** | 0.806 | 0.341 |
| words shuffled > random lexicon words | **0.998 PASS** | 0.696 | 0.750 |
| random lexicon words > random letters | **0.998 PASS** | 1.000 PASS | 1.000 PASS |
| random letters > one phrase repeated | **1.000 PASS** | 0.999 PASS | 0.559 |
| **pairs passed** | **5 of 5** | 2 of 5 | 1 of 5 |

All v2 intervals are disjoint. Rung means (SCRIBE v2, 95% interval): human 0.916 [0.902, 0.930]; sentences shuffled 0.354 [0.341, 0.366]; words shuffled 0.003; random lexicon words 0.001; random letters 0.000 (mean D 53.5); one phrase repeated 0.000 (mean D 305.8). Trie samples 0.017 [0.012, 0.024], between words shuffled and random letters. Short human samples 0.954, standard 0.916.

V4 on this ladder: Pearson of SCRIBE with bytes 0.009 (standard), 0.041 (short), −0.124 pooled (Spearman 0.010 pooled): short samples score a little higher than standard ones.
