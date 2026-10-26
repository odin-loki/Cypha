# SCRIBE v1: second control ladder, with real model output

**Odin Loch** · 2026-10-27

**Questions.**
1. After the v0 fixes (real model output, C6 sentence vectors, an I gate, a better E null, length parity), does statistical SCRIBE order the PLAN 6.4 V1 ladder on a fresh set of human text?
2. Does the old surface score `S = H·D·L` do any better?
3. Is SCRIBE stable against sample length on human text (V4)?

**Answers.**
1. No. On the fresh ladder (human text from [64, 72) MiB of enwik8, new seed, not tuned on) SCRIBE passes **3 of 6** adjacent pairs, up from 1 of 6 in v0. It passes human > trie samples, random lexicon words > random letters, and random letters > one repeated phrase. It fails the other three: trie > sentences shuffled (0.035), sentences shuffled > words shuffled (0.829) and words shuffled > random lexicon words (0.722). SCRIBE v1 must not judge anything yet.
2. No, and by less. Surface `S` passes **2 of 6** (human > trie at 0.963, and random lexicon > random letters). It rates random lexicon words 0.899 and word-shuffled text 0.926, as high as held-out human text (0.919), and a repeated phrase is not separated from random letters (0.556).
3. Yes, for length. The correlation of SCRIBE with length on human text is 0.023 (400–600 B), 0.107 (60–260 B) and −0.009 pooled (v0: −0.221 pooled), and short human text now scores 1.02 times standard human text (v0: 1.18). Seed variance is small. The E null (set level) is still not calibrated on every region of the corpus.

One of the three failed pairs is not the metric's fault: the order-4 trie writes word salad, which really is worse than shuffled human sentences, so "trie > sentences shuffled" cannot hold for a correct metric on this model (see "What still fails").

Code: [`native/include/cypha/eval/scribe.hpp`](../../native/include/cypha/eval/scribe.hpp), [`native/tools/scribe_ladder.cpp`](../../native/tools/scribe_ladder.cpp), [`native/tools/trie_sample.cpp`](../../native/tools/trie_sample.cpp). Spec: [`../PLAN.md`](../PLAN.md) section 6. Previous report: [`SCRIBE_V0.md`](SCRIBE_V0.md).

---

## What changed from v0

| v0 next step | Done in v1 |
|---|---|
| 1. Real model output | `trie_sample` trains the packed trie of `trie_chat.cpp` on wiki bytes only, [0, 7,340,032) of enwik8, no Dolly, then samples 600-byte continuations of 64-byte paragraph starts from its holdout [7,340,032, 8,388,608) with nucleus 0.9 at T = 0.6, 0.8 and 1.0 (same 11-byte no-repeat rule as `trie_chat`; the match expert may only copy from before 7,340,032). 250 prompts per file, 3 temperatures each: 750 continuations for development and 750 for the fresh ladder, with disjoint prompts. Whole run: about 90 s. |
| 2. Coherence | PPMI-SVD word vectors (`ppmi_svd`, 300 d, on enwik8 [16, 48) MiB only: 3.79 M tokens, top 50,000 words), centred. Sentence vectors by SIF weighting (a = 1e-3) minus the first principal component (fitted on 15,680 training sentences). **C1** is now the mean cosine over **all** sentence pairs. New **C0** is the mean cosine over all pairs of distinct content words in the sample. C3 is kept. |
| 3. Gate on I | `I < 0.2` caps the sample at I, next to the S gate. |
| 4. E null | E3 is now the Kolmogorov distance between the set's sentence lengths and the human length distribution. The null is 300 human sets drawn from a random third of 20 disjoint stretches of the training range, so its width includes variation between stretches. The length distribution comes from the reference range [48, 56) MiB, not from the stretches (in v0 and in my first attempt the null sets were part of the data that defined the reference, which made them too similar to it). |
| 5. Length parity | C is defined for short samples. A sample with fewer than two embeddable sentences is cut into two halves of its tokens, and C1, C3 use the halves. C0 and C3 need only 4 content words. |

Other changes: the model rung is cleaned by `clean_generated_text` (same markup stripping as for human text; stray markup characters become spaces; the cut first word, the cut last word and the rest of the prompt's sentence are dropped when there is a sentence boundary to cut at). A text with no boundary at all is kept whole, so that S4 scores it. Only texts under 100 bytes after cleaning are dropped: 127 of 750 in each file (17%). That favours the trie slightly.

Everything else is as in v0: S1, S3, S4, I2, I3, E1, the per-bucket percentile scoring, the weights S 2, C 2, I 1, and the S gate at 0.2.

**What I tried and did not adopt** (on the development ladder):

| Variant | Effect (development ladder, pooled trie rung) |
|---|---|
| C scored one-sided instead of the PLAN's two-sided | human > sentences shuffled 0.876 → 0.898, sentences shuffled > words shuffled 0.835 → 0.825, words shuffled > random lexicon 0.666 → 0.668. Too small; the PLAN's scoring stays (`--c-one-sided`). |
| Scale instead of cap: `raw × min(1, S/0.2) × min(1, I/0.2)` | sentences shuffled > words shuffled 0.835 → 0.693, words shuffled > random lexicon 0.666 → 0.775. Worse on one pair, better on the other, neither passes (`--soft-gate`). |

## The ladders

Human text: 500 samples of 400–600 B (at least 3 sentences, cut at sentence boundaries) plus 500 short samples of 60–260 B. Reference CDFs from [48, 56) MiB (1,909 standard and 1,729 short samples). Controls as in v0 (each keeps sentence lengths, final punctuation and a capital first letter). Rungs, best to worst: held-out human > trie samples > sentences shuffled across documents > words shuffled within sentences > random lexicon words > random letters > one phrase repeated.

- **Development ladder:** human text from [56, 64) MiB (v0's ladder and seed 20261027), trie file 1. The design above was chosen on it.
- **Fresh ladder:** human text from [64, 72) MiB, seed 31415926, trie file 2 (different prompts). The configuration was frozen before it was first run. I did not change anything after seeing its numbers. Two more seeds were run afterwards for stability only.

Pass rule (PLAN 6.4): pairwise ordering accuracy ≥ 0.95 over all (upper, lower) pairs, ties as one half, **and** non-overlapping 95% bootstrap intervals of the rung means (1,000 resamples).

### Development-ladder results

| Rung | n | SCRIBE | Surface S |
|---|---|---|---|
| held-out human | 500 | 0.535 [0.517, 0.553] | 0.909 |
| trie samples (pooled) | 623 | 0.028 [0.025, 0.032] | 0.734 |
| sentences shuffled | 500 | 0.230 [0.220, 0.241] | 0.899 |
| words shuffled | 500 | 0.111 [0.104, 0.118] | 0.920 |
| random lexicon words | 500 | 0.067 [0.065, 0.069] | 0.895 |
| random letters | 500 | 0.006 | 0.035 |
| one phrase repeated | 500 | 0.001 | 0.029 |

| Upper > lower | SCRIBE acc | Surface S acc | v0 SCRIBE acc |
|---|---|---|---|
| human > trie | 0.995 PASS | 0.947 FAIL | (0.559, not comparable: copied Dolly text) |
| trie > sentences shuffled | 0.019 FAIL | 0.039 FAIL | (0.613, not comparable) |
| sentences shuffled > words shuffled | 0.835 FAIL | 0.354 FAIL | 0.931 |
| words shuffled > random lexicon | 0.666 FAIL | 0.720 FAIL | 0.414 |
| random lexicon > random letters | 1.000 PASS | 1.000 PASS | 1.000 |
| random letters > one phrase repeated | 0.996 PASS | 0.583 FAIL | 0.030 |

SCRIBE 3 of 6, surface S 1 of 6. Without the model rung, human > sentences shuffled is 0.876 for SCRIBE (v0: 0.837) and 0.598 for surface S. Sentences shuffled > words shuffled got worse than v0 (0.931 → 0.835), because the better coherence statistics push shuffled sentences down further while the S gate holds word-shuffled text where it was.

## Fresh-ladder verdict

Seed 31415926. 500 standard and 500 short human samples out of 1,731 candidates; trie rung 623 of 750 continuations kept, mean 341 B after cleaning.

| Rung | n | SCRIBE | Human-parity | Surface S |
|---|---|---|---|---|
| held-out human | 500 | 0.557 [0.539, 0.573] | 1.00 | 0.919 [0.914, 0.923] |
| trie samples (pooled) | 623 | 0.035 [0.031, 0.040] | 0.06 | 0.734 [0.726, 0.742] |
| sentences shuffled across docs | 500 | 0.227 [0.217, 0.236] | 0.41 | 0.902 [0.898, 0.907] |
| words shuffled within sentences | 500 | 0.120 [0.112, 0.127] | 0.21 | 0.926 [0.922, 0.930] |
| random lexicon words | 500 | 0.067 [0.065, 0.069] | 0.12 | 0.899 [0.898, 0.901] |
| random letters | 500 | 0.006 [0.006, 0.006] | 0.01 | 0.034 [0.033, 0.036] |
| one phrase repeated | 500 | 0.001 [0.001, 0.001] | 0.00 | 0.029 [0.029, 0.030] |
| *trie T = 0.6 / 0.8 / 1.0* | 230 / 203 / 190 | 0.035 / 0.043 / 0.026 | | 0.798 / 0.752 / 0.637 |

### Pass or fail per adjacent pair (pooled trie rung)

| Upper > lower | SCRIBE acc | Intervals disjoint | SCRIBE | Surface S acc | Intervals disjoint | Surface S |
|---|---|---|---|---|---|---|
| human > trie | 0.993 | yes | **PASS** | 0.963 | yes | **PASS** |
| trie > sentences shuffled | 0.035 | no | FAIL | 0.039 | no | FAIL |
| sentences shuffled > words shuffled | 0.829 | yes | FAIL | 0.348 | no | FAIL |
| words shuffled > random lexicon | 0.722 | yes | FAIL | 0.727 | yes | FAIL |
| random lexicon > random letters | 1.000 | yes | **PASS** | 1.000 | yes | **PASS** |
| random letters > one phrase repeated | 0.999 | yes | **PASS** | 0.556 | yes | FAIL |

**SCRIBE FAIL, 3 of 6. Surface S FAIL, 2 of 6.**

One rung per temperature gives the same SCRIBE result at each temperature: 3 of 6, with the same three pairs passing (human > trie: 0.994 at T = 0.6, 0.989 at 0.8, 0.996 at 1.0). Surface S passes 1 of 6 at T = 0.6 (human > trie 0.937 fails), and 2 of 6 at T = 0.8 and 1.0 (0.963, 0.994). Without any model rung, SCRIBE passes 2 of 5 pairs (human > sentences shuffled 0.912 fails; the other four are the same as above) and surface S 1 of 5 (human > sentences shuffled 0.620).

### Where the trie samples fall

Share of (trie, control) pairs in which the trie sample scores higher:

| Control | SCRIBE | Surface S |
|---|---|---|
| held-out human | 0.007 | 0.037 |
| sentences shuffled | 0.035 | 0.039 |
| words shuffled | 0.111 | 0.022 |
| random lexicon words | 0.117 | 0.023 |
| random letters | 0.801 | 1.000 |
| one phrase repeated | 0.997 | 1.000 |

Both metrics put the trie between random lexicon words and random letters, which is where it belongs: a cleaned sample from the development file reads `the component of the states and serve" REDIRECT Asian all invasion 10266 Conventury contributor 2006-02-22T05:30:09ZAlgeria (U.S. See also cally relation on the Charles are also one of the relation an all the possible …`. Even in that position SCRIBE would not pass, since random lexicon > trie is 0.883 and trie > random letters 0.801.

### Axis profiles (fresh ladder; mean over samples)

| Rung | S | C (n) | I | S gate | I gate | E1 | E3 | E |
|---|---|---|---|---|---|---|---|---|
| held-out human | 0.78 | 0.45 (500) | 0.66 | 3% | 3% | 0.43 | 1.00 | 0.65 |
| trie, pooled | 0.04 | 0.11 (623) | 0.71 | 96% | 0% | 0.00 | 0.00 | 0.00 |
| trie T = 0.6 | 0.04 | 0.11 (230) | 0.61 | 98% | 0% | | | |
| trie T = 0.8 | 0.05 | 0.11 (203) | 0.78 | 92% | 0% | | | |
| trie T = 1.0 | 0.03 | 0.10 (190) | 0.74 | 98% | 0% | | | |
| sentences shuffled | 0.80 | 0.05 (500) | 0.64 | 2% | 1% | 0.00 | 0.03 | 0.01 |
| words shuffled | 0.12 | 0.44 (500) | 0.69 | 96% | 0% | 0.43 | 0.46 | 0.44 |
| random lexicon words | 0.13 | 0.01 (500) | 0.59 | 96% | 0% | 0.03 | 1.00 | 0.17 |
| random letters | 0.01 | 0.11 (500) | 0.28 | 100% | 0% | 0.00 | 1.00 | 0.06 |
| one phrase repeated | 0.67 | 0.00 (500) | 0.00 | 20% | 100% | 0.00 | 0.00 | 0.00 |
| held-out human, short | 0.77 | 0.44 (500) | 0.73 | 2% | 4% | 0.14 | 0.94 | 0.36 |

The profile reads as it should for each control: shuffled sentences keep S and lose C; shuffled words keep C and lose S; random lexicon words lose both; a repeated phrase has S but no I. Raw statistics: S3 is −6.08 nats per token for human text, −6.11 for shuffled sentences, −8.11 for shuffled words, −8.17 for random lexicon words and −7.19 for the trie. Mean pairwise sentence cosine (C1) is 0.289 for human text, 0.006 for shuffled sentences and 0.293 for shuffled words.

E is printed for completeness and plays no part in the verdict. Human parity is mean per-sample SCRIBE divided by the human mean.

### Which score separates which pair (accuracy of upper > lower; fresh ladder; 0.50 is no information)

| Upper > lower | S1 | S3 | S4 | **S** | C0 | C1 | C3 | **C** | I2 | I3 | **I** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| human > trie | 0.98 | 0.87 | 0.94 | 1.00 | 0.92 | 0.84 | 0.66 | 0.92 | 0.37 | 0.49 | 0.43 |
| trie > sentences shuffled | 0.01 | 0.12 | 0.06 | 0.00 | 0.22 | 0.82 | 0.63 | 0.60 | 0.42 | 0.63 | 0.61 |
| sentences shuffled > words shuffled | 0.49 | 1.00 | 0.58 | 1.00 | 0.22 | 0.02 | 0.20 | 0.03 | 0.55 | 0.40 | 0.42 |
| words shuffled > random lexicon | 0.28 | 0.57 | 0.42 | 0.42 | 1.00 | 0.99 | 0.86 | 1.00 | 0.41 | 0.76 | 0.73 |
| random lexicon > random letters | 1.00 | 0.84 | 0.50 | 1.00 | 0.07 | 0.01 | 0.50 | 0.01 | 0.50 | 0.99 | 0.99 |
| random letters > one phrase repeated | 0.00 | 0.03 | 0.51 | 0.00 | 0.53 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### V4 stability

| Set | n | Pearson, SCRIBE vs bytes | Spearman | Pearson, surface S vs bytes |
|---|---|---|---|---|
| human 400–600 B | 500 | 0.023 | 0.030 | 0.035 |
| human 60–260 B | 500 | 0.107 | 0.135 | 0.039 |
| both buckets pooled | 1000 | −0.009 | 0.012 | 0.027 |

Mean SCRIBE of held-out human: 0.557 standard, 0.567 short (ratio 1.02; v0 0.558 and 0.659, ratio 1.18). Of the 500 short samples, 287 have one sentence and are scored through the two-halves definition.

Seed variance (fresh ladder, three seeds; the trie file is the same in all three, so only the human text and the controls vary):

| Upper > lower | SCRIBE acc, seeds 31415926 / 27182818 / 16180339 | Surface S acc |
|---|---|---|
| human > trie | 0.993 / 0.992 / 0.991 | 0.963 / 0.961 / 0.954 |
| trie > sentences shuffled | 0.035 / 0.029 / 0.033 | 0.039 / 0.033 / 0.041 |
| sentences shuffled > words shuffled | 0.829 / 0.842 / 0.812 | 0.348 / 0.321 / 0.353 |
| words shuffled > random lexicon | 0.722 / 0.688 / 0.700 | 0.727 / 0.777 / 0.739 |
| random lexicon > random letters | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| random letters > one phrase repeated | 0.999 / 0.998 / 0.998 | 0.556 / 0.604 / 0.545 |

Pass and fail are the same for all three seeds. The Pearson correlation with length on standard human text is 0.023, 0.048 and 0.005, on short text 0.107, 0.091 and 0.123, pooled −0.009, 0.013 and −0.090. Short/standard ratio 1.02, 1.00 and 1.08. The books and web check of V4 was not run.

### E null check

A test of E needs held-out human sets, so I drew 100 random sets of 250 human samples from each ladder's own pool and scored them against the null for 250-sample sets. A calibrated null gives a two-sided E1 score averaging 0.50 (10% of sets under 0.1) and a one-sided E3 score averaging 0.75 (5% under 0.1).

| Held-out region | E1 mean (share under 0.1) | E3 mean (share under 0.1) | The 500-sample set of the ladder: E1 / E3 / E |
|---|---|---|---|
| [64, 72) MiB (fresh) | 0.52 (10%) | 0.92 (0%) | 0.43 / 1.00 / 0.65 |
| [56, 64) MiB (development) | 0.40 (31%) | 0.62 (7%) | 0.43 / 0.03 / 0.11 |

On the fresh region the held-out human set lands near the middle (E = 0.65). On the development region it does not: the Zipf slope of held-out 250-sets averages −0.845 against −0.836 for the null (null sd 0.009), and the 500-sample set has a length distance of 0.049, where 250-sets average 0.036 against 0.031 for the null (sd 0.012). On the fresh region the slopes are −0.840 against −0.836 and the length distances 0.0245 against 0.0309. The two regions differ from each other by about as much as the null is wide, so a single held-out region cannot calibrate E. v0's figure for the same region was E = 0.20; it is now 0.11, so E3 is not better there.

## What still fails and why

1. **Trie > sentences shuffled (0.035).** This one is the ladder, not the metric. Shuffled human sentences are fluent sentences (S 0.80); the trie is word salad (S 0.04, S gate fires on 96%, S3 −7.19 against −6.11). S alone separates them at 1.00 and the C axis is close to uninformative (0.60). Any metric that passes this pair on this model is wrong. PLAN 6.4's rung order assumes a model better than shuffled human text. The pair has to be tested on CyphaLM samples, not on an order-4 byte trie.
2. **Sentences shuffled > words shuffled (0.829).** S3 separates them perfectly (1.00) and C works against the order (on C1 shuffled sentences beat shuffled words in 2% of pairs, on C0 in 22%, because word-shuffled text keeps its bag of words and a bag-of-words coherence statistic cannot see word order). The PLAN's gate caps word-shuffled text at S ≈ 0.12, but shuffled sentences are pulled down by their low C (mean SCRIBE 0.227 against a human 0.557), so the two distributions overlap. The geometric mean with C weighted 2 gives shuffled sentences about the same log-score as shuffled words. Because C got better in v1, this pair got worse than in v0 (0.931).
3. **Words shuffled > random lexicon words (0.722).** Only C separates them (C0 1.00, C1 0.99), S says the opposite (S1 0.28: random lexicon words are all valid, shuffled real text has names the lexicon lacks). Both are word salad, both are capped at S ≈ 0.12, and the cap throws the C information away. Without any gate this pair is 0.987, but the ungated score then fails random lexicon > random letters (0.81), random letters > repeated phrase (0.42) and sentences shuffled > words shuffled (0.38). The scaling gate gives 0.820 on this pair and 0.668 on sentences shuffled > words shuffled. I could not find a way of combining S, C and I from these statistics that satisfies the PLAN's gate rule and this rung order together. The rule "word salad cannot be rescued by other axes" and the rung order "words shuffled > random lexicon" conflict unless S can tell the two apart, and S, with no tagger or parser (S2, S5 are not built), cannot.
4. **Human > sentences shuffled (0.912, without the model rung).** Not a pair on the PLAN ladder, but it matters: the per-sample score is a geometric mean of about eight percentile sub-scores, each roughly uniform on human text, so one low sub-score drags a human sample down. Human SCRIBE has a wide spread, and 9% of human/shuffled pairs are in the wrong order although the raw mean C1 is 0.289 for human text and 0.006 for shuffled sentences, so the information is there and is lost in the combination. I did not measure the accuracy of C1 alone for this pair.
5. **SCRIBE cannot rank trie temperatures.** 92–98% of trie samples sit under the S gate, so T = 0.6, 0.8 and 1.0 get 0.035, 0.043 and 0.026 with overlapping intervals. Surface S orders them (0.798, 0.752, 0.637). This is a gap in resolution inside the gated region, and it is the same cause as item 3.
6. **Surface S** fails as in v0: it rates word salad of valid words as high as human text and cannot separate a loop from random letters. The new real-model rung does not change that; surface S does separate the trie from human text at T = 0.8 and 1.0 (0.963, 0.994) but not reliably at T = 0.6 (0.937).
7. **E** is not calibrated across regions (above). It plays no part in the verdict.
8. **Limits of the data.** The word vectors come from 3.79 M tokens (the C1 vector report used all of enwik8, 11.2 M); the sentence splitter and cleaner are heuristics; the trie rung is 17% smaller than it was generated (127 continuations of 750 left under 100 bytes of prose) and its stray markup characters were blanked, both of which favour the trie. Wikipedia only. S2, S5, C2, C4, C5, I1, I4, E2, E4 and E5 are still not implemented, and R, B, V2 (human ratings) and V3 (attacks) were not run.

## Next steps

1. **Replace the trie by CyphaLM samples** (several temperatures, at least 200 each). The rung order is only meaningful for a model that writes sentences. Keep the trie as a "word salad from a real model" control, placed by measurement between random lexicon words and random letters, and say so.
2. **Decide the gate question on a new development ladder.** Candidate: a two-stage score, where the S gate sets the class and C and I order samples inside the class (the ungated score fails when S is the only thing that tells classes apart; the capped one fails when it is not). Whatever is chosen is then run once on a new fresh range; [72, 96) MiB of enwik8 is still unused.
3. **Cut the per-sample noise** (item 4): fewer, better sub-scores, or a trimmed geometric mean, so that human samples are not drawn down by one weak statistic.
4. **Calibrate E across regions.** Use [72, 96) MiB and the reference range as held-out regions for the null check, and widen the null by region until each region lands in the middle.
5. **Word vectors on more text** than 3.79 M tokens (enwik8 [8, 16) MiB is unused and disjoint from the trie's training text; so is [72, 96) once the fresh ladder is spent), then check C0 and C1 for sensitivity to vector quality.
6. Add S2 (tag plausibility) so that S can tell shuffled words from random words; this is what item 3 needs.
7. R and B when the question and fact machinery exists; then V2 ratings, V3 attacks, and the books/web check of V4.

## How to reproduce

```
g++ -O2 -std=c++20 -I native/third_party native/tools/trie_sample.cpp -o trie_sample
g++ -O2 -std=c++20 native/tools/ppmi_svd.cpp -o ppmi_svd
g++ -O2 -std=c++20 -I native/include native/tools/scribe_ladder.cpp -o scribe_ladder
dd if=enwik8 of=train32.xml bs=1048576 skip=16 count=32       # enwik8 [16, 48) MiB
./ppmi_svd train32.xml none none - v32 --no-ri --no-ablation  # 50 s; the similarity files may be absent
./trie_sample enwik8 gen_dev.txt gen_fresh.txt                # 90 s
./scribe_ladder enwik8 --vectors v32 --gen-dev gen_dev.txt --gen-fresh gen_fresh.txt   # 10 s, 210 MB; add --fresh-seed N
./scribe_ladder --selftest                                    # 62 checks
```

The output is deterministic given the seeds (development 20261027, fresh 31415926). The ladder prints every table above, for the development and the fresh ladder, plus the gate-variant and per-temperature ladders.
