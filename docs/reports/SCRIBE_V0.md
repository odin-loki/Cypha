# SCRIBE v0: first control ladder

**Odin Loch** · 2026-10-27

**Questions.**
1. Does a statistical SCRIBE (axes S, C, I, E; no neural network) order the PLAN 6.4 V1 control ladder?
2. Does the old surface score `S = H·D·L` order it?
3. Is SCRIBE stable against sample length on human text (V4)?

**Answers.**
1. No. SCRIBE passes 1 of 6 adjacent rung pairs (random lexicon words > random letters). It ranks a repeated phrase above random letters, cannot tell shuffled real words from random lexicon words, and orders shuffled sentences below human text only 84% of the time. SCRIBE v0 must not judge anything yet.
2. No. Surface `S` also passes 1 of 6, and rates random lexicon words (0.894) and word-shuffled text (0.920) as high as held-out human text (0.909).
3. Partly. The correlation with length inside a bucket is small (0.05 at 400–600 B, −0.11 at 60–260 B), but short human samples score 0.659 against 0.558 for standard ones, so the pooled correlation is −0.22.

Code: [`native/include/cypha/eval/scribe.hpp`](../../native/include/cypha/eval/scribe.hpp), [`native/tools/scribe_ladder.cpp`](../../native/tools/scribe_ladder.cpp). Spec: [`../PLAN.md`](../PLAN.md) section 6.

---

## What is implemented

Scope is the open-continuation profile, axes **S, C, I, E**. Axes **R and B are not implemented**: they need question and fact machinery (question vectors, the knowledge store, the claim extractor) that does not exist yet. Their weights are 0 in this profile anyway except B (0.5), which is therefore also absent.

| Part | v0 implementation |
|---|---|
| Cleaner | Strips XML tags, `<ref>…</ref>`, `{{…}}` (nested), tables, `[[a\|b]]` to b, `[[a]]` to a, Image/File/Category and interlanguage links, external links, `''` and `'''`, HTML entities (twice, for `&amp;nbsp;`); keeps only paragraph lines of 60 bytes or more that start and end like prose and carry no leftover markup |
| S1 lexical validity | Fraction of tokens in the lexicon (words seen at least twice in the training text) or numeric |
| S3 word fluency | Mean log-probability per predicted token (words and end of sentence) under an interpolated Kneser–Ney word trigram; each out-of-vocabulary word pays an extra 8 nats |
| S4 well-formedness | Six checks: valid UTF-8, sentence-initial capital (fraction of sentences), terminal punctuation, balanced brackets, even number of double quotes, no mid-word cut at the end |
| C1 local cohesion | Cosine of adjacent sentences' TF-IDF bag-of-words vectors (sentence-level IDF), two-sided. Stand-in for the C6 sentence vectors |
| C3 topic drift | Jensen–Shannon divergence (bits) of first-half and second-half content-word distributions (words with sentence document frequency under 5%), two-sided |
| I2 self-repetition | Geometric mean of the one-sided scores of `loop8` and the repeated-sentence rate (lower is better) |
| I3 diversity | Geometric mean of two-sided distinct-1 to distinct-4 over word n-grams |
| E1 Zipf slope, E3 sentence lengths | Set level, scored against human sets of the same size and length bucket; E3 uses 11 length bins of 4 tokens |
| Normalising | Empirical CDF of human reference samples of the same length bucket (short under 300 B, standard 300 B and over), one-sided `min(1, 2F)` and two-sided `1 − |2F − 1|`, mid-rank ties, smoothed by `+0.5/(N+1)` |
| Combining | Axis = geometric mean of sub-scores; per sample = weighted geometric mean of S, C, I with weights 2, 2, 1 and ε = 0.01; gate S < 0.2 caps at S; C is dropped (weights renormalised) for one-sentence samples; set level = mean × E^1; bootstrap 95% interval, 1000 resamples |

Data, all disjoint byte ranges of `enwik8`: lexicon, trigram and IDF from cleaned text in [16, 48) MiB (2600 documents, 17.5 MB, lexicon 47,691 words, 2.05 M trigrams); reference CDFs from [48, 56) MiB (1909 standard and 1729 short samples, at most 4 and 3 per document); held-out human test samples from [56, 64) MiB (500 samples of 400–600 B, at least 3 sentences, cut at sentence boundaries). Set-level (E) nulls use human sets drawn from five stretches of the corpus (the reference range and four parts of the training range); E1 and E3 use only token counts and sentence lengths, never the trained model.

## The ladder

500 samples per rung except the model rung. All controls are built from the 500 held-out human samples and keep the surface form on purpose (sentence lengths, final punctuation, a capital first letter), so S4 and E3 cannot catch them:

- **sentences shuffled**: each sentence replaced by one from another document, similar length;
- **words shuffled**: words permuted inside each sentence, terminal punctuation kept at the end, first word capitalised;
- **random lexicon words**: every alphabetic word replaced in place by a frequency-weighted lexicon word, capitalisation and punctuation kept;
- **random letters**: every alphabetic word replaced by random letters of the same length;
- **one phrase repeated**: the first six words, as a sentence, repeated to the sample's length.

**Model rung.** `ctx.txt` has 50 `GEN` lines, 45 unique after cutting each at its last sentence terminator (the logger cut them at 150 bytes). `qa.txt` has **no** `GEN` lines; its `greedy:` and `sample:` lines (2 unique texts) are scored separately below. Most `ctx.txt` GEN lines are **verbatim Dolly answers copied by the trie** (`seen`, `heldout_dolly`), so they are human text of another genre, with chat fragments such as `User: …` in them. The pair "human > model" therefore measures a genre gap, not a model gap, and cannot pass or fail on merit.

### Rung means (95% bootstrap interval of the mean)

| Rung | n | SCRIBE | Human-parity | Surface S | S gate fired |
|---|---|---|---|---|---|
| held-out human | 500 | 0.558 [0.540, 0.577] | 1.00 | 0.909 [0.903, 0.914] | 5% |
| model output (ctx.txt GEN) | 45 | 0.469 [0.369, 0.566] | 0.84 | 0.882 [0.854, 0.908] | 33% |
| sentences shuffled across docs | 500 | 0.308 [0.298, 0.319] | 0.55 | 0.895 [0.892, 0.899] | 3% |
| words shuffled within sentences | 500 | 0.112 [0.104, 0.120] | 0.20 | 0.920 [0.915, 0.924] | 94% |
| random lexicon words | 500 | 0.120 [0.116, 0.124] | 0.21 | 0.894 [0.892, 0.895] | 95% |
| random letters | 500 | 0.006 [0.006, 0.006] | 0.01 | 0.034 [0.033, 0.036] | 100% |
| one phrase repeated | 500 | 0.051 [0.049, 0.054] | 0.09 | 0.029 [0.029, 0.030] | 24% |
| *model, as logged (150-byte cut)* | 45 | 0.076 [0.050, 0.110] | 0.14 | 0.872 [0.844, 0.900] | 93% |
| *qa.txt greedy / sample* | 2 | 0.041 / 0.023 | 0.07 / 0.04 | 0.037 / 0.815 | both |

Human-parity here is the ratio of mean per-sample SCRIBE to the human mean (the PLAN's headline ratio, without the set-level E factor).

### Pass condition: ordering accuracy ≥ 0.95 over all (upper, lower) pairs, ties as one half, and disjoint intervals

| Upper > lower | SCRIBE accuracy | Intervals disjoint | SCRIBE | Surface S accuracy | Surface S |
|---|---|---|---|---|---|
| human > model | 0.559 | no | FAIL | 0.574 | FAIL |
| model > sentences shuffled | 0.613 | yes | FAIL | 0.529 | FAIL |
| sentences shuffled > words shuffled | 0.931 | yes | FAIL | 0.331 | FAIL |
| words shuffled > random lexicon words | 0.414 | no | FAIL | 0.724 | FAIL |
| random lexicon words > random letters | 1.000 | yes | PASS | 1.000 | PASS |
| random letters > one phrase repeated | 0.030 | no | FAIL | 0.571 | FAIL |
| *human > sentences shuffled (model rung skipped)* | 0.837 | yes | FAIL | 0.621 | FAIL |

**Verdict: SCRIBE v0 FAILS the ladder (1 of 6). Surface S FAILS the ladder (1 of 6).**

Why SCRIBE fails, from the sub-scores:

- **Words shuffled vs random lexicon words (0.414).** Both have S3 ≈ 0, so both fall under the S < 0.2 gate and are capped at S, which cannot tell them apart. The C axis does (0.47 against 0.10) but the gate throws it away. A trigram cannot see topical coherence.
- **Random letters vs repeated phrase (0.030).** The repeated phrase has valid words and a valid form (S 0.61), so only C and I are low, and the PLAN has no gate on them. Random letters are capped at S ≈ 0.006.
- **Human vs sentences shuffled (0.837).** C1 on a bag-of-words is noisy: many true adjacent human sentences share no word, so cosine 0 occurs in both. This is the weakest part of v0 and the reason C6 vectors are in the PLAN.
- **Sentences shuffled vs words shuffled (0.931).** Close; about 6% of word-shuffled samples escape the S gate.

### Axis profile (mean; C over samples with at least 2 sentences)

| Rung | S | C (n) | I | E1 | E3 | E |
|---|---|---|---|---|---|---|
| held-out human | 0.74 | 0.48 (500) | 0.66 | 0.30 | 0.13 | 0.20 |
| model output (ctx.txt GEN) | 0.54 | 0.27 (9) | 0.70 | 0.05 | 0.00 | 0.02 |
| sentences shuffled | 0.75 | 0.11 (500) | 0.65 | 0.07 | 0.00 | 0.02 |
| words shuffled | 0.11 | 0.47 (500) | 0.68 | 0.30 | 1.00 | 0.55 |
| random lexicon words | 0.13 | 0.10 (500) | 0.57 | 0.17 | 0.00 | 0.03 |
| random letters | 0.01 | 0.02 (500) | 0.28 | 0.00 | 0.00 | 0.00 |
| one phrase repeated | 0.61 | 0.00 (480) | 0.00 | 0.00 | 0.00 | 0.00 |

Human sub-scores average 0.71 (S1), 0.76 (S3), 0.96 (S4), 0.52 (C1), 0.52 (C3), 0.83 (I2), 0.54 (I3). A one-sided score averages 0.75 and a two-sided one 0.50 on a human sample by construction, which is why human SCRIBE is 0.56 and not 1: the scale is "percentile among humans", and human-parity is the number to quote.

### V4 stability

| Set | n | Pearson, SCRIBE vs bytes | Spearman | Pearson, surface S vs bytes |
|---|---|---|---|---|
| human 400–600 B | 500 | 0.048 | 0.029 | −0.010 |
| human 60–260 B | 500 | −0.109 | −0.127 | 0.003 |
| both buckets pooled | 1000 | −0.221 | −0.229 | 0.018 |

Mean SCRIBE of held-out human is 0.659 for the short bucket and 0.558 for the standard bucket. Part of that gap is that 285 of 500 short samples have one sentence, so the C axis is absent and cannot pull the geometric mean down. V4 is not met across buckets. Seed variance and the books/web check were not run.

## Honest limits

- **The ladder fails, so SCRIBE v0 judges nothing.** The numbers above are the first measurement; none of the three failures is a bug I found, but each follows from a design choice, which is the point of running the ladder first.
- **The model rung is not a degraded-model rung.** Its 45 texts are mostly copied human answers, 117 bytes on average, in another genre. The only real generated texts on disk are the two in `qa.txt` (n = 2), and SCRIBE rates them 0.041 and 0.023 (S gate fires) where surface S rates the sample 0.815. That is one clear case, not a test.
- **E is unreliable.** A held-out human set gets E = 0.20 (E1 0.30, E3 0.13), and the word-shuffled set, whose sentence lengths are the human ones, gets E3 = 1.00. The null for 500-sample sets is far narrower than the difference between two stretches of the corpus or a small change in sentence splitting. E is printed but plays no part in the verdict, and the set-level SCRIBE column in the tool output should not be quoted.
- The sentence splitter and the cleaner are heuristics; the tokenizer lowercases ASCII only and treats hyphen as a separator. The trained lexicon is 17.5 MB of 2006 Wikipedia, so it is a Wikipedia-genre metric until trained on books and web.
- C1 uses bag-of-words TF-IDF, not sentence vectors; S2, S5, C2, C4, C5, I1, I4, E2, E4, E5 are not implemented.
- The OOV penalty (8 nats), content-word threshold (5% document frequency) and length bins are untuned first choices.
- V2 (human ratings), V3 (adversarial attacks) and V5 beyond the disjoint-range design were not run. Each statistic is deterministic given the seed (20261027); one run takes about 8 s and 150 MB.

## Proposals tried after seeing the failures (not in the PLAN, tuned on this ladder, so not evidence)

| Variant | Adjacent pairs passed | What changes |
|---|---|---|
| A: PLAN gate plus a cap at I when I < 0.2 | 2 of 6 | random letters > repeated phrase becomes 0.996 |
| B: `raw × min(1, S/0.2) × min(1, I/0.2)` (scales instead of capping) | 2 of 6 | repeated phrase 0.965; words shuffled > random lexicon rises only to 0.587 |

Both leave the same four pairs failing, so gates are not the main problem.

## Next steps

1. **Fix the control, then the metric.** Replace the model rung with real generations (CyphaLM samples at several temperatures, not trie copies), at least 200 texts, so "human > model" can be tested.
2. **Coherence.** Add C6 sentence vectors (PPMI-SVD) and use all sentence pairs, not just adjacent ones; this is the only axis that separates shuffled sentences and shuffled words from lexicon noise.
3. **A gate on I** (cap at I when I < 0.2) in PLAN 6.3, because a degenerate loop of valid words passes S; decide it on a fresh ladder, not on this one.
4. **E:** build the null from many more stretches of the corpus (all of enwik8), or drop E3 to a Kolmogorov distance on the lengths; check that a held-out human set lands near the middle.
5. **Length parity:** score short samples with the C axis defined for one sentence (for example, cohesion between the two halves) or restrict V4 to samples with at least 2 sentences.
6. Add R and B when the question and fact machinery exists; then V2 ratings and V3 attacks.
