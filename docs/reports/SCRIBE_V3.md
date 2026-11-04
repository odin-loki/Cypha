# SCRIBE v3

**Odin Loch** · 2026-11-04

SCRIBE v3 is the mean of three judges: v2 (unchanged), SURPRISE (information theory) and MATCH (string matching). Each gives a score in [0, 1], with human text near the top. v3 also reports the minimum of the three, and flags a text when they disagree.

## Why

CyphaLM's best writer, K5, gained a lot on SCRIBE v2 with a sentence gate that picks sentences by word co-occurrence with the prompt (docs/reports/K5_UPGRADES.md, step 4). Most of the gain is on v2's coherence axis, which also rests on co-occurrence (PPMI-SVD word vectors). Part of the gain may be agreement between two co-occurrence measures. The two new judges read no word vectors and no co-occurrence statistics.

## The three judges

**v2.** Frozen as in docs/reports/SCRIBE_V2.md. The v1 and v2 profiles reproduce their outputs byte for byte (dev A, dev B and the K5 ladder, checked with `cmp` against the build before v3).

**SURPRISE.** The code length of the text, word by word, under a model trained on the pack only.

- The model. The pack's interpolated Kneser–Ney word trigram (SCRIBE's `ScribeModel`, proper over its vocabulary). Words it calls `<unk>` are spelled by a character trigram (Witten–Bell, trained on the pack's once-seen word types); numbers by one trained on the pack's number types. So every string has a probability. On top, the text's own history: a unigram cache, p = 0.9 p_static + 0.1 c(w)/|H| (Kuhn and De Mori), and a match model: if the last k ≥ 2 words occurred earlier, the word that followed them gets weight k/(k + 1). Surprisal s(w) = −log₂ p(w) bits.
- The statistics, each per text:
  - `w_mean`, `w_sd`, `w_ac1`: the mean of s, its standard deviation, and its lag-1 autocorrelation (burstiness).
  - `w_lo`: the share of words below 4.35 bits (the 25th percentile of the reference words), meaning too predictable.
  - `w_hi`: the share above 17.45 bits (the 90th percentile), meaning salad.
  - `w_ctx`: the information the trigram context gives, as unigram bits minus trigram bits per word. It falls when word order breaks.
  - `w_spell`: bits spent spelling unknown words, per word. It rises for non-words.
  - `w_coh`: cohesion as code length. Each sentence's content words (unigram surprisal ≥ 8 bits) are coded with a cache of the text's other sentences; the statistic is the bits a word saves, averaged over the sentences.
  - `w_anti`: the share of words that the context makes more than 2 bits less likely. It is reported only.

**MATCH.** How the text's bytes match the pack, and the text itself.

- A suffix array over the pack's cleaned text (documents joined by a newline), sorted on the first 64 bytes of each suffix. From each byte of the text, the longest prefix of the rest of the text that occurs in the pack (matching statistics, up to 64 bytes). One binary search inside the bucket of the first two bytes. The same array counts how often a string occurs.
- The statistics:
  - `m_med`, `m_p90`: the median and the 90th percentile of the match lengths.
  - `m_long`: the share of bytes inside matches of ≥ 32 bytes (copying).
  - `m_short`: the share inside matches of ≤ 4 bytes only (novel junk).
  - `m_lz`: the text's own Lempel–Ziv (LZ76) complexity, as phrases × log₂ n / n. Loops make it fall; random letters make it rise.
  - `m_tri`: the share of word starts whose pack match covers 3 or more whole words. It falls when word order breaks.
  - `m_coh`: cohesion. From each word start of a sentence, the longest match (≥ 5 bytes) with the text's other sentences. Each covered byte counts the matched string's information in the pack, log₂((N + 1)/(count + 1)) / log₂(N + 1), above a floor of 0.45, so that strings like " the " count nothing. The statistic is the mean over the bytes, then over the sentences.
  - With a prompt: `m_echo`, the share of the text's bytes inside ≥ 8-byte matches with the prompt; and `m_echo32`, the same for ≥ 32-byte matches (copied prompt sentences).
  - `m_words`: the mean number of whole words a match covers. It is reported only.
- Why not `fm_infinigram.hpp`: the FM index lives in the LM library (a .cpp to link), while scribe.hpp is header-only with the standard library only. The suffix array takes 84 MB for the 17.5 MB pack (the FM index would take about 23 MB, by its documented 1.35 bytes a text byte). It builds in 0.9 s on 4 threads. Queries are one binary search; all v3 features cost 0.94 ms a text on one core.

**The score form (both new judges).** Each statistic is compared with the human reference samples of its length bucket by v2's ECDF z-score (`QuantileZ`), **two-sided**: each side of z feeds one group (u = the bad side's |z|):

| Group | What | Evidence | Term of D |
|---|---|---|---|
| L | not words | max(0, u − 4) | 30 · sat(L − thL) |
| S | not word order | max(0, u − 1) | 30 · sat(S − thS) |
| C | sentences unrelated, or off the prompt | max(0, u − 0) | wC · C |
| G | too predictable (graded) | max(0, u − dG) | wG · G |
| X | too predictable, beyond each statistic's own human 99th percentile | Σ max(0, e_k − th_k), e_k = max(0, u − 1) | wP · X |

Here sat(x) = 1 − exp(−max(0, x)/κ), thL and thS are the 99th percentiles of L and S over the reference, and score = exp(−D/τ). The sides:

| Statistic | high side | low side |
|---|---|---|
| w_mean, w_sd, w_hi, w_spell | L | too predictable |
| w_ctx, w_lo, w_ac1 | too predictable | S |
| w_coh | too predictable (repetition) | C |
| m_short | L | too predictable |
| m_p90, m_tri | too predictable (copying) | S |
| m_med, m_long | too predictable | — (m_p90 and m_tri carry it) |
| m_lz | C (weight 0.5, dead zone 2) | too predictable (loops) |
| m_coh | too predictable | C |
| m_echo | too predictable | C |
| m_echo32 | too predictable | — |

A statistic that is missing where at least 95% of human samples have it counts at z = 4 on its bad side (v2's rule).

**SCRIBE v3** = mean(v2, SURPRISE, MATCH). `min3` = the minimum of the three. A text is flagged `disagree` when the maximum minus the minimum exceeds 0.5. JSON fields: `scribe_v3`, `surprise`, `match`, `scribe_v2`, `min3`, `disagree`, and per judge its z-scores and D terms (`v3`, `raw3`). Commands: `scribe ladder --profile v3`, `scribe score --profile v3`, `scribe attacks`.

## Freeze record (written before the fresh region was read)

Development data: ladder A (enwik8 [56, 64) MiB, seed 20261027, `--chain-rng`, v2's dev A), ladder B ([64, 72) MiB, seed 31415926). Pack [16, 48), reference [48, 56), vectors vec32. The attack batteries are built on the same test ranges and seeds.

**Frozen design** (`V3Params` defaults in native/include/cypha/eval/scribe.hpp):

| | SURPRISE | MATCH |
|---|---|---|
| dead zones L, S, C, G, X | 4, 1, 0, 1.5, 1 | 4, 1, 0, 2, 1 |
| cap, κ | 30, 2 | 30, 0.25 |
| wC, wG, wP | 4, 1, 8 | 6, 1, 20 |
| thresholds | 99th percentiles | 99th percentiles |
| τ | 20 | 30 |
| other | λ 0.1, match-model β 1, k ≥ 2; content words ≥ 8 bits; w_lo / w_hi at the reference words' 25th / 90th percentiles | m_coh floor 0.45, matches ≥ 5 bytes; m_lz in C with weight 0.5 and dead zone 2 |

τ is set so that the mean score of dev A's human samples equals v2's (0.909), rounded. It changes no order within a judge; it sets the scale for the mean of three.

**Registered rules.**
1. On each of dev A, dev B and the fresh ladder, SURPRISE alone, MATCH alone and the v3 mean each pass all five control pairs (human > sentences shuffled > words shuffled > random lexicon words > random letters > one phrase repeated). A pair passes with accuracy ≥ 0.95 and disjoint 95% bootstrap intervals, as for v2.
2. The attack battery: 200 prompt / continuation pairs from the test range (one per document; the prompt is 100–300 B, the continuation 400–600 B and ≥ 3 sentences). Nine registered attacks:
   - `copy_pack`: a 400–600 B window of pack text.
   - `echo_repeat`: the prompt repeated.
   - `echo_copy`: the prompt's sentences followed by the true continuation.
   - `stuff_human`: another document's continuation with the prompt's five rarest content words inserted after every 4th word.
   - `stuff_generic`: the same insertion into v2's generic paragraph.
   - `loop`: the continuation's first two sentences repeated.
   - `generic`: v2's fixed generic paragraph, the same for every prompt.
   - `shuffle_docs`: each sentence replaced by one from another document (the ladder's control).
   - `cooc_pack`: pack sentences chosen for maximal word association with the prompt. The score is the mean cosine of a sentence's content-word vectors with the prompt's mean content-word vector (the PPMI-SVD vectors, i.e. co-occurrence); the best sentences are taken until 400–600 B: a crude K5 gate.

   Rule: v3 ranks human text above every registered attack with pair accuracy ≥ 0.95, and SURPRISE or MATCH catches each attack (pair accuracy ≥ 0.95). Three extras are reported, not ruled, because they are known blind spots:
   - `reorder`: the continuation's own sentences in another order.
   - `cooc_out`: the co-occurrence attack drawn from enwik8 [0, 16) MiB, which is outside the pack.
   - `offprompt`: another document's continuation.
3. The fresh verdict: enwik8 [72, 95.37) MiB, seed 1123581321, read once, with the commands below.

**Development results.** Pair accuracy; every PASS has disjoint intervals.

| Pair | dev A v2 | SURPRISE | MATCH | v3 | dev B v2 | SURPRISE | MATCH | v3 |
|---|---|---|---|---|---|---|---|---|
| human > sentences shuffled | 0.974 | 0.952 | 0.953 | 0.982 | 0.983 | 0.969 | 0.955 | 0.988 |
| sentences shuffled > words shuffled | 0.998 | 0.995 | 0.983 | 1.000 | 0.994 | 0.998 | 0.986 | 0.997 |
| words shuffled > random lexicon words | 0.996 | 0.956 | 0.952 | 0.973 | 1.000 | 0.963 | 0.957 | 0.983 |
| random lexicon words > random letters | 1.000 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **pairs passed** | 5 | **5** | **5** | **5** | 5 | **5** | **5** | **5** |

Rung means on dev A (95% interval):

| Rung | v2 | SURPRISE | MATCH | v3 |
|---|---|---|---|---|
| human | 0.909 [0.895, 0.924] | 0.910 [0.898, 0.920] | 0.910 [0.895, 0.921] | 0.910 [0.900, 0.919] |
| sentences shuffled | 0.352 | 0.662 | 0.665 | 0.560 |
| words shuffled | 0.003 | 0.225 | 0.341 | 0.190 |
| random lexicon words | 0.001 | 0.157 | 0.248 | 0.135 |
| random letters | 0.000 | 0.065 | 0.058 | 0.041 |
| one phrase repeated | 0.000 | 0.000 | 0.000 | 0.000 |
| human, short | 0.932 | 0.943 | 0.949 | 0.941 |

The new judges' margins on two pairs are thin: 0.952–0.969 on human > sentences shuffled, and 0.952–0.963 on words shuffled > random lexicon words. Both pairs are about whether the sentences and words belong together. Without co-occurrence, the judges see that only through exact repetition of content words and strings. The v3 mean has a comfortable margin on every pair (≥ 0.973).

Attack battery on development data: mean score, with P(human > attack) in brackets.

| Attack | dev A v2 | SURPRISE | MATCH | v3 | caught by | rule | dev B v3 | dev B caught by | rule |
|---|---|---|---|---|---|---|---|---|---|
| human | 0.890 | 0.905 | 0.824 | 0.873 | | | 0.878 | | |
| copy_pack | 0.101 (0.987) | 0.045 (1.000) | 0.000 (1.000) | 0.049 (1.000) | v2, S, M | PASS | (0.996) | v2, S, M | PASS |
| echo_repeat | 0.000 (1.000) | 0.006 (1.000) | 0.000 (1.000) | 0.002 (1.000) | v2, S, M | PASS | (1.000) | v2, S, M | PASS |
| echo_copy | 0.889 (0.501) | 0.904 (0.488) | 0.000 (1.000) | 0.598 (0.942) | M | FAIL | (0.949) | M | FAIL |
| stuff_human | 0.384 (0.840) | 0.632 (0.806) | 0.666 (0.642) | 0.560 (0.860) | none | FAIL | (0.856) | none | FAIL |
| stuff_generic | 0.970 (0.328) | 0.822 (0.697) | 0.852 (0.452) | 0.881 (0.536) | none | FAIL | (0.503) | none | FAIL |
| loop | 0.000 (1.000) | 0.019 (1.000) | 0.003 (0.979) | 0.007 (1.000) | v2, S, M | PASS | (1.000) | v2, S, M | PASS |
| generic | 0.835 (0.760) | 0.653 (0.950) | 0.475 (0.945) | 0.654 (0.923) | S | FAIL | (0.929) | S, M | FAIL |
| shuffle_docs | 0.352 (0.956) | 0.678 (0.925) | 0.537 (0.919) | 0.522 (0.964) | v2 | FAIL | (0.990) | v2, S, M | PASS |
| cooc_pack | 0.140 (0.982) | 0.089 (0.999) | 0.000 (1.000) | 0.076 (0.999) | v2, S, M | PASS | (0.996) | v2, S, M | PASS |
| reorder (extra) | 0.885 (0.508) | 0.906 (0.499) | 0.823 (0.500) | 0.871 (0.500) | none | — | (0.507) | none | — |
| cooc_out (extra) | 0.889 (0.477) | 0.834 (0.688) | 0.799 (0.538) | 0.841 (0.582) | none | — | (0.572) | none | — |
| offprompt (extra) | 0.890 (0.500) | 0.905 (0.500) | 0.713 (0.739) | 0.836 (0.647) | none | — | (0.654) | none | — |

The attack rule holds for 4 of 9 attacks on dev A and 5 of 9 on dev B. It is registered as it stands and expected to fail on the fresh region:

- `echo_copy` is caught by MATCH alone (1.000). The mean of three only reaches 0.942–0.949, because the other two judges rate the copied text as human.
- The generic paragraph is caught by SURPRISE (0.950–0.960; it has no rare words and too even a surprisal), where v2 is at 0.760–0.775. The mean stays below 0.95.
- Keyword stuffing is not caught by either new judge. Inserting a keyword after every 4th word moves several statistics by 2 to 3 standard deviations. Human text reaches that on single statistics, so the 99th-percentile thresholds absorb it.
- The design that caught `stuff_generic` (w_anti in the order group, 0.96–0.98) cost SURPRISE a ladder pair on dev A, so w_anti is reported only.

## Design log (development data only)

1. **Statistics as listed in the brief.** Two-sided single statistics order the hard pairs poorly (human > sentences shuffled: best 0.70; words shuffled > random words: best 0.83). Mean surprisal, its spread, burstiness, the low and high shares, and match lengths do not see whether sentences belong together.
2. **Cohesion as code length.** A cache of the whole text's past gives 0.84 on human > shuffled. A cache of the *other sentences only*, averaged per sentence, gives 0.975 (it no longer counts repeats within a sentence). Restricting it to content words gives 0.978, and 0.970 on words shuffled > random words.
3. **Cohesion as string matching.** Byte matches between sentences give 0.943 unweighted. Weighting each byte by its string's information in the pack (−log₂ frequency, from the suffix array), above a floor, gives 0.977. The floor stops " the " counting; higher floors make random-word texts tie at 0.
4. **Word order.** Mean surprisal moves only +1.8 σ under word shuffling. The context's information (`w_ctx`) moves −6.8 σ. For MATCH, the share of word starts whose match spans 3 words (`m_tri`) separates sentence from word shuffles at 0.998.
5. **One saturating "form" group mixed lexical and order evidence.** Words shuffled and random words then saturated noisily. Two saturating groups (lexical, order) as in v2, plus a wide dead zone (4) for the lexical group: random letters sit at z 17–33, while human tails at z 2–3 must not trip it.
6. **The too-predictable side.** A group sum with one threshold let Wikipedia boilerplate (a heavy tail of long pack matches in human text) hide every other predictable signal, such as prompt echo and over-repetition. Each statistic now counts beyond its own human 99th percentile (X), plus a small graded part (G).
7. **Echo.** The 8-byte echo reaches only z ≈ 3.4 on copied prompts, because human continuations repeat names from their prompt. The 32-byte echo separates at 1.000.
8. **Reported only:** `w_anti` and `m_words`. Tried and dropped: a stem-free whole-text self-match statistic, LZ "savings" against the pack, pack word rarity (the random-word control keeps capitals, so it goes the wrong way), higher cohesion floors, and per-statistic thresholds for the order group.

Python design scripts and dumps were kept outside the repository (scratch). They worked on the z-scores exported by `scribe ladder --profile v3 --export` and `scribe attacks --export`, with the C++ reproducing every chosen number.

## Fresh verdict

Commands, registered before the run (`E=/root/cdata/enwik8`, vectors `/root/work/scribe/vec32`):

```
scribe ladder  --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:72:95.37 --vectors vec32 --profile v3 --seed 1123581321 --label v3_fresh
scribe attacks --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:72:95.37 --vectors vec32 --seed 1123581321 --outside enwik:$E:0:16 --n 200 --label att_fresh
```

Read once, 2026-11-04, at the frozen commit. 500 standard samples from 5,839 candidates. The ladder took 14 s with a 509 MB peak; the battery 16 s.

| Pair | v2 | SURPRISE | MATCH | v3 |
|---|---|---|---|---|
| human > sentences shuffled | 0.975 PASS | 0.950 PASS | **0.940 FAIL** | 0.981 PASS |
| sentences shuffled > words shuffled | 0.999 PASS | 1.000 PASS | 0.979 PASS | 1.000 PASS |
| words shuffled > random lexicon words | 0.998 PASS | 0.961 PASS | 0.950 PASS | 0.975 PASS |
| random lexicon words > random letters | 0.998 PASS | 1.000 PASS | 1.000 PASS | 1.000 PASS |
| random letters > one phrase repeated | 1.000 PASS | 1.000 PASS | 1.000 PASS | 1.000 PASS |
| **pairs passed** | 5 of 5 | **5 of 5** | **4 of 5** | **5 of 5** |

Rung means (95% interval):

| Rung | v2 | SURPRISE | MATCH | v3 |
|---|---|---|---|---|
| human | 0.916 [0.901, 0.931] | 0.903 [0.891, 0.914] | 0.900 [0.887, 0.913] | 0.906 [0.896, 0.916] |
| sentences shuffled | 0.354 | 0.667 | 0.659 | 0.560 |
| words shuffled | 0.003 | 0.225 | 0.338 | 0.189 |
| random lexicon words | 0.001 | 0.158 | 0.249 | 0.136 |
| random letters | 0.000 | 0.066 | 0.058 | 0.041 |
| one phrase repeated | 0.000 | 0.000 | 0.000 | 0.000 |
| human, short | 0.954 | 0.942 | 0.951 | 0.949 |

v2 reproduces its own frozen fresh verdict (0.975, 0.999, 0.998, 0.998, 1.000).

**Rule 1 fails for MATCH**: 0.940 on human > sentences shuffled. Rule 1 holds for SURPRISE (just: 0.950) and for the v3 mean (0.975 or more on every pair). On this pair MATCH has only exact string repetition between sentences to go on, and on dev data it already had the thinnest margin (0.953, 0.955).

Attack battery (fresh):

| Attack | v2 | SURPRISE | MATCH | v3 | caught by | rule |
|---|---|---|---|---|---|---|
| human | 0.895 | 0.899 | 0.820 | 0.871 | | |
| copy_pack | 0.107 (0.991) | 0.040 (0.997) | 0.000 (1.000) | 0.049 (0.996) | v2, S, M | PASS |
| echo_repeat | 0.000 (1.000) | 0.005 (1.000) | 0.000 (1.000) | 0.002 (1.000) | v2, S, M | PASS |
| echo_copy | 0.900 (0.523) | 0.900 (0.498) | 0.000 (1.000) | 0.600 (0.948) | M | FAIL |
| stuff_human | 0.443 (0.807) | 0.658 (0.781) | 0.674 (0.624) | 0.592 (0.825) | none | FAIL |
| stuff_generic | 0.977 (0.329) | 0.834 (0.679) | 0.893 (0.354) | 0.901 (0.457) | none | FAIL |
| loop | 0.000 (1.000) | 0.018 (0.999) | 0.003 (0.985) | 0.007 (1.000) | v2, S, M | PASS |
| generic | 0.835 (0.790) | 0.653 (0.970) | 0.461 (0.967) | 0.649 (0.930) | S, M | FAIL |
| shuffle_docs | 0.336 (0.966) | 0.668 (0.939) | 0.511 (0.951) | 0.505 (0.981) | v2, M | PASS |
| cooc_pack | 0.149 (0.977) | 0.096 (0.995) | 0.000 (1.000) | 0.082 (0.995) | v2, S, M | PASS |
| reorder (extra) | 0.892 (0.511) | 0.897 (0.500) | 0.818 (0.504) | 0.869 (0.507) | none | — |
| cooc_out (extra) | 0.839 (0.536) | 0.809 (0.684) | 0.768 (0.533) | 0.805 (0.608) | none | — |
| offprompt (extra) | 0.895 (0.500) | 0.899 (0.500) | 0.701 (0.747) | 0.832 (0.634) | none | — |

**Rule 2 fails: 5 of 9 attacks pass.**

- Copying, echo, loops and the co-occurrence attack from the pack are caught by every judge.
- The new judges add three things v2 does not have:
  - MATCH alone catches copied prompt sentences (1.000, where v2 is at 0.523).
  - Both new judges catch the generic paragraph (0.970, 0.967, where v2 is at 0.790).
  - Both see the co-occurrence attack as copying (0.995, 1.000).
- In two of those cases the mean of three stays below 0.95 (0.948 and 0.930): one judge's catch is diluted by two judges that rate the text as human. `min3` and the `disagree` flag are there for this case. 99% of `echo_copy` texts are flagged.
- Keyword stuffing is caught by no judge, v2 included (0.807 and 0.329).

The extras behave as registered blind spots:

- Sentences reordered within the text are invisible to all three (0.50).
- The co-occurrence attack drawn from outside the pack escapes copy detection. It is caught by nothing (v3 0.608), but SURPRISE (0.684) and MATCH see more of it than v2 (0.536).
- Off-prompt human text is seen only through MATCH's echo (0.747).

## K5 cross-check (report only)

The GEN-A dev ladder (enwik8 [56, 64) MiB, seed 20261027, no `--chain-rng`), with K5's 250 dev continuations: old K5 (nucleus 0.9, no classes, no gate: /root/work/k5m/OLD.gen.txt) and new K5 (nucleus 0.8, 256 classes, the 8-candidate co-occurrence gate: NEW.gen.txt). MATCH's echo uses each GEN line's PROMPT. v2's numbers equal K5's records byte for byte.

| K5 dev samples | n | v2 | SURPRISE | MATCH | v3 | min of 3 | disagree |
|---|---|---|---|---|---|---|---|
| old K5 | 242 | 0.353 [0.323, 0.386] | 0.718 [0.699, 0.736] | 0.592 [0.579, 0.607] | 0.554 [0.537, 0.573] | 0.327 | 26% |
| new K5 (gate) | 244 | 0.498 [0.471, 0.527] | 0.612 [0.591, 0.634] | 0.583 [0.566, 0.599] | 0.565 [0.549, 0.580] | 0.419 | 7% |
| change | | **+0.145** | **−0.106** | −0.009 | +0.011 | +0.092 | |
| human (same ladder) | 500 | 0.923 | 0.921 | 0.922 | 0.922 | 0.859 | 2% |
| sentences shuffled | 500 | 0.344 | 0.665 | 0.661 | 0.557 | 0.339 | 3% |

**The new judges do not confirm v2's gain.**

- SURPRISE rates the new K5 lower: its deviance on the too-predictable side rises from 0.93 to 4.33. The new K5 text is more predictable than human text, and more pack-like:
  - mean surprisal 7.4 bits against old K5's 8.4 and human text's 9.5 (median z −2.0);
  - fewer rare words (w_hi z −1.8) and more very predictable ones (w_lo z +1.0);
  - more 3-word pack matches than human text (m_tri z +1.8, MATCH).
- MATCH does not move (−0.009, inside the intervals). The v3 mean moves by +0.011, also inside them.
- Cohesion measured without co-occurrence is far below human text for both K5 versions:
  - w_coh: −0.10 and −0.07 bits a word, against +1.09 for human text;
  - m_coh: 0.008 and 0.012, against 0.115.
- The gate's gain on v2's coherence axis (word vectors) does not show as content words recurring across sentences.
- On these judges both K5 versions sit at or below sentences shuffled across documents on cohesion.

The new K5 is more consistent across judges: 7% of its texts are flagged `disagree` against 26% for the old. The minimum of the three rises by 0.092, mostly because v2's low scores rose. A human read or a neural judge is still the right check.

## Cost

`scribe score` on the 2,600 texts of the dev A battery, one core:

| | texts/s | peak RSS | setup |
|---|---|---|---|
| v2 | 1,476 | 385 MB | 3.6 s |
| v3 (v2 + SURPRISE + MATCH) | 564 | 493 MB | + 0.4 s spelling models, + 0.9 s suffix array (4 threads, 84 MB) |

v3's own features take 0.94 ms a text. The ladder computes them on 4 threads, 7,138 texts in about 1.7 s.

## What v3 is, and is not

- **Use the parts, not only the mean.** The mean of three is robust when the judges agree. When only one judge catches a failure (copied prompt sentences, a generic paragraph), the mean dilutes it. Read `min3` and `disagree` next to `scribe_v3`.
- **SURPRISE** is the most independent judge. It is two-sided by construction, so text that is too predictable (copied, generic, formulaic) loses as surely as salad does. It passed all five ladder pairs on dev A, dev B and fresh data.
- **MATCH** is a sharp copy and echo detector. As a coherence judge it is the weakest of the three: it failed one fresh ladder pair (human > sentences shuffled, 0.940).
- **Blind spots that remain:**
  - sentence order within a text;
  - keyword stuffing at one inserted word in five;
  - sentences gathered by co-occurrence from text outside the pack;
  - truth (numbers, negations, as for v2).
- **Freeze discipline.** The design was chosen on dev A and its battery and confirmed on dev B, with no change after B. It was frozen in commit a6fb4a9 before the fresh read. The fresh region was read once. No setting was changed after that read.

## Reproduce

```
E=/root/cdata/enwik8; V=/root/work/scribe/vec32
scribe ladder  --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:56:64 --vectors $V --profile v3 --seed 20261027 --chain-rng   # dev A
scribe ladder  --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:64:72 --vectors $V --profile v3 --seed 31415926                # dev B
scribe attacks --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:56:64 --vectors $V --seed 20261027 --outside enwik:$E:0:16       # dev A battery
scribe ladder  --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:56:64 --vectors $V --profile v3 --seed 20261027 --gen-trie NEW.gen.txt   # K5
scribe score   --pack enwik:$E:16:48 --ref enwik:$E:48:56 --vectors $V --profile v3 --in texts.jsonl --out scores.jsonl
```

`scribe selftest` (CTest `native_scribe_selftest`) checks the v3 parts: LZ76 on the Lempel–Ziv example, the spelling model's normalisation, the suffix array against a brute-force scan, the cache and match model, copy and echo detection, both sides of SURPRISE, and the mean, minimum and flag.
