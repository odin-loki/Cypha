# SCRIBE v3.1

**Odin Loch** · 2026-11-04

SCRIBE v3.1 makes v3 ([SCRIBE_V3.md](SCRIBE_V3.md)) harder to fool. It keeps the three judges: v2 (unchanged), SURPRISE (information theory) and MATCH (string matching). It changes two things:

1. **The average.** v3.1 is the geometric mean of the three scores, not the plain mean. One judge's confident catch is no longer diluted by two judges that see nothing.
2. **The two new judges.** SURPRISE 3.1 and MATCH 3.1 add statistics for words that fit neither neighbour (keyword stuffing) and a group for text that is too bland (a generic paragraph). MATCH 3.1 counts copied prompt *sentences* (64-byte matches), not repeated names.

Command: `scribe ladder|attacks|score --profile v31`. Profiles v1, v2 and v3 are unchanged, byte for byte.

**Result** (fresh region, enwik9 [100, 124) MiB, read once):
- The ladder passes 5 of 5.
- The registered attacks pass 9 of 9 (v3: 5 of 9). Keyword stuffing passed only just (0.953).
- The fresh human mean is 0.904.
- The held-out attacks, never run before, pass **2 of 8**. Stuffing at a lower rate, one plain generic paragraph, a template with the prompt's names, and fluent sentences from elsewhere still get through.

v3.1 is harder to fool than v3. It is not fool-proof.

## What v3 missed

On its fresh read (enwik8 [72, 95.37) MiB), v3's attack rule passed 5 of 9:

- `echo_copy` (the prompt's sentences, then the true continuation): MATCH alone caught it (1.000), but the mean of three was 0.948. Dilution.
- `generic` (v2's fixed generic paragraph): SURPRISE 0.970 and MATCH 0.967 caught it, v2 did not (0.790). The mean was 0.930. Dilution.
- `stuff_human` (another document's continuation with the prompt's five rarest content words inserted after every 4th word): 0.825. No judge caught it.
- `stuff_generic` (the same insertion into the generic paragraph): 0.457.

## The design

### The average

SCRIBE v3.1 = (v2 · SURPRISE 3.1 · MATCH 3.1)^(1/3).

Each judge's score is exp(−D/τ), so the geometric mean is exp(−(D_v2/5 + D_S/τ_S + D_M/τ_M)/3): the three deviances add. A deviance far beyond any human text's (a copied prompt sentence costs MATCH a deviance of several hundred) drives the score to 0, whatever the other two judges say. The price is the other side of the same coin: a human text that one judge rejects confidently also scores low. `min3` and `disagree` are still reported.

### New statistics

All are two-sided ECDF z-scores against the human reference, as in v2 and v3.

| Statistic | Judge | What it measures |
|---|---|---|
| `w_fit` | SURPRISE | Per word: the code length of its sentence with the word, minus the code length with the word deleted, minus the word's own unigram code length (pack trigram, no history). That is minus the pointwise mutual information of the word with its two-sided context (left two words, right two words or the sentence end). Negative: the word belongs. Positive: the sentence reads better without it. `w_fit` is its mean. |
| `w_fithi` | SURPRISE | The share of words whose fit is above the reference words' 90th percentile (1.18 bits). |
| `w_anti` | SURPRISE | (v3, reported only there) The share of words the left context makes more than 2 bits less likely. |
| `w_pgain` | SURPRISE | With a prompt: the bits a unigram cache of the prompt's words saves on the text's content words, per word of the text. Two-sided: too much is stuffing or echo, too little is off the prompt. |
| `m_bridge` | MATCH | The share of words whose two pack bigrams (with the left and with the right neighbour) are both absent from the pack, while the bigram the two neighbours form without the word is present. |
| `m_pcoh` | MATCH | With a prompt: from each word start, the longest match (≥ 4 bytes) with the prompt; each covered byte counts the matched string's information in the pack above m_coh's floor. Two-sided, like `w_pgain`. |
| `m_echo64` | MATCH | The share of bytes inside full 64-byte matches with the prompt: a copied sentence. It replaces v3's `m_echo32` in MATCH 3.1. On the development battery, 3 of 200 human continuations repeat a 32–48-byte name or phrase of their prompt. `m_echo32` gave them z = 13–23 and a MATCH deviance of several hundred. None has a 64-byte match, and every copied-prompt text keeps a share of at least 0.18. |
| `w_lead` | SURPRISE | (Reported only.) The first sentence's prompt gain minus the mean of the other sentences'. A cheap probe of sentence order. |

### New groups

v3's deviance has the groups L (not words), S (not word order), C (cohesion, off the prompt) and P (too predictable). v3.1 adds two:

| Group | Sides | Form |
|---|---|---|
| **F**: foreign words | high `w_fit`, `w_fithi`, `w_anti` (SURPRISE); high `m_bridge` (MATCH) | saturating, like L and S: 60 · (1 − exp(−max(0, F − thF)/2)); evidence beyond a dead zone of 1; thF = the reference's 99th percentile |
| **B**: bland | low `w_mean`, `w_sd`, `w_hi`, `w_spell`, `w_fit`; high `w_lo`, `w_ctx` (SURPRISE). These sides were in P in v3. | linear beyond its threshold: 10 · max(0, B − thB); no dead zone; thB = the reference's 97.5th percentile |

Two more changes keep the ladder in order:

- **L caps at 90** in v3.1 (S stays at 30, F at 60). Word salad saturates both S and F. Without this, SURPRISE 3.1 ranked random letters above random real words (0.011 on dev A). Not being words must cost more than order plus foreign words together.
- The prompt statistics' low sides (`w_pgain`, `m_pcoh`) count in C from zero, like all other cohesion (graded: less relevant is worse at every level).

τ is set as in v3: the mean score of dev A's human samples is 0.909 for each new judge. That gives τ_S = 27 and τ_M = 32 (rounded).

| | SURPRISE 3.1 | MATCH 3.1 |
|---|---|---|
| L, S caps | 90, 30 (κ 2) | 90, 30 (κ 0.25) |
| F | cap 60, κ 2, dead zone 1, 99th percentile | the same |
| B | 10 per unit beyond the 97.5th percentile, dead zone 0 | (no bland side) |
| C, G, X | as v3 (wC 4, deadG 1.5, wP 8) | as v3 (wC 6, deadG 2, wP 20) |
| τ | 27 | 32 |
| reported only | `w_lead` | `m_echo32` |

Everything else is v3's (`V3Params::set_v31()` in native/include/cypha/eval/scribe.hpp).

## Held-out attacks

These eight were written and committed (commit 5defdb5) before any v3.1 design work. They are built only by `scribe attacks --heldout`, which was never run on development data. `scribe selftest` checks their construction on toy strings. They are read once, on fresh data, and reported with pass or fail at 0.95. No rule.

| Attack | Construction |
|---|---|
| `ho_stuff7_rare` | another document's continuation, the prompt's 5 rarest content words inserted after every **7th** word |
| `ho_stuff3_freq` | the same, the prompt's 5 **most frequent** content words after every **3rd** word |
| `ho_stuff_rand` | the 5 rarest words inserted at **random gaps** of 2 to 8 words |
| `ho_generic2` | one of **four other generic paragraphs** (none of v2's), in turn |
| `ho_template` | a fixed five-sentence template with the prompt's two most frequent capitalised words and two rarest lowercase content words **swapped in** |
| `ho_echo_scatter` | the true continuation with the prompt's sentences **scattered between** its sentences (c1 p1 c2 p2 …) |
| `ho_title_swap` | another document's continuation with its title entity (most frequent capitalised word) **replaced by the prompt's** |
| `ho_list_related` | a fluent **list of loosely related sentences**: other documents' sentences that share the most (rarity-weighted) content words with the prompt, one per document |

## Freeze record (written before the fresh region was read)

Development data, as for v3: ladder A (enwik8 [56, 64) MiB, seed 20261027, `--chain-rng`), ladder B ([64, 72), seed 31415926), and their attack batteries. Pack [16, 48), reference [48, 56), vectors vec32. The design was chosen on dev A and its battery only. Dev B was run once, after the design was frozen, and nothing was changed after it.

**Registered rules.**

1. v3.1 passes all five control pairs (human > sentences shuffled > words shuffled > random lexicon words > random letters > one phrase repeated) on dev A, dev B and the fresh ladder. A pair passes with accuracy ≥ 0.95 and disjoint 95% bootstrap intervals.
2. On the fresh battery, v3.1 ranks human text above each of the 9 registered attacks with pair accuracy ≥ 0.95.
3. The 8 held-out attacks are reported with pass or fail at 0.95. No rule.
4. The fresh ladder's human mean under v3.1 is within 0.05 of v3's fresh 0.906 (that is, in [0.856, 0.956]).

**Predictions** (written now):

- Rule 1 holds (about 3 in 4). The weak pair is human > sentences shuffled: 0.959 on dev A, 0.982 on dev B.
- Rule 2 is a coin flip (about 1 in 2). On dev A four attacks sit at 0.958–0.964 (`stuff_human`, `stuff_generic`, `generic`, `shuffle_docs`). The pair accuracy of one attack has a standard error of about 0.015 on 200 texts.
- Rule 4 holds (dev A 0.897, dev B 0.920).
- Held-out: `ho_echo_scatter` passes (64-byte echo). `ho_generic2` and `ho_template` probably pass (bland, off the prompt or repetitive). `ho_stuff3_freq` and `ho_stuff_rand` are borderline. `ho_stuff7_rare` fails (fewer inserted words). `ho_title_swap` and `ho_list_related` fail: they are fluent human sentences, the blind spots of `offprompt` and `cooc_out`. Expected: 3 to 5 of 8.

**Fresh data.** enwik8 is used up. The fresh region is enwik9 [100, 124) MiB, past enwik8's 10^8 bytes (enwik9's first 10^8 bytes are enwik8, checked with `cmp`). Seed 2718281828. enwik9 holds some pages twice, so the fresh read uses `--dedup`. It drops a test document when more than half of its bytes lie inside full 64-byte matches with the pack's or the reference's text. Checked before the read with `scribe dedup` (which scores nothing):

| Region | Documents dropped |
|---|---|
| dev A, enwik8 [56, 64) | 3 of 737 |
| dev B, enwik8 [64, 72) | 0 of 592 |
| v3's fresh, enwik8 [72, 95.37) | 1 of 1,868 |
| **v3.1's fresh, enwik9 [100, 124)** | **2 of 2,106** (1,699 of 12,655,049 bytes) |

Commands, registered (`E=/root/cdata/enwik8`, `E9=/root/work/enwik9_dl/enwik9`, vectors `/root/work/scribe/vec32`):

```
scribe ladder  --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E9:100:124 --vectors vec32 --profile v31 --seed 2718281828 --dedup --label v31_fresh
scribe attacks --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E9:100:124 --vectors vec32 --profile v31 --seed 2718281828 --dedup --heldout --outside enwik:$E:0:16 --n 200 --label att31_fresh
```

### Development results

Ladder, pair accuracy (every PASS has disjoint intervals):

| Pair | dev A v3 | SURPRISE 3.1 | MATCH 3.1 | **v3.1** | dev B v3 | SURPRISE 3.1 | MATCH 3.1 | **v3.1** |
|---|---|---|---|---|---|---|---|---|
| human > sentences shuffled | 0.982 | 0.915 | 0.948 | **0.959** | 0.988 | 0.961 | 0.953 | **0.982** |
| sentences shuffled > words shuffled | 1.000 | 1.000 | 0.981 | **0.998** | 0.997 | 1.000 | 0.986 | **0.994** |
| words shuffled > random lexicon words | 0.973 | 0.955 | 0.873 | **0.985** | 0.983 | 0.968 | 0.858 | **0.998** |
| random lexicon words > random letters | 1.000 | 0.992 | 1.000 | **1.000** | 1.000 | 1.000 | 1.000 | **1.000** |
| random letters > one phrase repeated | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 | 1.000 | **1.000** |
| **pairs passed** | 5 | 4 | 3 | **5** | 5 | 5 | 4 | **5** |

Rung means of v3.1 (dev A / dev B): human 0.897 [0.883, 0.909] / 0.920 [0.909, 0.930]; sentences shuffled 0.550 / 0.548; words shuffled 0.034 / 0.034; random lexicon words 0.018 / 0.018; random letters 0.002 / 0.002; one phrase repeated 0.000 / 0.000; human, short 0.929 / 0.943.

v3.1 passes all five pairs, but its first pair is thinner than v3's (0.959 against 0.982 on dev A). The new judges alone do not pass the ladder on dev A (SURPRISE 3.1 4 of 5, MATCH 3.1 3 of 5). F and B charge a few odd human texts (lists, glossaries, plain passages) that shuffling dilutes, and `m_bridge` charges shuffled words more than random words.

Attack battery, mean score with P(human > attack) in brackets:

| Attack | dev A v2 | v3 | SURPRISE 3.1 | MATCH 3.1 | **v3.1** | dev B v3 | **v3.1** |
|---|---|---|---|---|---|---|---|
| human | 0.890 | 0.873 | 0.854 | 0.782 | 0.828 | 0.878 | 0.842 |
| copy_pack | 0.101 (0.987) | 0.049 (1.000) | 0.019 (0.999) | 0.000 (1.000) | **0.000 (1.000)** | (0.996) | **(1.000)** |
| echo_repeat | 0.000 (1.000) | 0.002 (1.000) | 0.015 (1.000) | 0.000 (1.000) | **0.000 (1.000)** | (1.000) | **(1.000)** |
| echo_copy | 0.889 (0.501) | 0.598 (0.942) | 0.843 (0.537) | 0.000 (1.000) | **0.000 (1.000)** | (0.949) | **(1.000)** |
| stuff_human | 0.384 (0.840) | 0.560 (0.860) | 0.132 (0.982) | 0.508 (0.757) | **0.250 (0.964)** | (0.856) | **(0.974)** |
| stuff_generic | 0.970 (0.328) | 0.881 (0.536) | 0.102 (0.993) | 0.659 (0.650) | **0.387 (0.963)** | (0.503) | **(0.979)** |
| loop | 0.000 (1.000) | 0.007 (1.000) | 0.120 (0.988) | 0.003 (0.993) | **0.000 (1.000)** | (1.000) | **(1.000)** |
| generic | 0.835 (0.760) | 0.654 (0.923) | 0.318 (0.965) | 0.374 (0.954) | **0.462 (0.960)** | (0.929) | **(0.973)** |
| shuffle_docs | 0.352 (0.956) | 0.522 (0.964) | 0.587 (0.913) | 0.412 (0.933) | **0.428 (0.958)** | (0.990) | **(0.979)** |
| cooc_pack | 0.140 (0.982) | 0.076 (0.999) | 0.039 (0.997) | 0.000 (1.000) | **0.000 (1.000)** | (0.996) | **(1.000)** |
| reorder (extra) | 0.885 (0.508) | 0.871 (0.500) | 0.855 (0.500) | 0.782 (0.500) | 0.827 (0.500) | (0.507) | (0.505) |
| cooc_out (extra) | 0.889 (0.477) | 0.841 (0.582) | 0.817 (0.621) | 0.760 (0.536) | 0.804 (0.553) | (0.572) | (0.543) |
| offprompt (extra) | 0.890 (0.500) | 0.836 (0.647) | 0.717 (0.814) | 0.533 (0.843) | 0.688 (0.798) | (0.654) | (0.833) |
| **registered, v3.1 ≥ 0.95** | | | | | **9 of 9** | | **9 of 9** |

The v3.1 rule passes 9 of 9 on both development batteries. v3's own rule (v3 ≥ 0.95 and caught by SURPRISE or MATCH) passes 4 of 9 on dev A and 5 of 9 on dev B. The margins on dev A are thin: four attacks sit at 0.958–0.964.

### Design log (development data only)

Each arm is scored on dev A: the number of the 9 registered attacks with P(human > attack) ≥ 0.95 under the combination, the worst of them, and the ladder. Python scripts outside the repository recomputed the deviance from the z-scores that `--export` writes. They reproduce the C++ to within 10^−5.

| # | Arm | Attacks | Worst | Ladder | Outcome |
|---|---|---|---|---|---|
| 1 | v3 as frozen (mean of three) | 5 of 9 | 0.536 (stuff_generic) | 5 of 5 | the starting point |
| 2 | v3's judges, geometric mean | 6 of 9 | 0.542 | 5 of 5 (min 0.969) | fixes `echo_copy` (1.000) |
| 3 | v3's judges, power means with exponent −1, −2, −4; mean–minimum mixtures (λ 0.25–1) | ≤ 6 of 9 | ≤ 0.58 | 5 of 5 | no better for `generic` (0.904–0.912); the human mean falls to 0.83–0.89 |
| 4 | new statistics in v3's groups (fit in S, prompt gain P and C) | 6 of 9 | 0.828 | 5 of 5 | stuffing 0.93, generic 0.92 |
| 5 | fit statistics in a linear (not saturating) group | — | — | SURPRISE inverts random words > random letters (0.000) | dropped: insertion evidence must saturate |
| 6 | the whole "too predictable" sum, counted beyond its own 99th percentile | 5–6 of 9 | 0.86–0.88 | 5 of 5 | no help for `generic`: Wikipedia boilerplate sets the threshold high |
| 7 | caps of 60 and 120 for L and S | 5–7 of 9 | 0.85–0.91 | 3–5 of 5 | stuffing up a little, `shuffle_docs` and the ladder down |
| 8 | judge weights (τ × 0.5 to 1) | 5–7 of 9 | ≤ 0.92 | 5 of 5 | `generic` stays at 0.88–0.92 |
| 9 | Mahalanobis distance of the z-vector from the reference | — | — | — | as a lone statistic: `generic` 0.61 (SURPRISE) to 0.965 (MATCH), stuffing 0.65–0.99; dropped |
| 10 | bland group B at the 99th percentile | 5–6 of 9 | 0.91–0.94 | 5 of 5 | the generic paragraph is only just past the blandest 1% of Wikipedia |
| 11 | B at the 97.5th percentile, weight 10–30 | 8–9 of 9 | 0.92–0.96 | 4–5 of 5 | `generic` 0.955–0.98 |
| 12 | `m_echo64` in place of `m_echo32` | +0.015 on every attack | | | removes 3 of 200 human false positives |
| 13 | saturating F, cap 60, κ 2 | 9 of 9 | 0.954 | 5 of 5 | stuffing 0.97 |
| 14 | L cap 90 | | | SURPRISE 3.1: random words > random letters 0.011 → 0.992 | kept |
| 15 | dead zones in C (0.5, 1), `m_echo`'s low side off, prompt C dead zone 1 or 2 | 6–9 of 9 | 0.93–0.96 | 3–5 of 5 | no gain; τ swings |
| 16 | "isolated sentence" share (no content word shared with any other sentence or the prompt) | — | — | — | `shuffle_docs` 0.946 alone, the same evidence as cohesion; not added |
| 17 | mean of three times a veto beyond each judge's human 95th–99.5th percentile | 8–9 of 9 | 0.94–0.96 | 4–5 of 5 | the same frontier as the geometric mean |
| 18 | 108 settings of F and B (cap, κ, q, weight) | | | | the best min(first ladder pair, worst attack) is 0.960: the frontier is flat |
| — | **frozen** | **9 of 9** | **0.958** | **5 of 5 (min 0.959)** | |

The trade-off is real. Every failure mode a judge learns adds its own false positives on odd but human texts (data lists, citations, glossaries, plain passages). Under the geometric mean, these are not diluted either. On dev A no setting gives more than about 0.96 on both the first ladder pair and the worst attack.

One-step changes of the frozen settings (dev A, worst registered attack; the ladder keeps 5 of 5 in every case): B at the 99th percentile 0.923 (`generic`); B weight 5 0.936 (`generic`); F cap 45 0.949, F cap 90 0.950, F κ 1 0.950; prompt C dead zone 2 0.950; every other one-step change 0.952 or more. The design is fragile at the 0.95 line, and `generic` hangs on B.

**Sentence order (stretch, reported only).** `w_lead` ranks human text above reordered text at 0.556 on dev A. A check for a pronoun or "this/these/however" as the first word finds nothing (24% of human continuations, 20.5% of reordered ones). Sentence order stays invisible.

## K5 cross-check (report only)

As in SCRIBE_V3.md: the GEN-A dev ladder (enwik8 [56, 64), seed 20261027), with K5's 250 dev continuations (old K5: /root/work/k5m/OLD.gen.txt; new K5 with the sentence gate: NEW.gen.txt), `--gen-trie`. The prompt statistics use each GEN line's PROMPT.

| K5 dev samples | n | v2 | v3 | SURPRISE 3.1 | MATCH 3.1 | **v3.1** |
|---|---|---|---|---|---|---|
| old K5 | 242 | 0.353 [0.323, 0.386] | 0.554 [0.537, 0.573] | 0.644 [0.629, 0.658] | 0.456 [0.444, 0.469] | **0.435 [0.415, 0.455]** |
| new K5 (gate) | 244 | 0.498 [0.471, 0.527] | 0.565 [0.549, 0.580] | 0.554 [0.527, 0.578] | 0.464 [0.451, 0.478] | **0.483 [0.466, 0.501]** |
| change | | +0.145 | +0.011 | **−0.090** | +0.008 | **+0.048** |
| human (same ladder) | 500 | 0.923 | 0.922 | 0.930 | 0.924 | 0.919 |
| sentences shuffled | 500 | 0.344 | 0.557 | 0.736 | 0.677 | 0.546 |

- v3.1 sees a small gain for the gate (+0.048, intervals disjoint). It comes from v2: in a geometric mean, v2's large relative change (0.353 → 0.498) counts for more than in a plain mean.
- SURPRISE 3.1 again rates the new K5 lower (−0.090), as SURPRISE did (−0.106): the gated text is more predictable than human text.
- MATCH 3.1 does not move.
- Both K5 versions stay far below human text (0.919) and below sentences shuffled across documents (0.546) on v3.1.

## Cost

`scribe score` on the 2,600 texts of the dev A battery, one core:

| | texts/s | peak RSS |
|---|---|---|
| v2 | 1,379 | 385 MB |
| v3 | 518 | 493 MB |
| v3.1 (v2, v3's judges and v3.1's judges) | 448 | 491 MB |

v3.1's own statistics cost about 0.3 ms a text: four more trigram look-ups per word for the fit, one pack match per word for the bridge, and a scan of the prompt. Setup is the same as v3's. `scribe dedup` over a 24 MiB region takes about 30 s on 4 threads.

## Byte-for-byte checks

Profiles v1, v2 and v3 reproduce the build of main (6aed3eb) byte for byte, checked with `cmp`. The checks cover the dev A and dev B ladders in each profile, the K5 ladders (old and new, v2 and v3), both v3 attack batteries, and every export file (scores, texts, references, v3raw).

## Fresh verdict

Read once, 2026-11-04, with the registered commands, at the frozen commit 28ff62c (plus `--export` to keep the per-text scores). enwik9 [100, 124) MiB, seed 2718281828. The filter dropped 2 of 2,106 documents. 500 standard samples from 6,116 candidates. The ladder took 20 s and the battery 23 s. Nothing was changed after the read.

**Ladder** (pair accuracy):

| Pair | v2 | SURPRISE | MATCH | v3 | SURPRISE 3.1 | MATCH 3.1 | **v3.1** |
|---|---|---|---|---|---|---|---|
| human > sentences shuffled | 0.976 PASS | 0.935 FAIL | 0.940 FAIL | 0.979 PASS | 0.924 FAIL | 0.935 FAIL | **0.968 PASS** |
| sentences shuffled > words shuffled | 0.998 PASS | 1.000 PASS | 0.978 PASS | 1.000 PASS | 1.000 PASS | 0.978 PASS | **1.000 PASS** |
| words shuffled > random lexicon words | 0.996 PASS | 0.955 PASS | 0.951 PASS | 0.972 PASS | 0.956 PASS | 0.863 FAIL | **0.992 PASS** |
| random lexicon words > random letters | 1.000 PASS | 1.000 PASS | 1.000 PASS | 1.000 PASS | 1.000 PASS | 1.000 PASS | **1.000 PASS** |
| random letters > one phrase repeated | 1.000 PASS | 1.000 PASS | 1.000 PASS | 1.000 PASS | 1.000 PASS | 1.000 PASS | **1.000 PASS** |
| **pairs passed** | 5 | 4 | 4 | 5 | 4 | 3 | **5** |

Rung means of v3.1 (95% interval): human 0.904 [0.892, 0.916]; sentences shuffled 0.546; words shuffled 0.034; random lexicon words 0.018; random letters 0.002; one phrase repeated 0.000; human, short 0.941. On the same samples v3 has a human mean of 0.908.

**Battery** (mean score, P(human > attack) in brackets):

| Attack | v2 | SURPRISE | MATCH | v3 | SURPRISE 3.1 | MATCH 3.1 | **v3.1** | |
|---|---|---|---|---|---|---|---|---|
| human | 0.893 | 0.899 | 0.827 | 0.873 | 0.860 | 0.791 | 0.833 | |
| copy_pack | 0.104 (0.982) | 0.042 (0.996) | 0.000 (1.000) | 0.049 (0.995) | 0.009 (0.996) | 0.000 (1.000) | **0.000 (0.995)** | PASS |
| echo_repeat | 0.000 (1.000) | 0.005 (1.000) | 0.000 (1.000) | 0.002 (1.000) | 0.015 (0.995) | 0.000 (1.000) | **0.000 (1.000)** | PASS |
| echo_copy | 0.887 (0.500) | 0.892 (0.507) | 0.000 (0.997) | 0.593 (0.954) | 0.846 (0.530) | 0.000 (0.997) | **0.000 (0.996)** | PASS |
| stuff_human | 0.477 (0.791) | 0.676 (0.770) | 0.689 (0.637) | 0.614 (0.825) | 0.180 (0.967) | 0.547 (0.745) | **0.305 (0.953)** | PASS |
| stuff_generic | 0.987 (0.299) | 0.853 (0.653) | 0.872 (0.408) | 0.904 (0.467) | 0.106 (0.995) | 0.725 (0.566) | **0.412 (0.967)** | PASS |
| loop | 0.000 (1.000) | 0.019 (0.998) | 0.003 (0.983) | 0.007 (0.998) | 0.121 (0.989) | 0.004 (0.993) | **0.000 (0.995)** | PASS |
| generic | 0.835 (0.770) | 0.653 (0.950) | 0.475 (0.944) | 0.654 (0.937) | 0.318 (0.978) | 0.374 (0.951) | **0.462 (0.964)** | PASS |
| shuffle_docs | 0.336 (0.968) | 0.662 (0.936) | 0.519 (0.932) | 0.506 (0.975) | 0.573 (0.935) | 0.400 (0.947) | **0.417 (0.966)** | PASS |
| cooc_pack | 0.127 (0.983) | 0.092 (0.995) | 0.000 (0.995) | 0.073 (0.995) | 0.046 (0.994) | 0.000 (0.995) | **0.000 (0.995)** | PASS |
| reorder (extra) | 0.888 (0.506) | 0.900 (0.499) | 0.828 (0.499) | 0.872 (0.504) | 0.860 (0.501) | 0.792 (0.498) | 0.832 (0.503) | — |
| cooc_out (extra) | 0.861 (0.491) | 0.796 (0.693) | 0.765 (0.563) | 0.807 (0.592) | 0.777 (0.649) | 0.724 (0.571) | 0.770 (0.573) | — |
| offprompt (extra) | 0.893 (0.500) | 0.899 (0.500) | 0.705 (0.751) | 0.833 (0.651) | 0.718 (0.815) | 0.529 (0.866) | 0.687 (0.809) | — |
| ho_stuff7_rare | 0.833 (0.515) | 0.900 (0.446) | 0.938 (0.230) | 0.890 (0.392) | 0.624 (0.679) | 0.898 (0.283) | **0.739 (0.585)** | FAIL |
| ho_stuff3_freq | 0.107 (0.965) | 0.455 (0.932) | 0.587 (0.727) | 0.383 (0.960) | 0.066 (0.995) | 0.579 (0.711) | **0.140 (0.984)** | PASS |
| ho_stuff_rand | 0.702 (0.603) | 0.815 (0.598) | 0.859 (0.370) | 0.792 (0.570) | 0.350 (0.886) | 0.762 (0.481) | **0.516 (0.831)** | FAIL |
| ho_generic2 | 0.838 (0.670) | 0.492 (0.902) | 0.578 (0.864) | 0.636 (0.882) | 0.275 (0.981) | 0.453 (0.911) | **0.458 (0.943)** | FAIL |
| ho_template | 0.999 (0.269) | 0.721 (0.822) | 0.988 (0.166) | 0.903 (0.507) | 0.592 (0.816) | 0.979 (0.168) | **0.817 (0.578)** | FAIL |
| ho_echo_scatter | 0.467 (0.759) | 0.743 (0.677) | 0.000 (0.999) | 0.403 (0.973) | 0.743 (0.647) | 0.003 (0.998) | **0.006 (0.996)** | PASS |
| ho_title_swap | 0.891 (0.500) | 0.907 (0.490) | 0.749 (0.682) | 0.849 (0.603) | 0.815 (0.628) | 0.627 (0.751) | **0.757 (0.680)** | FAIL |
| ho_list_related | 0.562 (0.859) | 0.720 (0.888) | 0.659 (0.800) | 0.647 (0.898) | 0.735 (0.807) | 0.638 (0.746) | **0.618 (0.856)** | FAIL |

**Verdicts.**

| Rule | Result |
|---|---|
| 1. v3.1 passes all five ladder pairs on dev A, dev B and fresh | **holds**: 5 of 5 on each (fresh first pair 0.968) |
| 2. v3.1 ≥ 0.95 on each of the 9 registered attacks (fresh) | **holds**: 9 of 9. `stuff_human` only just (0.953), then `generic` 0.964, `shuffle_docs` 0.966, `stuff_generic` 0.967 |
| 3. held-out attacks (no rule) | **2 of 8 pass**: `ho_stuff3_freq` 0.984, `ho_echo_scatter` 0.996 |
| 4. fresh human mean within 0.05 of 0.906 | **holds**: 0.904 |

On the same fresh battery, v3's own rule passes 5 of 9, as it did on enwik8.

**Predictions, scored.** Rules 1, 2 and 4 held, as predicted for 1 and 4; I had called rule 2 a coin flip. For the held-out attacks I predicted 3 to 5 of 8; the result is 2. The calls on `ho_echo_scatter` (pass), `ho_stuff7_rare`, `ho_title_swap` and `ho_list_related` (fail) were right. `ho_stuff3_freq` (called borderline) passed and `ho_stuff_rand` (borderline) failed. `ho_generic2` (called a pass) missed by 0.007, and `ho_template` (called a pass) failed clearly.

**Why the held-out attacks got through** (looked at after the read; nothing was changed):

- **Stuffing is caught by its rate, not its pattern.** One inserted word in 4 (frequent words) is caught at 0.984, one in 5 at 0.953, about one in 6 at random gaps at 0.831, and one in 8 at 0.585. No statistic looks at the spacing. The fit and bridge statistics are averages over the text. At one word in 8 they move by about 1.8 standard deviations, inside the human range.
- **Generic paragraphs:** three of the four new paragraphs are caught (0.965–0.975). The first ("There are many things to say about this subject…") is not (0.862): v2 rates it 1.000, and SURPRISE 3.1 alone reaches only 0.942.
- **The template** is bland (`w_mean` z −2.6), but it repeats the prompt's names and its own words. MATCH 3.1 sees an on-topic, cohesive text and rates it above human text (0.979). Its cohesion and prompt statistics charge every shortfall (graded, from zero). An excess is charged only beyond a dead zone of 2 standard deviations, or beyond the human 99th percentile. v2 rates it 0.999.
- **Title swap and loosely related sentences** are fluent human sentences. They are the same blind spot as `offprompt` (0.809) and `cooc_out` (0.573).

## What v3.1 is, and is not

- **It does not dilute a confident catch.** Copied prompt sentences, copying from the pack, loops and echo all go to 0 under the geometric mean. That holds even when two judges rate the text as human (`echo_copy`: 0.948 under v3, 0.996 under v3.1; `ho_echo_scatter` 0.996).
- **It catches keyword stuffing at one inserted word in five or more**, by a principled statistic: words that fit neither neighbour, as code length (`w_fit`) and as strings (`m_bridge`). It does not catch it at one word in six to eight (0.83, 0.59). So it is not fool-proof against a stuffer who inserts less.
- **It catches v2's generic paragraph and most bland paragraphs** (0.964; three of four new ones at 0.965–0.975), through the bland group B. One plain paragraph of the same kind got through (0.862). B is the most tuned part of the design: at the 99th percentile instead of the 97.5th it would have missed `generic` on dev A.
- **It does not see:**
  - sentence order (`reorder` 0.50; `w_lead` 0.556 on dev A);
  - fluent human sentences from elsewhere, whether gathered by co-occurrence, by shared words, or with the title swapped in (`cooc_out` 0.57, `ho_list_related` 0.86, `ho_title_swap` 0.68, `offprompt` 0.81);
  - a template that repeats the prompt's names (0.58);
  - truth (numbers, negations).
- **Its margins are thin.** On dev A the first ladder pair is 0.959 and four attacks sit at 0.958–0.964. On fresh data `stuff_human` passed at 0.953. Each failure mode a judge learns adds false positives on odd human texts (lists, citations, glossaries, plain passages), and the geometric mean does not dilute those either. On dev A no setting of the new groups gave more than about 0.96 on both the first ladder pair and the worst attack.
- **The new judges are weaker on their own than v3's.** SURPRISE 3.1 alone passes 4 of 5 ladder pairs on fresh data, MATCH 3.1 alone 3 of 5. Only the geometric mean passes 5 of 5. Read v3.1 as one score, with v2, `min3` and `disagree` next to it.
- **With a prompt, scores are lower.** The prompt statistics' graded shortfalls charge human continuations too: the battery's human mean is 0.833 against the ladder's 0.904. Compare prompted texts with prompted texts.
- **K5:** v3.1 sees a small gain from the sentence gate (+0.048), carried by v2. SURPRISE 3.1 rates the gated text lower.
- **Freeze discipline.** The design was chosen on dev A and its battery. Dev B was read once after the freeze. The rules, predictions, fresh region and filter were committed in 28ff62c before the fresh read. The fresh region was read once, and nothing was changed after it. The held-out attacks were committed (5defdb5) before any design work and never run on development data.

## Reproduce

```
E=/root/cdata/enwik8; V=/root/work/scribe/vec32
scribe ladder  --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:56:64 --vectors $V --profile v31 --seed 20261027 --chain-rng   # dev A
scribe ladder  --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:64:72 --vectors $V --profile v31 --seed 31415926                # dev B
scribe attacks --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:56:64 --vectors $V --profile v31 --seed 20261027 --outside enwik:$E:0:16   # dev A battery
scribe attacks --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:64:72 --vectors $V --profile v31 --seed 31415926 --outside enwik:$E:0:16   # dev B battery
scribe ladder  --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:56:64 --vectors $V --profile v31 --seed 20261027 --gen-trie NEW.gen.txt   # K5
scribe dedup   --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:/root/work/enwik9_dl/enwik9:100:124                                    # the filter's count
scribe score   --pack enwik:$E:16:48 --ref enwik:$E:48:56 --vectors $V --profile v31 --in texts.jsonl --out scores.jsonl
```

`scribe selftest` (CTest `native_scribe_selftest`, 119 checks) covers the v3.1 parts:
- an inserted word's fit (positive, above every word of its sentence);
- `m_bridge` counting exactly the inserted word;
- prompt gain and prompt cohesion rising with the prompt's words;
- the F and B groups (saturation, the bland charge, v3 ignoring them, L's cap);
- the geometric mean not diluting a confident catch;
- the parameter parser;
- the near-duplicate filter;
- the construction of all eight held-out attacks on toy strings.
