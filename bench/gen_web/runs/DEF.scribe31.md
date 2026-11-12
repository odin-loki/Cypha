# SCRIBE DEF ladder: raw:/mnt/d/cypha_llm/genw/rungs.jsonl (profile v31)

Near-duplicate filter (--dedup): dropped 55 of 4910 test documents (67280 of 10185347 bytes) with more than half their bytes inside 64-byte matches with the pack or the reference.

Pack raw:/mnt/d/cypha_llm/genw/pack.jsonl: 18830 docs, 38951819 bytes, lexicon 71797 words, 4348179 trigrams; vectors: /mnt/d/cypha_llm/genw/vecW. Reference raw:/mnt/d/cypha_llm/genw/ref.jsonl: 9394 standard + 12715 short samples (seed 20261027). Test: 500 standard samples (of 9408 candidates), 500 short (seed 20261112).

Trie rung: 250 GEN lines, 250 kept after cleaning (0 under 100 B dropped).

## Rung means (95% bootstrap interval of the mean, 1000 resamples)

| rung | n | SCRIBE | parity | surface S | mean bytes |
|---|---|---|---|---|---|
| held-out human | 500 | 0.901 [0.888, 0.915] | 1.00 | 0.921 [0.913, 0.927] | 525 |
| trie samples (pooled) | 250 | 0.622 [0.605, 0.638] | 0.69 | 0.933 [0.929, 0.938] | 439 |
| sentences shuffled across docs | 500 | 0.592 [0.584, 0.600] | 0.66 | 0.904 [0.901, 0.907] | 523 |
| words shuffled within sentences | 500 | 0.057 [0.056, 0.059] | 0.06 | 0.930 [0.924, 0.935] | 525 |
| random lexicon words | 500 | 0.031 [0.030, 0.031] | 0.03 | 0.897 [0.895, 0.899] | 514 |
| random letters | 500 | 0.005 [0.004, 0.005] | 0.01 | 0.067 [0.064, 0.069] | 525 |
| one phrase repeated | 500 | 0.000 [0.000, 0.000] | 0.00 | 0.028 [0.028, 0.029] | 542 |
| trie samples T=1.0 | 250 | 0.622 [0.605, 0.639] | 0.69 | 0.933 [0.930, 0.937] | 439 |
| held-out human, short (60-260 B) | 500 | 0.938 [0.928, 0.948] | 1.04 | 0.922 [0.914, 0.929] | 179 |

## Ladders: pairwise ordering accuracy over all (upper, lower) pairs (ties = 1/2) and disjoint 95% intervals; PASS needs both (accuracy >= 0.95)

### trie pooled

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > trie samples (pooled) | 0.919 | yes | FAIL | 0.507 | no | FAIL |
| trie samples (pooled) > sentences shuffled across docs | 0.554 | yes | FAIL | 0.738 | yes | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 1.000 | yes | PASS | 0.280 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.965 | yes | PASS | 0.793 | yes | FAIL |
| random lexicon words > random letters | 1.000 | yes | PASS | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.917 | yes | FAIL |

Verdict (trie pooled): SCRIBE FAIL (4 of 6 pairs), surface S FAIL (1 of 6 pairs)

### trie samples T=1.0

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > trie samples T=1.0 | 0.919 | yes | FAIL | 0.507 | no | FAIL |
| trie samples T=1.0 > sentences shuffled across docs | 0.554 | yes | FAIL | 0.738 | yes | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 1.000 | yes | PASS | 0.280 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.965 | yes | PASS | 0.793 | yes | FAIL |
| random lexicon words > random letters | 1.000 | yes | PASS | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.917 | yes | FAIL |

Verdict (trie samples T=1.0): SCRIBE FAIL (4 of 6 pairs), surface S FAIL (1 of 6 pairs)

### control ladder (no model rung)

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > sentences shuffled across docs | 0.954 | yes | PASS | 0.686 | yes | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 1.000 | yes | PASS | 0.280 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.965 | yes | PASS | 0.793 | yes | FAIL |
| random lexicon words > random letters | 1.000 | yes | PASS | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.917 | yes | FAIL |

Verdict (control ladder (no model rung)): SCRIBE PASS (5 of 5 pairs), surface S FAIL (1 of 5 pairs)

## Axis profile (mean over samples)

| rung | n | SCRIBE | mean D | D_L | D_S | D_C | D_I | D_R | share D_L > 0 | share D_S > 0 | share D_I > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out human | 500 | 0.914 | 0.74 | 0.12 | 0.09 | 0.45 | 0.07 | 0.00 | 1% | 2% | 0% |
| trie samples (pooled) | 250 | 0.691 | 2.11 | 0.00 | 0.00 | 2.11 | 0.00 | 0.00 | 0% | 0% | 0% |
| sentences shuffled across docs | 500 | 0.404 | 4.88 | 0.00 | 0.01 | 4.85 | 0.02 | 0.00 | 0% | 0% | 0% |
| words shuffled within sentences | 500 | 0.007 | 25.18 | 0.12 | 24.62 | 0.44 | 0.00 | 0.00 | 1% | 100% | 0% |
| random lexicon words | 500 | 0.003 | 31.46 | 0.00 | 25.77 | 5.69 | 0.00 | 0.00 | 0% | 100% | 0% |
| random letters | 500 | 0.001 | 39.83 | 27.04 | 10.17 | 2.63 | 0.00 | 0.00 | 100% | 94% | 0% |
| one phrase repeated | 500 | 0.000 | 153.75 | 0.64 | 2.31 | 1.75 | 149.05 | 0.00 | 12% | 20% | 100% |
| trie samples T=1.0 | 250 | 0.691 | 2.11 | 0.00 | 0.00 | 2.11 | 0.00 | 0.00 | 0% | 0% | 0% |
| held-out human, short (60-260 B) | 500 | 0.944 | 0.42 | 0.02 | 0.03 | 0.26 | 0.11 | 0.00 | 1% | 1% | 1% |

## V4: Pearson (Spearman) of SCRIBE with bytes on held-out human text

standard 0.078 (0.106); short 0.142 (0.150); pooled -0.099 (-0.078). Mean SCRIBE standard 0.901, short 0.938.

## SCRIBE v3: the three judges and their mean

v3 = mean(v2, SURPRISE, MATCH). Reference word thresholds for SURPRISE: 4.74 and 16.97 bits; MATCH's echo reference: 1000 standard + 0 short prompted continuations.

v3.1 (docs/reports/SCRIBE_V31.md) = the power mean (exponent 0; 0 = geometric) of v2, SURPRISE 3.1 and MATCH 3.1, the new judges with the v3.1 statistics; fit threshold 1.27 bits.

| rung | n | v2 | SURPRISE | MATCH | v3 | min of 3 | SURPRISE 3.1 | MATCH 3.1 | v3.1 | disagree |
|---|---|---|---|---|---|---|---|---|---|---|
| held-out human | 500 | 0.914 [0.899, 0.928] | 0.901 [0.888, 0.913] | 0.903 [0.888, 0.917] | 0.906 [0.895, 0.917] | 0.840 [0.823, 0.856] | 0.911 [0.896, 0.924] | 0.903 [0.888, 0.917] | 0.901 [0.886, 0.915] | 3% |
| trie samples (pooled) | 250 | 0.691 [0.663, 0.717] | 0.754 [0.737, 0.768] | 0.648 [0.632, 0.664] | 0.698 [0.682, 0.714] | 0.591 [0.571, 0.610] | 0.682 [0.667, 0.697] | 0.530 [0.512, 0.549] | 0.622 [0.606, 0.640] | 0% |
| sentences shuffled across docs | 500 | 0.404 [0.391, 0.418] | 0.697 [0.691, 0.702] | 0.694 [0.688, 0.701] | 0.598 [0.592, 0.605] | 0.396 [0.385, 0.409] | 0.762 [0.757, 0.768] | 0.708 [0.702, 0.714] | 0.592 [0.584, 0.601] | 5% |
| words shuffled within sentences | 500 | 0.007 [0.007, 0.008] | 0.253 [0.250, 0.256] | 0.743 [0.722, 0.763] | 0.334 [0.327, 0.341] | 0.007 [0.007, 0.008] | 0.040 [0.040, 0.041] | 0.721 [0.701, 0.741] | 0.057 [0.056, 0.059] | 76% |
| random lexicon words | 500 | 0.003 [0.002, 0.006] | 0.182 [0.180, 0.185] | 0.524 [0.506, 0.540] | 0.237 [0.231, 0.242] | 0.002 [0.002, 0.003] | 0.031 [0.031, 0.031] | 0.518 [0.502, 0.536] | 0.031 [0.030, 0.031] | 54% |
| random letters | 500 | 0.001 [0.001, 0.001] | 0.104 [0.102, 0.106] | 0.063 [0.063, 0.064] | 0.056 [0.055, 0.057] | 0.001 [0.001, 0.001] | 0.020 [0.020, 0.021] | 0.012 [0.011, 0.012] | 0.005 [0.004, 0.005] | 0% |
| one phrase repeated | 500 | 0.000 [0.000, 0.000] | 0.002 [0.001, 0.002] | 0.007 [0.006, 0.007] | 0.003 [0.003, 0.003] | 0.000 [0.000, 0.000] | 0.004 [0.004, 0.005] | 0.008 [0.007, 0.008] | 0.000 [0.000, 0.000] | 0% |
| trie samples T=1.0 | 250 | 0.691 [0.664, 0.717] | 0.754 [0.739, 0.769] | 0.648 [0.633, 0.667] | 0.698 [0.683, 0.715] | 0.591 [0.572, 0.610] | 0.682 [0.667, 0.696] | 0.530 [0.511, 0.549] | 0.622 [0.606, 0.639] | 0% |
| held-out human, short (60-260 B) | 500 | 0.944 [0.933, 0.955] | 0.942 [0.933, 0.952] | 0.947 [0.939, 0.956] | 0.945 [0.937, 0.952] | 0.888 [0.874, 0.902] | 0.946 [0.935, 0.957] | 0.944 [0.934, 0.954] | 0.938 [0.928, 0.948] | 3% |

Deviance terms (mean over samples; L lexical, S order, C cohesion, P too predictable):

| rung | SURPRISE D | D_L | D_S | D_C | D_P | MATCH D | D_L | D_S | D_C | D_P |
|---|---|---|---|---|---|---|---|---|---|---|
| held-out human | 2.48 | 0.15 | 0.09 | 1.71 | 0.52 | 5.01 | 0.29 | 0.12 | 2.50 | 2.10 |
| trie samples (pooled) | 5.93 | 0.00 | 0.02 | 5.33 | 0.57 | 13.58 | 0.00 | 0.00 | 13.57 | 0.01 |
| sentences shuffled across docs | 7.31 | 0.00 | 0.01 | 7.20 | 0.10 | 11.15 | 0.06 | 0.00 | 10.86 | 0.22 |
| words shuffled within sentences | 27.75 | 0.17 | 26.30 | 0.93 | 0.35 | 11.07 | 0.30 | 7.17 | 3.55 | 0.05 |
| random lexicon words | 34.20 | 0.09 | 27.71 | 6.16 | 0.24 | 21.67 | 0.18 | 10.50 | 10.99 | 0.00 |
| random letters | 45.97 | 30.00 | 8.48 | 7.28 | 0.20 | 82.90 | 30.00 | 30.00 | 22.90 | 0.00 |
| one phrase repeated | 135.12 | 0.48 | 1.04 | 0.71 | 132.89 | 154.06 | 0.87 | 4.78 | 1.06 | 147.36 |
| trie samples T=1.0 | 5.93 | 0.00 | 0.02 | 5.33 | 0.57 | 13.58 | 0.00 | 0.00 | 13.57 | 0.01 |
| held-out human, short (60-260 B) | 1.36 | 0.06 | 0.08 | 0.86 | 0.35 | 1.88 | 0.11 | 0.11 | 1.30 | 0.36 |

Deviance terms of the v3.1 judges (mean over samples; L lexical, S order, C cohesion, P too predictable, F foreign words, B bland):

| rung | SURPRISE D | D_L | D_S | D_C | D_P | D_F | D_B | MATCH D | D_L | D_S | D_C | D_P | D_F | D_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out human | 3.78 | 0.46 | 0.09 | 1.71 | 0.21 | 0.33 | 0.98 | 5.72 | 0.88 | 0.12 | 2.50 | 2.10 | 0.12 | 0.00 |
| trie samples (pooled) | 10.77 | 0.00 | 0.02 | 9.93 | 0.25 | 0.00 | 0.57 | 21.46 | 0.00 | 0.00 | 21.44 | 0.01 | 0.00 | 0.00 |
| sentences shuffled across docs | 7.46 | 0.00 | 0.01 | 7.20 | 0.06 | 0.07 | 0.12 | 11.33 | 0.18 | 0.00 | 10.86 | 0.22 | 0.06 | 0.00 |
| words shuffled within sentences | 87.20 | 0.50 | 26.30 | 0.93 | 0.16 | 59.30 | 0.00 | 13.65 | 0.89 | 7.17 | 3.55 | 0.05 | 1.99 | 0.00 |
| random lexicon words | 93.89 | 0.26 | 27.71 | 6.16 | 0.01 | 59.76 | 0.00 | 24.06 | 0.54 | 10.50 | 10.99 | 0.00 | 2.04 | 0.00 |
| random letters | 106.00 | 90.00 | 8.48 | 7.28 | 0.01 | 0.23 | 0.00 | 142.90 | 90.00 | 30.00 | 22.90 | 0.00 | 0.00 | 0.00 |
| one phrase repeated | 167.34 | 1.45 | 1.04 | 0.71 | 25.45 | 10.96 | 127.73 | 163.78 | 2.60 | 4.78 | 1.06 | 147.36 | 7.99 | 0.00 |
| trie samples T=1.0 | 10.77 | 0.00 | 0.02 | 9.93 | 0.25 | 0.00 | 0.57 | 21.46 | 0.00 | 0.00 | 21.44 | 0.01 | 0.00 | 0.00 |
| held-out human, short (60-260 B) | 1.96 | 0.17 | 0.08 | 0.86 | 0.16 | 0.11 | 0.58 | 2.36 | 0.34 | 0.11 | 1.30 | 0.36 | 0.24 | 0.00 |

### v3 parts: trie pooled

| upper > lower | v2 | | SURPRISE | | MATCH | | v3 | | min of 3 | | SURPRISE 3.1 | | MATCH 3.1 | | v3.1 | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out human > trie samples (pooled) | 0.817 | FAIL | 0.816 | FAIL | 0.901 | FAIL | 0.875 | FAIL | 0.870 | FAIL | 0.915 | FAIL | 0.933 | FAIL | 0.919 | FAIL |
| trie samples (pooled) > sentences shuffled across docs | 0.859 | FAIL | 0.642 | FAIL | 0.302 | FAIL | 0.734 | FAIL | 0.822 | FAIL | 0.206 | FAIL | 0.122 | FAIL | 0.554 | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 1.000 | PASS | 1.000 | PASS | 0.363 | FAIL | 0.999 | PASS | 1.000 | PASS | 1.000 | PASS | 0.409 | FAIL | 1.000 | PASS |
| words shuffled within sentences > random lexicon words | 0.983 | PASS | 0.943 | FAIL | 0.774 | FAIL | 0.812 | FAIL | 0.983 | PASS | 0.950 | FAIL | 0.748 | FAIL | 0.965 | PASS |
| random lexicon words > random letters | 0.901 | FAIL | 0.988 | PASS | 1.000 | PASS | 1.000 | PASS | 0.901 | FAIL | 0.990 | PASS | 1.000 | PASS | 1.000 | PASS |
| random letters > one phrase repeated | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 0.967 | PASS | 0.890 | FAIL | 1.000 | PASS |

Pairs passed (trie pooled): v2 3 of 6; SURPRISE 3 of 6; MATCH 2 of 6; v3 3 of 6; min of 3 3 of 6; SURPRISE 3.1 3 of 6; MATCH 3.1 1 of 6; v3.1 4 of 6

### v3 parts: trie samples T=1.0

| upper > lower | v2 | | SURPRISE | | MATCH | | v3 | | min of 3 | | SURPRISE 3.1 | | MATCH 3.1 | | v3.1 | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out human > trie samples T=1.0 | 0.817 | FAIL | 0.816 | FAIL | 0.901 | FAIL | 0.875 | FAIL | 0.870 | FAIL | 0.915 | FAIL | 0.933 | FAIL | 0.919 | FAIL |
| trie samples T=1.0 > sentences shuffled across docs | 0.859 | FAIL | 0.642 | FAIL | 0.302 | FAIL | 0.734 | FAIL | 0.822 | FAIL | 0.206 | FAIL | 0.122 | FAIL | 0.554 | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 1.000 | PASS | 1.000 | PASS | 0.363 | FAIL | 0.999 | PASS | 1.000 | PASS | 1.000 | PASS | 0.409 | FAIL | 1.000 | PASS |
| words shuffled within sentences > random lexicon words | 0.983 | PASS | 0.943 | FAIL | 0.774 | FAIL | 0.812 | FAIL | 0.983 | PASS | 0.950 | FAIL | 0.748 | FAIL | 0.965 | PASS |
| random lexicon words > random letters | 0.901 | FAIL | 0.988 | PASS | 1.000 | PASS | 1.000 | PASS | 0.901 | FAIL | 0.990 | PASS | 1.000 | PASS | 1.000 | PASS |
| random letters > one phrase repeated | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 0.967 | PASS | 0.890 | FAIL | 1.000 | PASS |

Pairs passed (trie samples T=1.0): v2 3 of 6; SURPRISE 3 of 6; MATCH 2 of 6; v3 3 of 6; min of 3 3 of 6; SURPRISE 3.1 3 of 6; MATCH 3.1 1 of 6; v3.1 4 of 6

### v3 parts: control ladder (no model rung)

| upper > lower | v2 | | SURPRISE | | MATCH | | v3 | | min of 3 | | SURPRISE 3.1 | | MATCH 3.1 | | v3.1 | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out human > sentences shuffled across docs | 0.967 | PASS | 0.925 | FAIL | 0.916 | FAIL | 0.962 | PASS | 0.953 | PASS | 0.913 | FAIL | 0.910 | FAIL | 0.954 | PASS |
| sentences shuffled across docs > words shuffled within sentences | 1.000 | PASS | 1.000 | PASS | 0.363 | FAIL | 0.999 | PASS | 1.000 | PASS | 1.000 | PASS | 0.409 | FAIL | 1.000 | PASS |
| words shuffled within sentences > random lexicon words | 0.983 | PASS | 0.943 | FAIL | 0.774 | FAIL | 0.812 | FAIL | 0.983 | PASS | 0.950 | FAIL | 0.748 | FAIL | 0.965 | PASS |
| random lexicon words > random letters | 0.901 | FAIL | 0.988 | PASS | 1.000 | PASS | 1.000 | PASS | 0.901 | FAIL | 0.990 | PASS | 1.000 | PASS | 1.000 | PASS |
| random letters > one phrase repeated | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 0.967 | PASS | 0.890 | FAIL | 1.000 | PASS |

Pairs passed (control ladder (no model rung)): v2 4 of 5; SURPRISE 3 of 5; MATCH 2 of 5; v3 4 of 5; min of 3 4 of 5; SURPRISE 3.1 3 of 5; MATCH 3.1 1 of 5; v3.1 5 of 5
