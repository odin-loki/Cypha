# SCRIBE DG2 ladder: enwik:/mnt/d/cypha_llm/enwik8:56:64 (profile v31)

Pack enwik:/mnt/d/cypha_llm/enwik8:16:48: 2600 docs, 17521535 bytes, lexicon 47691 words, 2051016 trigrams; vectors: /mnt/d/cypha_llm/scribe/v32. Reference enwik:/mnt/d/cypha_llm/enwik8:48:56: 1909 standard + 1729 short samples (seed 20261027). Test: 500 standard samples (of 2135 candidates), 500 short (seed 20261027).

Trie rung: 250 GEN lines, 244 kept after cleaning (6 under 100 B dropped).

## Rung means (95% bootstrap interval of the mean, 1000 resamples)

| rung | n | SCRIBE | parity | surface S | mean bytes |
|---|---|---|---|---|---|
| held-out human | 500 | 0.919 [0.909, 0.930] | 1.00 | 0.912 [0.906, 0.918] | 531 |
| trie samples (pooled) | 244 | 0.545 [0.519, 0.570] | 0.59 | 0.934 [0.930, 0.938] | 432 |
| sentences shuffled across docs | 500 | 0.546 [0.539, 0.553] | 0.59 | 0.905 [0.901, 0.908] | 532 |
| words shuffled within sentences | 500 | 0.035 [0.034, 0.035] | 0.04 | 0.923 [0.919, 0.927] | 531 |
| random lexicon words | 500 | 0.018 [0.018, 0.019] | 0.02 | 0.896 [0.894, 0.898] | 523 |
| random letters | 500 | 0.001 [0.001, 0.002] | 0.00 | 0.035 [0.034, 0.037] | 531 |
| one phrase repeated | 500 | 0.000 [0.000, 0.000] | 0.00 | 0.029 [0.028, 0.029] | 550 |
| trie samples T=1.0 | 244 | 0.545 [0.517, 0.572] | 0.59 | 0.934 [0.930, 0.938] | 432 |
| held-out human, short (60-260 B) | 500 | 0.931 [0.917, 0.943] | 1.01 | 0.909 [0.901, 0.915] | 188 |

## Ladders: pairwise ordering accuracy over all (upper, lower) pairs (ties = 1/2) and disjoint 95% intervals; PASS needs both (accuracy >= 0.95)

### trie pooled

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > trie samples (pooled) | 0.932 | yes | FAIL | 0.432 | no | FAIL |
| trie samples (pooled) > sentences shuffled across docs | 0.444 | no | FAIL | 0.701 | yes | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 0.998 | yes | PASS | 0.372 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.998 | yes | PASS | 0.724 | yes | FAIL |
| random lexicon words > random letters | 1.000 | yes | PASS | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.590 | yes | FAIL |

Verdict (trie pooled): SCRIBE FAIL (4 of 6 pairs), surface S FAIL (1 of 6 pairs)

### trie samples T=1.0

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > trie samples T=1.0 | 0.932 | yes | FAIL | 0.432 | no | FAIL |
| trie samples T=1.0 > sentences shuffled across docs | 0.444 | no | FAIL | 0.701 | yes | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 0.998 | yes | PASS | 0.372 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.998 | yes | PASS | 0.724 | yes | FAIL |
| random lexicon words > random letters | 1.000 | yes | PASS | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.590 | yes | FAIL |

Verdict (trie samples T=1.0): SCRIBE FAIL (4 of 6 pairs), surface S FAIL (1 of 6 pairs)

### control ladder (no model rung)

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > sentences shuffled across docs | 0.979 | yes | PASS | 0.592 | no | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 0.998 | yes | PASS | 0.372 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.998 | yes | PASS | 0.724 | yes | FAIL |
| random lexicon words > random letters | 1.000 | yes | PASS | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.590 | yes | FAIL |

Verdict (control ladder (no model rung)): SCRIBE PASS (5 of 5 pairs), surface S FAIL (1 of 5 pairs)

## Axis profile (mean over samples)

| rung | n | SCRIBE | mean D | D_L | D_S | D_C | D_I | D_R | share D_L > 0 | share D_S > 0 | share D_I > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out human | 500 | 0.923 | 0.54 | 0.02 | 0.00 | 0.38 | 0.13 | 0.00 | 1% | 0% | 1% |
| trie samples (pooled) | 244 | 0.599 | 3.03 | 0.00 | 0.20 | 2.82 | 0.00 | 0.00 | 0% | 8% | 0% |
| sentences shuffled across docs | 500 | 0.344 | 5.75 | 0.00 | 0.02 | 5.58 | 0.15 | 0.00 | 0% | 0% | 1% |
| words shuffled within sentences | 500 | 0.003 | 28.58 | 0.02 | 28.17 | 0.39 | 0.00 | 0.00 | 1% | 100% | 0% |
| random lexicon words | 500 | 0.001 | 35.52 | 0.00 | 28.45 | 7.08 | 0.00 | 0.00 | 0% | 100% | 0% |
| random letters | 500 | 0.000 | 53.39 | 27.64 | 23.43 | 2.32 | 0.00 | 0.00 | 100% | 100% | 0% |
| one phrase repeated | 500 | 0.000 | 306.60 | 1.03 | 3.66 | 2.10 | 299.81 | 0.00 | 22% | 27% | 100% |
| trie samples T=1.0 | 244 | 0.599 | 3.03 | 0.00 | 0.20 | 2.82 | 0.00 | 0.00 | 0% | 8% | 0% |
| held-out human, short (60-260 B) | 500 | 0.935 | 0.57 | 0.05 | 0.21 | 0.24 | 0.06 | 0.00 | 2% | 3% | 1% |

## V4: Pearson (Spearman) of SCRIBE with bytes on held-out human text

standard 0.050 (0.034); short 0.086 (0.110); pooled -0.022 (-0.092). Mean SCRIBE standard 0.919, short 0.931.

## SCRIBE v3: the three judges and their mean

v3 = mean(v2, SURPRISE, MATCH). Reference word thresholds for SURPRISE: 4.35 and 17.45 bits; MATCH's echo reference: 524 standard + 0 short prompted continuations.

v3.1 (docs/reports/SCRIBE_V31.md) = the power mean (exponent 0; 0 = geometric) of v2, SURPRISE 3.1 and MATCH 3.1, the new judges with the v3.1 statistics; fit threshold 1.18 bits.

| rung | n | v2 | SURPRISE | MATCH | v3 | min of 3 | SURPRISE 3.1 | MATCH 3.1 | v3.1 | disagree |
|---|---|---|---|---|---|---|---|---|---|---|
| held-out human | 500 | 0.923 [0.909, 0.935] | 0.921 [0.910, 0.931] | 0.922 [0.910, 0.933] | 0.922 [0.911, 0.931] | 0.859 [0.844, 0.873] | 0.930 [0.918, 0.941] | 0.924 [0.911, 0.934] | 0.919 [0.908, 0.930] | 2% |
| trie samples (pooled) | 244 | 0.599 [0.567, 0.630] | 0.650 [0.622, 0.678] | 0.643 [0.619, 0.666] | 0.631 [0.608, 0.655] | 0.509 [0.482, 0.534] | 0.557 [0.518, 0.589] | 0.544 [0.517, 0.570] | 0.545 [0.518, 0.572] | 4% |
| sentences shuffled across docs | 500 | 0.344 [0.332, 0.355] | 0.665 [0.661, 0.669] | 0.661 [0.656, 0.666] | 0.557 [0.552, 0.562] | 0.339 [0.329, 0.350] | 0.736 [0.730, 0.740] | 0.677 [0.671, 0.682] | 0.546 [0.540, 0.553] | 3% |
| words shuffled within sentences | 500 | 0.003 [0.003, 0.003] | 0.226 [0.224, 0.229] | 0.341 [0.337, 0.345] | 0.190 [0.188, 0.192] | 0.003 [0.003, 0.003] | 0.037 [0.036, 0.037] | 0.345 [0.339, 0.351] | 0.035 [0.034, 0.035] | 1% |
| random lexicon words | 500 | 0.001 [0.001, 0.001] | 0.157 [0.155, 0.159] | 0.249 [0.247, 0.251] | 0.136 [0.135, 0.137] | 0.001 [0.001, 0.001] | 0.029 [0.028, 0.029] | 0.262 [0.259, 0.265] | 0.018 [0.018, 0.019] | 0% |
| random letters | 500 | 0.000 [0.000, 0.000] | 0.065 [0.064, 0.066] | 0.059 [0.058, 0.059] | 0.041 [0.041, 0.042] | 0.000 [0.000, 0.000] | 0.013 [0.013, 0.014] | 0.011 [0.011, 0.011] | 0.001 [0.001, 0.002] | 0% |
| one phrase repeated | 500 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0% |
| trie samples T=1.0 | 244 | 0.599 [0.568, 0.631] | 0.650 [0.622, 0.678] | 0.643 [0.619, 0.665] | 0.631 [0.606, 0.655] | 0.509 [0.481, 0.534] | 0.557 [0.523, 0.589] | 0.544 [0.520, 0.570] | 0.545 [0.519, 0.574] | 4% |
| held-out human, short (60-260 B) | 500 | 0.935 [0.921, 0.948] | 0.943 [0.931, 0.953] | 0.947 [0.935, 0.958] | 0.942 [0.932, 0.950] | 0.879 [0.864, 0.896] | 0.940 [0.926, 0.953] | 0.946 [0.934, 0.958] | 0.931 [0.918, 0.942] | 4% |

Deviance terms (mean over samples; L lexical, S order, C cohesion, P too predictable):

| rung | SURPRISE D | D_L | D_S | D_C | D_P | MATCH D | D_L | D_S | D_C | D_P |
|---|---|---|---|---|---|---|---|---|---|---|
| held-out human | 2.04 | 0.01 | 0.04 | 1.31 | 0.68 | 4.94 | 0.01 | 0.10 | 2.09 | 2.73 |
| trie samples (pooled) | 10.03 | 0.00 | 0.00 | 5.45 | 4.58 | 14.73 | 0.00 | 0.00 | 12.67 | 2.06 |
| sentences shuffled across docs | 8.19 | 0.00 | 0.04 | 8.01 | 0.13 | 12.81 | 0.02 | 0.08 | 12.25 | 0.46 |
| words shuffled within sentences | 29.85 | 0.03 | 28.51 | 0.74 | 0.58 | 32.55 | 0.00 | 29.67 | 2.88 | 0.01 |
| random lexicon words | 37.18 | 0.00 | 29.08 | 6.94 | 1.16 | 41.86 | 0.00 | 29.97 | 11.90 | 0.00 |
| random letters | 54.80 | 30.00 | 17.01 | 7.75 | 0.04 | 85.18 | 30.00 | 30.00 | 25.18 | 0.00 |
| one phrase repeated | 198.75 | 0.76 | 1.18 | 0.64 | 196.17 | 285.95 | 0.82 | 13.18 | 0.96 | 270.98 |
| trie samples T=1.0 | 10.03 | 0.00 | 0.00 | 5.45 | 4.58 | 14.73 | 0.00 | 0.00 | 12.67 | 2.06 |
| held-out human, short (60-260 B) | 1.49 | 0.05 | 0.18 | 0.65 | 0.61 | 2.94 | 0.12 | 0.35 | 0.97 | 1.50 |

Deviance terms of the v3.1 judges (mean over samples; L lexical, S order, C cohesion, P too predictable, F foreign words, B bland):

| rung | SURPRISE D | D_L | D_S | D_C | D_P | D_F | D_B | MATCH D | D_L | D_S | D_C | D_P | D_F | D_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out human | 2.80 | 0.04 | 0.04 | 1.31 | 0.13 | 0.35 | 0.93 | 5.02 | 0.02 | 0.10 | 2.09 | 2.73 | 0.07 | 0.00 |
| trie samples (pooled) | 21.32 | 0.00 | 0.00 | 9.74 | 0.71 | 0.00 | 10.87 | 21.76 | 0.00 | 0.00 | 19.70 | 2.06 | 0.00 | 0.00 |
| sentences shuffled across docs | 8.40 | 0.00 | 0.04 | 8.01 | 0.11 | 0.18 | 0.06 | 12.89 | 0.06 | 0.08 | 12.25 | 0.46 | 0.04 | 0.00 |
| words shuffled within sentences | 89.39 | 0.09 | 28.51 | 0.74 | 0.10 | 59.95 | 0.00 | 34.92 | 0.00 | 29.67 | 2.88 | 0.01 | 2.37 | 0.00 |
| random lexicon words | 96.02 | 0.00 | 29.08 | 6.94 | 0.01 | 59.99 | 0.00 | 43.27 | 0.00 | 29.97 | 11.90 | 0.00 | 1.41 | 0.00 |
| random letters | 118.04 | 90.00 | 17.01 | 7.75 | 0.01 | 3.27 | 0.00 | 145.18 | 90.00 | 30.00 | 25.18 | 0.00 | 0.00 | 0.00 |
| one phrase repeated | 242.25 | 2.27 | 1.18 | 0.64 | 38.73 | 13.83 | 185.60 | 295.21 | 2.47 | 13.18 | 0.96 | 270.98 | 7.61 | 0.00 |
| trie samples T=1.0 | 21.32 | 0.00 | 0.00 | 9.74 | 0.71 | 0.00 | 10.87 | 21.76 | 0.00 | 0.00 | 19.70 | 2.06 | 0.00 | 0.00 |
| held-out human, short (60-260 B) | 2.84 | 0.15 | 0.18 | 0.65 | 0.18 | 0.60 | 1.07 | 3.33 | 0.36 | 0.35 | 0.97 | 1.50 | 0.15 | 0.00 |

### v3 parts: trie pooled

| upper > lower | v2 | | SURPRISE | | MATCH | | v3 | | min of 3 | | SURPRISE 3.1 | | MATCH 3.1 | | v3.1 | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out human > trie samples (pooled) | 0.861 | FAIL | 0.859 | FAIL | 0.888 | FAIL | 0.901 | FAIL | 0.893 | FAIL | 0.925 | FAIL | 0.927 | FAIL | 0.932 | FAIL |
| trie samples (pooled) > sentences shuffled across docs | 0.815 | FAIL | 0.452 | FAIL | 0.394 | FAIL | 0.575 | FAIL | 0.740 | FAIL | 0.280 | FAIL | 0.249 | FAIL | 0.444 | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 0.998 | PASS | 1.000 | PASS | 0.990 | PASS | 1.000 | PASS | 0.996 | PASS | 1.000 | PASS | 0.987 | PASS | 0.998 | PASS |
| words shuffled within sentences > random lexicon words | 0.998 | PASS | 0.957 | PASS | 0.968 | PASS | 0.984 | PASS | 0.998 | PASS | 0.968 | PASS | 0.877 | FAIL | 0.998 | PASS |
| random lexicon words > random letters | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS |
| random letters > one phrase repeated | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS |

Pairs passed (trie pooled): v2 4 of 6; SURPRISE 4 of 6; MATCH 4 of 6; v3 4 of 6; min of 3 4 of 6; SURPRISE 3.1 4 of 6; MATCH 3.1 3 of 6; v3.1 4 of 6

### v3 parts: trie samples T=1.0

| upper > lower | v2 | | SURPRISE | | MATCH | | v3 | | min of 3 | | SURPRISE 3.1 | | MATCH 3.1 | | v3.1 | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out human > trie samples T=1.0 | 0.861 | FAIL | 0.859 | FAIL | 0.888 | FAIL | 0.901 | FAIL | 0.893 | FAIL | 0.925 | FAIL | 0.927 | FAIL | 0.932 | FAIL |
| trie samples T=1.0 > sentences shuffled across docs | 0.815 | FAIL | 0.452 | FAIL | 0.394 | FAIL | 0.575 | FAIL | 0.740 | FAIL | 0.280 | FAIL | 0.249 | FAIL | 0.444 | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 0.998 | PASS | 1.000 | PASS | 0.990 | PASS | 1.000 | PASS | 0.996 | PASS | 1.000 | PASS | 0.987 | PASS | 0.998 | PASS |
| words shuffled within sentences > random lexicon words | 0.998 | PASS | 0.957 | PASS | 0.968 | PASS | 0.984 | PASS | 0.998 | PASS | 0.968 | PASS | 0.877 | FAIL | 0.998 | PASS |
| random lexicon words > random letters | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS |
| random letters > one phrase repeated | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS |

Pairs passed (trie samples T=1.0): v2 4 of 6; SURPRISE 4 of 6; MATCH 4 of 6; v3 4 of 6; min of 3 4 of 6; SURPRISE 3.1 4 of 6; MATCH 3.1 3 of 6; v3.1 4 of 6

### v3 parts: control ladder (no model rung)

| upper > lower | v2 | | SURPRISE | | MATCH | | v3 | | min of 3 | | SURPRISE 3.1 | | MATCH 3.1 | | v3.1 | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out human > sentences shuffled across docs | 0.983 | PASS | 0.961 | PASS | 0.963 | PASS | 0.989 | PASS | 0.978 | PASS | 0.939 | FAIL | 0.961 | PASS | 0.979 | PASS |
| sentences shuffled across docs > words shuffled within sentences | 0.998 | PASS | 1.000 | PASS | 0.990 | PASS | 1.000 | PASS | 0.996 | PASS | 1.000 | PASS | 0.987 | PASS | 0.998 | PASS |
| words shuffled within sentences > random lexicon words | 0.998 | PASS | 0.957 | PASS | 0.968 | PASS | 0.984 | PASS | 0.998 | PASS | 0.968 | PASS | 0.877 | FAIL | 0.998 | PASS |
| random lexicon words > random letters | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS |
| random letters > one phrase repeated | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS | 1.000 | PASS |

Pairs passed (control ladder (no model rung)): v2 5 of 5; SURPRISE 5 of 5; MATCH 5 of 5; v3 5 of 5; min of 3 5 of 5; SURPRISE 3.1 4 of 5; MATCH 3.1 4 of 5; v3.1 5 of 5
