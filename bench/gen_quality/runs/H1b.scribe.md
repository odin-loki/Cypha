# SCRIBE H1b ladder: enwik:/mnt/d/cypha_llm/enwik8:56:64 (profile v2)

Pack enwik:/mnt/d/cypha_llm/enwik8:16:48: 2600 docs, 17521535 bytes, lexicon 47691 words, 2051016 trigrams; vectors: /mnt/d/cypha_llm/scribe/v32. Reference enwik:/mnt/d/cypha_llm/enwik8:48:56: 1909 standard + 1729 short samples (seed 20261027). Test: 500 standard samples (of 2135 candidates), 500 short (seed 20261027).

Trie rung: 250 GEN lines, 246 kept after cleaning (4 under 100 B dropped).

## Rung means (95% bootstrap interval of the mean, 1000 resamples)

| rung | n | SCRIBE | parity | surface S | mean bytes |
|---|---|---|---|---|---|
| held-out human | 500 | 0.923 [0.910, 0.935] | 1.00 | 0.912 [0.906, 0.918] | 531 |
| trie samples (pooled) | 246 | 0.551 [0.521, 0.585] | 0.60 | 0.941 [0.936, 0.944] | 436 |
| sentences shuffled across docs | 500 | 0.344 [0.333, 0.356] | 0.37 | 0.905 [0.901, 0.908] | 532 |
| words shuffled within sentences | 500 | 0.003 [0.003, 0.003] | 0.00 | 0.923 [0.919, 0.927] | 531 |
| random lexicon words | 500 | 0.001 [0.001, 0.001] | 0.00 | 0.896 [0.894, 0.898] | 523 |
| random letters | 500 | 0.000 [0.000, 0.000] | 0.00 | 0.035 [0.034, 0.037] | 531 |
| one phrase repeated | 500 | 0.000 [0.000, 0.000] | 0.00 | 0.029 [0.028, 0.029] | 550 |
| trie samples T=1.0 | 246 | 0.551 [0.520, 0.585] | 0.60 | 0.941 [0.937, 0.945] | 436 |
| held-out human, short (60-260 B) | 500 | 0.935 [0.920, 0.948] | 1.01 | 0.909 [0.901, 0.915] | 188 |

## Ladders: pairwise ordering accuracy over all (upper, lower) pairs (ties = 1/2) and disjoint 95% intervals; PASS needs both (accuracy >= 0.95)

### trie pooled

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > trie samples (pooled) | 0.892 | yes | FAIL | 0.382 | no | FAIL |
| trie samples (pooled) > sentences shuffled across docs | 0.760 | yes | FAIL | 0.751 | yes | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 0.998 | yes | PASS | 0.372 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.998 | yes | PASS | 0.724 | yes | FAIL |
| random lexicon words > random letters | 1.000 | yes | PASS | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.590 | yes | FAIL |

Verdict (trie pooled): SCRIBE FAIL (4 of 6 pairs), surface S FAIL (1 of 6 pairs)

### trie samples T=1.0

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > trie samples T=1.0 | 0.892 | yes | FAIL | 0.382 | no | FAIL |
| trie samples T=1.0 > sentences shuffled across docs | 0.760 | yes | FAIL | 0.751 | yes | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 0.998 | yes | PASS | 0.372 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.998 | yes | PASS | 0.724 | yes | FAIL |
| random lexicon words > random letters | 1.000 | yes | PASS | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.590 | yes | FAIL |

Verdict (trie samples T=1.0): SCRIBE FAIL (4 of 6 pairs), surface S FAIL (1 of 6 pairs)

### control ladder (no model rung)

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > sentences shuffled across docs | 0.983 | yes | PASS | 0.592 | no | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 0.998 | yes | PASS | 0.372 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.998 | yes | PASS | 0.724 | yes | FAIL |
| random lexicon words > random letters | 1.000 | yes | PASS | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.590 | yes | FAIL |

Verdict (control ladder (no model rung)): SCRIBE PASS (5 of 5 pairs), surface S FAIL (1 of 5 pairs)

## Axis profile (mean over samples)

| rung | n | SCRIBE | mean D | D_L | D_S | D_C | D_I | D_R | share D_L > 0 | share D_S > 0 | share D_I > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out human | 500 | 0.923 | 0.54 | 0.02 | 0.00 | 0.38 | 0.13 | 0.00 | 1% | 0% | 1% |
| trie samples (pooled) | 246 | 0.551 | 3.56 | 0.00 | 0.65 | 2.91 | 0.00 | 0.00 | 0% | 20% | 0% |
| sentences shuffled across docs | 500 | 0.344 | 5.75 | 0.00 | 0.02 | 5.58 | 0.15 | 0.00 | 0% | 0% | 1% |
| words shuffled within sentences | 500 | 0.003 | 28.58 | 0.02 | 28.17 | 0.39 | 0.00 | 0.00 | 1% | 100% | 0% |
| random lexicon words | 500 | 0.001 | 35.52 | 0.00 | 28.45 | 7.08 | 0.00 | 0.00 | 0% | 100% | 0% |
| random letters | 500 | 0.000 | 53.39 | 27.64 | 23.43 | 2.32 | 0.00 | 0.00 | 100% | 100% | 0% |
| one phrase repeated | 500 | 0.000 | 306.60 | 1.03 | 3.66 | 2.10 | 299.81 | 0.00 | 22% | 27% | 100% |
| trie samples T=1.0 | 246 | 0.551 | 3.56 | 0.00 | 0.65 | 2.91 | 0.00 | 0.00 | 0% | 20% | 0% |
| held-out human, short (60-260 B) | 500 | 0.935 | 0.57 | 0.05 | 0.21 | 0.24 | 0.06 | 0.00 | 2% | 3% | 1% |

## V4: Pearson (Spearman) of SCRIBE with bytes on held-out human text

standard -0.012 (-0.051); short 0.051 (0.183); pooled -0.032 (0.039). Mean SCRIBE standard 0.923, short 0.935.
