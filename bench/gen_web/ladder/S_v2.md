# SCRIBE S_v2 ladder: raw:/mnt/d/cypha_llm/genw/ladder_S.jsonl (profile v2)

Near-duplicate filter (--dedup): dropped 192 of 19302 test documents (169068 of 38885676 bytes) with more than half their bytes inside 64-byte matches with the pack or the reference.

Pack raw:/mnt/d/cypha_llm/genw/pack.jsonl: 18830 docs, 38951819 bytes, lexicon 71797 words, 4348179 trigrams; vectors: /mnt/d/cypha_llm/genw/vecW. Reference raw:/mnt/d/cypha_llm/genw/ref.jsonl: 9394 standard + 12715 short samples (seed 20261027). Test: 500 standard samples (of 36246 candidates), 500 short (seed 20261112).

## Rung means (95% bootstrap interval of the mean, 1000 resamples)

| rung | n | SCRIBE | parity | surface S | mean bytes |
|---|---|---|---|---|---|
| held-out human | 500 | 0.908 [0.891, 0.924] | 1.00 | 0.916 [0.907, 0.924] | 518 |
| sentences shuffled across docs | 500 | 0.398 [0.385, 0.412] | 0.44 | 0.903 [0.900, 0.906] | 512 |
| words shuffled within sentences | 500 | 0.008 [0.007, 0.008] | 0.01 | 0.929 [0.924, 0.935] | 518 |
| random lexicon words | 500 | 0.002 [0.002, 0.002] | 0.00 | 0.897 [0.895, 0.898] | 514 |
| random letters | 500 | 0.001 [0.001, 0.001] | 0.00 | 0.069 [0.066, 0.071] | 518 |
| one phrase repeated | 500 | 0.000 [0.000, 0.000] | 0.00 | 0.028 [0.028, 0.029] | 534 |
| held-out human, short (60-260 B) | 500 | 0.950 [0.939, 0.961] | 1.05 | 0.923 [0.916, 0.930] | 181 |

## Ladders: pairwise ordering accuracy over all (upper, lower) pairs (ties = 1/2) and disjoint 95% intervals; PASS needs both (accuracy >= 0.95)

### control ladder (no model rung)

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > sentences shuffled across docs | 0.958 | yes | PASS | 0.685 | yes | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 1.000 | yes | PASS | 0.284 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.975 | yes | PASS | 0.784 | yes | FAIL |
| random lexicon words > random letters | 0.886 | yes | FAIL | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.923 | yes | FAIL |

Verdict (control ladder (no model rung)): SCRIBE FAIL (4 of 5 pairs), surface S FAIL (1 of 5 pairs)

## Axis profile (mean over samples)

| rung | n | SCRIBE | mean D | D_L | D_S | D_C | D_I | D_R | share D_L > 0 | share D_S > 0 | share D_I > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out human | 500 | 0.908 | 0.83 | 0.15 | 0.06 | 0.44 | 0.17 | 0.00 | 1% | 1% | 1% |
| sentences shuffled across docs | 500 | 0.398 | 4.93 | 0.04 | 0.00 | 4.89 | 0.00 | 0.00 | 1% | 0% | 0% |
| words shuffled within sentences | 500 | 0.008 | 25.11 | 0.15 | 24.52 | 0.44 | 0.00 | 0.00 | 1% | 100% | 0% |
| random lexicon words | 500 | 0.002 | 31.53 | 0.00 | 25.85 | 5.68 | 0.00 | 0.00 | 0% | 100% | 0% |
| random letters | 500 | 0.001 | 39.40 | 26.92 | 9.82 | 2.66 | 0.00 | 0.00 | 100% | 94% | 0% |
| one phrase repeated | 500 | 0.000 | 154.49 | 0.58 | 2.65 | 1.71 | 149.56 | 0.00 | 11% | 23% | 100% |
| held-out human, short (60-260 B) | 500 | 0.950 | 0.40 | 0.11 | 0.05 | 0.23 | 0.02 | 0.00 | 1% | 1% | 0% |

## V4: Pearson (Spearman) of SCRIBE with bytes on held-out human text

standard 0.094 (0.094); short 0.071 (0.229); pooled -0.106 (0.026). Mean SCRIBE standard 0.908, short 0.950.
