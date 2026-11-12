# SCRIBE DEF ladder: raw:/mnt/d/cypha_llm/genw/rungs.jsonl (profile v2)

Near-duplicate filter (--dedup): dropped 55 of 4910 test documents (67280 of 10185347 bytes) with more than half their bytes inside 64-byte matches with the pack or the reference.

Pack raw:/mnt/d/cypha_llm/genw/pack.jsonl: 18830 docs, 38951819 bytes, lexicon 71797 words, 4348179 trigrams; vectors: /mnt/d/cypha_llm/genw/vecW. Reference raw:/mnt/d/cypha_llm/genw/ref.jsonl: 9394 standard + 12715 short samples (seed 20261027). Test: 500 standard samples (of 9408 candidates), 500 short (seed 20261112).

Trie rung: 250 GEN lines, 250 kept after cleaning (0 under 100 B dropped).

## Rung means (95% bootstrap interval of the mean, 1000 resamples)

| rung | n | SCRIBE | parity | surface S | mean bytes |
|---|---|---|---|---|---|
| held-out human | 500 | 0.914 [0.899, 0.927] | 1.00 | 0.921 [0.913, 0.927] | 525 |
| trie samples (pooled) | 250 | 0.691 [0.665, 0.718] | 0.76 | 0.933 [0.929, 0.938] | 439 |
| sentences shuffled across docs | 500 | 0.404 [0.390, 0.418] | 0.44 | 0.904 [0.901, 0.907] | 523 |
| words shuffled within sentences | 500 | 0.007 [0.007, 0.008] | 0.01 | 0.930 [0.924, 0.935] | 525 |
| random lexicon words | 500 | 0.003 [0.002, 0.006] | 0.00 | 0.897 [0.895, 0.899] | 514 |
| random letters | 500 | 0.001 [0.001, 0.001] | 0.00 | 0.067 [0.064, 0.069] | 525 |
| one phrase repeated | 500 | 0.000 [0.000, 0.000] | 0.00 | 0.028 [0.028, 0.029] | 542 |
| trie samples T=1.0 | 250 | 0.691 [0.666, 0.717] | 0.76 | 0.933 [0.930, 0.937] | 439 |
| held-out human, short (60-260 B) | 500 | 0.944 [0.932, 0.955] | 1.03 | 0.922 [0.914, 0.929] | 179 |

## Ladders: pairwise ordering accuracy over all (upper, lower) pairs (ties = 1/2) and disjoint 95% intervals; PASS needs both (accuracy >= 0.95)

### trie pooled

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > trie samples (pooled) | 0.817 | yes | FAIL | 0.507 | no | FAIL |
| trie samples (pooled) > sentences shuffled across docs | 0.859 | yes | FAIL | 0.738 | yes | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 1.000 | yes | PASS | 0.280 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.983 | yes | PASS | 0.793 | yes | FAIL |
| random lexicon words > random letters | 0.901 | yes | FAIL | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.917 | yes | FAIL |

Verdict (trie pooled): SCRIBE FAIL (3 of 6 pairs), surface S FAIL (1 of 6 pairs)

### trie samples T=1.0

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > trie samples T=1.0 | 0.817 | yes | FAIL | 0.507 | no | FAIL |
| trie samples T=1.0 > sentences shuffled across docs | 0.859 | yes | FAIL | 0.738 | yes | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 1.000 | yes | PASS | 0.280 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.983 | yes | PASS | 0.793 | yes | FAIL |
| random lexicon words > random letters | 0.901 | yes | FAIL | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.917 | yes | FAIL |

Verdict (trie samples T=1.0): SCRIBE FAIL (3 of 6 pairs), surface S FAIL (1 of 6 pairs)

### control ladder (no model rung)

| upper > lower | SCRIBE acc | disjoint | | surface S acc | disjoint | |
|---|---|---|---|---|---|---|
| held-out human > sentences shuffled across docs | 0.967 | yes | PASS | 0.686 | yes | FAIL |
| sentences shuffled across docs > words shuffled within sentences | 1.000 | yes | PASS | 0.280 | no | FAIL |
| words shuffled within sentences > random lexicon words | 0.983 | yes | PASS | 0.793 | yes | FAIL |
| random lexicon words > random letters | 0.901 | yes | FAIL | 1.000 | yes | PASS |
| random letters > one phrase repeated | 1.000 | yes | PASS | 0.917 | yes | FAIL |

Verdict (control ladder (no model rung)): SCRIBE FAIL (4 of 5 pairs), surface S FAIL (1 of 5 pairs)

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

standard 0.021 (0.076); short 0.168 (0.272); pooled -0.075 (0.042). Mean SCRIBE standard 0.914, short 0.944.
