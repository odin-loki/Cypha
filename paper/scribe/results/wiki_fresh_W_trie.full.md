# Where the trie falls, all per-text metrics on W

Analysis `wiki_fresh_W_trie` (1. Fresh Wikipedia control ladder). meta.py `ladder`.
Inputs: SP/run/baselines/subsets/wiki_fresh_W.jsonl, SP/run/fresh/wiki_fresh/scores.jsonl, SP/run/baselines/wiki_fresh_W/stats.jsonl, SP/run/baselines/wiki_fresh_W/ppl_gpt2.jsonl, SP/run/baselines/wiki_fresh_W/cola.jsonl, SP/run/baselines/wiki_fresh_W/coherence.jsonl, SP/run/baselines/wiki_fresh_W/detector.jsonl.

1050 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.021 [0.009, 0.035] |
| held-out human | 150 | 0 | 0.907 [0.883, 0.931] |
| sentences shuffled | 150 | 0 | 0.362 [0.342, 0.383] |
| words shuffled | 150 | 0 | 0.003 [0.003, 0.003] |
| random lexicon words | 150 | 0 | 0.001 [0.001, 0.001] |
| random letters | 150 | 0 | 0.000 [0.000, 0.000] |
| one phrase repeated | 150 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.002 | [0.000, 0.004] | no | FAIL |
| held-out human > sentences shuffled | 0.978 | [0.956, 0.995] | yes | **PASS** |
| sentences shuffled > words shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| words shuffled > random lexicon words | 1.000 | [0.999, 1.000] | yes | **PASS** |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.002 | 0.019 | 0.413 | 0.703 | 0.995 | 1.000 |
| held-out human |  | 0.978 | 1.000 | 1.000 | 1.000 | 1.000 |
| sentences shuffled |  |  | 1.000 | 1.000 | 1.000 | 1.000 |
| words shuffled |  |  |  | 1.000 | 1.000 | 1.000 |
| random lexicon words |  |  |  |  | 1.000 | 1.000 |
| random letters |  |  |  |  |  | 1.000 |

### SCRIBE v1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.034 [0.027, 0.042] |
| held-out human | 150 | 0 | 0.551 [0.515, 0.583] |
| sentences shuffled | 150 | 0 | 0.239 [0.221, 0.257] |
| words shuffled | 150 | 0 | 0.124 [0.109, 0.140] |
| random lexicon words | 150 | 0 | 0.067 [0.064, 0.069] |
| random letters | 150 | 0 | 0.006 [0.006, 0.006] |
| one phrase repeated | 150 | 0 | 0.001 [0.001, 0.001] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.010 | [0.003, 0.019] | no | FAIL |
| held-out human > sentences shuffled | 0.875 | [0.827, 0.916] | yes | FAIL |
| sentences shuffled > words shuffled | 0.835 | [0.789, 0.878] | yes | FAIL |
| words shuffled > random lexicon words | 0.707 | [0.637, 0.768] | yes | FAIL |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 0.999 | [0.998, 1.000] | yes | **PASS** |

Passed 2 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.010 | 0.021 | 0.124 | 0.124 | 0.776 | 0.997 |
| held-out human |  | 0.875 | 0.942 | 0.979 | 1.000 | 1.000 |
| sentences shuffled |  |  | 0.835 | 0.996 | 1.000 | 1.000 |
| words shuffled |  |  |  | 0.707 | 1.000 | 1.000 |
| random lexicon words |  |  |  |  | 1.000 | 1.000 |
| random letters |  |  |  |  |  | 0.999 |

### surface S (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.740 [0.724, 0.756] |
| held-out human | 150 | 0 | 0.909 [0.897, 0.919] |
| sentences shuffled | 150 | 0 | 0.900 [0.894, 0.905] |
| words shuffled | 150 | 0 | 0.920 [0.911, 0.927] |
| random lexicon words | 150 | 0 | 0.893 [0.890, 0.897] |
| random letters | 150 | 0 | 0.035 [0.032, 0.039] |
| one phrase repeated | 150 | 0 | 0.029 [0.028, 0.030] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.059 | [0.033, 0.091] | no | FAIL |
| held-out human > sentences shuffled | 0.630 | [0.570, 0.689] | no | FAIL |
| sentences shuffled > words shuffled | 0.339 | [0.282, 0.403] | no | FAIL |
| words shuffled > random lexicon words | 0.726 | [0.663, 0.783] | yes | FAIL |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 0.539 | [0.468, 0.616] | yes | FAIL |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.059 | 0.031 | 0.037 | 0.024 | 1.000 | 1.000 |
| held-out human |  | 0.630 | 0.471 | 0.688 | 1.000 | 1.000 |
| sentences shuffled |  |  | 0.339 | 0.581 | 1.000 | 1.000 |
| words shuffled |  |  |  | 0.726 | 1.000 | 1.000 |
| random lexicon words |  |  |  |  | 1.000 | 1.000 |
| random letters |  |  |  |  |  | 0.539 |

### distinct-1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.749 [0.733, 0.768] |
| held-out human | 150 | 0 | 0.724 [0.713, 0.735] |
| sentences shuffled | 150 | 0 | 0.782 [0.774, 0.790] |
| words shuffled | 150 | 0 | 0.724 [0.712, 0.735] |
| random lexicon words | 150 | 0 | 0.815 [0.807, 0.822] |
| random letters | 150 | 0 | 0.995 [0.993, 0.996] |
| one phrase repeated | 150 | 0 | 0.068 [0.066, 0.071] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.575 | [0.509, 0.639] | no | FAIL |
| held-out human > sentences shuffled | 0.258 | [0.206, 0.311] | no | FAIL |
| sentences shuffled > words shuffled | 0.742 | [0.684, 0.795] | yes | FAIL |
| words shuffled > random lexicon words | 0.123 | [0.088, 0.163] | no | FAIL |
| random lexicon words > random letters | 0.000 | [0.000, 0.000] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.575 | 0.410 | 0.575 | 0.310 | 0.009 | 1.000 |
| held-out human |  | 0.258 | 0.500 | 0.123 | 0.000 | 1.000 |
| sentences shuffled |  |  | 0.742 | 0.315 | 0.000 | 1.000 |
| words shuffled |  |  |  | 0.123 | 0.000 | 1.000 |
| random lexicon words |  |  |  |  | 0.000 | 1.000 |
| random letters |  |  |  |  |  | 1.000 |

### distinct-2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.961 [0.955, 0.967] |
| held-out human | 150 | 0 | 0.960 [0.954, 0.967] |
| sentences shuffled | 150 | 0 | 0.983 [0.980, 0.986] |
| words shuffled | 150 | 0 | 0.991 [0.989, 0.993] |
| random lexicon words | 150 | 0 | 0.996 [0.994, 0.997] |
| random letters | 150 | 0 | 1.000 [0.999, 1.000] |
| one phrase repeated | 150 | 0 | 0.070 [0.068, 0.072] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.537 | [0.469, 0.604] | no | FAIL |
| held-out human > sentences shuffled | 0.273 | [0.216, 0.329] | no | FAIL |
| sentences shuffled > words shuffled | 0.342 | [0.284, 0.402] | no | FAIL |
| words shuffled > random lexicon words | 0.410 | [0.356, 0.469] | no | FAIL |
| random lexicon words > random letters | 0.360 | [0.321, 0.400] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.537 | 0.380 | 0.290 | 0.243 | 0.183 | 1.000 |
| held-out human |  | 0.273 | 0.156 | 0.089 | 0.042 | 1.000 |
| sentences shuffled |  |  | 0.342 | 0.253 | 0.158 | 1.000 |
| words shuffled |  |  |  | 0.410 | 0.290 | 1.000 |
| random lexicon words |  |  |  |  | 0.360 | 1.000 |
| random letters |  |  |  |  |  | 1.000 |

### distinct-3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 1.000 [1.000, 1.000] |
| held-out human | 150 | 0 | 0.991 [0.987, 0.994] |
| sentences shuffled | 150 | 0 | 0.998 [0.997, 0.999] |
| words shuffled | 150 | 0 | 1.000 [1.000, 1.000] |
| random lexicon words | 150 | 0 | 1.000 [1.000, 1.000] |
| random letters | 150 | 0 | 1.000 [1.000, 1.000] |
| one phrase repeated | 150 | 0 | 0.071 [0.069, 0.073] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.673 | [0.637, 0.710] | yes | FAIL |
| held-out human > sentences shuffled | 0.382 | [0.337, 0.427] | no | FAIL |
| sentences shuffled > words shuffled | 0.449 | [0.419, 0.479] | no | FAIL |
| words shuffled > random lexicon words | 0.493 | [0.480, 0.507] | no | FAIL |
| random lexicon words > random letters | 0.497 | [0.490, 0.500] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.673 | 0.560 | 0.510 | 0.503 | 0.500 | 1.000 |
| held-out human |  | 0.382 | 0.334 | 0.330 | 0.327 | 1.000 |
| sentences shuffled |  |  | 0.449 | 0.443 | 0.440 | 1.000 |
| words shuffled |  |  |  | 0.493 | 0.490 | 1.000 |
| random lexicon words |  |  |  |  | 0.497 | 1.000 |
| random letters |  |  |  |  |  | 1.000 |

### distinct-4 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 1.000 [1.000, 1.000] |
| held-out human | 150 | 0 | 0.997 [0.995, 0.999] |
| sentences shuffled | 150 | 0 | 0.999 [0.999, 1.000] |
| words shuffled | 150 | 0 | 1.000 [1.000, 1.000] |
| random lexicon words | 150 | 0 | 1.000 [1.000, 1.000] |
| random letters | 150 | 0 | 1.000 [1.000, 1.000] |
| one phrase repeated | 150 | 0 | 0.072 [0.070, 0.074] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.557 | [0.533, 0.583] | yes | FAIL |
| held-out human > sentences shuffled | 0.463 | [0.432, 0.492] | no | FAIL |
| sentences shuffled > words shuffled | 0.480 | [0.463, 0.493] | no | FAIL |
| words shuffled > random lexicon words | 0.500 | [0.500, 0.500] | no | FAIL |
| random lexicon words > random letters | 0.500 | [0.500, 0.500] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.557 | 0.520 | 0.500 | 0.500 | 0.500 | 1.000 |
| held-out human |  | 0.463 | 0.443 | 0.443 | 0.443 | 1.000 |
| sentences shuffled |  |  | 0.480 | 0.480 | 0.480 | 1.000 |
| words shuffled |  |  |  | 0.500 | 0.500 | 1.000 |
| random lexicon words |  |  |  |  | 0.500 | 1.000 |
| random letters |  |  |  |  |  | 1.000 |

### rep-2 (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.039 [0.033, 0.045] |
| held-out human | 150 | 0 | 0.040 [0.033, 0.046] |
| sentences shuffled | 150 | 0 | 0.017 [0.014, 0.020] |
| words shuffled | 150 | 0 | 0.009 [0.007, 0.011] |
| random lexicon words | 150 | 0 | 0.004 [0.003, 0.006] |
| random letters | 150 | 0 | 0.000 [0.000, 0.001] |
| one phrase repeated | 150 | 0 | 0.930 [0.928, 0.932] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.537 | [0.469, 0.604] | no | FAIL |
| held-out human > sentences shuffled | 0.273 | [0.216, 0.329] | no | FAIL |
| sentences shuffled > words shuffled | 0.342 | [0.284, 0.402] | no | FAIL |
| words shuffled > random lexicon words | 0.410 | [0.356, 0.469] | no | FAIL |
| random lexicon words > random letters | 0.360 | [0.321, 0.400] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.537 | 0.380 | 0.290 | 0.243 | 0.183 | 1.000 |
| held-out human |  | 0.273 | 0.156 | 0.089 | 0.042 | 1.000 |
| sentences shuffled |  |  | 0.342 | 0.253 | 0.158 | 1.000 |
| words shuffled |  |  |  | 0.410 | 0.290 | 1.000 |
| random lexicon words |  |  |  |  | 0.360 | 1.000 |
| random letters |  |  |  |  |  | 1.000 |

### rep-3 (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.000 [0.000, 0.000] |
| held-out human | 150 | 0 | 0.009 [0.006, 0.013] |
| sentences shuffled | 150 | 0 | 0.002 [0.001, 0.003] |
| words shuffled | 150 | 0 | 0.000 [0.000, 0.000] |
| random lexicon words | 150 | 0 | 0.000 [0.000, 0.000] |
| random letters | 150 | 0 | 0.000 [0.000, 0.000] |
| one phrase repeated | 150 | 0 | 0.929 [0.927, 0.931] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.673 | [0.637, 0.710] | yes | FAIL |
| held-out human > sentences shuffled | 0.382 | [0.337, 0.427] | no | FAIL |
| sentences shuffled > words shuffled | 0.449 | [0.419, 0.479] | no | FAIL |
| words shuffled > random lexicon words | 0.493 | [0.480, 0.507] | no | FAIL |
| random lexicon words > random letters | 0.497 | [0.490, 0.500] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.673 | 0.560 | 0.510 | 0.503 | 0.500 | 1.000 |
| held-out human |  | 0.382 | 0.334 | 0.330 | 0.327 | 1.000 |
| sentences shuffled |  |  | 0.449 | 0.443 | 0.440 | 1.000 |
| words shuffled |  |  |  | 0.493 | 0.490 | 1.000 |
| random lexicon words |  |  |  |  | 0.497 | 1.000 |
| random letters |  |  |  |  |  | 1.000 |

### rep-4 (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.000 [0.000, 0.000] |
| held-out human | 150 | 0 | 0.003 [0.001, 0.005] |
| sentences shuffled | 150 | 0 | 0.001 [0.000, 0.001] |
| words shuffled | 150 | 0 | 0.000 [0.000, 0.000] |
| random lexicon words | 150 | 0 | 0.000 [0.000, 0.000] |
| random letters | 150 | 0 | 0.000 [0.000, 0.000] |
| one phrase repeated | 150 | 0 | 0.928 [0.926, 0.930] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.557 | [0.533, 0.583] | yes | FAIL |
| held-out human > sentences shuffled | 0.463 | [0.432, 0.492] | no | FAIL |
| sentences shuffled > words shuffled | 0.480 | [0.463, 0.493] | no | FAIL |
| words shuffled > random lexicon words | 0.500 | [0.500, 0.500] | no | FAIL |
| random lexicon words > random letters | 0.500 | [0.500, 0.500] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.557 | 0.520 | 0.500 | 0.500 | 0.500 | 1.000 |
| held-out human |  | 0.463 | 0.443 | 0.443 | 0.443 | 1.000 |
| sentences shuffled |  |  | 0.480 | 0.480 | 0.480 | 1.000 |
| words shuffled |  |  |  | 0.500 | 0.500 | 1.000 |
| random lexicon words |  |  |  |  | 0.500 | 1.000 |
| random letters |  |  |  |  |  | 1.000 |

### seq-rep-4 (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.000 [0.000, 0.000] |
| held-out human | 150 | 0 | 0.003 [0.001, 0.005] |
| sentences shuffled | 150 | 0 | 0.001 [0.000, 0.001] |
| words shuffled | 150 | 0 | 0.000 [0.000, 0.000] |
| random lexicon words | 150 | 0 | 0.000 [0.000, 0.000] |
| random letters | 150 | 0 | 0.000 [0.000, 0.000] |
| one phrase repeated | 150 | 0 | 0.928 [0.926, 0.930] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.557 | [0.533, 0.583] | yes | FAIL |
| held-out human > sentences shuffled | 0.463 | [0.432, 0.492] | no | FAIL |
| sentences shuffled > words shuffled | 0.480 | [0.463, 0.493] | no | FAIL |
| words shuffled > random lexicon words | 0.500 | [0.500, 0.500] | no | FAIL |
| random lexicon words > random letters | 0.500 | [0.500, 0.500] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.557 | 0.520 | 0.500 | 0.500 | 0.500 | 1.000 |
| held-out human |  | 0.463 | 0.443 | 0.443 | 0.443 | 1.000 |
| sentences shuffled |  |  | 0.480 | 0.480 | 0.480 | 1.000 |
| words shuffled |  |  |  | 0.500 | 0.500 | 1.000 |
| random lexicon words |  |  |  |  | 0.500 | 1.000 |
| random letters |  |  |  |  |  | 1.000 |

### diversity (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.961 [0.955, 0.967] |
| held-out human | 150 | 0 | 0.950 [0.940, 0.959] |
| sentences shuffled | 150 | 0 | 0.981 [0.976, 0.984] |
| words shuffled | 150 | 0 | 0.991 [0.988, 0.993] |
| random lexicon words | 150 | 0 | 0.996 [0.994, 0.997] |
| random letters | 150 | 0 | 1.000 [0.999, 1.000] |
| one phrase repeated | 150 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.566 | [0.499, 0.632] | no | FAIL |
| held-out human > sentences shuffled | 0.273 | [0.214, 0.331] | no | FAIL |
| sentences shuffled > words shuffled | 0.338 | [0.280, 0.398] | no | FAIL |
| words shuffled > random lexicon words | 0.410 | [0.356, 0.470] | no | FAIL |
| random lexicon words > random letters | 0.360 | [0.321, 0.400] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.566 | 0.392 | 0.292 | 0.244 | 0.183 | 1.000 |
| held-out human |  | 0.273 | 0.151 | 0.089 | 0.042 | 1.000 |
| sentences shuffled |  |  | 0.338 | 0.253 | 0.158 | 1.000 |
| words shuffled |  |  |  | 0.410 | 0.290 | 1.000 |
| random lexicon words |  |  |  |  | 0.360 | 1.000 |
| random letters |  |  |  |  |  | 1.000 |

### type-token ratio (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.749 [0.733, 0.768] |
| held-out human | 150 | 0 | 0.724 [0.713, 0.735] |
| sentences shuffled | 150 | 0 | 0.782 [0.774, 0.790] |
| words shuffled | 150 | 0 | 0.724 [0.712, 0.735] |
| random lexicon words | 150 | 0 | 0.815 [0.807, 0.822] |
| random letters | 150 | 0 | 0.995 [0.993, 0.996] |
| one phrase repeated | 150 | 0 | 0.068 [0.066, 0.071] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.575 | [0.509, 0.639] | no | FAIL |
| held-out human > sentences shuffled | 0.258 | [0.206, 0.311] | no | FAIL |
| sentences shuffled > words shuffled | 0.742 | [0.684, 0.795] | yes | FAIL |
| words shuffled > random lexicon words | 0.123 | [0.088, 0.163] | no | FAIL |
| random lexicon words > random letters | 0.000 | [0.000, 0.000] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.575 | 0.410 | 0.575 | 0.310 | 0.009 | 1.000 |
| held-out human |  | 0.258 | 0.500 | 0.123 | 0.000 | 1.000 |
| sentences shuffled |  |  | 0.742 | 0.315 | 0.000 | 1.000 |
| words shuffled |  |  |  | 0.123 | 0.000 | 1.000 |
| random lexicon words |  |  |  |  | 0.000 | 1.000 |
| random letters |  |  |  |  |  | 1.000 |

### gzip ratio (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.731 [0.713, 0.751] |
| held-out human | 150 | 0 | 0.621 [0.615, 0.627] |
| sentences shuffled | 150 | 0 | 0.659 [0.655, 0.664] |
| words shuffled | 150 | 0 | 0.636 [0.630, 0.642] |
| random lexicon words | 150 | 0 | 0.667 [0.663, 0.671] |
| random letters | 150 | 0 | 0.718 [0.715, 0.722] |
| one phrase repeated | 150 | 0 | 0.117 [0.114, 0.119] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.820 | [0.771, 0.869] | yes | FAIL |
| held-out human > sentences shuffled | 0.214 | [0.164, 0.267] | no | FAIL |
| sentences shuffled > words shuffled | 0.693 | [0.631, 0.753] | yes | FAIL |
| words shuffled > random lexicon words | 0.226 | [0.171, 0.281] | no | FAIL |
| random lexicon words > random letters | 0.060 | [0.035, 0.088] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.820 | 0.698 | 0.776 | 0.664 | 0.465 | 1.000 |
| held-out human |  | 0.214 | 0.385 | 0.146 | 0.013 | 1.000 |
| sentences shuffled |  |  | 0.693 | 0.412 | 0.049 | 1.000 |
| words shuffled |  |  |  | 0.226 | 0.019 | 1.000 |
| random lexicon words |  |  |  |  | 0.060 | 1.000 |
| random letters |  |  |  |  |  | 1.000 |

### GPT-2 PPL (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 1577.409 [1419.505, 1767.124] |
| held-out human | 150 | 0 | 36.469 [33.919, 39.254] |
| sentences shuffled | 150 | 0 | 66.184 [62.470, 70.465] |
| words shuffled | 150 | 0 | 1034.817 [960.240, 1111.744] |
| random lexicon words | 150 | 0 | 2449.814 [2304.278, 2594.924] |
| random letters | 150 | 0 | 515.325 [502.597, 528.932] |
| one phrase repeated | 150 | 0 | 1.725 [1.686, 1.768] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.000 | [0.000, 0.000] | no | FAIL |
| held-out human > sentences shuffled | 0.866 | [0.825, 0.904] | yes | FAIL |
| sentences shuffled > words shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| words shuffled > random lexicon words | 0.948 | [0.924, 0.970] | yes | FAIL |
| random lexicon words > random letters | 0.000 | [0.000, 0.000] | no | FAIL |
| random letters > one phrase repeated | 0.000 | [0.000, 0.000] | no | FAIL |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.000 | 0.000 | 0.347 | 0.800 | 0.071 | 0.000 |
| held-out human |  | 0.866 | 1.000 | 1.000 | 1.000 | 0.000 |
| sentences shuffled |  |  | 1.000 | 1.000 | 1.000 | 0.000 |
| words shuffled |  |  |  | 0.948 | 0.095 | 0.000 |
| random lexicon words |  |  |  |  | 0.000 | 0.000 |
| random letters |  |  |  |  |  | 0.000 |

### CoLA acceptability, mean (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.089 [0.081, 0.098] |
| held-out human | 150 | 0 | 0.874 [0.860, 0.886] |
| sentences shuffled | 150 | 0 | 0.863 [0.848, 0.878] |
| words shuffled | 150 | 0 | 0.076 [0.069, 0.083] |
| random lexicon words | 150 | 0 | 0.057 [0.054, 0.061] |
| random letters | 150 | 0 | 0.495 [0.489, 0.501] |
| one phrase repeated | 150 | 0 | 0.477 [0.430, 0.527] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.000 | [0.000, 0.000] | no | FAIL |
| held-out human > sentences shuffled | 0.527 | [0.459, 0.589] | no | FAIL |
| sentences shuffled > words shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| words shuffled > random lexicon words | 0.715 | [0.656, 0.774] | yes | FAIL |
| random lexicon words > random letters | 0.000 | [0.000, 0.000] | no | FAIL |
| random letters > one phrase repeated | 0.556 | [0.481, 0.628] | no | FAIL |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.000 | 0.000 | 0.683 | 0.884 | 0.000 | 0.073 |
| held-out human |  | 0.527 | 1.000 | 1.000 | 1.000 | 0.821 |
| sentences shuffled |  |  | 1.000 | 1.000 | 0.995 | 0.812 |
| words shuffled |  |  |  | 0.715 | 0.000 | 0.057 |
| random lexicon words |  |  |  |  | 0.000 | 0.029 |
| random letters |  |  |  |  |  | 0.556 |

### CoLA acceptability, min (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.071 [0.068, 0.075] |
| held-out human | 150 | 0 | 0.720 [0.682, 0.755] |
| sentences shuffled | 150 | 0 | 0.681 [0.639, 0.718] |
| words shuffled | 150 | 0 | 0.045 [0.043, 0.047] |
| random lexicon words | 150 | 0 | 0.040 [0.039, 0.041] |
| random letters | 150 | 0 | 0.411 [0.401, 0.421] |
| one phrase repeated | 150 | 0 | 0.471 [0.422, 0.520] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.007 | [0.000, 0.020] | no | FAIL |
| held-out human > sentences shuffled | 0.542 | [0.475, 0.600] | no | FAIL |
| sentences shuffled > words shuffled | 0.999 | [0.996, 1.000] | yes | **PASS** |
| words shuffled > random lexicon words | 0.621 | [0.557, 0.681] | yes | FAIL |
| random lexicon words > random letters | 0.000 | [0.000, 0.000] | no | FAIL |
| random letters > one phrase repeated | 0.505 | [0.426, 0.577] | no | FAIL |

Passed 1 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.007 | 0.011 | 0.906 | 0.964 | 0.003 | 0.059 |
| held-out human |  | 0.542 | 0.999 | 1.000 | 0.888 | 0.708 |
| sentences shuffled |  |  | 0.999 | 1.000 | 0.821 | 0.676 |
| words shuffled |  |  |  | 0.621 | 0.000 | 0.015 |
| random lexicon words |  |  |  |  | 0.000 | 0.007 |
| random letters |  |  |  |  |  | 0.505 |

### MiniLM coherence, adjacent (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 45 | 105 | 0.161 [0.139, 0.182] |
| held-out human | 150 | 0 | 0.391 [0.372, 0.412] |
| sentences shuffled | 150 | 0 | 0.052 [0.042, 0.062] |
| words shuffled | 150 | 0 | 0.351 [0.333, 0.371] |
| random lexicon words | 150 | 0 | 0.152 [0.144, 0.160] |
| random letters | 150 | 0 | 0.567 [0.553, 0.579] |
| one phrase repeated | 150 | 0 | 0.985 [0.966, 1.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.055 | [0.027, 0.090] | no | FAIL |
| held-out human > sentences shuffled | 0.992 | [0.981, 0.998] | yes | **PASS** |
| sentences shuffled > words shuffled | 0.007 | [0.002, 0.013] | no | FAIL |
| words shuffled > random lexicon words | 0.956 | [0.933, 0.976] | yes | **PASS** |
| random lexicon words > random letters | 0.000 | [0.000, 0.000] | no | FAIL |
| random letters > one phrase repeated | 0.020 | [0.000, 0.047] | no | FAIL |

Passed 2 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.055 | 0.879 | 0.074 | 0.552 | 0.000 | 0.003 |
| held-out human |  | 0.992 | 0.594 | 0.966 | 0.128 | 0.017 |
| sentences shuffled |  |  | 0.007 | 0.101 | 0.000 | 0.000 |
| words shuffled |  |  |  | 0.956 | 0.070 | 0.015 |
| random lexicon words |  |  |  |  | 0.000 | 0.001 |
| random letters |  |  |  |  |  | 0.020 |

### MiniLM coherence, all pairs (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 45 | 105 | 0.166 [0.144, 0.186] |
| held-out human | 150 | 0 | 0.372 [0.353, 0.392] |
| sentences shuffled | 150 | 0 | 0.047 [0.040, 0.054] |
| words shuffled | 150 | 0 | 0.334 [0.317, 0.352] |
| random lexicon words | 150 | 0 | 0.151 [0.145, 0.158] |
| random letters | 150 | 0 | 0.567 [0.555, 0.579] |
| one phrase repeated | 150 | 0 | 0.992 [0.982, 1.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.064 | [0.031, 0.100] | no | FAIL |
| held-out human > sentences shuffled | 0.997 | [0.994, 1.000] | yes | **PASS** |
| sentences shuffled > words shuffled | 0.003 | [0.001, 0.007] | no | FAIL |
| words shuffled > random lexicon words | 0.954 | [0.930, 0.976] | yes | **PASS** |
| random lexicon words > random letters | 0.000 | [0.000, 0.000] | no | FAIL |
| random letters > one phrase repeated | 0.006 | [0.000, 0.015] | no | FAIL |

Passed 2 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.064 | 0.928 | 0.095 | 0.576 | 0.000 | 0.000 |
| held-out human |  | 0.997 | 0.585 | 0.969 | 0.103 | 0.001 |
| sentences shuffled |  |  | 0.003 | 0.047 | 0.000 | 0.000 |
| words shuffled |  |  |  | 0.954 | 0.054 | 0.000 |
| random lexicon words |  |  |  |  | 0.000 | 0.000 |
| random letters |  |  |  |  |  | 0.006 |

### GPT-2 detector P(human) (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| trie samples (pooled) | 150 | 0 | 0.145 [0.106, 0.186] |
| held-out human | 150 | 0 | 0.818 [0.770, 0.865] |
| sentences shuffled | 150 | 0 | 0.695 [0.636, 0.753] |
| words shuffled | 150 | 0 | 0.710 [0.659, 0.760] |
| random lexicon words | 150 | 0 | 0.077 [0.055, 0.100] |
| random letters | 150 | 0 | 0.267 [0.230, 0.304] |
| one phrase repeated | 150 | 0 | 0.074 [0.041, 0.113] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| trie samples (pooled) > held-out human | 0.060 | [0.038, 0.082] | no | FAIL |
| held-out human > sentences shuffled | 0.654 | [0.592, 0.713] | yes | FAIL |
| sentences shuffled > words shuffled | 0.520 | [0.455, 0.583] | no | FAIL |
| words shuffled > random lexicon words | 0.943 | [0.914, 0.968] | yes | FAIL |
| random lexicon words > random letters | 0.163 | [0.118, 0.209] | no | FAIL |
| random letters > one phrase repeated | 0.895 | [0.851, 0.935] | yes | FAIL |

Passed 0 of 6 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | held-out human | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|---|
| trie samples (pooled) | 0.060 | 0.115 | 0.097 | 0.496 | 0.233 | 0.747 |
| held-out human |  | 0.654 | 0.691 | 0.961 | 0.890 | 0.961 |
| sentences shuffled |  |  | 0.520 | 0.916 | 0.817 | 0.935 |
| words shuffled |  |  |  | 0.943 | 0.848 | 0.941 |
| random lexicon words |  |  |  |  | 0.163 | 0.764 |
| random letters |  |  |  |  |  | 0.895 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | trie samples (pooled) > held-out human | held-out human > sentences shuffled | sentences shuffled > words shuffled | words shuffled > random lexicon words | random lexicon words > random letters | random letters > one phrase repeated | passed |
|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.002 FAIL | 0.978 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 5 of 6 |
| SCRIBE v1 | 0.010 FAIL | 0.875 FAIL | 0.835 FAIL | 0.707 FAIL | 1.000 **PASS** | 0.999 **PASS** | 2 of 6 |
| surface S | 0.059 FAIL | 0.630 FAIL | 0.339 FAIL | 0.726 FAIL | 1.000 **PASS** | 0.539 FAIL | 1 of 6 |
| distinct-1 | 0.575 FAIL | 0.258 FAIL | 0.742 FAIL | 0.123 FAIL | 0.000 FAIL | 1.000 **PASS** | 1 of 6 |
| distinct-2 | 0.537 FAIL | 0.273 FAIL | 0.342 FAIL | 0.410 FAIL | 0.360 FAIL | 1.000 **PASS** | 1 of 6 |
| distinct-3 | 0.673 FAIL | 0.382 FAIL | 0.449 FAIL | 0.493 FAIL | 0.497 FAIL | 1.000 **PASS** | 1 of 6 |
| distinct-4 | 0.557 FAIL | 0.463 FAIL | 0.480 FAIL | 0.500 FAIL | 0.500 FAIL | 1.000 **PASS** | 1 of 6 |
| rep-2 | 0.537 FAIL | 0.273 FAIL | 0.342 FAIL | 0.410 FAIL | 0.360 FAIL | 1.000 **PASS** | 1 of 6 |
| rep-3 | 0.673 FAIL | 0.382 FAIL | 0.449 FAIL | 0.493 FAIL | 0.497 FAIL | 1.000 **PASS** | 1 of 6 |
| rep-4 | 0.557 FAIL | 0.463 FAIL | 0.480 FAIL | 0.500 FAIL | 0.500 FAIL | 1.000 **PASS** | 1 of 6 |
| seq-rep-4 | 0.557 FAIL | 0.463 FAIL | 0.480 FAIL | 0.500 FAIL | 0.500 FAIL | 1.000 **PASS** | 1 of 6 |
| diversity | 0.566 FAIL | 0.273 FAIL | 0.338 FAIL | 0.410 FAIL | 0.360 FAIL | 1.000 **PASS** | 1 of 6 |
| type-token ratio | 0.575 FAIL | 0.258 FAIL | 0.742 FAIL | 0.123 FAIL | 0.000 FAIL | 1.000 **PASS** | 1 of 6 |
| gzip ratio | 0.820 FAIL | 0.214 FAIL | 0.693 FAIL | 0.226 FAIL | 0.060 FAIL | 1.000 **PASS** | 1 of 6 |
| GPT-2 PPL | 0.000 FAIL | 0.866 FAIL | 1.000 **PASS** | 0.948 FAIL | 0.000 FAIL | 0.000 FAIL | 1 of 6 |
| CoLA acceptability, mean | 0.000 FAIL | 0.527 FAIL | 1.000 **PASS** | 0.715 FAIL | 0.000 FAIL | 0.556 FAIL | 1 of 6 |
| CoLA acceptability, min | 0.007 FAIL | 0.542 FAIL | 0.999 **PASS** | 0.621 FAIL | 0.000 FAIL | 0.505 FAIL | 1 of 6 |
| MiniLM coherence, adjacent | 0.055 FAIL | 0.992 **PASS** | 0.007 FAIL | 0.956 **PASS** | 0.000 FAIL | 0.020 FAIL | 2 of 6 |
| MiniLM coherence, all pairs | 0.064 FAIL | 0.997 **PASS** | 0.003 FAIL | 0.954 **PASS** | 0.000 FAIL | 0.006 FAIL | 2 of 6 |
| GPT-2 detector P(human) | 0.060 FAIL | 0.654 FAIL | 0.520 FAIL | 0.943 FAIL | 0.163 FAIL | 0.895 FAIL | 0 of 6 |
