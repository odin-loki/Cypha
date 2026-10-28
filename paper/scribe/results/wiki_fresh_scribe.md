# SCRIBE v2, v1, surface S and the v2 deviance parts; 500 texts per rung

Analysis `wiki_fresh_scribe` (1. Fresh Wikipedia control ladder). meta.py `ladder`.
Inputs: SP/run/fresh/wiki_fresh/texts.jsonl, SP/run/fresh/wiki_fresh/scores.jsonl.

3000 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.916 [0.901, 0.930] |
| sentences shuffled | 500 | 0 | 0.354 [0.342, 0.366] |
| words shuffled | 500 | 0 | 0.003 [0.003, 0.003] |
| random lexicon words | 500 | 0 | 0.001 [0.001, 0.001] |
| random letters | 500 | 0 | 0.000 [0.000, 0.000] |
| one phrase repeated | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.975 | [0.963, 0.985] | yes | **PASS** |
| sentences shuffled > words shuffled | 0.999 | [0.996, 1.000] | yes | **PASS** |
| words shuffled > random lexicon words | 0.998 | [0.996, 1.000] | yes | **PASS** |
| random lexicon words > random letters | 0.998 | [0.994, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 |
| sentences shuffled |  | 0.999 | 1.000 | 1.000 | 1.000 |
| words shuffled |  |  | 0.998 | 1.000 | 1.000 |
| random lexicon words |  |  |  | 0.998 | 1.000 |
| random letters |  |  |  |  | 1.000 |

### SCRIBE v1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.556 [0.538, 0.573] |
| sentences shuffled | 500 | 0 | 0.235 [0.225, 0.245] |
| words shuffled | 500 | 0 | 0.130 [0.120, 0.140] |
| random lexicon words | 500 | 0 | 0.068 [0.066, 0.070] |
| random letters | 500 | 0 | 0.006 [0.006, 0.006] |
| one phrase repeated | 500 | 0 | 0.001 [0.001, 0.001] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.888 | [0.864, 0.909] | yes | FAIL |
| sentences shuffled > words shuffled | 0.806 | [0.778, 0.834] | yes | FAIL |
| words shuffled > random lexicon words | 0.696 | [0.659, 0.734] | yes | FAIL |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 0.999 | [0.998, 0.999] | yes | **PASS** |

Passed 2 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.888 | 0.947 | 0.986 | 1.000 | 1.000 |
| sentences shuffled |  | 0.806 | 0.987 | 1.000 | 1.000 |
| words shuffled |  |  | 0.696 | 0.996 | 1.000 |
| random lexicon words |  |  |  | 1.000 | 1.000 |
| random letters |  |  |  |  | 0.999 |

### surface S (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.915 [0.910, 0.921] |
| sentences shuffled | 500 | 0 | 0.905 [0.901, 0.908] |
| words shuffled | 500 | 0 | 0.925 [0.920, 0.929] |
| random lexicon words | 500 | 0 | 0.895 [0.893, 0.897] |
| random letters | 500 | 0 | 0.035 [0.034, 0.037] |
| one phrase repeated | 500 | 0 | 0.029 [0.029, 0.030] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.627 | [0.591, 0.658] | yes | FAIL |
| sentences shuffled > words shuffled | 0.341 | [0.307, 0.374] | no | FAIL |
| words shuffled > random lexicon words | 0.750 | [0.720, 0.783] | yes | FAIL |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 0.559 | [0.520, 0.599] | yes | FAIL |

Passed 1 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.627 | 0.474 | 0.709 | 1.000 | 1.000 |
| sentences shuffled |  | 0.341 | 0.603 | 1.000 | 1.000 |
| words shuffled |  |  | 0.750 | 1.000 | 1.000 |
| random lexicon words |  |  |  | 1.000 | 1.000 |
| random letters |  |  |  |  | 0.559 |

### v2 deviance D (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.603 [0.468, 0.748] |
| sentences shuffled | 500 | 0 | 5.569 [5.390, 5.754] |
| words shuffled | 500 | 0 | 28.575 [28.468, 28.676] |
| random lexicon words | 500 | 0 | 35.527 [35.401, 35.650] |
| random letters | 500 | 0 | 53.488 [53.251, 53.746] |
| one phrase repeated | 500 | 0 | 305.765 [304.612, 306.935] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.975 | [0.963, 0.985] | yes | **PASS** |
| sentences shuffled > words shuffled | 0.999 | [0.996, 1.000] | yes | **PASS** |
| words shuffled > random lexicon words | 0.998 | [0.996, 1.000] | yes | **PASS** |
| random lexicon words > random letters | 0.998 | [0.994, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 |
| sentences shuffled |  | 0.999 | 1.000 | 1.000 | 1.000 |
| words shuffled |  |  | 0.998 | 1.000 | 1.000 |
| random lexicon words |  |  |  | 0.998 | 1.000 |
| random letters |  |  |  |  | 1.000 |

### v2 D_L (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.031 [0.003, 0.068] |
| sentences shuffled | 500 | 0 | 0.000 [0.000, 0.001] |
| words shuffled | 500 | 0 | 0.031 [0.003, 0.067] |
| random lexicon words | 500 | 0 | 0.000 [0.000, 0.000] |
| random letters | 500 | 0 | 27.646 [27.603, 27.689] |
| one phrase repeated | 500 | 0 | 0.796 [0.549, 1.101] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.496 | [0.491, 0.500] | no | FAIL |
| sentences shuffled > words shuffled | 0.504 | [0.500, 0.510] | yes | FAIL |
| words shuffled > random lexicon words | 0.495 | [0.490, 0.499] | no | FAIL |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 0.000 | [0.000, 0.000] | no | FAIL |

Passed 1 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.496 | 0.500 | 0.495 | 1.000 | 0.588 |
| sentences shuffled |  | 0.504 | 0.499 | 1.000 | 0.592 |
| words shuffled |  |  | 0.495 | 1.000 | 0.588 |
| random lexicon words |  |  |  | 1.000 | 0.593 |
| random letters |  |  |  |  | 0.000 |

### v2 D_S (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.012 [0.000, 0.035] |
| sentences shuffled | 500 | 0 | 0.004 [0.000, 0.013] |
| words shuffled | 500 | 0 | 28.143 [28.072, 28.217] |
| random lexicon words | 500 | 0 | 28.428 [28.368, 28.484] |
| random letters | 500 | 0 | 23.535 [23.310, 23.750] |
| one phrase repeated | 500 | 0 | 3.965 [3.347, 4.625] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.499 | [0.495, 0.502] | no | FAIL |
| sentences shuffled > words shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| words shuffled > random lexicon words | 0.596 | [0.564, 0.630] | yes | FAIL |
| random lexicon words > random letters | 0.005 | [0.002, 0.009] | no | FAIL |
| random letters > one phrase repeated | 0.030 | [0.019, 0.042] | no | FAIL |

Passed 1 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.499 | 1.000 | 1.000 | 1.000 | 0.645 |
| sentences shuffled |  | 1.000 | 1.000 | 1.000 | 0.646 |
| words shuffled |  |  | 0.596 | 0.014 | 0.002 |
| random lexicon words |  |  |  | 0.005 | 0.002 |
| random letters |  |  |  |  | 0.030 |

### v2 D_C (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.401 [0.332, 0.478] |
| sentences shuffled | 500 | 0 | 5.481 [5.331, 5.630] |
| words shuffled | 500 | 0 | 0.401 [0.333, 0.475] |
| random lexicon words | 500 | 0 | 7.099 [6.986, 7.209] |
| random letters | 500 | 0 | 2.307 [2.228, 2.395] |
| one phrase repeated | 500 | 0 | 2.104 [1.949, 2.254] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.990 | [0.985, 0.994] | yes | **PASS** |
| sentences shuffled > words shuffled | 0.010 | [0.006, 0.015] | no | FAIL |
| words shuffled > random lexicon words | 1.000 | [0.999, 1.000] | yes | **PASS** |
| random lexicon words > random letters | 0.007 | [0.003, 0.012] | no | FAIL |
| random letters > one phrase repeated | 0.536 | [0.496, 0.577] | no | FAIL |

Passed 2 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.990 | 0.496 | 0.999 | 0.945 | 0.754 |
| sentences shuffled |  | 0.010 | 0.773 | 0.069 | 0.086 |
| words shuffled |  |  | 1.000 | 0.944 | 0.755 |
| random lexicon words |  |  |  | 0.007 | 0.016 |
| random letters |  |  |  |  | 0.536 |

### v2 D_I (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.160 [0.051, 0.293] |
| sentences shuffled | 500 | 0 | 0.084 [0.000, 0.233] |
| words shuffled | 500 | 0 | 0.000 [0.000, 0.000] |
| random lexicon words | 500 | 0 | 0.000 [0.000, 0.000] |
| random letters | 500 | 0 | 0.000 [0.000, 0.000] |
| one phrase repeated | 500 | 0 | 298.900 [297.695, 299.986] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.493 | [0.487, 0.499] | no | FAIL |
| sentences shuffled > words shuffled | 0.498 | [0.495, 0.500] | no | FAIL |
| words shuffled > random lexicon words | 0.500 | [0.500, 0.500] | no | FAIL |
| random lexicon words > random letters | 0.500 | [0.500, 0.500] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 1 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.493 | 0.491 | 0.491 | 0.491 | 1.000 |
| sentences shuffled |  | 0.498 | 0.498 | 0.498 | 1.000 |
| words shuffled |  |  | 0.500 | 0.500 | 1.000 |
| random lexicon words |  |  |  | 0.500 | 1.000 |
| random letters |  |  |  |  | 1.000 |

### v2 D_R (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.000 [0.000, 0.000] |
| sentences shuffled | 500 | 0 | 0.000 [0.000, 0.000] |
| words shuffled | 500 | 0 | 0.000 [0.000, 0.000] |
| random lexicon words | 500 | 0 | 0.000 [0.000, 0.000] |
| random letters | 500 | 0 | 0.000 [0.000, 0.000] |
| one phrase repeated | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.500 | [0.500, 0.500] | no | FAIL |
| sentences shuffled > words shuffled | 0.500 | [0.500, 0.500] | no | FAIL |
| words shuffled > random lexicon words | 0.500 | [0.500, 0.500] | no | FAIL |
| random lexicon words > random letters | 0.500 | [0.500, 0.500] | no | FAIL |
| random letters > one phrase repeated | 0.500 | [0.500, 0.500] | no | FAIL |

Passed 0 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| sentences shuffled |  | 0.500 | 0.500 | 0.500 | 0.500 |
| words shuffled |  |  | 0.500 | 0.500 | 0.500 |
| random lexicon words |  |  |  | 0.500 | 0.500 |
| random letters |  |  |  |  | 0.500 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | held-out human > sentences shuffled | sentences shuffled > words shuffled | words shuffled > random lexicon words | random lexicon words > random letters | random letters > one phrase repeated | passed |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.975 **PASS** | 0.999 **PASS** | 0.998 **PASS** | 0.998 **PASS** | 1.000 **PASS** | 5 of 5 |
| SCRIBE v1 | 0.888 FAIL | 0.806 FAIL | 0.696 FAIL | 1.000 **PASS** | 0.999 **PASS** | 2 of 5 |
| surface S | 0.627 FAIL | 0.341 FAIL | 0.750 FAIL | 1.000 **PASS** | 0.559 FAIL | 1 of 5 |
| v2 deviance D | 0.975 **PASS** | 0.999 **PASS** | 0.998 **PASS** | 0.998 **PASS** | 1.000 **PASS** | 5 of 5 |
| v2 D_L | 0.496 FAIL | 0.504 FAIL | 0.495 FAIL | 1.000 **PASS** | 0.000 FAIL | 1 of 5 |
| v2 D_S | 0.499 FAIL | 1.000 **PASS** | 0.596 FAIL | 0.005 FAIL | 0.030 FAIL | 1 of 5 |
| v2 D_C | 0.990 **PASS** | 0.010 FAIL | 1.000 **PASS** | 0.007 FAIL | 0.536 FAIL | 2 of 5 |
| v2 D_I | 0.493 FAIL | 0.498 FAIL | 0.500 FAIL | 0.500 FAIL | 1.000 **PASS** | 1 of 5 |
| v2 D_R | 0.500 FAIL | 0.500 FAIL | 0.500 FAIL | 0.500 FAIL | 0.500 FAIL | 0 of 5 |
