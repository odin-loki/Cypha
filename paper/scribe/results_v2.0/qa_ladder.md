# Control ladder

2400 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2p (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.442 [0.411, 0.472] |
| other | 600 | 0 | 0.005 [0.004, 0.007] |
| echo | 600 | 0 | 0.007 [0.006, 0.008] |
| shuffled | 600 | 0 | 0.058 [0.047, 0.068] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.961 | [0.949, 0.972] | yes | **PASS** |
| other > echo | 0.592 | [0.553, 0.626] | no | FAIL |
| echo > shuffled | 0.259 | [0.234, 0.286] | no | FAIL |

Passed 1 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.961 | 0.957 | 0.864 |
| other |  | 0.592 | 0.206 |
| echo |  |  | 0.259 |

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.898 [0.879, 0.914] |
| other | 600 | 0 | 0.879 [0.861, 0.896] |
| echo | 600 | 0 | 0.378 [0.344, 0.416] |
| shuffled | 600 | 0 | 0.135 [0.114, 0.155] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.521 | [0.489, 0.551] | no | FAIL |
| other > echo | 0.777 | [0.752, 0.801] | yes | FAIL |
| echo > shuffled | 0.412 | [0.375, 0.452] | yes | FAIL |

Passed 0 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.521 | 0.786 | 0.939 |
| other |  | 0.777 | 0.934 |
| echo |  |  | 0.412 |

### SCRIBE v3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.781 [0.765, 0.796] |
| other | 600 | 0 | 0.817 [0.805, 0.829] |
| echo | 600 | 0 | 0.249 [0.227, 0.273] |
| shuffled | 600 | 0 | 0.370 [0.354, 0.387] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.470 | [0.434, 0.505] | no | FAIL |
| other > echo | 0.964 | [0.955, 0.973] | yes | **PASS** |
| echo > shuffled | 0.326 | [0.294, 0.362] | no | FAIL |

Passed 1 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.470 | 0.926 | 0.907 |
| other |  | 0.964 | 0.939 |
| echo |  |  | 0.326 |

### SCRIBE v3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.552 [0.517, 0.583] |
| other | 600 | 0 | 0.648 [0.636, 0.660] |
| echo | 600 | 0 | 0.000 [0.000, 0.000] |
| shuffled | 600 | 0 | 0.113 [0.101, 0.126] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.536 | [0.496, 0.573] | no | FAIL |
| other > echo | 0.999 | [0.998, 1.000] | yes | **PASS** |
| echo > shuffled | 0.001 | [0.000, 0.002] | no | FAIL |

Passed 1 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.536 | 0.952 | 0.752 |
| other |  | 0.999 | 0.960 |
| echo |  |  | 0.001 |

### SURPRISE (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.912 [0.899, 0.923] |
| other | 600 | 0 | 0.901 [0.888, 0.912] |
| echo | 600 | 0 | 0.368 [0.335, 0.406] |
| shuffled | 600 | 0 | 0.535 [0.512, 0.559] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.516 | [0.486, 0.548] | no | FAIL |
| other > echo | 0.787 | [0.758, 0.812] | yes | FAIL |
| echo > shuffled | 0.316 | [0.284, 0.352] | no | FAIL |

Passed 0 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.516 | 0.792 | 0.817 |
| other |  | 0.787 | 0.808 |
| echo |  |  | 0.316 |

### MATCH (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.534 [0.498, 0.568] |
| other | 600 | 0 | 0.672 [0.659, 0.685] |
| echo | 600 | 0 | 0.000 [0.000, 0.000] |
| shuffled | 600 | 0 | 0.441 [0.414, 0.469] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.486 | [0.447, 0.521] | no | FAIL |
| other > echo | 0.998 | [0.994, 1.000] | yes | **PASS** |
| echo > shuffled | 0.000 | [0.000, 0.000] | no | FAIL |

Passed 1 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.486 | 0.981 | 0.528 |
| other |  | 0.998 | 0.692 |
| echo |  |  | 0.000 |

### SURPRISE 3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.746 [0.722, 0.765] |
| other | 600 | 0 | 0.688 [0.676, 0.698] |
| echo | 600 | 0 | 0.228 [0.207, 0.252] |
| shuffled | 600 | 0 | 0.138 [0.124, 0.155] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.679 | [0.642, 0.709] | yes | FAIL |
| other > echo | 0.895 | [0.873, 0.915] | yes | FAIL |
| echo > shuffled | 0.488 | [0.453, 0.524] | yes | FAIL |

Passed 0 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.679 | 0.894 | 0.933 |
| other |  | 0.895 | 0.951 |
| echo |  |  | 0.488 |

### MATCH 3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.484 [0.450, 0.518] |
| other | 600 | 0 | 0.477 [0.465, 0.489] |
| echo | 600 | 0 | 0.000 [0.000, 0.000] |
| shuffled | 600 | 0 | 0.370 [0.343, 0.396] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.535 | [0.496, 0.573] | no | FAIL |
| other > echo | 0.999 | [0.997, 1.000] | yes | **PASS** |
| echo > shuffled | 0.000 | [0.000, 0.000] | no | FAIL |

Passed 1 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.535 | 0.942 | 0.539 |
| other |  | 0.999 | 0.664 |
| echo |  |  | 0.000 |

### min of 3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.505 [0.471, 0.539] |
| other | 600 | 0 | 0.640 [0.626, 0.654] |
| echo | 600 | 0 | 0.000 [0.000, 0.000] |
| shuffled | 600 | 0 | 0.070 [0.058, 0.084] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.472 | [0.435, 0.507] | no | FAIL |
| other > echo | 0.997 | [0.993, 1.000] | yes | **PASS** |
| echo > shuffled | 0.000 | [0.000, 0.000] | no | FAIL |

Passed 1 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.472 | 0.985 | 0.737 |
| other |  | 0.997 | 0.960 |
| echo |  |  | 0.000 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | correct > other | other > echo | echo > shuffled | passed |
|---|---|---|---|---|
| SCRIBE v2p | 0.961 **PASS** | 0.592 FAIL | 0.259 FAIL | 1 of 3 |
| SCRIBE v2 | 0.521 FAIL | 0.777 FAIL | 0.412 FAIL | 0 of 3 |
| SCRIBE v3 | 0.470 FAIL | 0.964 **PASS** | 0.326 FAIL | 1 of 3 |
| SCRIBE v3.1 | 0.536 FAIL | 0.999 **PASS** | 0.001 FAIL | 1 of 3 |
| SURPRISE | 0.516 FAIL | 0.787 FAIL | 0.316 FAIL | 0 of 3 |
| MATCH | 0.486 FAIL | 0.998 **PASS** | 0.000 FAIL | 1 of 3 |
| SURPRISE 3.1 | 0.679 FAIL | 0.895 FAIL | 0.488 FAIL | 0 of 3 |
| MATCH 3.1 | 0.535 FAIL | 0.999 **PASS** | 0.000 FAIL | 1 of 3 |
| min of 3 | 0.472 FAIL | 0.997 **PASS** | 0.000 FAIL | 1 of 3 |
