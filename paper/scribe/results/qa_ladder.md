# correct > other answer > prompt echo > shuffled words

Analysis `qa_ladder` (9. Q/A ladder). meta.py `ladder`.
Inputs: SP/run/qa/test.jsonl, SP/run/qa/test.v2.jsonl, SP/run/qa/test.v2p.jsonl.

2400 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.894 [0.875, 0.910] |
| other | 600 | 0 | 0.875 [0.857, 0.892] |
| echo | 600 | 0 | 0.372 [0.339, 0.409] |
| shuffled | 600 | 0 | 0.135 [0.114, 0.155] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.519 | [0.490, 0.547] | no | FAIL |
| other > echo | 0.804 | [0.779, 0.826] | yes | FAIL |
| echo > shuffled | 0.417 | [0.381, 0.457] | yes | FAIL |

Passed 0 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.519 | 0.813 | 0.937 |
| other |  | 0.804 | 0.933 |
| echo |  |  | 0.417 |

### SCRIBE v2p (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.431 [0.399, 0.460] |
| other | 600 | 0 | 0.005 [0.004, 0.007] |
| echo | 600 | 0 | 0.006 [0.006, 0.007] |
| shuffled | 600 | 0 | 0.058 [0.047, 0.068] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.961 | [0.949, 0.972] | yes | **PASS** |
| other > echo | 0.587 | [0.548, 0.621] | no | FAIL |
| echo > shuffled | 0.254 | [0.228, 0.281] | no | FAIL |

Passed 1 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.961 | 0.957 | 0.863 |
| other |  | 0.587 | 0.208 |
| echo |  |  | 0.254 |

### SCRIBE v1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.463 [0.445, 0.481] |
| other | 600 | 0 | 0.445 [0.427, 0.462] |
| echo | 600 | 0 | 0.143 [0.126, 0.162] |
| shuffled | 600 | 0 | 0.304 [0.291, 0.318] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.526 | [0.493, 0.559] | no | FAIL |
| other > echo | 0.846 | [0.822, 0.867] | yes | FAIL |
| echo > shuffled | 0.228 | [0.200, 0.260] | no | FAIL |

Passed 0 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.526 | 0.856 | 0.712 |
| other |  | 0.846 | 0.691 |
| echo |  |  | 0.228 |

### surface S (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| correct | 600 | 0 | 0.857 [0.847, 0.866] |
| other | 600 | 0 | 0.856 [0.847, 0.865] |
| echo | 600 | 0 | 0.354 [0.327, 0.386] |
| shuffled | 600 | 0 | 0.875 [0.866, 0.883] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| correct > other | 0.506 | [0.474, 0.540] | no | FAIL |
| other > echo | 0.846 | [0.821, 0.867] | yes | FAIL |
| echo > shuffled | 0.135 | [0.115, 0.157] | no | FAIL |

Passed 0 of 3 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | other | echo | shuffled |
|---|---|---|---|
| correct | 0.506 | 0.849 | 0.460 |
| other |  | 0.846 | 0.454 |
| echo |  |  | 0.135 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | correct > other | other > echo | echo > shuffled | passed |
|---|---|---|---|---|
| SCRIBE v2 | 0.519 FAIL | 0.804 FAIL | 0.417 FAIL | 0 of 3 |
| SCRIBE v2p | 0.961 **PASS** | 0.587 FAIL | 0.254 FAIL | 1 of 3 |
| SCRIBE v1 | 0.526 FAIL | 0.846 FAIL | 0.228 FAIL | 0 of 3 |
| surface S | 0.506 FAIL | 0.846 FAIL | 0.135 FAIL | 0 of 3 |
