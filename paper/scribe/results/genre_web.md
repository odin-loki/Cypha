# Web ladder, each genre with its own pack and reference, SCRIBE only

Analysis `genre_web` (2. Genre control ladders). meta.py `ladder`.
Inputs: SP/run/genre/web/texts.jsonl, SP/run/genre/web/scores.jsonl.

3000 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.923 [0.909, 0.936] |
| sentences shuffled | 500 | 0 | 0.431 [0.417, 0.445] |
| words shuffled | 500 | 0 | 0.004 [0.004, 0.004] |
| random lexicon words | 500 | 0 | 0.001 [0.001, 0.002] |
| random letters | 500 | 0 | 0.000 [0.000, 0.000] |
| one phrase repeated | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.966 | [0.953, 0.977] | yes | **PASS** |
| sentences shuffled > words shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| words shuffled > random lexicon words | 0.985 | [0.978, 0.992] | yes | **PASS** |
| random lexicon words > random letters | 0.999 | [0.998, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.966 | 0.996 | 0.996 | 0.999 | 1.000 |
| sentences shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| words shuffled |  |  | 0.985 | 1.000 | 1.000 |
| random lexicon words |  |  |  | 0.999 | 1.000 |
| random letters |  |  |  |  | 1.000 |

### SCRIBE v1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.545 [0.527, 0.561] |
| sentences shuffled | 500 | 0 | 0.279 [0.268, 0.290] |
| words shuffled | 500 | 0 | 0.204 [0.194, 0.216] |
| random lexicon words | 500 | 0 | 0.100 [0.096, 0.103] |
| random letters | 500 | 0 | 0.002 [0.002, 0.002] |
| one phrase repeated | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.862 | [0.837, 0.887] | yes | FAIL |
| sentences shuffled > words shuffled | 0.680 | [0.645, 0.712] | yes | FAIL |
| words shuffled > random lexicon words | 0.772 | [0.744, 0.802] | yes | FAIL |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 0.995 | [0.990, 0.998] | yes | **PASS** |

Passed 2 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.862 | 0.910 | 0.969 | 1.000 | 1.000 |
| sentences shuffled |  | 0.680 | 0.941 | 1.000 | 1.000 |
| words shuffled |  |  | 0.772 | 1.000 | 1.000 |
| random lexicon words |  |  |  | 1.000 | 1.000 |
| random letters |  |  |  |  | 0.995 |

### surface S (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.932 [0.926, 0.937] |
| sentences shuffled | 500 | 0 | 0.920 [0.917, 0.923] |
| words shuffled | 500 | 0 | 0.943 [0.939, 0.946] |
| random lexicon words | 500 | 0 | 0.914 [0.913, 0.916] |
| random letters | 500 | 0 | 0.052 [0.050, 0.054] |
| one phrase repeated | 500 | 0 | 0.023 [0.023, 0.024] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.663 | [0.627, 0.695] | yes | FAIL |
| sentences shuffled > words shuffled | 0.295 | [0.263, 0.327] | no | FAIL |
| words shuffled > random lexicon words | 0.776 | [0.743, 0.805] | yes | FAIL |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 0.870 | [0.844, 0.894] | yes | FAIL |

Passed 1 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.663 | 0.473 | 0.728 | 1.000 | 1.000 |
| sentences shuffled |  | 0.295 | 0.576 | 1.000 | 1.000 |
| words shuffled |  |  | 0.776 | 1.000 | 1.000 |
| random lexicon words |  |  |  | 1.000 | 1.000 |
| random letters |  |  |  |  | 0.870 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | held-out human > sentences shuffled | sentences shuffled > words shuffled | words shuffled > random lexicon words | random lexicon words > random letters | random letters > one phrase repeated | passed |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.966 **PASS** | 1.000 **PASS** | 0.985 **PASS** | 0.999 **PASS** | 1.000 **PASS** | 5 of 5 |
| SCRIBE v1 | 0.862 FAIL | 0.680 FAIL | 0.772 FAIL | 1.000 **PASS** | 0.995 **PASS** | 2 of 5 |
| surface S | 0.663 FAIL | 0.295 FAIL | 0.776 FAIL | 1.000 **PASS** | 0.870 FAIL | 1 of 5 |
