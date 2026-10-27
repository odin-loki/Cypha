# Books ladder, each genre with its own pack and reference, SCRIBE only

Analysis `genre_books` (2. Genre control ladders). meta.py `ladder`.
Inputs: SP/run/genre/books/texts.jsonl, SP/run/genre/books/scores.jsonl.

3000 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.906 [0.893, 0.919] |
| sentences shuffled | 500 | 0 | 0.747 [0.730, 0.761] |
| words shuffled | 500 | 0 | 0.003 [0.003, 0.003] |
| random lexicon words | 500 | 0 | 0.002 [0.002, 0.002] |
| random letters | 500 | 0 | 0.000 [0.000, 0.000] |
| one phrase repeated | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.785 | [0.756, 0.813] | yes | FAIL |
| sentences shuffled > words shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| words shuffled > random lexicon words | 0.894 | [0.873, 0.913] | yes | FAIL |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.785 | 1.000 | 1.000 | 1.000 | 1.000 |
| sentences shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| words shuffled |  |  | 0.894 | 1.000 | 1.000 |
| random lexicon words |  |  |  | 1.000 | 1.000 |
| random letters |  |  |  |  | 1.000 |

### SCRIBE v1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.510 [0.492, 0.527] |
| sentences shuffled | 500 | 0 | 0.453 [0.438, 0.466] |
| words shuffled | 500 | 0 | 0.077 [0.073, 0.080] |
| random lexicon words | 500 | 0 | 0.080 [0.078, 0.083] |
| random letters | 500 | 0 | 0.006 [0.006, 0.006] |
| one phrase repeated | 500 | 0 | 0.001 [0.001, 0.001] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.607 | [0.571, 0.641] | yes | FAIL |
| sentences shuffled > words shuffled | 0.981 | [0.972, 0.989] | yes | **PASS** |
| words shuffled > random lexicon words | 0.452 | [0.416, 0.490] | no | FAIL |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 0.997 | [0.996, 0.999] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.607 | 0.975 | 0.975 | 1.000 | 1.000 |
| sentences shuffled |  | 0.981 | 0.980 | 1.000 | 1.000 |
| words shuffled |  |  | 0.452 | 0.991 | 1.000 |
| random lexicon words |  |  |  | 1.000 | 1.000 |
| random letters |  |  |  |  | 0.997 |

### surface S (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 500 | 0 | 0.949 [0.946, 0.952] |
| sentences shuffled | 500 | 0 | 0.941 [0.939, 0.944] |
| words shuffled | 500 | 0 | 0.951 [0.949, 0.953] |
| random lexicon words | 500 | 0 | 0.944 [0.943, 0.946] |
| random letters | 500 | 0 | 0.019 [0.018, 0.021] |
| one phrase repeated | 500 | 0 | 0.018 [0.017, 0.018] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 0.602 | [0.565, 0.638] | yes | FAIL |
| sentences shuffled > words shuffled | 0.401 | [0.367, 0.436] | no | FAIL |
| words shuffled > random lexicon words | 0.597 | [0.559, 0.632] | yes | FAIL |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 0.481 | [0.439, 0.519] | yes | FAIL |

Passed 1 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 0.602 | 0.508 | 0.600 | 1.000 | 1.000 |
| sentences shuffled |  | 0.401 | 0.480 | 1.000 | 1.000 |
| words shuffled |  |  | 0.597 | 1.000 | 1.000 |
| random lexicon words |  |  |  | 1.000 | 1.000 |
| random letters |  |  |  |  | 0.481 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | held-out human > sentences shuffled | sentences shuffled > words shuffled | words shuffled > random lexicon words | random lexicon words > random letters | random letters > one phrase repeated | passed |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.785 FAIL | 1.000 **PASS** | 0.894 FAIL | 1.000 **PASS** | 1.000 **PASS** | 3 of 5 |
| SCRIBE v1 | 0.607 FAIL | 0.981 **PASS** | 0.452 FAIL | 1.000 **PASS** | 0.997 **PASS** | 3 of 5 |
| surface S | 0.602 FAIL | 0.401 FAIL | 0.597 FAIL | 1.000 **PASS** | 0.481 FAIL | 1 of 5 |
