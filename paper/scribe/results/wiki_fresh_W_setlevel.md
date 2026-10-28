# Set-level metrics on W, one set per rung (q: 100 texts per rung; MAUVE with GPT-2 small features against 100 other human texts)

Analysis `wiki_fresh_W_setlevel` (1. Fresh Wikipedia control ladder). meta.py `setladder`.
Inputs: SP/run/baselines/wiki_fresh_W_mauve/setlevel.jsonl.

6 subsets. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

Units are subsets: accuracy is over all (upper subset, lower subset) pairs and the bootstrap resamples subsets within each rung.

### MAUVE (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 1 | 0 | 0.994 [0.994, 0.994] |
| sentences shuffled | 1 | 0 | 0.979 [0.979, 0.979] |
| words shuffled | 1 | 0 | 0.006 [0.006, 0.006] |
| random lexicon words | 1 | 0 | 0.005 [0.005, 0.005] |
| random letters | 1 | 0 | 0.005 [0.005, 0.005] |
| one phrase repeated | 1 | 0 | 0.036 [0.036, 0.036] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| sentences shuffled > words shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| words shuffled > random lexicon words | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 0.000 | [0.000, 0.000] | no | FAIL |

Passed 4 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sentences shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| words shuffled |  |  | 1.000 | 1.000 | 0.000 |
| random lexicon words |  |  |  | 1.000 | 0.000 |
| random letters |  |  |  |  | 0.000 |

### self-BLEU (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 1 | 0 | 0.052 [0.052, 0.052] |
| sentences shuffled | 1 | 0 | 0.289 [0.289, 0.289] |
| words shuffled | 1 | 0 | 0.042 [0.042, 0.042] |
| random lexicon words | 1 | 0 | 0.046 [0.046, 0.046] |
| random letters | 1 | 0 | 0.014 [0.014, 0.014] |
| one phrase repeated | 1 | 0 | 0.031 [0.031, 0.031] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| sentences shuffled > words shuffled | 0.000 | [0.000, 0.000] | no | FAIL |
| words shuffled > random lexicon words | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random lexicon words > random letters | 0.000 | [0.000, 0.000] | no | FAIL |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| sentences shuffled |  | 0.000 | 0.000 | 0.000 | 0.000 |
| words shuffled |  |  | 1.000 | 0.000 | 0.000 |
| random lexicon words |  |  |  | 0.000 | 0.000 |
| random letters |  |  |  |  | 1.000 |

### |Zipf − human Zipf| (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 1 | 0 | 0.000 [0.000, 0.000] |
| sentences shuffled | 1 | 0 | 0.020 [0.020, 0.020] |
| words shuffled | 1 | 0 | 0.020 [0.020, 0.020] |
| random lexicon words | 1 | 0 | 0.037 [0.037, 0.037] |
| random letters | 1 | 0 | 0.207 [0.207, 0.207] |
| one phrase repeated | 1 | 0 | 0.241 [0.241, 0.241] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > sentences shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| sentences shuffled > words shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| words shuffled > random lexicon words | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random lexicon words > random letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random letters > one phrase repeated | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sentences shuffled | words shuffled | random lexicon words | random letters | one phrase repeated |
|---|---|---|---|---|---|
| held-out human | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sentences shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| words shuffled |  |  | 1.000 | 1.000 | 1.000 |
| random lexicon words |  |  |  | 1.000 | 1.000 |
| random letters |  |  |  |  | 1.000 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | held-out human > sentences shuffled | sentences shuffled > words shuffled | words shuffled > random lexicon words | random lexicon words > random letters | random letters > one phrase repeated | passed |
|---|---|---|---|---|---|---|
| MAUVE | 1.000 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 0.000 FAIL | 4 of 5 |
| self-BLEU | 1.000 **PASS** | 0.000 FAIL | 1.000 **PASS** | 0.000 FAIL | 1.000 **PASS** | 3 of 5 |
| \|Zipf − human Zipf\| | 1.000 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 5 of 5 |
