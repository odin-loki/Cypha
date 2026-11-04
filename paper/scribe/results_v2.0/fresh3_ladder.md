# Control ladder

3000 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.909 [0.896, 0.924] |
| sent_shuffled | 500 | 0 | 0.353 [0.343, 0.364] |
| word_shuffled | 500 | 0 | 0.003 [0.003, 0.004] |
| random_lexicon | 500 | 0 | 0.001 [0.001, 0.001] |
| random_letters | 500 | 0 | 0.000 [0.000, 0.000] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.974 | [0.962, 0.984] | yes | **PASS** |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.998 | [0.995, 0.999] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.974 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.998 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SCRIBE v3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.913 [0.903, 0.923] |
| sent_shuffled | 500 | 0 | 0.560 [0.555, 0.565] |
| word_shuffled | 500 | 0 | 0.189 [0.187, 0.191] |
| random_lexicon | 500 | 0 | 0.135 [0.134, 0.136] |
| random_letters | 500 | 0 | 0.042 [0.041, 0.042] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.982 | [0.972, 0.990] | yes | **PASS** |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.976 | [0.966, 0.984] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.982 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.976 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SCRIBE v3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.908 [0.897, 0.920] |
| sent_shuffled | 500 | 0 | 0.551 [0.544, 0.557] |
| word_shuffled | 500 | 0 | 0.034 [0.034, 0.035] |
| random_lexicon | 500 | 0 | 0.018 [0.018, 0.018] |
| random_letters | 500 | 0 | 0.002 [0.001, 0.002] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.966 | [0.951, 0.979] | yes | **PASS** |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.991 | [0.984, 0.997] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.966 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.991 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SURPRISE (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.920 [0.909, 0.932] |
| sent_shuffled | 500 | 0 | 0.665 [0.661, 0.671] |
| word_shuffled | 500 | 0 | 0.228 [0.225, 0.230] |
| random_lexicon | 500 | 0 | 0.159 [0.157, 0.161] |
| random_letters | 500 | 0 | 0.067 [0.066, 0.067] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.951 | [0.934, 0.967] | yes | **PASS** |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.955 | [0.941, 0.967] | yes | **PASS** |
| random_lexicon > random_letters | 0.998 | [0.994, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.951 | 0.998 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.955 | 0.998 | 1.000 |
| random_lexicon |  |  |  | 0.998 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### MATCH (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.908 [0.896, 0.920] |
| sent_shuffled | 500 | 0 | 0.661 [0.656, 0.667] |
| word_shuffled | 500 | 0 | 0.335 [0.330, 0.340] |
| random_lexicon | 500 | 0 | 0.246 [0.245, 0.248] |
| random_letters | 500 | 0 | 0.058 [0.058, 0.058] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.941 | [0.923, 0.958] | yes | FAIL |
| sent_shuffled > word_shuffled | 0.989 | [0.980, 0.997] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.950 | [0.936, 0.963] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.941 | 0.990 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 0.989 | 0.994 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.950 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SURPRISE 3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.926 [0.913, 0.938] |
| sent_shuffled | 500 | 0 | 0.733 [0.727, 0.739] |
| word_shuffled | 500 | 0 | 0.036 [0.036, 0.037] |
| random_lexicon | 500 | 0 | 0.029 [0.029, 0.029] |
| random_letters | 500 | 0 | 0.013 [0.013, 0.014] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.934 | [0.914, 0.954] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.949 | [0.934, 0.963] | yes | FAIL |
| random_lexicon > random_letters | 0.996 | [0.990, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.934 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.949 | 0.992 | 1.000 |
| random_lexicon |  |  |  | 0.996 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### MATCH 3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.909 [0.897, 0.921] |
| sent_shuffled | 500 | 0 | 0.675 [0.669, 0.680] |
| word_shuffled | 500 | 0 | 0.336 [0.329, 0.343] |
| random_lexicon | 500 | 0 | 0.261 [0.258, 0.264] |
| random_letters | 500 | 0 | 0.011 [0.011, 0.011] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.935 | [0.916, 0.953] | yes | FAIL |
| sent_shuffled > word_shuffled | 0.987 | [0.978, 0.996] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.846 | [0.816, 0.875] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.935 | 0.989 | 0.998 | 1.000 | 1.000 |
| sent_shuffled |  | 0.987 | 0.994 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.846 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### min of 3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.845 [0.828, 0.861] |
| sent_shuffled | 500 | 0 | 0.349 [0.339, 0.359] |
| word_shuffled | 500 | 0 | 0.003 [0.003, 0.004] |
| random_lexicon | 500 | 0 | 0.001 [0.001, 0.001] |
| random_letters | 500 | 0 | 0.000 [0.000, 0.000] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.967 | [0.954, 0.979] | yes | **PASS** |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.998 | [0.995, 0.999] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.998 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | human > sent_shuffled | sent_shuffled > word_shuffled | word_shuffled > random_lexicon | random_lexicon > random_letters | random_letters > repeated_phrase | passed |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.974 **PASS** | 1.000 **PASS** | 0.998 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 5 of 5 |
| SCRIBE v3 | 0.982 **PASS** | 1.000 **PASS** | 0.976 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 5 of 5 |
| SCRIBE v3.1 | 0.966 **PASS** | 1.000 **PASS** | 0.991 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 5 of 5 |
| SURPRISE | 0.951 **PASS** | 1.000 **PASS** | 0.955 **PASS** | 0.998 **PASS** | 1.000 **PASS** | 5 of 5 |
| MATCH | 0.941 FAIL | 0.989 **PASS** | 0.950 FAIL | 1.000 **PASS** | 1.000 **PASS** | 3 of 5 |
| SURPRISE 3.1 | 0.934 FAIL | 1.000 **PASS** | 0.949 FAIL | 0.996 **PASS** | 1.000 **PASS** | 3 of 5 |
| MATCH 3.1 | 0.935 FAIL | 0.987 **PASS** | 0.846 FAIL | 1.000 **PASS** | 1.000 **PASS** | 3 of 5 |
| min of 3 | 0.967 **PASS** | 1.000 **PASS** | 0.998 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 5 of 5 |
