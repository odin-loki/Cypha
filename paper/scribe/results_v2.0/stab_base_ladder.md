# Control ladder

3000 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.916 [0.901, 0.930] |
| sent_shuffled | 500 | 0 | 0.354 [0.342, 0.366] |
| word_shuffled | 500 | 0 | 0.003 [0.003, 0.003] |
| random_lexicon | 500 | 0 | 0.001 [0.001, 0.001] |
| random_letters | 500 | 0 | 0.000 [0.000, 0.000] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.975 | [0.963, 0.985] | yes | **PASS** |
| sent_shuffled > word_shuffled | 0.999 | [0.996, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.998 | [0.996, 1.000] | yes | **PASS** |
| random_lexicon > random_letters | 0.998 | [0.994, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 0.999 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.998 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 0.998 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SCRIBE v3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.906 [0.895, 0.917] |
| sent_shuffled | 500 | 0 | 0.560 [0.554, 0.566] |
| word_shuffled | 500 | 0 | 0.189 [0.187, 0.191] |
| random_lexicon | 500 | 0 | 0.136 [0.135, 0.137] |
| random_letters | 500 | 0 | 0.041 [0.041, 0.042] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.981 | [0.972, 0.989] | yes | **PASS** |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.975 | [0.965, 0.984] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.981 | 0.996 | 0.996 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.975 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SCRIBE v3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.902 [0.888, 0.913] |
| sent_shuffled | 500 | 0 | 0.548 [0.540, 0.556] |
| word_shuffled | 500 | 0 | 0.034 [0.034, 0.035] |
| random_lexicon | 500 | 0 | 0.018 [0.018, 0.019] |
| random_letters | 500 | 0 | 0.001 [0.001, 0.002] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.965 | [0.950, 0.978] | yes | **PASS** |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.996 | [0.994, 0.998] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.965 | 0.992 | 0.992 | 0.995 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.996 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SURPRISE (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.903 [0.890, 0.914] |
| sent_shuffled | 500 | 0 | 0.667 [0.663, 0.671] |
| word_shuffled | 500 | 0 | 0.225 [0.223, 0.228] |
| random_lexicon | 500 | 0 | 0.158 [0.156, 0.160] |
| random_letters | 500 | 0 | 0.066 [0.065, 0.067] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.950 | [0.935, 0.965] | yes | **PASS** |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.961 | [0.949, 0.972] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.950 | 0.993 | 0.994 | 0.996 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.961 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### MATCH (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.900 [0.887, 0.914] |
| sent_shuffled | 500 | 0 | 0.659 [0.650, 0.667] |
| word_shuffled | 500 | 0 | 0.338 [0.333, 0.343] |
| random_lexicon | 500 | 0 | 0.249 [0.247, 0.251] |
| random_letters | 500 | 0 | 0.058 [0.058, 0.059] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.940 | [0.922, 0.957] | yes | FAIL |
| sent_shuffled > word_shuffled | 0.979 | [0.967, 0.992] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.950 | [0.936, 0.963] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 4 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.940 | 0.985 | 0.990 | 0.990 | 0.992 |
| sent_shuffled |  | 0.979 | 0.984 | 0.990 | 1.000 |
| word_shuffled |  |  | 0.950 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SURPRISE 3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.914 [0.900, 0.926] |
| sent_shuffled | 500 | 0 | 0.738 [0.733, 0.742] |
| word_shuffled | 500 | 0 | 0.036 [0.036, 0.037] |
| random_lexicon | 500 | 0 | 0.029 [0.029, 0.029] |
| random_letters | 500 | 0 | 0.013 [0.013, 0.014] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.935 | [0.914, 0.953] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.962 | [0.949, 0.973] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 4 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.935 | 0.996 | 0.996 | 0.997 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.962 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### MATCH 3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.905 [0.892, 0.918] |
| sent_shuffled | 500 | 0 | 0.674 [0.666, 0.682] |
| word_shuffled | 500 | 0 | 0.341 [0.334, 0.347] |
| random_lexicon | 500 | 0 | 0.259 [0.255, 0.262] |
| random_letters | 500 | 0 | 0.011 [0.011, 0.011] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.940 | [0.922, 0.957] | yes | FAIL |
| sent_shuffled > word_shuffled | 0.978 | [0.966, 0.990] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.865 | [0.840, 0.889] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.940 | 0.986 | 0.990 | 0.990 | 0.992 |
| sent_shuffled |  | 0.978 | 0.985 | 0.990 | 1.000 |
| word_shuffled |  |  | 0.865 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### min of 3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.833 [0.816, 0.848] |
| sent_shuffled | 500 | 0 | 0.345 [0.334, 0.356] |
| word_shuffled | 500 | 0 | 0.003 [0.003, 0.003] |
| random_lexicon | 500 | 0 | 0.001 [0.001, 0.001] |
| random_letters | 500 | 0 | 0.000 [0.000, 0.000] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.964 | [0.951, 0.976] | yes | **PASS** |
| sent_shuffled > word_shuffled | 0.992 | [0.984, 0.998] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.998 | [0.996, 1.000] | yes | **PASS** |
| random_lexicon > random_letters | 0.998 | [0.994, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.964 | 0.990 | 0.990 | 0.992 | 1.000 |
| sent_shuffled |  | 0.992 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.998 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 0.998 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | human > sent_shuffled | sent_shuffled > word_shuffled | word_shuffled > random_lexicon | random_lexicon > random_letters | random_letters > repeated_phrase | passed |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.975 **PASS** | 0.999 **PASS** | 0.998 **PASS** | 0.998 **PASS** | 1.000 **PASS** | 5 of 5 |
| SCRIBE v3 | 0.981 **PASS** | 1.000 **PASS** | 0.975 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 5 of 5 |
| SCRIBE v3.1 | 0.965 **PASS** | 1.000 **PASS** | 0.996 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 5 of 5 |
| SURPRISE | 0.950 **PASS** | 1.000 **PASS** | 0.961 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 5 of 5 |
| MATCH | 0.940 FAIL | 0.979 **PASS** | 0.950 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 4 of 5 |
| SURPRISE 3.1 | 0.935 FAIL | 1.000 **PASS** | 0.962 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 4 of 5 |
| MATCH 3.1 | 0.940 FAIL | 0.978 **PASS** | 0.865 FAIL | 1.000 **PASS** | 1.000 **PASS** | 3 of 5 |
| min of 3 | 0.964 **PASS** | 0.992 **PASS** | 0.998 **PASS** | 0.998 **PASS** | 1.000 **PASS** | 5 of 5 |
