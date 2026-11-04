# Control ladder

3000 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.839 [0.818, 0.860] |
| sent_shuffled | 500 | 0 | 0.431 [0.414, 0.450] |
| word_shuffled | 500 | 0 | 0.004 [0.004, 0.004] |
| random_lexicon | 500 | 0 | 0.001 [0.001, 0.001] |
| random_letters | 500 | 0 | 0.000 [0.000, 0.000] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.890 | [0.867, 0.913] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.998 | [0.997, 0.999] | yes | **PASS** |
| random_lexicon > random_letters | 0.999 | [0.998, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 4 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.890 | 0.994 | 0.994 | 0.998 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.998 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 0.999 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SCRIBE v3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.855 [0.841, 0.870] |
| sent_shuffled | 500 | 0 | 0.582 [0.573, 0.590] |
| word_shuffled | 500 | 0 | 0.184 [0.182, 0.185] |
| random_lexicon | 500 | 0 | 0.137 [0.136, 0.138] |
| random_letters | 500 | 0 | 0.042 [0.042, 0.042] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.909 | [0.887, 0.930] | yes | FAIL |
| sent_shuffled > word_shuffled | 0.997 | [0.993, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.961 | [0.948, 0.972] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 4 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.909 | 0.998 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 0.997 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.961 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SCRIBE v3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.836 [0.817, 0.854] |
| sent_shuffled | 500 | 0 | 0.573 [0.562, 0.584] |
| word_shuffled | 500 | 0 | 0.035 [0.034, 0.035] |
| random_lexicon | 500 | 0 | 0.018 [0.018, 0.019] |
| random_letters | 500 | 0 | 0.002 [0.002, 0.002] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.880 | [0.853, 0.903] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.987 | [0.977, 0.994] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 4 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.880 | 0.994 | 0.998 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.987 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SURPRISE (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.892 [0.878, 0.905] |
| sent_shuffled | 500 | 0 | 0.677 [0.670, 0.684] |
| word_shuffled | 500 | 0 | 0.226 [0.224, 0.228] |
| random_lexicon | 500 | 0 | 0.158 [0.156, 0.160] |
| random_letters | 500 | 0 | 0.067 [0.066, 0.068] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.897 | [0.874, 0.921] | yes | FAIL |
| sent_shuffled > word_shuffled | 0.998 | [0.995, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.963 | [0.950, 0.974] | yes | **PASS** |
| random_lexicon > random_letters | 0.998 | [0.994, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 4 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.897 | 0.998 | 0.999 | 1.000 | 1.000 |
| sent_shuffled |  | 0.998 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.963 | 0.997 | 1.000 |
| random_lexicon |  |  |  | 0.998 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### MATCH (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.835 [0.816, 0.853] |
| sent_shuffled | 500 | 0 | 0.636 [0.626, 0.647] |
| word_shuffled | 500 | 0 | 0.321 [0.317, 0.325] |
| random_lexicon | 500 | 0 | 0.252 [0.250, 0.254] |
| random_letters | 500 | 0 | 0.059 [0.059, 0.060] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.841 | [0.812, 0.868] | yes | FAIL |
| sent_shuffled > word_shuffled | 0.938 | [0.918, 0.956] | yes | FAIL |
| word_shuffled > random_lexicon | 0.910 | [0.892, 0.928] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 2 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.841 | 0.957 | 0.986 | 1.000 | 1.000 |
| sent_shuffled |  | 0.938 | 0.971 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.910 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SURPRISE 3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.881 [0.864, 0.898] |
| sent_shuffled | 500 | 0 | 0.727 [0.716, 0.738] |
| word_shuffled | 500 | 0 | 0.036 [0.036, 0.036] |
| random_lexicon | 500 | 0 | 0.029 [0.029, 0.029] |
| random_letters | 500 | 0 | 0.014 [0.014, 0.014] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.862 | [0.834, 0.888] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.959 | [0.946, 0.971] | yes | **PASS** |
| random_lexicon > random_letters | 0.998 | [0.994, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 4 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.862 | 0.998 | 0.998 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.959 | 0.996 | 1.000 |
| random_lexicon |  |  |  | 0.998 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### MATCH 3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.838 [0.821, 0.857] |
| sent_shuffled | 500 | 0 | 0.653 [0.643, 0.663] |
| word_shuffled | 500 | 0 | 0.329 [0.323, 0.335] |
| random_lexicon | 500 | 0 | 0.266 [0.262, 0.270] |
| random_letters | 500 | 0 | 0.011 [0.011, 0.011] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.839 | [0.810, 0.866] | yes | FAIL |
| sent_shuffled > word_shuffled | 0.943 | [0.925, 0.960] | yes | FAIL |
| word_shuffled > random_lexicon | 0.848 | [0.821, 0.871] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 2 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.839 | 0.958 | 0.983 | 1.000 | 1.000 |
| sent_shuffled |  | 0.943 | 0.973 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.848 | 0.998 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### min of 3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.745 [0.723, 0.767] |
| sent_shuffled | 500 | 0 | 0.406 [0.392, 0.421] |
| word_shuffled | 500 | 0 | 0.004 [0.004, 0.004] |
| random_lexicon | 500 | 0 | 0.001 [0.001, 0.001] |
| random_letters | 500 | 0 | 0.000 [0.000, 0.000] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.856 | [0.830, 0.881] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.998 | [0.997, 0.999] | yes | **PASS** |
| random_lexicon > random_letters | 0.999 | [0.998, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 4 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.856 | 0.994 | 0.994 | 0.998 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.998 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 0.999 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | human > sent_shuffled | sent_shuffled > word_shuffled | word_shuffled > random_lexicon | random_lexicon > random_letters | random_letters > repeated_phrase | passed |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.890 FAIL | 1.000 **PASS** | 0.998 **PASS** | 0.999 **PASS** | 1.000 **PASS** | 4 of 5 |
| SCRIBE v3 | 0.909 FAIL | 0.997 **PASS** | 0.961 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 4 of 5 |
| SCRIBE v3.1 | 0.880 FAIL | 1.000 **PASS** | 0.987 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 4 of 5 |
| SURPRISE | 0.897 FAIL | 0.998 **PASS** | 0.963 **PASS** | 0.998 **PASS** | 1.000 **PASS** | 4 of 5 |
| MATCH | 0.841 FAIL | 0.938 FAIL | 0.910 FAIL | 1.000 **PASS** | 1.000 **PASS** | 2 of 5 |
| SURPRISE 3.1 | 0.862 FAIL | 1.000 **PASS** | 0.959 **PASS** | 0.998 **PASS** | 1.000 **PASS** | 4 of 5 |
| MATCH 3.1 | 0.839 FAIL | 0.943 FAIL | 0.848 FAIL | 1.000 **PASS** | 1.000 **PASS** | 2 of 5 |
| min of 3 | 0.856 FAIL | 1.000 **PASS** | 0.998 **PASS** | 0.999 **PASS** | 1.000 **PASS** | 4 of 5 |
