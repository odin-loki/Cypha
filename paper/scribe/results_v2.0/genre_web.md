# Control ladder

3000 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.918 [0.903, 0.932] |
| sent_shuffled | 500 | 0 | 0.432 [0.418, 0.446] |
| word_shuffled | 500 | 0 | 0.004 [0.004, 0.005] |
| random_lexicon | 500 | 0 | 0.001 [0.001, 0.002] |
| random_letters | 500 | 0 | 0.000 [0.000, 0.000] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.959 | [0.945, 0.972] | yes | **PASS** |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.987 | [0.981, 0.992] | yes | **PASS** |
| random_lexicon > random_letters | 0.998 | [0.995, 0.999] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.959 | 0.998 | 0.998 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.987 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 0.998 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SCRIBE v3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.912 [0.901, 0.922] |
| sent_shuffled | 500 | 0 | 0.600 [0.594, 0.606] |
| word_shuffled | 500 | 0 | 0.198 [0.195, 0.201] |
| random_lexicon | 500 | 0 | 0.145 [0.143, 0.146] |
| random_letters | 500 | 0 | 0.043 [0.042, 0.043] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.965 | [0.951, 0.978] | yes | **PASS** |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.971 | [0.959, 0.980] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.965 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.971 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SCRIBE v3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.903 [0.890, 0.917] |
| sent_shuffled | 500 | 0 | 0.599 [0.592, 0.607] |
| word_shuffled | 500 | 0 | 0.038 [0.037, 0.038] |
| random_lexicon | 500 | 0 | 0.022 [0.022, 0.022] |
| random_letters | 500 | 0 | 0.002 [0.002, 0.002] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.948 | [0.930, 0.966] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [0.999, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.979 | [0.969, 0.987] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 4 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.948 | 0.996 | 0.996 | 0.999 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.979 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SURPRISE (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.914 [0.904, 0.924] |
| sent_shuffled | 500 | 0 | 0.686 [0.681, 0.690] |
| word_shuffled | 500 | 0 | 0.238 [0.235, 0.240] |
| random_lexicon | 500 | 0 | 0.176 [0.174, 0.178] |
| random_letters | 500 | 0 | 0.070 [0.069, 0.071] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.951 | [0.934, 0.965] | yes | **PASS** |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.956 | [0.942, 0.969] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 5 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.951 | 0.998 | 0.999 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.956 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### MATCH (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.903 [0.888, 0.918] |
| sent_shuffled | 500 | 0 | 0.682 [0.675, 0.689] |
| word_shuffled | 500 | 0 | 0.353 [0.344, 0.362] |
| random_lexicon | 500 | 0 | 0.256 [0.254, 0.259] |
| random_letters | 500 | 0 | 0.058 [0.058, 0.059] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.934 | [0.916, 0.954] | yes | FAIL |
| sent_shuffled > word_shuffled | 0.961 | [0.944, 0.977] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.936 | [0.918, 0.950] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.934 | 0.967 | 0.978 | 0.988 | 0.994 |
| sent_shuffled |  | 0.961 | 0.993 | 0.994 | 0.998 |
| word_shuffled |  |  | 0.936 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SURPRISE 3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.925 [0.914, 0.936] |
| sent_shuffled | 500 | 0 | 0.755 [0.750, 0.758] |
| word_shuffled | 500 | 0 | 0.038 [0.037, 0.038] |
| random_lexicon | 500 | 0 | 0.030 [0.030, 0.031] |
| random_letters | 500 | 0 | 0.015 [0.014, 0.015] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.934 | [0.914, 0.951] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.961 | [0.949, 0.972] | yes | **PASS** |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 4 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.934 | 0.998 | 0.999 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.961 | 0.996 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### MATCH 3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.905 [0.889, 0.920] |
| sent_shuffled | 500 | 0 | 0.697 [0.690, 0.704] |
| word_shuffled | 500 | 0 | 0.343 [0.333, 0.355] |
| random_lexicon | 500 | 0 | 0.257 [0.253, 0.262] |
| random_letters | 500 | 0 | 0.011 [0.011, 0.011] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.933 | [0.915, 0.952] | yes | FAIL |
| sent_shuffled > word_shuffled | 0.964 | [0.948, 0.979] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.808 | [0.776, 0.836] | yes | FAIL |
| random_lexicon > random_letters | 0.998 | [0.994, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.933 | 0.964 | 0.975 | 0.992 | 0.994 |
| sent_shuffled |  | 0.964 | 0.990 | 0.997 | 0.999 |
| word_shuffled |  |  | 0.808 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 0.998 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### min of 3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.841 [0.824, 0.858] |
| sent_shuffled | 500 | 0 | 0.418 [0.406, 0.431] |
| word_shuffled | 500 | 0 | 0.004 [0.004, 0.005] |
| random_lexicon | 500 | 0 | 0.001 [0.001, 0.002] |
| random_letters | 500 | 0 | 0.000 [0.000, 0.000] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.946 | [0.929, 0.963] | yes | FAIL |
| sent_shuffled > word_shuffled | 0.998 | [0.994, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.987 | [0.981, 0.992] | yes | **PASS** |
| random_lexicon > random_letters | 0.998 | [0.995, 0.999] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 4 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.946 | 0.990 | 0.991 | 0.996 | 1.000 |
| sent_shuffled |  | 0.998 | 0.998 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.987 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 0.998 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | human > sent_shuffled | sent_shuffled > word_shuffled | word_shuffled > random_lexicon | random_lexicon > random_letters | random_letters > repeated_phrase | passed |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.959 **PASS** | 1.000 **PASS** | 0.987 **PASS** | 0.998 **PASS** | 1.000 **PASS** | 5 of 5 |
| SCRIBE v3 | 0.965 **PASS** | 1.000 **PASS** | 0.971 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 5 of 5 |
| SCRIBE v3.1 | 0.948 FAIL | 1.000 **PASS** | 0.979 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 4 of 5 |
| SURPRISE | 0.951 **PASS** | 1.000 **PASS** | 0.956 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 5 of 5 |
| MATCH | 0.934 FAIL | 0.961 **PASS** | 0.936 FAIL | 1.000 **PASS** | 1.000 **PASS** | 3 of 5 |
| SURPRISE 3.1 | 0.934 FAIL | 1.000 **PASS** | 0.961 **PASS** | 1.000 **PASS** | 1.000 **PASS** | 4 of 5 |
| MATCH 3.1 | 0.933 FAIL | 0.964 **PASS** | 0.808 FAIL | 0.998 **PASS** | 1.000 **PASS** | 3 of 5 |
| min of 3 | 0.946 FAIL | 0.998 **PASS** | 0.987 **PASS** | 0.998 **PASS** | 1.000 **PASS** | 4 of 5 |
