# Control ladder

3000 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.910 [0.896, 0.923] |
| sent_shuffled | 500 | 0 | 0.756 [0.739, 0.771] |
| word_shuffled | 500 | 0 | 0.003 [0.003, 0.003] |
| random_lexicon | 500 | 0 | 0.002 [0.002, 0.002] |
| random_letters | 500 | 0 | 0.000 [0.000, 0.000] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.782 | [0.752, 0.811] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.882 | [0.860, 0.901] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.782 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.882 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SCRIBE v3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.902 [0.894, 0.910] |
| sent_shuffled | 500 | 0 | 0.776 [0.768, 0.784] |
| word_shuffled | 500 | 0 | 0.183 [0.181, 0.184] |
| random_lexicon | 500 | 0 | 0.162 [0.161, 0.164] |
| random_letters | 500 | 0 | 0.019 [0.019, 0.020] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.837 | [0.812, 0.864] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.797 | [0.769, 0.822] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.837 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.797 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SCRIBE v3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.903 [0.894, 0.912] |
| sent_shuffled | 500 | 0 | 0.788 [0.779, 0.796] |
| word_shuffled | 500 | 0 | 0.032 [0.031, 0.032] |
| random_lexicon | 500 | 0 | 0.027 [0.027, 0.027] |
| random_letters | 500 | 0 | 0.001 [0.001, 0.001] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.821 | [0.794, 0.850] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.803 | [0.776, 0.830] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.821 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.803 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SURPRISE (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.898 [0.887, 0.908] |
| sent_shuffled | 500 | 0 | 0.791 [0.783, 0.800] |
| word_shuffled | 500 | 0 | 0.218 [0.215, 0.220] |
| random_lexicon | 500 | 0 | 0.199 [0.196, 0.201] |
| random_letters | 500 | 0 | 0.009 [0.009, 0.010] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.772 | [0.740, 0.802] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.775 | [0.743, 0.806] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.772 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.775 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### MATCH (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.899 [0.888, 0.909] |
| sent_shuffled | 500 | 0 | 0.781 [0.772, 0.789] |
| word_shuffled | 500 | 0 | 0.328 [0.325, 0.332] |
| random_lexicon | 500 | 0 | 0.286 [0.283, 0.289] |
| random_letters | 500 | 0 | 0.049 [0.049, 0.049] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.788 | [0.759, 0.817] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.771 | [0.744, 0.798] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.788 | 0.998 | 0.998 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.771 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### SURPRISE 3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.916 [0.905, 0.926] |
| sent_shuffled | 500 | 0 | 0.839 [0.832, 0.846] |
| word_shuffled | 500 | 0 | 0.035 [0.034, 0.035] |
| random_lexicon | 500 | 0 | 0.033 [0.032, 0.033] |
| random_letters | 500 | 0 | 0.015 [0.015, 0.015] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.766 | [0.734, 0.797] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.773 | [0.741, 0.803] | yes | FAIL |
| random_lexicon > random_letters | 0.999 | [0.997, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.766 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.773 | 0.986 | 1.000 |
| random_lexicon |  |  |  | 0.999 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### MATCH 3.1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.903 [0.892, 0.913] |
| sent_shuffled | 500 | 0 | 0.792 [0.784, 0.800] |
| word_shuffled | 500 | 0 | 0.315 [0.308, 0.322] |
| random_lexicon | 500 | 0 | 0.289 [0.284, 0.294] |
| random_letters | 500 | 0 | 0.009 [0.009, 0.009] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.786 | [0.757, 0.814] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.648 | [0.615, 0.683] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.786 | 0.998 | 0.998 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.648 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

### min of 3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 500 | 0 | 0.818 [0.804, 0.832] |
| sent_shuffled | 500 | 0 | 0.681 [0.668, 0.692] |
| word_shuffled | 500 | 0 | 0.003 [0.003, 0.003] |
| random_lexicon | 500 | 0 | 0.002 [0.002, 0.002] |
| random_letters | 500 | 0 | 0.000 [0.000, 0.000] |
| repeated_phrase | 500 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| human > sent_shuffled | 0.780 | [0.750, 0.809] | yes | FAIL |
| sent_shuffled > word_shuffled | 1.000 | [1.000, 1.000] | yes | **PASS** |
| word_shuffled > random_lexicon | 0.882 | [0.860, 0.901] | yes | FAIL |
| random_lexicon > random_letters | 1.000 | [1.000, 1.000] | yes | **PASS** |
| random_letters > repeated_phrase | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 5 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | sent_shuffled | word_shuffled | random_lexicon | random_letters | repeated_phrase |
|---|---|---|---|---|---|
| human | 0.780 | 1.000 | 1.000 | 1.000 | 1.000 |
| sent_shuffled |  | 1.000 | 1.000 | 1.000 | 1.000 |
| word_shuffled |  |  | 0.882 | 1.000 | 1.000 |
| random_lexicon |  |  |  | 1.000 | 1.000 |
| random_letters |  |  |  |  | 1.000 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | human > sent_shuffled | sent_shuffled > word_shuffled | word_shuffled > random_lexicon | random_lexicon > random_letters | random_letters > repeated_phrase | passed |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.782 FAIL | 1.000 **PASS** | 0.882 FAIL | 1.000 **PASS** | 1.000 **PASS** | 3 of 5 |
| SCRIBE v3 | 0.837 FAIL | 1.000 **PASS** | 0.797 FAIL | 1.000 **PASS** | 1.000 **PASS** | 3 of 5 |
| SCRIBE v3.1 | 0.821 FAIL | 1.000 **PASS** | 0.803 FAIL | 1.000 **PASS** | 1.000 **PASS** | 3 of 5 |
| SURPRISE | 0.772 FAIL | 1.000 **PASS** | 0.775 FAIL | 1.000 **PASS** | 1.000 **PASS** | 3 of 5 |
| MATCH | 0.788 FAIL | 1.000 **PASS** | 0.771 FAIL | 1.000 **PASS** | 1.000 **PASS** | 3 of 5 |
| SURPRISE 3.1 | 0.766 FAIL | 1.000 **PASS** | 0.773 FAIL | 0.999 **PASS** | 1.000 **PASS** | 3 of 5 |
| MATCH 3.1 | 0.786 FAIL | 1.000 **PASS** | 0.648 FAIL | 1.000 **PASS** | 1.000 **PASS** | 3 of 5 |
| min of 3 | 0.780 FAIL | 1.000 **PASS** | 0.882 FAIL | 1.000 **PASS** | 1.000 **PASS** | 3 of 5 |
