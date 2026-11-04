# Agreement with human ratings

1000 texts, ratings under `meta.human`, criteria: overall. Item level: Pearson, Spearman and Kendall tau-b between the oriented metric and the mean rating; percentile intervals from 1000 resamples of the items (seed 20261028). Δ Kendall is the item-level Kendall minus that of the reference metric (SCRIBE v2) on the same items; its p-value is a two-sided paired bootstrap test (centred: p = (1 + #|d* − d| >= |d|) / (B + 1)), Benjamini-Hochberg adjusted across metrics within each criterion. System level: Kendall and Spearman between per-system mean ratings and per-system mean metric scores, intervals from resampling items within systems. Human ceiling: one rating of each item, drawn at random, against the mean of its other ratings (leave one out), averaged over 20 draws; it compares one rater with k − 1, so it is a conservative reference for a metric compared with the mean of all k.

## All stories (1000 texts)

### overall, all stories

Human ceiling (leave one out, 1000 items with two or more ratings): Pearson 0.797, Spearman 0.803, Kendall 0.709.

| metric | n | Pearson | Spearman | Kendall | Δ Kendall vs SCRIBE v2 | p | p (BH) |
|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 1000 | 0.050 [−0.006, 0.107] | 0.061 [0.001, 0.120] | 0.044 [0.001, 0.088] | 0.000 | n/a | n/a |
| SCRIBE v3 | 1000 | 0.089 [0.027, 0.152] | 0.105 [0.043, 0.164] | 0.074 [0.031, 0.117] | 0.030 | 0.106 | 0.247 |
| SCRIBE v3.1 | 1000 | 0.030 [−0.034, 0.095] | 0.075 [0.013, 0.132] | 0.053 [0.009, 0.092] | 0.008 | 0.660 | 0.770 |
| SURPRISE | 1000 | 0.046 [−0.024, 0.117] | 0.066 [0.003, 0.126] | 0.046 [0.002, 0.089] | 0.001 | 0.963 | 0.963 |
| MATCH | 1000 | 0.115 [0.052, 0.180] | 0.142 [0.082, 0.202] | 0.100 [0.059, 0.142] | 0.056 | 0.023 | 0.080 |
| SURPRISE 3.1 | 1000 | −0.045 [−0.107, 0.020] | 0.043 [−0.018, 0.101] | 0.029 [−0.015, 0.070] | −0.015 | 0.547 | 0.766 |
| MATCH 3.1 | 1000 | 0.117 [0.054, 0.180] | 0.146 [0.087, 0.207] | 0.103 [0.061, 0.145] | 0.059 | 0.017 | 0.080 |
| min of 3 | 1000 | 0.066 [0.000, 0.132] | 0.087 [0.023, 0.148] | 0.061 [0.016, 0.103] | 0.017 | 0.486 | 0.766 |

| metric | systems | system Kendall | system Spearman |
|---|---|---|---|
| SCRIBE v2 | 5 | 0.200 [0.200, 0.600] | 0.500 [0.300, 0.800] |
| SCRIBE v3 | 5 | 0.600 [0.200, 1.000] | 0.700 [0.500, 1.000] |
| SCRIBE v3.1 | 5 | 0.400 [0.200, 0.800] | 0.600 [0.500, 0.900] |
| SURPRISE | 5 | 0.800 [0.400, 1.000] | 0.900 [0.600, 1.000] |
| MATCH | 5 | 0.600 [0.200, 0.800] | 0.700 [0.500, 0.900] |
| SURPRISE 3.1 | 5 | 0.600 [−0.400, 1.000] | 0.700 [−0.500, 1.000] |
| MATCH 3.1 | 5 | 0.600 [0.200, 0.800] | 0.700 [0.500, 0.900] |
| min of 3 | 5 | 0.600 [0.200, 0.800] | 0.700 [0.500, 0.900] |

### Summary, all stories

Item-level Kendall tau-b:

| metric | overall |
|---|---|
| SCRIBE v2 | 0.044 |
| SCRIBE v3 | 0.074 |
| SCRIBE v3.1 | 0.053 |
| SURPRISE | 0.046 |
| MATCH | 0.100 |
| SURPRISE 3.1 | 0.029 |
| MATCH 3.1 | 0.103 |
| min of 3 | 0.061 |
| human ceiling (LOO) | 0.709 |

System-level Kendall tau-b:

| metric | overall |
|---|---|
| SCRIBE v2 | 0.200 |
| SCRIBE v3 | 0.600 |
| SCRIBE v3.1 | 0.400 |
| SURPRISE | 0.800 |
| MATCH | 0.600 |
| SURPRISE 3.1 | 0.600 |
| MATCH 3.1 | 0.600 |
| min of 3 | 0.600 |
