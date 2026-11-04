# Agreement with human ratings

1000 texts, ratings under `meta.human`, criteria: overall. Item level: Pearson, Spearman and Kendall tau-b between the oriented metric and the mean rating; percentile intervals from 1000 resamples of the items (seed 20261028). Δ Kendall is the item-level Kendall minus that of the reference metric (SCRIBE v2) on the same items; its p-value is a two-sided paired bootstrap test (centred: p = (1 + #|d* − d| >= |d|) / (B + 1)), Benjamini-Hochberg adjusted across metrics within each criterion. System level: Kendall and Spearman between per-system mean ratings and per-system mean metric scores, intervals from resampling items within systems. Human ceiling: one rating of each item, drawn at random, against the mean of its other ratings (leave one out), averaged over 20 draws; it compares one rater with k − 1, so it is a conservative reference for a metric compared with the mean of all k.

## All stories (1000 texts)

### overall, all stories

Human ceiling (leave one out, 1000 items with two or more ratings): Pearson 0.797, Spearman 0.803, Kendall 0.709.

| metric | n | Pearson | Spearman | Kendall | Δ Kendall vs SCRIBE v2 | p | p (BH) |
|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 1000 | 0.050 [−0.006, 0.107] | 0.061 [0.001, 0.120] | 0.044 [0.001, 0.088] | 0.000 | n/a | n/a |
| SCRIBE v3 | 1000 | 0.112 [0.048, 0.178] | 0.122 [0.062, 0.184] | 0.086 [0.044, 0.130] | 0.041 | 0.034 | 0.056 |
| SCRIBE v3.1 | 1000 | 0.113 [0.043, 0.178] | 0.127 [0.063, 0.189] | 0.089 [0.044, 0.132] | 0.044 | 0.040 | 0.056 |
| SURPRISE | 1000 | 0.046 [−0.024, 0.117] | 0.066 [0.003, 0.126] | 0.046 [0.002, 0.089] | 0.001 | 0.963 | 0.963 |
| MATCH | 1000 | 0.157 [0.092, 0.220] | 0.167 [0.105, 0.226] | 0.117 [0.074, 0.159] | 0.072 | 0.008 | 0.028 |
| SURPRISE 3.1 | 1000 | 0.029 [−0.041, 0.097] | 0.087 [0.023, 0.151] | 0.060 [0.015, 0.105] | 0.015 | 0.532 | 0.621 |
| MATCH 3.1 | 1000 | 0.170 [0.105, 0.231] | 0.167 [0.100, 0.225] | 0.116 [0.070, 0.157] | 0.072 | 0.005 | 0.028 |
| min of 3 | 1000 | 0.133 [0.067, 0.198] | 0.144 [0.082, 0.205] | 0.101 [0.057, 0.144] | 0.057 | 0.024 | 0.056 |

| metric | systems | system Kendall | system Spearman |
|---|---|---|---|
| SCRIBE v2 | 5 | 0.200 [0.200, 0.600] | 0.500 [0.300, 0.800] |
| SCRIBE v3 | 5 | 0.600 [0.200, 1.000] | 0.800 [0.500, 1.000] |
| SCRIBE v3.1 | 5 | 0.800 [0.400, 1.000] | 0.900 [0.600, 1.000] |
| SURPRISE | 5 | 0.800 [0.400, 1.000] | 0.900 [0.600, 1.000] |
| MATCH | 5 | 0.800 [0.400, 1.000] | 0.900 [0.600, 1.000] |
| SURPRISE 3.1 | 5 | 1.000 [0.000, 1.000] | 1.000 [−0.100, 1.000] |
| MATCH 3.1 | 5 | 0.800 [0.600, 1.000] | 0.900 [0.800, 1.000] |
| min of 3 | 5 | 0.600 [0.200, 1.000] | 0.800 [0.500, 1.000] |

### Summary, all stories

Item-level Kendall tau-b:

| metric | overall |
|---|---|
| SCRIBE v2 | 0.044 |
| SCRIBE v3 | 0.086 |
| SCRIBE v3.1 | 0.089 |
| SURPRISE | 0.046 |
| MATCH | 0.117 |
| SURPRISE 3.1 | 0.060 |
| MATCH 3.1 | 0.116 |
| min of 3 | 0.101 |
| human ceiling (LOO) | 0.709 |

System-level Kendall tau-b:

| metric | overall |
|---|---|
| SCRIBE v2 | 0.200 |
| SCRIBE v3 | 0.600 |
| SCRIBE v3.1 | 0.800 |
| SURPRISE | 0.800 |
| MATCH | 0.800 |
| SURPRISE 3.1 | 1.000 |
| MATCH 3.1 | 0.800 |
| min of 3 | 0.600 |
