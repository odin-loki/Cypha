# Agreement with human ratings

1000 texts, ratings under `meta.human`, criteria: overall. Item level: Pearson, Spearman and Kendall tau-b between the oriented metric and the mean rating; percentile intervals from 1000 resamples of the items (seed 20261028). Δ Kendall is the item-level Kendall minus that of the reference metric (SCRIBE v2) on the same items; its p-value is a two-sided paired bootstrap test (centred: p = (1 + #|d* − d| >= |d|) / (B + 1)), Benjamini-Hochberg adjusted across metrics within each criterion. System level: Kendall and Spearman between per-system mean ratings and per-system mean metric scores, intervals from resampling items within systems. Human ceiling: one rating of each item, drawn at random, against the mean of its other ratings (leave one out), averaged over 20 draws; it compares one rater with k − 1, so it is a conservative reference for a metric compared with the mean of all k.

## All stories (1000 texts)

### overall, all stories

Human ceiling (leave one out, 1000 items with two or more ratings): Pearson 0.848, Spearman 0.842, Kendall 0.737.

| metric | n | Pearson | Spearman | Kendall | Δ Kendall vs SCRIBE v2 | p | p (BH) |
|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 1000 | 0.189 [0.131, 0.238] | 0.199 [0.136, 0.254] | 0.146 [0.100, 0.186] | 0.000 | n/a | n/a |
| SCRIBE v3 | 1000 | 0.186 [0.129, 0.233] | 0.187 [0.125, 0.239] | 0.129 [0.087, 0.166] | −0.016 | 0.252 | 0.252 |
| SCRIBE v3.1 | 1000 | 0.161 [0.102, 0.211] | 0.178 [0.114, 0.232] | 0.123 [0.078, 0.161] | −0.023 | 0.117 | 0.185 |
| SURPRISE | 1000 | 0.082 [0.028, 0.135] | 0.129 [0.068, 0.187] | 0.092 [0.050, 0.133] | −0.054 | 0.020 | 0.070 |
| MATCH | 1000 | 0.145 [0.084, 0.199] | 0.147 [0.083, 0.203] | 0.110 [0.063, 0.152] | −0.036 | 0.133 | 0.185 |
| SURPRISE 3.1 | 1000 | 0.038 [−0.015, 0.089] | 0.101 [0.040, 0.161] | 0.073 [0.030, 0.115] | −0.072 | 0.003 | 0.021 |
| MATCH 3.1 | 1000 | 0.140 [0.077, 0.195] | 0.145 [0.081, 0.202] | 0.109 [0.062, 0.152] | −0.037 | 0.122 | 0.185 |
| min of 3 | 1000 | 0.168 [0.113, 0.218] | 0.175 [0.112, 0.228] | 0.123 [0.079, 0.160] | −0.023 | 0.159 | 0.185 |

| metric | systems | system Kendall | system Spearman |
|---|---|---|---|
| SCRIBE v2 | 5 | 0.200 [0.000, 0.400] | 0.500 [0.100, 0.600] |
| SCRIBE v3 | 5 | 0.400 [0.200, 0.400] | 0.600 [0.500, 0.600] |
| SCRIBE v3.1 | 5 | 0.200 [0.200, 0.400] | 0.500 [0.200, 0.600] |
| SURPRISE | 5 | 0.400 [0.200, 0.800] | 0.600 [0.500, 0.900] |
| MATCH | 5 | 0.400 [0.200, 0.800] | 0.600 [0.500, 0.900] |
| SURPRISE 3.1 | 5 | 0.200 [0.200, 0.600] | 0.500 [0.200, 0.800] |
| MATCH 3.1 | 5 | 0.400 [0.200, 0.600] | 0.600 [0.500, 0.800] |
| min of 3 | 5 | 0.200 [0.200, 0.400] | 0.500 [0.200, 0.600] |

### Summary, all stories

Item-level Kendall tau-b:

| metric | overall |
|---|---|
| SCRIBE v2 | 0.146 |
| SCRIBE v3 | 0.129 |
| SCRIBE v3.1 | 0.123 |
| SURPRISE | 0.092 |
| MATCH | 0.110 |
| SURPRISE 3.1 | 0.073 |
| MATCH 3.1 | 0.109 |
| min of 3 | 0.123 |
| human ceiling (LOO) | 0.737 |

System-level Kendall tau-b:

| metric | overall |
|---|---|
| SCRIBE v2 | 0.200 |
| SCRIBE v3 | 0.400 |
| SCRIBE v3.1 | 0.200 |
| SURPRISE | 0.400 |
| MATCH | 0.400 |
| SURPRISE 3.1 | 0.200 |
| MATCH 3.1 | 0.400 |
| min of 3 | 0.200 |
