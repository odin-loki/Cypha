# OpenMEVA MANS ROC, overall quality (5 ratings per story)

Analysis `mans_roc_corr` (5. Human ratings (HANNA, OpenMEVA)). meta.py `corr`.
Inputs: SP/data/openmeva/mans_roc.jsonl, SP/run/human/mans_roc.v2.jsonl, SP/run/human/mans_roc.v2p.jsonl, SP/run/ts/mans_roc.ts.jsonl, SP/run/baselines/mans_roc/stats.jsonl.
Not available yet (left out): R/mans_roc/ppl_gpt2.jsonl.

1000 texts, ratings under `meta.human`, criteria: overall. Item level: Pearson, Spearman and Kendall tau-b between the oriented metric and the mean rating; percentile intervals from 1000 resamples of the items (seed 20261028). Δ Kendall is the item-level Kendall minus that of the reference metric (SCRIBE v2) on the same items; its p-value is a two-sided paired bootstrap test (centred: p = (1 + #|d* − d| >= |d|) / (B + 1)), Benjamini-Hochberg adjusted across metrics within each criterion. System level: Kendall and Spearman between per-system mean ratings and per-system mean metric scores, intervals from resampling items within systems. Human ceiling: one rating of each item, drawn at random, against the mean of its other ratings (leave one out), averaged over 20 draws; it compares one rater with k − 1, so it is a conservative reference for a metric compared with the mean of all k.

## All stories (1000 texts)

### overall, all stories

Human ceiling (leave one out, 1000 items with two or more ratings): Pearson 0.848, Spearman 0.842, Kendall 0.737.

| metric | n | Pearson | Spearman | Kendall | Δ Kendall vs SCRIBE v2 | p | p (BH) |
|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 1000 | 0.189 [0.131, 0.238] | 0.199 [0.136, 0.254] | 0.146 [0.100, 0.186] | 0.000 | n/a | n/a |
| SCRIBE v2p | 1000 | 0.054 [−0.013, 0.118] | 0.064 [−0.002, 0.130] | 0.044 [−0.001, 0.090] | −0.101 | 0.001 | 0.002 |
| SCRIBE v2, two-sided (exploratory) | 1000 | 0.098 [0.038, 0.153] | 0.070 [0.005, 0.126] | 0.048 [0.003, 0.088] | −0.097 | 0.001 | 0.002 |
| SCRIBE v1 | 1000 | 0.108 [0.049, 0.161] | 0.101 [0.038, 0.156] | 0.069 [0.026, 0.107] | −0.076 | 0.003 | 0.004 |
| surface S | 1000 | 0.065 [0.000, 0.122] | −0.007 [−0.069, 0.057] | −0.005 [−0.049, 0.039] | −0.150 | 0.001 | 0.002 |
| distinct-1 | 1000 | 0.123 [0.066, 0.179] | 0.075 [0.015, 0.136] | 0.054 [0.013, 0.096] | −0.091 | 0.005 | 0.006 |
| distinct-2 | 1000 | 0.090 [0.028, 0.144] | 0.005 [−0.058, 0.062] | 0.004 [−0.042, 0.047] | −0.142 | 0.001 | 0.002 |
| distinct-3 | 1000 | 0.104 [0.048, 0.153] | 0.043 [−0.022, 0.100] | 0.035 [−0.018, 0.082] | −0.111 | 0.001 | 0.002 |
| distinct-4 | 1000 | 0.109 [0.059, 0.158] | 0.103 [0.041, 0.163] | 0.086 [0.034, 0.135] | −0.060 | 0.028 | 0.028 |
| rep-2 | 1000 | 0.090 [0.028, 0.144] | 0.005 [−0.058, 0.062] | 0.004 [−0.042, 0.047] | −0.142 | 0.001 | 0.002 |
| rep-3 | 1000 | 0.104 [0.048, 0.153] | 0.043 [−0.022, 0.100] | 0.035 [−0.018, 0.082] | −0.111 | 0.001 | 0.002 |
| rep-4 | 1000 | 0.109 [0.059, 0.158] | 0.103 [0.041, 0.163] | 0.086 [0.034, 0.135] | −0.060 | 0.028 | 0.028 |
| seq-rep-4 | 1000 | 0.109 [0.059, 0.158] | 0.103 [0.041, 0.163] | 0.086 [0.034, 0.135] | −0.060 | 0.028 | 0.028 |
| diversity | 1000 | 0.098 [0.039, 0.149] | 0.002 [−0.061, 0.059] | 0.002 [−0.045, 0.044] | −0.144 | 0.001 | 0.002 |
| type-token ratio | 1000 | 0.123 [0.066, 0.179] | 0.075 [0.015, 0.136] | 0.054 [0.013, 0.096] | −0.091 | 0.005 | 0.006 |
| gzip ratio | 1000 | 0.079 [0.018, 0.139] | 0.045 [−0.016, 0.107] | 0.030 [−0.012, 0.073] | −0.115 | 0.002 | 0.003 |

| metric | systems | system Kendall | system Spearman |
|---|---|---|---|
| SCRIBE v2 | 5 | 0.200 [0.000, 0.400] | 0.500 [0.100, 0.600] |
| SCRIBE v2p | 5 | −0.400 [−0.400, 0.200] | −0.400 [−0.400, 0.200] |
| SCRIBE v2, two-sided (exploratory) | 5 | 0.000 [−0.200, 0.200] | 0.100 [−0.200, 0.500] |
| SCRIBE v1 | 5 | 0.000 [−0.200, 0.200] | 0.100 [−0.200, 0.300] |
| surface S | 5 | −0.400 [−0.400, 0.000] | −0.400 [−0.400, −0.100] |
| distinct-1 | 5 | 0.000 [−0.200, 0.000] | −0.100 [−0.300, −0.100] |
| distinct-2 | 5 | −0.200 [−0.400, −0.200] | −0.300 [−0.400, −0.300] |
| distinct-3 | 5 | −0.400 [−0.400, 0.000] | −0.400 [−0.400, −0.100] |
| distinct-4 | 5 | −0.400 [−0.400, 0.005] | −0.400 [−0.400, −0.098] |
| rep-2 | 5 | −0.200 [−0.400, −0.200] | −0.300 [−0.400, −0.300] |
| rep-3 | 5 | −0.400 [−0.400, 0.000] | −0.400 [−0.400, −0.100] |
| rep-4 | 5 | −0.400 [−0.400, 0.005] | −0.400 [−0.400, −0.098] |
| seq-rep-4 | 5 | −0.400 [−0.400, 0.005] | −0.400 [−0.400, −0.098] |
| diversity | 5 | −0.200 [−0.400, −0.200] | −0.300 [−0.400, −0.300] |
| type-token ratio | 5 | 0.000 [−0.200, 0.000] | −0.100 [−0.300, −0.100] |
| gzip ratio | 5 | −0.200 [−0.400, 0.000] | 0.000 [−0.400, 0.100] |

### Summary, all stories

Item-level Kendall tau-b:

| metric | overall |
|---|---|
| SCRIBE v2 | 0.146 |
| SCRIBE v2p | 0.044 |
| SCRIBE v2, two-sided (exploratory) | 0.048 |
| SCRIBE v1 | 0.069 |
| surface S | −0.005 |
| distinct-1 | 0.054 |
| distinct-2 | 0.004 |
| distinct-3 | 0.035 |
| distinct-4 | 0.086 |
| rep-2 | 0.004 |
| rep-3 | 0.035 |
| rep-4 | 0.086 |
| seq-rep-4 | 0.086 |
| diversity | 0.002 |
| type-token ratio | 0.054 |
| gzip ratio | 0.030 |
| human ceiling (LOO) | 0.737 |

System-level Kendall tau-b:

| metric | overall |
|---|---|
| SCRIBE v2 | 0.200 |
| SCRIBE v2p | −0.400 |
| SCRIBE v2, two-sided (exploratory) | 0.000 |
| SCRIBE v1 | 0.000 |
| surface S | −0.400 |
| distinct-1 | 0.000 |
| distinct-2 | −0.200 |
| distinct-3 | −0.400 |
| distinct-4 | −0.400 |
| rep-2 | −0.200 |
| rep-3 | −0.400 |
| rep-4 | −0.400 |
| seq-rep-4 | −0.400 |
| diversity | −0.200 |
| type-token ratio | 0.000 |
| gzip ratio | −0.200 |
