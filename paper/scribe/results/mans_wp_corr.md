# OpenMEVA MANS WP, overall quality (5 ratings per story)

Analysis `mans_wp_corr` (5. Human ratings (HANNA, OpenMEVA)). meta.py `corr`.
Inputs: SP/data/openmeva/mans_wp.jsonl, SP/run/human/mans_wp.v2.jsonl, SP/run/human/mans_wp.v2p.jsonl, SP/run/ts/mans_wp.ts.jsonl, SP/run/baselines/mans_wp/stats.jsonl, SP/run/baselines/mans_wp/ppl_gpt2.jsonl.

1000 texts, ratings under `meta.human`, criteria: overall. Item level: Pearson, Spearman and Kendall tau-b between the oriented metric and the mean rating; percentile intervals from 1000 resamples of the items (seed 20261028). Δ Kendall is the item-level Kendall minus that of the reference metric (SCRIBE v2) on the same items; its p-value is a two-sided paired bootstrap test (centred: p = (1 + #|d* − d| >= |d|) / (B + 1)), Benjamini-Hochberg adjusted across metrics within each criterion. System level: Kendall and Spearman between per-system mean ratings and per-system mean metric scores, intervals from resampling items within systems. Human ceiling: one rating of each item, drawn at random, against the mean of its other ratings (leave one out), averaged over 20 draws; it compares one rater with k − 1, so it is a conservative reference for a metric compared with the mean of all k.

## All stories (1000 texts)

### overall, all stories

Human ceiling (leave one out, 1000 items with two or more ratings): Pearson 0.797, Spearman 0.803, Kendall 0.709.

| metric | n | Pearson | Spearman | Kendall | Δ Kendall vs SCRIBE v2 | p | p (BH) |
|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 1000 | 0.057 [0.000, 0.115] | 0.068 [0.010, 0.128] | 0.049 [0.007, 0.093] | 0.000 | n/a | n/a |
| SCRIBE v2p | 1000 | 0.047 [−0.015, 0.109] | 0.057 [−0.002, 0.120] | 0.040 [−0.002, 0.084] | −0.009 | 0.755 | 0.755 |
| SCRIBE v2, two-sided (exploratory) | 1000 | −0.041 [−0.108, 0.026] | −0.022 [−0.088, 0.040] | −0.015 [−0.062, 0.028] | −0.064 | 0.015 | 0.017 |
| SCRIBE v1 | 1000 | −0.022 [−0.084, 0.040] | −0.010 [−0.073, 0.052] | −0.007 [−0.051, 0.037] | −0.056 | 0.052 | 0.055 |
| surface S | 1000 | −0.070 [−0.127, −0.015] | −0.066 [−0.128, −0.004] | −0.045 [−0.088, −0.002] | −0.094 | 0.006 | 0.007 |
| distinct-1 | 1000 | −0.140 [−0.204, −0.082] | −0.139 [−0.200, −0.082] | −0.097 [−0.140, −0.058] | −0.146 | 0.001 | 0.002 |
| distinct-2 | 1000 | −0.121 [−0.176, −0.068] | −0.137 [−0.195, −0.077] | −0.094 [−0.134, −0.053] | −0.143 | 0.001 | 0.002 |
| distinct-3 | 1000 | −0.061 [−0.116, −0.007] | −0.106 [−0.165, −0.043] | −0.079 [−0.124, −0.032] | −0.128 | 0.003 | 0.005 |
| distinct-4 | 1000 | −0.022 [−0.080, 0.029] | −0.057 [−0.118, 0.004] | −0.046 [−0.095, 0.003] | −0.095 | 0.005 | 0.007 |
| rep-2 | 1000 | −0.121 [−0.176, −0.068] | −0.137 [−0.195, −0.077] | −0.094 [−0.134, −0.053] | −0.143 | 0.001 | 0.002 |
| rep-3 | 1000 | −0.061 [−0.116, −0.007] | −0.106 [−0.165, −0.043] | −0.079 [−0.124, −0.032] | −0.128 | 0.003 | 0.005 |
| rep-4 | 1000 | −0.022 [−0.080, 0.029] | −0.057 [−0.118, 0.004] | −0.046 [−0.095, 0.003] | −0.095 | 0.005 | 0.007 |
| seq-rep-4 | 1000 | −0.022 [−0.080, 0.029] | −0.057 [−0.118, 0.004] | −0.046 [−0.095, 0.003] | −0.095 | 0.005 | 0.007 |
| diversity | 1000 | −0.102 [−0.156, −0.049] | −0.142 [−0.201, −0.083] | −0.098 [−0.139, −0.057] | −0.147 | 0.001 | 0.002 |
| type-token ratio | 1000 | −0.140 [−0.204, −0.082] | −0.139 [−0.200, −0.082] | −0.097 [−0.140, −0.058] | −0.146 | 0.001 | 0.002 |
| gzip ratio | 1000 | −0.155 [−0.221, −0.093] | −0.157 [−0.218, −0.101] | −0.110 [−0.153, −0.071] | −0.160 | 0.001 | 0.002 |
| GPT-2 PPL | 1000 | 0.301 [0.247, 0.356] | 0.325 [0.269, 0.378] | 0.226 [0.186, 0.263] | 0.177 | 0.001 | 0.002 |

| metric | systems | system Kendall | system Spearman |
|---|---|---|---|
| SCRIBE v2 | 5 | 0.200 [0.200, 0.600] | 0.500 [0.498, 0.800] |
| SCRIBE v2p | 5 | −0.800 [−1.000, 0.400] | −0.900 [−1.000, 0.500] |
| SCRIBE v2, two-sided (exploratory) | 5 | −0.400 [−0.800, 0.200] | −0.500 [−0.900, 0.100] |
| SCRIBE v1 | 5 | −0.800 [−1.000, −0.400] | −0.900 [−1.000, −0.600] |
| surface S | 5 | −0.800 [−1.000, −0.600] | −0.900 [−1.000, −0.800] |
| distinct-1 | 5 | −0.600 [−0.800, −0.400] | −0.800 [−0.900, −0.600] |
| distinct-2 | 5 | −0.400 [−0.805, −0.200] | −0.600 [−0.902, −0.500] |
| distinct-3 | 5 | −0.800 [−1.000, −0.400] | −0.900 [−1.000, −0.600] |
| distinct-4 | 5 | −0.800 [−1.000, −0.600] | −0.900 [−1.000, −0.800] |
| rep-2 | 5 | −0.400 [−0.805, −0.200] | −0.600 [−0.902, −0.500] |
| rep-3 | 5 | −0.800 [−1.000, −0.400] | −0.900 [−1.000, −0.600] |
| rep-4 | 5 | −0.800 [−1.000, −0.600] | −0.900 [−1.000, −0.800] |
| seq-rep-4 | 5 | −0.800 [−1.000, −0.600] | −0.900 [−1.000, −0.800] |
| diversity | 5 | −0.600 [−1.000, −0.400] | −0.800 [−1.000, −0.600] |
| type-token ratio | 5 | −0.600 [−0.800, −0.400] | −0.800 [−0.900, −0.600] |
| gzip ratio | 5 | −0.600 [−1.000, −0.400] | −0.800 [−1.000, −0.600] |
| GPT-2 PPL | 5 | 0.600 [0.600, 1.000] | 0.800 [0.800, 1.000] |

### Summary, all stories

Item-level Kendall tau-b:

| metric | overall |
|---|---|
| SCRIBE v2 | 0.049 |
| SCRIBE v2p | 0.040 |
| SCRIBE v2, two-sided (exploratory) | −0.015 |
| SCRIBE v1 | −0.007 |
| surface S | −0.045 |
| distinct-1 | −0.097 |
| distinct-2 | −0.094 |
| distinct-3 | −0.079 |
| distinct-4 | −0.046 |
| rep-2 | −0.094 |
| rep-3 | −0.079 |
| rep-4 | −0.046 |
| seq-rep-4 | −0.046 |
| diversity | −0.098 |
| type-token ratio | −0.097 |
| gzip ratio | −0.110 |
| GPT-2 PPL | 0.226 |
| human ceiling (LOO) | 0.709 |

System-level Kendall tau-b:

| metric | overall |
|---|---|
| SCRIBE v2 | 0.200 |
| SCRIBE v2p | −0.800 |
| SCRIBE v2, two-sided (exploratory) | −0.400 |
| SCRIBE v1 | −0.800 |
| surface S | −0.800 |
| distinct-1 | −0.600 |
| distinct-2 | −0.400 |
| distinct-3 | −0.800 |
| distinct-4 | −0.800 |
| rep-2 | −0.400 |
| rep-3 | −0.800 |
| rep-4 | −0.800 |
| seq-rep-4 | −0.800 |
| diversity | −0.600 |
| type-token ratio | −0.600 |
| gzip ratio | −0.600 |
| GPT-2 PPL | 0.600 |
