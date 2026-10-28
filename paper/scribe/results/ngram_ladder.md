# KN word n-grams: human > 5 > 4 > 3 > 2 > 1

Analysis `ngram_ladder` (6. Model ladders (known order)). meta.py `models`.
Inputs: SP/run/models/ladder_ngram.fresh.jsonl, SP/run/models/ladder_ngram.fresh.scribe.jsonl.

1491 texts in 6 systems, known order best first: human, ngram5, ngram4, ngram3, ngram2, ngram1. Spearman and Kendall (tau-b) are between the oriented system means and the known order (1 = the known order). Bootstrap: 1000 resamples of the texts within each system, seed 20261028; P(perfect) is the share of resamples whose system means are in exactly the known order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.

### SCRIBE v2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 250 | 0 | 0.911 [0.891, 0.930] |
| ngram5 | 250 | 0 | 0.089 [0.076, 0.104] |
| ngram4 | 248 | 0 | 0.081 [0.065, 0.097] |
| ngram3 | 246 | 0 | 0.081 [0.069, 0.096] |
| ngram2 | 248 | 0 | 0.051 [0.037, 0.066] |
| ngram1 | 249 | 0 | 0.001 [0.001, 0.001] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > ngram5 | 0.994 | [0.984, 1.000] | yes |
| ngram5 > ngram4 | 0.501 | [0.455, 0.554] | no |
| ngram4 > ngram3 | 0.476 | [0.424, 0.525] | no |
| ngram3 > ngram2 | 0.667 | [0.619, 0.714] | yes |
| ngram2 > ngram1 | 0.993 | [0.988, 0.997] | yes |

Spearman 0.943 [0.771, 1.000], Kendall 0.867 [0.600, 1.000], in the known order: no, P(perfect) 0.327.

### SCRIBE v1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 250 | 0 | 0.522 [0.497, 0.547] |
| ngram5 | 250 | 0 | 0.075 [0.071, 0.080] |
| ngram4 | 248 | 0 | 0.076 [0.071, 0.083] |
| ngram3 | 246 | 0 | 0.080 [0.074, 0.086] |
| ngram2 | 248 | 0 | 0.078 [0.071, 0.084] |
| ngram1 | 249 | 0 | 0.019 [0.018, 0.021] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > ngram5 | 0.982 | [0.969, 0.992] | yes |
| ngram5 > ngram4 | 0.498 | [0.447, 0.550] | no |
| ngram4 > ngram3 | 0.465 | [0.411, 0.517] | no |
| ngram3 > ngram2 | 0.543 | [0.492, 0.599] | no |
| ngram2 > ngram1 | 0.959 | [0.942, 0.975] | yes |

Spearman 0.486 [0.429, 0.943], Kendall 0.333 [0.200, 0.867], in the known order: no, P(perfect) 0.005.

### surface S (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 250 | 0 | 0.911 [0.903, 0.919] |
| ngram5 | 250 | 0 | 0.873 [0.869, 0.878] |
| ngram4 | 248 | 0 | 0.870 [0.866, 0.874] |
| ngram3 | 246 | 0 | 0.872 [0.868, 0.876] |
| ngram2 | 248 | 0 | 0.871 [0.866, 0.875] |
| ngram1 | 249 | 0 | 0.866 [0.861, 0.870] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > ngram5 | 0.761 | [0.716, 0.805] | yes |
| ngram5 > ngram4 | 0.534 | [0.486, 0.582] | no |
| ngram4 > ngram3 | 0.482 | [0.430, 0.535] | no |
| ngram3 > ngram2 | 0.513 | [0.467, 0.564] | no |
| ngram2 > ngram1 | 0.541 | [0.487, 0.591] | no |

Spearman 0.829 [0.429, 1.000], Kendall 0.733 [0.333, 1.000], in the known order: no, P(perfect) 0.057.

## Summary

| metric | Spearman [95% interval] | Kendall [95% interval] | in order | P(perfect) | human > ngram5 | ngram5 > ngram4 | ngram4 > ngram3 | ngram3 > ngram2 | ngram2 > ngram1 |
|---|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.943 [0.771, 1.000] | 0.867 [0.600, 1.000] | no | 0.327 | 0.994 | 0.501 | 0.476 | 0.667 | 0.993 |
| SCRIBE v1 | 0.486 [0.429, 0.943] | 0.333 [0.200, 0.867] | no | 0.005 | 0.982 | 0.498 | 0.465 | 0.543 | 0.959 |
| surface S | 0.829 [0.429, 1.000] | 0.733 [0.333, 1.000] | no | 0.057 | 0.761 | 0.534 | 0.482 | 0.513 | 0.541 |
