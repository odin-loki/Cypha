# GPT-2 output dataset, pure sampling, per-text metrics on gpt2out_500 (stats only under the plan in force)

Analysis `gpt2out_pure_500_all` (6. Model ladders (known order)). meta.py `models`.
Inputs: SP/run/baselines/subsets/gpt2out_500.jsonl, SP/run/gpt2out/samples.scribe.jsonl, SP/run/gpt2out/samples.ts.jsonl, SP/run/baselines/gpt2out_500/stats.jsonl.

2500 texts in 5 systems, known order best first: WebText (human), GPT-2 xl, GPT-2 large, GPT-2 medium, GPT-2 small. Spearman and Kendall (tau-b) are between the oriented system means and the known order (1 = the known order). Bootstrap: 1000 resamples of the texts within each system, seed 20261028; P(perfect) is the share of resamples whose system means are in exactly the known order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.

### SCRIBE v2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.927 [0.913, 0.937] |
| GPT-2 xl | 500 | 0 | 0.834 [0.815, 0.854] |
| GPT-2 large | 500 | 0 | 0.818 [0.798, 0.837] |
| GPT-2 medium | 500 | 0 | 0.711 [0.686, 0.735] |
| GPT-2 small | 500 | 0 | 0.757 [0.735, 0.778] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.647 | [0.612, 0.681] | yes |
| GPT-2 xl > GPT-2 large | 0.519 | [0.487, 0.554] | no |
| GPT-2 large > GPT-2 medium | 0.608 | [0.573, 0.642] | yes |
| GPT-2 medium > GPT-2 small | 0.465 | [0.431, 0.498] | no |

Spearman 0.900 [0.800, 0.900], Kendall 0.800 [0.600, 0.800], in the known order: no, P(perfect) 0.000.

### SCRIBE v2, two-sided (exploratory) (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.860 [0.844, 0.876] |
| GPT-2 xl | 500 | 0 | 0.819 [0.800, 0.837] |
| GPT-2 large | 500 | 0 | 0.802 [0.782, 0.819] |
| GPT-2 medium | 500 | 0 | 0.729 [0.706, 0.751] |
| GPT-2 small | 500 | 0 | 0.756 [0.736, 0.777] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.566 | [0.531, 0.603] | yes |
| GPT-2 xl > GPT-2 large | 0.524 | [0.490, 0.560] | no |
| GPT-2 large > GPT-2 medium | 0.573 | [0.537, 0.606] | yes |
| GPT-2 medium > GPT-2 small | 0.487 | [0.451, 0.520] | no |

Spearman 0.900 [0.800, 1.000], Kendall 0.800 [0.600, 1.000], in the known order: no, P(perfect) 0.029.

### SCRIBE v1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.543 [0.524, 0.561] |
| GPT-2 xl | 500 | 0 | 0.489 [0.470, 0.507] |
| GPT-2 large | 500 | 0 | 0.466 [0.447, 0.486] |
| GPT-2 medium | 500 | 0 | 0.362 [0.343, 0.381] |
| GPT-2 small | 500 | 0 | 0.422 [0.404, 0.441] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.580 | [0.545, 0.618] | yes |
| GPT-2 xl > GPT-2 large | 0.532 | [0.496, 0.568] | no |
| GPT-2 large > GPT-2 medium | 0.636 | [0.600, 0.670] | yes |
| GPT-2 medium > GPT-2 small | 0.420 | [0.386, 0.452] | no |

Spearman 0.900 [0.800, 0.900], Kendall 0.800 [0.600, 0.800], in the known order: no, P(perfect) 0.000.

### surface S (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.933 [0.928, 0.937] |
| GPT-2 xl | 500 | 0 | 0.931 [0.927, 0.935] |
| GPT-2 large | 500 | 0 | 0.926 [0.920, 0.931] |
| GPT-2 medium | 500 | 0 | 0.904 [0.899, 0.909] |
| GPT-2 small | 500 | 0 | 0.914 [0.909, 0.918] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.546 | [0.511, 0.583] | no |
| GPT-2 xl > GPT-2 large | 0.525 | [0.490, 0.560] | no |
| GPT-2 large > GPT-2 medium | 0.632 | [0.599, 0.667] | yes |
| GPT-2 medium > GPT-2 small | 0.455 | [0.419, 0.490] | no |

Spearman 0.900 [0.700, 0.900], Kendall 0.800 [0.600, 0.800], in the known order: no, P(perfect) 0.002.

### distinct-1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.747 [0.741, 0.752] |
| GPT-2 xl | 500 | 0 | 0.779 [0.774, 0.784] |
| GPT-2 large | 500 | 0 | 0.781 [0.775, 0.786] |
| GPT-2 medium | 500 | 0 | 0.825 [0.820, 0.830] |
| GPT-2 small | 500 | 0 | 0.810 [0.806, 0.815] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.348 | [0.315, 0.381] | no |
| GPT-2 xl > GPT-2 large | 0.493 | [0.459, 0.528] | no |
| GPT-2 large > GPT-2 medium | 0.293 | [0.262, 0.325] | no |
| GPT-2 medium > GPT-2 small | 0.580 | [0.545, 0.614] | yes |

Spearman −0.900 [−0.900, −0.800], Kendall −0.800 [−0.800, −0.600], in the known order: no, P(perfect) 0.000.

### distinct-2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.966 [0.963, 0.970] |
| GPT-2 xl | 500 | 0 | 0.977 [0.975, 0.980] |
| GPT-2 large | 500 | 0 | 0.976 [0.973, 0.979] |
| GPT-2 medium | 500 | 0 | 0.986 [0.983, 0.988] |
| GPT-2 small | 500 | 0 | 0.986 [0.984, 0.988] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.399 | [0.365, 0.435] | no |
| GPT-2 xl > GPT-2 large | 0.480 | [0.447, 0.515] | no |
| GPT-2 large > GPT-2 medium | 0.383 | [0.348, 0.415] | no |
| GPT-2 medium > GPT-2 small | 0.530 | [0.497, 0.562] | no |

Spearman −0.800 [−1.000, −0.800], Kendall −0.600 [−1.000, −0.600], in the known order: no, P(perfect) 0.000.

### distinct-3 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.990 [0.988, 0.992] |
| GPT-2 xl | 500 | 0 | 0.995 [0.993, 0.996] |
| GPT-2 large | 500 | 0 | 0.993 [0.991, 0.995] |
| GPT-2 medium | 500 | 0 | 0.996 [0.994, 0.998] |
| GPT-2 small | 500 | 0 | 0.997 [0.996, 0.998] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.442 | [0.416, 0.467] | no |
| GPT-2 xl > GPT-2 large | 0.507 | [0.482, 0.532] | no |
| GPT-2 large > GPT-2 medium | 0.454 | [0.432, 0.479] | no |
| GPT-2 medium > GPT-2 small | 0.499 | [0.480, 0.517] | no |

Spearman −0.900 [−1.000, −0.500], Kendall −0.800 [−1.000, −0.400], in the known order: no, P(perfect) 0.000.

### distinct-4 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.995 [0.994, 0.997] |
| GPT-2 xl | 500 | 0 | 0.998 [0.996, 0.999] |
| GPT-2 large | 500 | 0 | 0.997 [0.995, 0.998] |
| GPT-2 medium | 500 | 0 | 0.997 [0.995, 0.999] |
| GPT-2 small | 500 | 0 | 0.999 [0.998, 0.999] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.476 | [0.458, 0.494] | no |
| GPT-2 xl > GPT-2 large | 0.509 | [0.492, 0.525] | no |
| GPT-2 large > GPT-2 medium | 0.476 | [0.462, 0.491] | no |
| GPT-2 medium > GPT-2 small | 0.503 | [0.491, 0.516] | no |

Spearman −0.700 [−1.000, −0.200], Kendall −0.600 [−1.000, −0.200], in the known order: no, P(perfect) 0.000.

### rep-2 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.034 [0.030, 0.037] |
| GPT-2 xl | 500 | 0 | 0.023 [0.020, 0.025] |
| GPT-2 large | 500 | 0 | 0.024 [0.021, 0.027] |
| GPT-2 medium | 500 | 0 | 0.014 [0.012, 0.017] |
| GPT-2 small | 500 | 0 | 0.014 [0.012, 0.016] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.399 | [0.365, 0.435] | no |
| GPT-2 xl > GPT-2 large | 0.480 | [0.447, 0.515] | no |
| GPT-2 large > GPT-2 medium | 0.383 | [0.348, 0.415] | no |
| GPT-2 medium > GPT-2 small | 0.530 | [0.497, 0.562] | no |

Spearman −0.800 [−1.000, −0.800], Kendall −0.600 [−1.000, −0.600], in the known order: no, P(perfect) 0.000.

### rep-3 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.010 [0.008, 0.012] |
| GPT-2 xl | 500 | 0 | 0.005 [0.004, 0.007] |
| GPT-2 large | 500 | 0 | 0.007 [0.005, 0.009] |
| GPT-2 medium | 500 | 0 | 0.004 [0.002, 0.006] |
| GPT-2 small | 500 | 0 | 0.003 [0.002, 0.004] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.442 | [0.416, 0.467] | no |
| GPT-2 xl > GPT-2 large | 0.507 | [0.482, 0.532] | no |
| GPT-2 large > GPT-2 medium | 0.454 | [0.432, 0.479] | no |
| GPT-2 medium > GPT-2 small | 0.499 | [0.480, 0.517] | no |

Spearman −0.900 [−1.000, −0.500], Kendall −0.800 [−1.000, −0.400], in the known order: no, P(perfect) 0.000.

### rep-4 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.005 [0.003, 0.006] |
| GPT-2 xl | 500 | 0 | 0.002 [0.001, 0.004] |
| GPT-2 large | 500 | 0 | 0.003 [0.002, 0.005] |
| GPT-2 medium | 500 | 0 | 0.003 [0.001, 0.005] |
| GPT-2 small | 500 | 0 | 0.001 [0.001, 0.002] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.476 | [0.458, 0.494] | no |
| GPT-2 xl > GPT-2 large | 0.509 | [0.492, 0.525] | no |
| GPT-2 large > GPT-2 medium | 0.476 | [0.462, 0.491] | no |
| GPT-2 medium > GPT-2 small | 0.503 | [0.491, 0.516] | no |

Spearman −0.700 [−1.000, −0.200], Kendall −0.600 [−1.000, −0.200], in the known order: no, P(perfect) 0.000.

### seq-rep-4 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.005 [0.003, 0.006] |
| GPT-2 xl | 500 | 0 | 0.002 [0.001, 0.004] |
| GPT-2 large | 500 | 0 | 0.003 [0.002, 0.005] |
| GPT-2 medium | 500 | 0 | 0.003 [0.001, 0.005] |
| GPT-2 small | 500 | 0 | 0.001 [0.001, 0.002] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.476 | [0.458, 0.494] | no |
| GPT-2 xl > GPT-2 large | 0.509 | [0.492, 0.525] | no |
| GPT-2 large > GPT-2 medium | 0.476 | [0.462, 0.491] | no |
| GPT-2 medium > GPT-2 small | 0.503 | [0.491, 0.516] | no |

Spearman −0.700 [−1.000, −0.200], Kendall −0.600 [−1.000, −0.200], in the known order: no, P(perfect) 0.000.

### diversity (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.954 [0.948, 0.960] |
| GPT-2 xl | 500 | 0 | 0.971 [0.967, 0.975] |
| GPT-2 large | 500 | 0 | 0.969 [0.963, 0.974] |
| GPT-2 medium | 500 | 0 | 0.981 [0.975, 0.986] |
| GPT-2 small | 500 | 0 | 0.982 [0.978, 0.985] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.401 | [0.366, 0.435] | no |
| GPT-2 xl > GPT-2 large | 0.480 | [0.448, 0.516] | no |
| GPT-2 large > GPT-2 medium | 0.385 | [0.351, 0.418] | no |
| GPT-2 medium > GPT-2 small | 0.530 | [0.496, 0.562] | no |

Spearman −0.900 [−1.000, −0.800], Kendall −0.800 [−1.000, −0.600], in the known order: no, P(perfect) 0.000.

### type-token ratio (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.747 [0.741, 0.752] |
| GPT-2 xl | 500 | 0 | 0.779 [0.774, 0.784] |
| GPT-2 large | 500 | 0 | 0.781 [0.775, 0.786] |
| GPT-2 medium | 500 | 0 | 0.825 [0.820, 0.830] |
| GPT-2 small | 500 | 0 | 0.810 [0.806, 0.815] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.348 | [0.315, 0.381] | no |
| GPT-2 xl > GPT-2 large | 0.493 | [0.459, 0.528] | no |
| GPT-2 large > GPT-2 medium | 0.293 | [0.262, 0.325] | no |
| GPT-2 medium > GPT-2 small | 0.580 | [0.545, 0.614] | yes |

Spearman −0.900 [−0.900, −0.800], Kendall −0.800 [−0.800, −0.600], in the known order: no, P(perfect) 0.000.

### gzip ratio (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.634 [0.631, 0.637] |
| GPT-2 xl | 500 | 0 | 0.645 [0.643, 0.648] |
| GPT-2 large | 500 | 0 | 0.646 [0.644, 0.649] |
| GPT-2 medium | 500 | 0 | 0.659 [0.656, 0.662] |
| GPT-2 small | 500 | 0 | 0.658 [0.656, 0.661] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.406 | [0.373, 0.441] | no |
| GPT-2 xl > GPT-2 large | 0.485 | [0.450, 0.521] | no |
| GPT-2 large > GPT-2 medium | 0.376 | [0.342, 0.410] | no |
| GPT-2 medium > GPT-2 small | 0.504 | [0.469, 0.539] | no |

Spearman −0.900 [−1.000, −0.800], Kendall −0.800 [−1.000, −0.600], in the known order: no, P(perfect) 0.000.

## Summary

| metric | Spearman [95% interval] | Kendall [95% interval] | in order | P(perfect) | WebText (human) > GPT-2 xl | GPT-2 xl > GPT-2 large | GPT-2 large > GPT-2 medium | GPT-2 medium > GPT-2 small |
|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.900 [0.800, 0.900] | 0.800 [0.600, 0.800] | no | 0.000 | 0.647 | 0.519 | 0.608 | 0.465 |
| SCRIBE v2, two-sided (exploratory) | 0.900 [0.800, 1.000] | 0.800 [0.600, 1.000] | no | 0.029 | 0.566 | 0.524 | 0.573 | 0.487 |
| SCRIBE v1 | 0.900 [0.800, 0.900] | 0.800 [0.600, 0.800] | no | 0.000 | 0.580 | 0.532 | 0.636 | 0.420 |
| surface S | 0.900 [0.700, 0.900] | 0.800 [0.600, 0.800] | no | 0.002 | 0.546 | 0.525 | 0.632 | 0.455 |
| distinct-1 | −0.900 [−0.900, −0.800] | −0.800 [−0.800, −0.600] | no | 0.000 | 0.348 | 0.493 | 0.293 | 0.580 |
| distinct-2 | −0.800 [−1.000, −0.800] | −0.600 [−1.000, −0.600] | no | 0.000 | 0.399 | 0.480 | 0.383 | 0.530 |
| distinct-3 | −0.900 [−1.000, −0.500] | −0.800 [−1.000, −0.400] | no | 0.000 | 0.442 | 0.507 | 0.454 | 0.499 |
| distinct-4 | −0.700 [−1.000, −0.200] | −0.600 [−1.000, −0.200] | no | 0.000 | 0.476 | 0.509 | 0.476 | 0.503 |
| rep-2 | −0.800 [−1.000, −0.800] | −0.600 [−1.000, −0.600] | no | 0.000 | 0.399 | 0.480 | 0.383 | 0.530 |
| rep-3 | −0.900 [−1.000, −0.500] | −0.800 [−1.000, −0.400] | no | 0.000 | 0.442 | 0.507 | 0.454 | 0.499 |
| rep-4 | −0.700 [−1.000, −0.200] | −0.600 [−1.000, −0.200] | no | 0.000 | 0.476 | 0.509 | 0.476 | 0.503 |
| seq-rep-4 | −0.700 [−1.000, −0.200] | −0.600 [−1.000, −0.200] | no | 0.000 | 0.476 | 0.509 | 0.476 | 0.503 |
| diversity | −0.900 [−1.000, −0.800] | −0.800 [−1.000, −0.600] | no | 0.000 | 0.401 | 0.480 | 0.385 | 0.530 |
| type-token ratio | −0.900 [−0.900, −0.800] | −0.800 [−0.800, −0.600] | no | 0.000 | 0.348 | 0.493 | 0.293 | 0.580 |
| gzip ratio | −0.900 [−1.000, −0.800] | −0.800 [−1.000, −0.600] | no | 0.000 | 0.406 | 0.485 | 0.376 | 0.504 |
