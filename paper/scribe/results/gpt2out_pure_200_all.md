# GPT-2 output dataset, pure sampling, all per-text metrics on gpt2out_200

Analysis `gpt2out_pure_200_all` (6. Model ladders (known order)). meta.py `models`.
Inputs: SP/run/baselines/subsets/gpt2out_200.jsonl, SP/run/gpt2out/samples.scribe.jsonl, SP/run/gpt2out/samples.ts.jsonl, SP/run/baselines/gpt2out_200/stats.jsonl.
Not available yet (left out): R/gpt2out_200/ppl_gpt2.jsonl.

1000 texts in 5 systems, known order best first: WebText (human), GPT-2 xl, GPT-2 large, GPT-2 medium, GPT-2 small. Spearman and Kendall (tau-b) are between the oriented system means and the known order (1 = the known order). Bootstrap: 1000 resamples of the texts within each system, seed 20261028; P(perfect) is the share of resamples whose system means are in exactly the known order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.

### SCRIBE v2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.929 [0.909, 0.948] |
| GPT-2 xl | 200 | 0 | 0.830 [0.800, 0.861] |
| GPT-2 large | 200 | 0 | 0.799 [0.768, 0.831] |
| GPT-2 medium | 200 | 0 | 0.710 [0.670, 0.751] |
| GPT-2 small | 200 | 0 | 0.755 [0.721, 0.789] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.648 | [0.595, 0.697] | yes |
| GPT-2 xl > GPT-2 large | 0.560 | [0.506, 0.614] | no |
| GPT-2 large > GPT-2 medium | 0.575 | [0.521, 0.635] | yes |
| GPT-2 medium > GPT-2 small | 0.475 | [0.415, 0.531] | no |

Spearman 0.900 [0.700, 1.000], Kendall 0.800 [0.600, 1.000], in the known order: no, P(perfect) 0.052.

### SCRIBE v2, two-sided (exploratory) (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.880 [0.858, 0.902] |
| GPT-2 xl | 200 | 0 | 0.818 [0.789, 0.848] |
| GPT-2 large | 200 | 0 | 0.783 [0.753, 0.814] |
| GPT-2 medium | 200 | 0 | 0.716 [0.679, 0.752] |
| GPT-2 small | 200 | 0 | 0.753 [0.722, 0.786] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.587 | [0.535, 0.640] | yes |
| GPT-2 xl > GPT-2 large | 0.569 | [0.514, 0.625] | no |
| GPT-2 large > GPT-2 medium | 0.555 | [0.502, 0.614] | yes |
| GPT-2 medium > GPT-2 small | 0.477 | [0.418, 0.533] | no |

Spearman 0.900 [0.700, 1.000], Kendall 0.800 [0.600, 1.000], in the known order: no, P(perfect) 0.070.

### SCRIBE v1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.572 [0.544, 0.600] |
| GPT-2 xl | 200 | 0 | 0.487 [0.459, 0.514] |
| GPT-2 large | 200 | 0 | 0.444 [0.416, 0.474] |
| GPT-2 medium | 200 | 0 | 0.350 [0.321, 0.379] |
| GPT-2 small | 200 | 0 | 0.417 [0.388, 0.449] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.620 | [0.562, 0.676] | yes |
| GPT-2 xl > GPT-2 large | 0.563 | [0.507, 0.617] | no |
| GPT-2 large > GPT-2 medium | 0.623 | [0.573, 0.682] | yes |
| GPT-2 medium > GPT-2 small | 0.411 | [0.350, 0.468] | no |

Spearman 0.900 [0.700, 0.900], Kendall 0.800 [0.600, 0.800], in the known order: no, P(perfect) 0.001.

### surface S (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.936 [0.929, 0.944] |
| GPT-2 xl | 200 | 0 | 0.930 [0.923, 0.936] |
| GPT-2 large | 200 | 0 | 0.923 [0.915, 0.930] |
| GPT-2 medium | 200 | 0 | 0.900 [0.891, 0.908] |
| GPT-2 small | 200 | 0 | 0.915 [0.909, 0.922] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.558 | [0.500, 0.612] | no |
| GPT-2 xl > GPT-2 large | 0.541 | [0.486, 0.600] | no |
| GPT-2 large > GPT-2 medium | 0.635 | [0.582, 0.691] | yes |
| GPT-2 medium > GPT-2 small | 0.433 | [0.378, 0.491] | no |

Spearman 0.900 [0.700, 0.900], Kendall 0.800 [0.600, 0.800], in the known order: no, P(perfect) 0.001.

### distinct-1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.753 [0.745, 0.763] |
| GPT-2 xl | 200 | 0 | 0.782 [0.774, 0.789] |
| GPT-2 large | 200 | 0 | 0.781 [0.771, 0.790] |
| GPT-2 medium | 200 | 0 | 0.822 [0.813, 0.830] |
| GPT-2 small | 200 | 0 | 0.813 [0.806, 0.820] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.369 | [0.318, 0.421] | no |
| GPT-2 xl > GPT-2 large | 0.497 | [0.441, 0.557] | no |
| GPT-2 large > GPT-2 medium | 0.313 | [0.259, 0.367] | no |
| GPT-2 medium > GPT-2 small | 0.547 | [0.490, 0.602] | no |

Spearman −0.800 [−1.000, −0.800], Kendall −0.600 [−1.000, −0.600], in the known order: no, P(perfect) 0.000.

### distinct-2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.969 [0.964, 0.974] |
| GPT-2 xl | 200 | 0 | 0.975 [0.971, 0.979] |
| GPT-2 large | 200 | 0 | 0.976 [0.971, 0.981] |
| GPT-2 medium | 200 | 0 | 0.983 [0.978, 0.987] |
| GPT-2 small | 200 | 0 | 0.987 [0.984, 0.990] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.459 | [0.405, 0.515] | no |
| GPT-2 xl > GPT-2 large | 0.451 | [0.395, 0.508] | no |
| GPT-2 large > GPT-2 medium | 0.407 | [0.354, 0.464] | no |
| GPT-2 medium > GPT-2 small | 0.482 | [0.426, 0.534] | no |

Spearman −1.000 [−1.000, −0.700], Kendall −1.000 [−1.000, −0.600], in the known order: no, P(perfect) 0.000.

### distinct-3 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.992 [0.988, 0.995] |
| GPT-2 xl | 200 | 0 | 0.994 [0.991, 0.997] |
| GPT-2 large | 200 | 0 | 0.994 [0.989, 0.996] |
| GPT-2 medium | 200 | 0 | 0.994 [0.990, 0.998] |
| GPT-2 small | 200 | 0 | 0.998 [0.996, 0.999] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.472 | [0.431, 0.510] | no |
| GPT-2 xl > GPT-2 large | 0.510 | [0.471, 0.550] | no |
| GPT-2 large > GPT-2 medium | 0.456 | [0.421, 0.491] | no |
| GPT-2 medium > GPT-2 small | 0.488 | [0.457, 0.518] | no |

Spearman −0.700 [−1.000, −0.100], Kendall −0.600 [−1.000, 0.000], in the known order: no, P(perfect) 0.000.

### distinct-4 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.996 [0.993, 0.998] |
| GPT-2 xl | 200 | 0 | 0.997 [0.995, 0.999] |
| GPT-2 large | 200 | 0 | 0.997 [0.993, 0.999] |
| GPT-2 medium | 200 | 0 | 0.996 [0.992, 0.999] |
| GPT-2 small | 200 | 0 | 0.999 [0.998, 1.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.490 | [0.460, 0.518] | no |
| GPT-2 xl > GPT-2 large | 0.504 | [0.478, 0.530] | no |
| GPT-2 large > GPT-2 medium | 0.481 | [0.454, 0.507] | no |
| GPT-2 medium > GPT-2 small | 0.502 | [0.480, 0.523] | no |

Spearman −0.300 [−1.000, 0.100], Kendall −0.200 [−1.000, 0.200], in the known order: no, P(perfect) 0.000.

### rep-2 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.031 [0.026, 0.036] |
| GPT-2 xl | 200 | 0 | 0.025 [0.021, 0.029] |
| GPT-2 large | 200 | 0 | 0.024 [0.019, 0.029] |
| GPT-2 medium | 200 | 0 | 0.017 [0.013, 0.022] |
| GPT-2 small | 200 | 0 | 0.013 [0.010, 0.016] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.459 | [0.405, 0.515] | no |
| GPT-2 xl > GPT-2 large | 0.451 | [0.395, 0.508] | no |
| GPT-2 large > GPT-2 medium | 0.407 | [0.354, 0.464] | no |
| GPT-2 medium > GPT-2 small | 0.482 | [0.426, 0.534] | no |

Spearman −1.000 [−1.000, −0.700], Kendall −1.000 [−1.000, −0.600], in the known order: no, P(perfect) 0.000.

### rep-3 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.008 [0.005, 0.012] |
| GPT-2 xl | 200 | 0 | 0.006 [0.003, 0.009] |
| GPT-2 large | 200 | 0 | 0.006 [0.004, 0.011] |
| GPT-2 medium | 200 | 0 | 0.006 [0.002, 0.010] |
| GPT-2 small | 200 | 0 | 0.002 [0.001, 0.004] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.472 | [0.431, 0.510] | no |
| GPT-2 xl > GPT-2 large | 0.510 | [0.471, 0.550] | no |
| GPT-2 large > GPT-2 medium | 0.456 | [0.421, 0.491] | no |
| GPT-2 medium > GPT-2 small | 0.488 | [0.457, 0.518] | no |

Spearman −0.700 [−1.000, −0.100], Kendall −0.600 [−1.000, 0.000], in the known order: no, P(perfect) 0.000.

### rep-4 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.004 [0.002, 0.007] |
| GPT-2 xl | 200 | 0 | 0.003 [0.001, 0.005] |
| GPT-2 large | 200 | 0 | 0.003 [0.001, 0.007] |
| GPT-2 medium | 200 | 0 | 0.004 [0.001, 0.008] |
| GPT-2 small | 200 | 0 | 0.001 [0.000, 0.002] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.490 | [0.460, 0.518] | no |
| GPT-2 xl > GPT-2 large | 0.504 | [0.478, 0.530] | no |
| GPT-2 large > GPT-2 medium | 0.481 | [0.454, 0.507] | no |
| GPT-2 medium > GPT-2 small | 0.502 | [0.480, 0.523] | no |

Spearman −0.300 [−1.000, 0.100], Kendall −0.200 [−1.000, 0.200], in the known order: no, P(perfect) 0.000.

### seq-rep-4 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.004 [0.002, 0.007] |
| GPT-2 xl | 200 | 0 | 0.003 [0.001, 0.005] |
| GPT-2 large | 200 | 0 | 0.003 [0.001, 0.007] |
| GPT-2 medium | 200 | 0 | 0.004 [0.001, 0.008] |
| GPT-2 small | 200 | 0 | 0.001 [0.000, 0.002] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.490 | [0.460, 0.518] | no |
| GPT-2 xl > GPT-2 large | 0.504 | [0.478, 0.530] | no |
| GPT-2 large > GPT-2 medium | 0.481 | [0.454, 0.507] | no |
| GPT-2 medium > GPT-2 small | 0.502 | [0.480, 0.523] | no |

Spearman −0.300 [−1.000, 0.100], Kendall −0.200 [−1.000, 0.200], in the known order: no, P(perfect) 0.000.

### diversity (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.959 [0.949, 0.968] |
| GPT-2 xl | 200 | 0 | 0.968 [0.960, 0.975] |
| GPT-2 large | 200 | 0 | 0.969 [0.959, 0.977] |
| GPT-2 medium | 200 | 0 | 0.976 [0.965, 0.984] |
| GPT-2 small | 200 | 0 | 0.984 [0.979, 0.988] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.459 | [0.403, 0.516] | no |
| GPT-2 xl > GPT-2 large | 0.456 | [0.400, 0.513] | no |
| GPT-2 large > GPT-2 medium | 0.407 | [0.354, 0.462] | no |
| GPT-2 medium > GPT-2 small | 0.482 | [0.427, 0.534] | no |

Spearman −1.000 [−1.000, −0.600], Kendall −1.000 [−1.000, −0.400], in the known order: no, P(perfect) 0.000.

### type-token ratio (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.753 [0.745, 0.763] |
| GPT-2 xl | 200 | 0 | 0.782 [0.774, 0.789] |
| GPT-2 large | 200 | 0 | 0.781 [0.771, 0.790] |
| GPT-2 medium | 200 | 0 | 0.822 [0.813, 0.830] |
| GPT-2 small | 200 | 0 | 0.813 [0.806, 0.820] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.369 | [0.318, 0.421] | no |
| GPT-2 xl > GPT-2 large | 0.497 | [0.441, 0.557] | no |
| GPT-2 large > GPT-2 medium | 0.313 | [0.259, 0.367] | no |
| GPT-2 medium > GPT-2 small | 0.547 | [0.490, 0.602] | no |

Spearman −0.800 [−1.000, −0.800], Kendall −0.600 [−1.000, −0.600], in the known order: no, P(perfect) 0.000.

### gzip ratio (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.640 [0.636, 0.644] |
| GPT-2 xl | 200 | 0 | 0.644 [0.640, 0.648] |
| GPT-2 large | 200 | 0 | 0.648 [0.644, 0.652] |
| GPT-2 medium | 200 | 0 | 0.659 [0.655, 0.663] |
| GPT-2 small | 200 | 0 | 0.660 [0.656, 0.664] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.480 | [0.425, 0.531] | no |
| GPT-2 xl > GPT-2 large | 0.450 | [0.391, 0.512] | no |
| GPT-2 large > GPT-2 medium | 0.391 | [0.333, 0.451] | no |
| GPT-2 medium > GPT-2 small | 0.498 | [0.438, 0.555] | no |

Spearman −1.000 [−1.000, −0.800], Kendall −1.000 [−1.000, −0.600], in the known order: no, P(perfect) 0.000.

## Summary

| metric | Spearman [95% interval] | Kendall [95% interval] | in order | P(perfect) | WebText (human) > GPT-2 xl | GPT-2 xl > GPT-2 large | GPT-2 large > GPT-2 medium | GPT-2 medium > GPT-2 small |
|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.900 [0.700, 1.000] | 0.800 [0.600, 1.000] | no | 0.052 | 0.648 | 0.560 | 0.575 | 0.475 |
| SCRIBE v2, two-sided (exploratory) | 0.900 [0.700, 1.000] | 0.800 [0.600, 1.000] | no | 0.070 | 0.587 | 0.569 | 0.555 | 0.477 |
| SCRIBE v1 | 0.900 [0.700, 0.900] | 0.800 [0.600, 0.800] | no | 0.001 | 0.620 | 0.563 | 0.623 | 0.411 |
| surface S | 0.900 [0.700, 0.900] | 0.800 [0.600, 0.800] | no | 0.001 | 0.558 | 0.541 | 0.635 | 0.433 |
| distinct-1 | −0.800 [−1.000, −0.800] | −0.600 [−1.000, −0.600] | no | 0.000 | 0.369 | 0.497 | 0.313 | 0.547 |
| distinct-2 | −1.000 [−1.000, −0.700] | −1.000 [−1.000, −0.600] | no | 0.000 | 0.459 | 0.451 | 0.407 | 0.482 |
| distinct-3 | −0.700 [−1.000, −0.100] | −0.600 [−1.000, 0.000] | no | 0.000 | 0.472 | 0.510 | 0.456 | 0.488 |
| distinct-4 | −0.300 [−1.000, 0.100] | −0.200 [−1.000, 0.200] | no | 0.000 | 0.490 | 0.504 | 0.481 | 0.502 |
| rep-2 | −1.000 [−1.000, −0.700] | −1.000 [−1.000, −0.600] | no | 0.000 | 0.459 | 0.451 | 0.407 | 0.482 |
| rep-3 | −0.700 [−1.000, −0.100] | −0.600 [−1.000, 0.000] | no | 0.000 | 0.472 | 0.510 | 0.456 | 0.488 |
| rep-4 | −0.300 [−1.000, 0.100] | −0.200 [−1.000, 0.200] | no | 0.000 | 0.490 | 0.504 | 0.481 | 0.502 |
| seq-rep-4 | −0.300 [−1.000, 0.100] | −0.200 [−1.000, 0.200] | no | 0.000 | 0.490 | 0.504 | 0.481 | 0.502 |
| diversity | −1.000 [−1.000, −0.600] | −1.000 [−1.000, −0.400] | no | 0.000 | 0.459 | 0.456 | 0.407 | 0.482 |
| type-token ratio | −0.800 [−1.000, −0.800] | −0.600 [−1.000, −0.600] | no | 0.000 | 0.369 | 0.497 | 0.313 | 0.547 |
| gzip ratio | −1.000 [−1.000, −0.800] | −1.000 [−1.000, −0.600] | no | 0.000 | 0.480 | 0.450 | 0.391 | 0.498 |
