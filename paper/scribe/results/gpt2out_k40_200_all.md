# GPT-2 output dataset, top-k 40, all per-text metrics on gpt2out_200

Analysis `gpt2out_k40_200_all` (6. Model ladders (known order)). meta.py `models`.
Inputs: SP/run/baselines/subsets/gpt2out_200.jsonl, SP/run/gpt2out/samples.scribe.jsonl, SP/run/gpt2out/samples.ts.jsonl, SP/run/baselines/gpt2out_200/stats.jsonl.
Not available yet (left out): R/gpt2out_200/ppl_gpt2.jsonl.

1000 texts in 5 systems, known order best first: WebText (human), GPT-2 xl, top-k 40, GPT-2 large, top-k 40, GPT-2 medium, top-k 40, GPT-2 small, top-k 40. Spearman and Kendall (tau-b) are between the oriented system means and the known order (1 = the known order). Bootstrap: 1000 resamples of the texts within each system, seed 20261028; P(perfect) is the share of resamples whose system means are in exactly the known order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.

### SCRIBE v2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.929 [0.909, 0.948] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.963 [0.943, 0.980] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.972 [0.955, 0.986] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.961 [0.939, 0.977] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.970 [0.954, 0.984] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.400 | [0.353, 0.447] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.475 | [0.436, 0.517] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.543 | [0.502, 0.581] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.454 | [0.416, 0.496] | no |

Spearman −0.500 [−1.000, −0.100], Kendall −0.400 [−1.000, 0.000], in the known order: no, P(perfect) 0.000.

### SCRIBE v2, two-sided (exploratory) (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.880 [0.858, 0.902] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.807 [0.776, 0.836] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.818 [0.791, 0.847] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.809 [0.778, 0.839] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.756 [0.727, 0.784] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.616 | [0.562, 0.670] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.484 | [0.430, 0.540] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.510 | [0.451, 0.568] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.583 | [0.525, 0.639] | no |

Spearman 0.700 [0.600, 1.000], Kendall 0.600 [0.400, 1.000], in the known order: no, P(perfect) 0.154.

### SCRIBE v1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.572 [0.544, 0.600] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.525 [0.495, 0.554] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.530 [0.504, 0.556] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.556 [0.527, 0.580] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.484 [0.456, 0.513] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.558 | [0.501, 0.615] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.505 | [0.450, 0.563] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.463 | [0.408, 0.523] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.601 | [0.540, 0.654] | yes |

Spearman 0.600 [0.300, 0.900], Kendall 0.400 [0.200, 0.800], in the known order: no, P(perfect) 0.012.

### surface S (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.936 [0.929, 0.944] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.919 [0.907, 0.929] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.935 [0.926, 0.944] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.937 [0.926, 0.947] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.926 [0.915, 0.936] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.522 | [0.463, 0.579] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.449 | [0.391, 0.503] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.481 | [0.424, 0.537] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.541 | [0.484, 0.600] | no |

Spearman 0.000 [−0.600, 0.700], Kendall 0.000 [−0.400, 0.600], in the known order: no, P(perfect) 0.001.

### distinct-1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.753 [0.745, 0.763] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.687 [0.678, 0.696] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.692 [0.683, 0.700] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.712 [0.703, 0.721] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.690 [0.680, 0.700] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.762 | [0.717, 0.809] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.485 | [0.427, 0.541] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.402 | [0.349, 0.462] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.572 | [0.519, 0.629] | yes |

Spearman 0.300 [0.100, 0.700], Kendall 0.200 [0.000, 0.600], in the known order: no, P(perfect) 0.000.

### distinct-2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.969 [0.964, 0.974] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.946 [0.939, 0.953] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.949 [0.943, 0.954] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.957 [0.950, 0.963] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.949 [0.942, 0.955] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.678 | [0.624, 0.733] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.509 | [0.451, 0.563] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.410 | [0.355, 0.468] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.549 | [0.493, 0.607] | no |

Spearman 0.100 [0.000, 0.700], Kendall 0.000 [−0.200, 0.600], in the known order: no, P(perfect) 0.001.

### distinct-3 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.992 [0.988, 0.995] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.982 [0.977, 0.987] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.986 [0.983, 0.989] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.988 [0.981, 0.993] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.984 [0.980, 0.989] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.621 | [0.575, 0.670] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.505 | [0.451, 0.558] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.437 | [0.383, 0.489] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.567 | [0.512, 0.619] | no |

Spearman 0.300 [−0.200, 0.800], Kendall 0.200 [−0.200, 0.600], in the known order: no, P(perfect) 0.004.

### distinct-4 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.996 [0.993, 0.998] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.992 [0.988, 0.995] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.994 [0.991, 0.997] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.994 [0.988, 0.998] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.993 [0.989, 0.997] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.556 | [0.519, 0.592] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.491 | [0.455, 0.534] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.476 | [0.437, 0.514] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.529 | [0.491, 0.567] | no |

Spearman 0.400 [−0.600, 0.900], Kendall 0.400 [−0.400, 0.800], in the known order: no, P(perfect) 0.008.

### rep-2 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.031 [0.026, 0.036] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.054 [0.047, 0.061] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.051 [0.046, 0.057] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.043 [0.037, 0.050] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.051 [0.045, 0.058] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.678 | [0.624, 0.733] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.509 | [0.451, 0.563] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.410 | [0.355, 0.468] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.549 | [0.493, 0.607] | no |

Spearman 0.100 [0.000, 0.700], Kendall 0.000 [−0.200, 0.600], in the known order: no, P(perfect) 0.001.

### rep-3 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.008 [0.005, 0.012] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.018 [0.013, 0.023] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.014 [0.011, 0.017] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.012 [0.007, 0.019] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.016 [0.011, 0.020] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.621 | [0.575, 0.670] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.505 | [0.451, 0.558] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.437 | [0.383, 0.489] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.567 | [0.512, 0.619] | no |

Spearman 0.300 [−0.200, 0.800], Kendall 0.200 [−0.200, 0.600], in the known order: no, P(perfect) 0.004.

### rep-4 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.004 [0.002, 0.007] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.008 [0.005, 0.012] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.006 [0.003, 0.009] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.006 [0.002, 0.012] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.007 [0.003, 0.011] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.556 | [0.519, 0.592] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.491 | [0.455, 0.534] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.476 | [0.437, 0.514] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.529 | [0.491, 0.567] | no |

Spearman 0.400 [−0.600, 0.900], Kendall 0.400 [−0.400, 0.800], in the known order: no, P(perfect) 0.008.

### seq-rep-4 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.004 [0.002, 0.007] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.008 [0.005, 0.012] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.006 [0.003, 0.009] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.006 [0.002, 0.012] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.007 [0.003, 0.011] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.556 | [0.519, 0.592] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.491 | [0.455, 0.534] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.476 | [0.437, 0.514] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.529 | [0.491, 0.567] | no |

Spearman 0.400 [−0.600, 0.900], Kendall 0.400 [−0.400, 0.800], in the known order: no, P(perfect) 0.008.

### diversity (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.959 [0.949, 0.968] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.924 [0.912, 0.936] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.933 [0.924, 0.941] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.944 [0.931, 0.954] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.931 [0.919, 0.942] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.675 | [0.621, 0.731] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.511 | [0.450, 0.567] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.412 | [0.356, 0.470] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.558 | [0.501, 0.617] | no |

Spearman 0.300 [0.000, 0.700], Kendall 0.200 [−0.200, 0.600], in the known order: no, P(perfect) 0.001.

### type-token ratio (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.753 [0.745, 0.763] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.687 [0.678, 0.696] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.692 [0.683, 0.700] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.712 [0.703, 0.721] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.690 [0.680, 0.700] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.762 | [0.717, 0.809] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.485 | [0.427, 0.541] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.402 | [0.349, 0.462] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.572 | [0.519, 0.629] | yes |

Spearman 0.300 [0.100, 0.700], Kendall 0.200 [0.000, 0.600], in the known order: no, P(perfect) 0.000.

### gzip ratio (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 200 | 0 | 0.640 [0.636, 0.644] |
| GPT-2 xl, top-k 40 | 200 | 0 | 0.611 [0.606, 0.616] |
| GPT-2 large, top-k 40 | 200 | 0 | 0.612 [0.608, 0.617] |
| GPT-2 medium, top-k 40 | 200 | 0 | 0.614 [0.609, 0.619] |
| GPT-2 small, top-k 40 | 200 | 0 | 0.606 [0.601, 0.611] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.738 | [0.690, 0.785] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.493 | [0.435, 0.550] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.469 | [0.414, 0.523] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.575 | [0.517, 0.627] | no |

Spearman 0.600 [0.300, 1.000], Kendall 0.400 [0.200, 1.000], in the known order: no, P(perfect) 0.035.

## Summary

| metric | Spearman [95% interval] | Kendall [95% interval] | in order | P(perfect) | WebText (human) > GPT-2 xl, top-k 40 | GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 |
|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | −0.500 [−1.000, −0.100] | −0.400 [−1.000, 0.000] | no | 0.000 | 0.400 | 0.475 | 0.543 | 0.454 |
| SCRIBE v2, two-sided (exploratory) | 0.700 [0.600, 1.000] | 0.600 [0.400, 1.000] | no | 0.154 | 0.616 | 0.484 | 0.510 | 0.583 |
| SCRIBE v1 | 0.600 [0.300, 0.900] | 0.400 [0.200, 0.800] | no | 0.012 | 0.558 | 0.505 | 0.463 | 0.601 |
| surface S | 0.000 [−0.600, 0.700] | 0.000 [−0.400, 0.600] | no | 0.001 | 0.522 | 0.449 | 0.481 | 0.541 |
| distinct-1 | 0.300 [0.100, 0.700] | 0.200 [0.000, 0.600] | no | 0.000 | 0.762 | 0.485 | 0.402 | 0.572 |
| distinct-2 | 0.100 [0.000, 0.700] | 0.000 [−0.200, 0.600] | no | 0.001 | 0.678 | 0.509 | 0.410 | 0.549 |
| distinct-3 | 0.300 [−0.200, 0.800] | 0.200 [−0.200, 0.600] | no | 0.004 | 0.621 | 0.505 | 0.437 | 0.567 |
| distinct-4 | 0.400 [−0.600, 0.900] | 0.400 [−0.400, 0.800] | no | 0.008 | 0.556 | 0.491 | 0.476 | 0.529 |
| rep-2 | 0.100 [0.000, 0.700] | 0.000 [−0.200, 0.600] | no | 0.001 | 0.678 | 0.509 | 0.410 | 0.549 |
| rep-3 | 0.300 [−0.200, 0.800] | 0.200 [−0.200, 0.600] | no | 0.004 | 0.621 | 0.505 | 0.437 | 0.567 |
| rep-4 | 0.400 [−0.600, 0.900] | 0.400 [−0.400, 0.800] | no | 0.008 | 0.556 | 0.491 | 0.476 | 0.529 |
| seq-rep-4 | 0.400 [−0.600, 0.900] | 0.400 [−0.400, 0.800] | no | 0.008 | 0.556 | 0.491 | 0.476 | 0.529 |
| diversity | 0.300 [0.000, 0.700] | 0.200 [−0.200, 0.600] | no | 0.001 | 0.675 | 0.511 | 0.412 | 0.558 |
| type-token ratio | 0.300 [0.100, 0.700] | 0.200 [0.000, 0.600] | no | 0.000 | 0.762 | 0.485 | 0.402 | 0.572 |
| gzip ratio | 0.600 [0.300, 1.000] | 0.400 [0.200, 1.000] | no | 0.035 | 0.738 | 0.493 | 0.469 | 0.575 |
