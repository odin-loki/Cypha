# GPT-2 output dataset, top-k 40, per-text metrics on gpt2out_500 (stats only under the plan in force)

Analysis `gpt2out_k40_500_all` (6. Model ladders (known order)). meta.py `models`.
Inputs: SP/run/baselines/subsets/gpt2out_500.jsonl, SP/run/gpt2out/samples.scribe.jsonl, SP/run/gpt2out/samples.ts.jsonl, SP/run/baselines/gpt2out_500/stats.jsonl.

2500 texts in 5 systems, known order best first: WebText (human), GPT-2 xl, top-k 40, GPT-2 large, top-k 40, GPT-2 medium, top-k 40, GPT-2 small, top-k 40. Spearman and Kendall (tau-b) are between the oriented system means and the known order (1 = the known order). Bootstrap: 1000 resamples of the texts within each system, seed 20261028; P(perfect) is the share of resamples whose system means are in exactly the known order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.

### SCRIBE v2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.927 [0.913, 0.937] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.973 [0.964, 0.981] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.969 [0.958, 0.978] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.964 [0.952, 0.974] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.969 [0.957, 0.979] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.365 | [0.336, 0.393] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.505 | [0.479, 0.530] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.519 | [0.495, 0.546] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.463 | [0.439, 0.488] | no |

Spearman −0.300 [−0.900, 0.000], Kendall −0.200 [−0.800, 0.200], in the known order: no, P(perfect) 0.000.

### SCRIBE v2, two-sided (exploratory) (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.860 [0.844, 0.876] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.824 [0.806, 0.841] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.813 [0.794, 0.830] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.806 [0.787, 0.824] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.746 [0.724, 0.767] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.557 | [0.522, 0.596] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.512 | [0.475, 0.551] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.512 | [0.476, 0.546] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.581 | [0.545, 0.615] | yes |

Spearman 1.000 [0.700, 1.000], Kendall 1.000 [0.600, 1.000], in the known order: yes, P(perfect) 0.506.

### SCRIBE v1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.543 [0.524, 0.561] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.537 [0.517, 0.556] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.521 [0.502, 0.538] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.548 [0.530, 0.565] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.473 [0.453, 0.491] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.505 | [0.469, 0.540] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.529 | [0.492, 0.564] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.464 | [0.428, 0.500] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.604 | [0.568, 0.639] | yes |

Spearman 0.400 [0.100, 0.900], Kendall 0.400 [0.000, 0.800], in the known order: no, P(perfect) 0.004.

### surface S (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.933 [0.928, 0.937] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.926 [0.920, 0.932] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.927 [0.921, 0.934] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.936 [0.929, 0.942] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.920 [0.912, 0.927] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.499 | [0.465, 0.534] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.493 | [0.456, 0.529] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.457 | [0.423, 0.492] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.551 | [0.512, 0.586] | yes |

Spearman 0.300 [−0.200, 0.700], Kendall 0.200 [−0.200, 0.600], in the known order: no, P(perfect) 0.001.

### distinct-1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.747 [0.741, 0.752] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.695 [0.689, 0.701] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.694 [0.688, 0.700] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.708 [0.703, 0.713] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.682 [0.675, 0.689] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.721 | [0.687, 0.751] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.504 | [0.467, 0.539] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.437 | [0.402, 0.474] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.599 | [0.565, 0.631] | yes |

Spearman 0.700 [0.600, 0.700], Kendall 0.600 [0.400, 0.600], in the known order: no, P(perfect) 0.000.

### distinct-2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.966 [0.963, 0.970] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.949 [0.945, 0.953] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.949 [0.944, 0.952] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.956 [0.952, 0.959] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.945 [0.940, 0.950] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.652 | [0.615, 0.685] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.510 | [0.476, 0.546] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.436 | [0.401, 0.473] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.552 | [0.515, 0.586] | yes |

Spearman 0.700 [0.100, 0.700], Kendall 0.600 [0.000, 0.600], in the known order: no, P(perfect) 0.001.

### distinct-3 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.990 [0.988, 0.992] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.984 [0.982, 0.987] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.985 [0.982, 0.987] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.988 [0.984, 0.990] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.982 [0.977, 0.985] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.592 | [0.559, 0.622] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.515 | [0.482, 0.548] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.444 | [0.412, 0.476] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.554 | [0.519, 0.585] | no |

Spearman 0.600 [0.100, 0.900], Kendall 0.400 [0.000, 0.800], in the known order: no, P(perfect) 0.016.

### distinct-4 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.995 [0.994, 0.997] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.993 [0.991, 0.995] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.993 [0.991, 0.995] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.994 [0.991, 0.996] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.991 [0.987, 0.994] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.538 | [0.515, 0.561] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.512 | [0.487, 0.537] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.470 | [0.444, 0.494] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.535 | [0.507, 0.559] | no |

Spearman 0.600 [0.000, 1.000], Kendall 0.400 [−0.005, 1.000], in the known order: no, P(perfect) 0.072.

### rep-2 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.034 [0.030, 0.037] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.051 [0.047, 0.055] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.051 [0.048, 0.056] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.044 [0.041, 0.048] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.055 [0.050, 0.060] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.652 | [0.615, 0.685] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.510 | [0.476, 0.546] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.436 | [0.401, 0.473] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.552 | [0.515, 0.586] | yes |

Spearman 0.700 [0.100, 0.700], Kendall 0.600 [0.000, 0.600], in the known order: no, P(perfect) 0.001.

### rep-3 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.010 [0.008, 0.012] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.016 [0.013, 0.018] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.015 [0.013, 0.018] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.012 [0.010, 0.016] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.018 [0.015, 0.023] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.592 | [0.559, 0.622] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.515 | [0.482, 0.548] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.444 | [0.412, 0.476] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.554 | [0.519, 0.585] | no |

Spearman 0.600 [0.100, 0.900], Kendall 0.400 [0.000, 0.800], in the known order: no, P(perfect) 0.016.

### rep-4 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.005 [0.003, 0.006] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.007 [0.005, 0.009] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.007 [0.005, 0.009] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.006 [0.004, 0.009] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.009 [0.006, 0.013] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.538 | [0.515, 0.561] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.512 | [0.487, 0.537] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.470 | [0.444, 0.494] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.535 | [0.507, 0.559] | no |

Spearman 0.600 [0.000, 1.000], Kendall 0.400 [−0.005, 1.000], in the known order: no, P(perfect) 0.072.

### seq-rep-4 (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.005 [0.003, 0.006] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.007 [0.005, 0.009] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.007 [0.005, 0.009] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.006 [0.004, 0.009] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.009 [0.006, 0.013] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.538 | [0.515, 0.561] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.512 | [0.487, 0.537] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.470 | [0.444, 0.494] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.535 | [0.507, 0.559] | no |

Spearman 0.600 [0.000, 1.000], Kendall 0.400 [−0.005, 1.000], in the known order: no, P(perfect) 0.072.

### diversity (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.954 [0.948, 0.960] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.930 [0.923, 0.937] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.930 [0.923, 0.937] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.941 [0.934, 0.947] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.925 [0.915, 0.933] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.647 | [0.611, 0.681] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.514 | [0.479, 0.550] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.433 | [0.397, 0.469] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.555 | [0.518, 0.588] | yes |

Spearman 0.600 [0.100, 0.700], Kendall 0.400 [0.000, 0.600], in the known order: no, P(perfect) 0.002.

### type-token ratio (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.747 [0.741, 0.752] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.695 [0.689, 0.701] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.694 [0.688, 0.700] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.708 [0.703, 0.713] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.682 [0.675, 0.689] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.721 | [0.687, 0.751] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.504 | [0.467, 0.539] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.437 | [0.402, 0.474] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.599 | [0.565, 0.631] | yes |

Spearman 0.700 [0.600, 0.700], Kendall 0.600 [0.400, 0.600], in the known order: no, P(perfect) 0.000.

### gzip ratio (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 500 | 0 | 0.634 [0.631, 0.637] |
| GPT-2 xl, top-k 40 | 500 | 0 | 0.612 [0.609, 0.615] |
| GPT-2 large, top-k 40 | 500 | 0 | 0.611 [0.608, 0.615] |
| GPT-2 medium, top-k 40 | 500 | 0 | 0.614 [0.611, 0.617] |
| GPT-2 small, top-k 40 | 500 | 0 | 0.604 [0.600, 0.607] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.690 | [0.656, 0.722] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.506 | [0.469, 0.543] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.473 | [0.439, 0.511] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.581 | [0.549, 0.616] | yes |

Spearman 0.700 [0.600, 1.000], Kendall 0.600 [0.400, 1.000], in the known order: no, P(perfect) 0.042.

## Summary

| metric | Spearman [95% interval] | Kendall [95% interval] | in order | P(perfect) | WebText (human) > GPT-2 xl, top-k 40 | GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 |
|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | −0.300 [−0.900, 0.000] | −0.200 [−0.800, 0.200] | no | 0.000 | 0.365 | 0.505 | 0.519 | 0.463 |
| SCRIBE v2, two-sided (exploratory) | 1.000 [0.700, 1.000] | 1.000 [0.600, 1.000] | yes | 0.506 | 0.557 | 0.512 | 0.512 | 0.581 |
| SCRIBE v1 | 0.400 [0.100, 0.900] | 0.400 [0.000, 0.800] | no | 0.004 | 0.505 | 0.529 | 0.464 | 0.604 |
| surface S | 0.300 [−0.200, 0.700] | 0.200 [−0.200, 0.600] | no | 0.001 | 0.499 | 0.493 | 0.457 | 0.551 |
| distinct-1 | 0.700 [0.600, 0.700] | 0.600 [0.400, 0.600] | no | 0.000 | 0.721 | 0.504 | 0.437 | 0.599 |
| distinct-2 | 0.700 [0.100, 0.700] | 0.600 [0.000, 0.600] | no | 0.001 | 0.652 | 0.510 | 0.436 | 0.552 |
| distinct-3 | 0.600 [0.100, 0.900] | 0.400 [0.000, 0.800] | no | 0.016 | 0.592 | 0.515 | 0.444 | 0.554 |
| distinct-4 | 0.600 [0.000, 1.000] | 0.400 [−0.005, 1.000] | no | 0.072 | 0.538 | 0.512 | 0.470 | 0.535 |
| rep-2 | 0.700 [0.100, 0.700] | 0.600 [0.000, 0.600] | no | 0.001 | 0.652 | 0.510 | 0.436 | 0.552 |
| rep-3 | 0.600 [0.100, 0.900] | 0.400 [0.000, 0.800] | no | 0.016 | 0.592 | 0.515 | 0.444 | 0.554 |
| rep-4 | 0.600 [0.000, 1.000] | 0.400 [−0.005, 1.000] | no | 0.072 | 0.538 | 0.512 | 0.470 | 0.535 |
| seq-rep-4 | 0.600 [0.000, 1.000] | 0.400 [−0.005, 1.000] | no | 0.072 | 0.538 | 0.512 | 0.470 | 0.535 |
| diversity | 0.600 [0.100, 0.700] | 0.400 [0.000, 0.600] | no | 0.002 | 0.647 | 0.514 | 0.433 | 0.555 |
| type-token ratio | 0.700 [0.600, 0.700] | 0.600 [0.400, 0.600] | no | 0.000 | 0.721 | 0.504 | 0.437 | 0.599 |
| gzip ratio | 0.700 [0.600, 1.000] | 0.600 [0.400, 1.000] | no | 0.042 | 0.690 | 0.506 | 0.473 | 0.581 |
