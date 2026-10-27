# GPT-2 output dataset, pure sampling: human > xl > large > medium > small, SCRIBE on 1,000 texts per system

Analysis `gpt2out_pure_scribe` (6. Model ladders (known order)). meta.py `models`.
Inputs: SP/run/gpt2out/samples.jsonl, SP/run/gpt2out/samples.scribe.jsonl, SP/run/gpt2out/samples.ts.jsonl.

5000 texts in 5 systems, known order best first: WebText (human), GPT-2 xl, GPT-2 large, GPT-2 medium, GPT-2 small. Spearman and Kendall (tau-b) are between the oriented system means and the known order (1 = the known order). Bootstrap: 1000 resamples of the texts within each system, seed 20261028; P(perfect) is the share of resamples whose system means are in exactly the known order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.

### SCRIBE v2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 1000 | 0 | 0.925 [0.915, 0.934] |
| GPT-2 xl | 1000 | 0 | 0.826 [0.814, 0.840] |
| GPT-2 large | 1000 | 0 | 0.816 [0.802, 0.830] |
| GPT-2 medium | 1000 | 0 | 0.706 [0.690, 0.723] |
| GPT-2 small | 1000 | 0 | 0.767 [0.751, 0.782] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.653 | [0.631, 0.676] | yes |
| GPT-2 xl > GPT-2 large | 0.512 | [0.486, 0.535] | no |
| GPT-2 large > GPT-2 medium | 0.617 | [0.592, 0.641] | yes |
| GPT-2 medium > GPT-2 small | 0.444 | [0.420, 0.468] | no |

Spearman 0.900 [0.800, 0.900], Kendall 0.800 [0.600, 0.800], in the known order: no, P(perfect) 0.000.

### SCRIBE v2, two-sided (exploratory) (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 1000 | 0 | 0.855 [0.844, 0.865] |
| GPT-2 xl | 1000 | 0 | 0.807 [0.795, 0.820] |
| GPT-2 large | 1000 | 0 | 0.800 [0.786, 0.814] |
| GPT-2 medium | 1000 | 0 | 0.723 [0.708, 0.737] |
| GPT-2 small | 1000 | 0 | 0.765 [0.751, 0.778] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.571 | [0.548, 0.597] | yes |
| GPT-2 xl > GPT-2 large | 0.508 | [0.482, 0.533] | no |
| GPT-2 large > GPT-2 medium | 0.586 | [0.559, 0.611] | yes |
| GPT-2 medium > GPT-2 small | 0.460 | [0.438, 0.485] | no |

Spearman 0.900 [0.800, 0.900], Kendall 0.800 [0.600, 0.800], in the known order: no, P(perfect) 0.000.

### SCRIBE v1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 1000 | 0 | 0.534 [0.522, 0.547] |
| GPT-2 xl | 1000 | 0 | 0.479 [0.465, 0.492] |
| GPT-2 large | 1000 | 0 | 0.471 [0.457, 0.484] |
| GPT-2 medium | 1000 | 0 | 0.364 [0.352, 0.377] |
| GPT-2 small | 1000 | 0 | 0.426 [0.413, 0.439] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.580 | [0.556, 0.606] | yes |
| GPT-2 xl > GPT-2 large | 0.512 | [0.485, 0.537] | no |
| GPT-2 large > GPT-2 medium | 0.640 | [0.616, 0.664] | yes |
| GPT-2 medium > GPT-2 small | 0.417 | [0.393, 0.442] | no |

Spearman 0.900 [0.800, 0.900], Kendall 0.800 [0.600, 0.800], in the known order: no, P(perfect) 0.000.

### surface S (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 1000 | 0 | 0.931 [0.927, 0.934] |
| GPT-2 xl | 1000 | 0 | 0.929 [0.925, 0.932] |
| GPT-2 large | 1000 | 0 | 0.927 [0.923, 0.930] |
| GPT-2 medium | 1000 | 0 | 0.904 [0.901, 0.908] |
| GPT-2 small | 1000 | 0 | 0.915 [0.912, 0.918] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.551 | [0.525, 0.577] | no |
| GPT-2 xl > GPT-2 large | 0.504 | [0.478, 0.531] | no |
| GPT-2 large > GPT-2 medium | 0.645 | [0.622, 0.669] | yes |
| GPT-2 medium > GPT-2 small | 0.442 | [0.418, 0.467] | no |

Spearman 0.900 [0.600, 0.900], Kendall 0.800 [0.400, 0.800], in the known order: no, P(perfect) 0.000.

## Summary

| metric | Spearman [95% interval] | Kendall [95% interval] | in order | P(perfect) | WebText (human) > GPT-2 xl | GPT-2 xl > GPT-2 large | GPT-2 large > GPT-2 medium | GPT-2 medium > GPT-2 small |
|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.900 [0.800, 0.900] | 0.800 [0.600, 0.800] | no | 0.000 | 0.653 | 0.512 | 0.617 | 0.444 |
| SCRIBE v2, two-sided (exploratory) | 0.900 [0.800, 0.900] | 0.800 [0.600, 0.800] | no | 0.000 | 0.571 | 0.508 | 0.586 | 0.460 |
| SCRIBE v1 | 0.900 [0.800, 0.900] | 0.800 [0.600, 0.800] | no | 0.000 | 0.580 | 0.512 | 0.640 | 0.417 |
| surface S | 0.900 [0.600, 0.900] | 0.800 [0.400, 0.800] | no | 0.000 | 0.551 | 0.504 | 0.645 | 0.442 |
