# GPT-2 output dataset, top-k 40: human > xl > large > medium > small, SCRIBE on 1,000 texts per system

Analysis `gpt2out_k40_scribe` (6. Model ladders (known order)). meta.py `models`.
Inputs: SP/run/gpt2out/samples.jsonl, SP/run/gpt2out/samples.scribe.jsonl, SP/run/gpt2out/samples.ts.jsonl.

5000 texts in 5 systems, known order best first: WebText (human), GPT-2 xl, top-k 40, GPT-2 large, top-k 40, GPT-2 medium, top-k 40, GPT-2 small, top-k 40. Spearman and Kendall (tau-b) are between the oriented system means and the known order (1 = the known order). Bootstrap: 1000 resamples of the texts within each system, seed 20261028; P(perfect) is the share of resamples whose system means are in exactly the known order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.

### SCRIBE v2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 1000 | 0 | 0.925 [0.915, 0.934] |
| GPT-2 xl, top-k 40 | 1000 | 0 | 0.969 [0.962, 0.976] |
| GPT-2 large, top-k 40 | 1000 | 0 | 0.968 [0.961, 0.976] |
| GPT-2 medium, top-k 40 | 1000 | 0 | 0.972 [0.965, 0.978] |
| GPT-2 small, top-k 40 | 1000 | 0 | 0.970 [0.962, 0.977] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.377 | [0.358, 0.399] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.496 | [0.479, 0.513] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.506 | [0.488, 0.524] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.475 | [0.459, 0.492] | no |

Spearman −0.800 [−1.000, −0.100], Kendall −0.600 [−1.000, 0.000], in the known order: no, P(perfect) 0.000.

### SCRIBE v2, two-sided (exploratory) (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 1000 | 0 | 0.855 [0.844, 0.865] |
| GPT-2 xl, top-k 40 | 1000 | 0 | 0.819 [0.806, 0.831] |
| GPT-2 large, top-k 40 | 1000 | 0 | 0.811 [0.796, 0.824] |
| GPT-2 medium, top-k 40 | 1000 | 0 | 0.811 [0.799, 0.824] |
| GPT-2 small, top-k 40 | 1000 | 0 | 0.746 [0.730, 0.761] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.552 | [0.527, 0.577] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.508 | [0.481, 0.534] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.504 | [0.477, 0.529] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.583 | [0.559, 0.608] | yes |

Spearman 0.900 [0.600, 1.000], Kendall 0.800 [0.400, 1.000], in the known order: no, P(perfect) 0.332.

### SCRIBE v1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 1000 | 0 | 0.534 [0.522, 0.547] |
| GPT-2 xl, top-k 40 | 1000 | 0 | 0.540 [0.528, 0.553] |
| GPT-2 large, top-k 40 | 1000 | 0 | 0.523 [0.510, 0.536] |
| GPT-2 medium, top-k 40 | 1000 | 0 | 0.549 [0.537, 0.561] |
| GPT-2 small, top-k 40 | 1000 | 0 | 0.478 [0.465, 0.490] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.490 | [0.466, 0.516] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.524 | [0.498, 0.548] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.469 | [0.443, 0.493] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.600 | [0.577, 0.625] | yes |

Spearman 0.300 [0.100, 0.700], Kendall 0.200 [0.000, 0.600], in the known order: no, P(perfect) 0.000.

### surface S (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 1000 | 0 | 0.931 [0.927, 0.934] |
| GPT-2 xl, top-k 40 | 1000 | 0 | 0.929 [0.925, 0.933] |
| GPT-2 large, top-k 40 | 1000 | 0 | 0.927 [0.922, 0.932] |
| GPT-2 medium, top-k 40 | 1000 | 0 | 0.936 [0.932, 0.941] |
| GPT-2 small, top-k 40 | 1000 | 0 | 0.922 [0.917, 0.927] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 0.483 | [0.457, 0.509] | no |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.498 | [0.475, 0.523] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.465 | [0.439, 0.490] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 0.542 | [0.515, 0.567] | yes |

Spearman 0.400 [0.000, 0.500], Kendall 0.400 [−0.200, 0.400], in the known order: no, P(perfect) 0.000.

## Summary

| metric | Spearman [95% interval] | Kendall [95% interval] | in order | P(perfect) | WebText (human) > GPT-2 xl, top-k 40 | GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 |
|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | −0.800 [−1.000, −0.100] | −0.600 [−1.000, 0.000] | no | 0.000 | 0.377 | 0.496 | 0.506 | 0.475 |
| SCRIBE v2, two-sided (exploratory) | 0.900 [0.600, 1.000] | 0.800 [0.400, 1.000] | no | 0.332 | 0.552 | 0.508 | 0.504 | 0.583 |
| SCRIBE v1 | 0.300 [0.100, 0.700] | 0.200 [0.000, 0.600] | no | 0.000 | 0.490 | 0.524 | 0.469 | 0.600 |
| surface S | 0.400 [0.000, 0.500] | 0.400 [−0.200, 0.400] | no | 0.000 | 0.483 | 0.498 | 0.465 | 0.542 |
