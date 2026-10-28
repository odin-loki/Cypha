# GPT-2 output dataset, top-k 40, set-level metrics (one set per system: self-BLEU and Zipf on gpt2out_200; MAUVE with GPT-2 small features if run)

Analysis `gpt2out_k40_setlevel` (6. Model ladders (known order)). meta.py `models`.
Inputs: SP/run/baselines/gpt2out_200/setlevel.jsonl.
Not available yet (left out): SP/run/baselines/gpt2out_mauve/setlevel.jsonl.

5 texts in 5 systems, known order best first: WebText (human), GPT-2 xl, top-k 40, GPT-2 large, top-k 40, GPT-2 medium, top-k 40, GPT-2 small, top-k 40. Spearman and Kendall (tau-b) are between the oriented system means and the known order (1 = the known order). Bootstrap: 1000 resamples of the texts within each system, seed 20261028; P(perfect) is the share of resamples whose system means are in exactly the known order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.

### self-BLEU (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 1 | 0 | 0.085 [0.085, 0.085] |
| GPT-2 xl, top-k 40 | 1 | 0 | 0.118 [0.118, 0.118] |
| GPT-2 large, top-k 40 | 1 | 0 | 0.118 [0.118, 0.118] |
| GPT-2 medium, top-k 40 | 1 | 0 | 0.115 [0.115, 0.115] |
| GPT-2 small, top-k 40 | 1 | 0 | 0.141 [0.141, 0.141] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 1.000 | [1.000, 1.000] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 1.000 | [1.000, 1.000] | yes |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.000 | [0.000, 0.000] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 1.000 | [1.000, 1.000] | yes |

Spearman 0.700 [0.700, 0.700], Kendall 0.600 [0.600, 0.600], in the known order: no, P(perfect) 0.000.

### |Zipf − human Zipf| (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 1 | 0 | 0.000 [0.000, 0.000] |
| GPT-2 xl, top-k 40 | 1 | 0 | 0.054 [0.054, 0.054] |
| GPT-2 large, top-k 40 | 1 | 0 | 0.051 [0.051, 0.051] |
| GPT-2 medium, top-k 40 | 1 | 0 | 0.044 [0.044, 0.044] |
| GPT-2 small, top-k 40 | 1 | 0 | 0.101 [0.101, 0.101] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl, top-k 40 | 1.000 | [1.000, 1.000] | yes |
| GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | 0.000 | [0.000, 0.000] | no |
| GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | 0.000 | [0.000, 0.000] | no |
| GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 | 1.000 | [1.000, 1.000] | yes |

Spearman 0.600 [0.600, 0.600], Kendall 0.400 [0.400, 0.400], in the known order: no, P(perfect) 0.000.

## Summary

| metric | Spearman [95% interval] | Kendall [95% interval] | in order | P(perfect) | WebText (human) > GPT-2 xl, top-k 40 | GPT-2 xl, top-k 40 > GPT-2 large, top-k 40 | GPT-2 large, top-k 40 > GPT-2 medium, top-k 40 | GPT-2 medium, top-k 40 > GPT-2 small, top-k 40 |
|---|---|---|---|---|---|---|---|---|
| self-BLEU | 0.700 [0.700, 0.700] | 0.600 [0.600, 0.600] | no | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| \|Zipf − human Zipf\| | 0.600 [0.600, 0.600] | 0.400 [0.400, 0.400] | no | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
