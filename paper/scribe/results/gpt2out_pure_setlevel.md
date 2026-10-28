# GPT-2 output dataset, pure sampling, set-level metrics (one set per system: self-BLEU and Zipf on gpt2out_200; MAUVE with GPT-2 small features if run)

Analysis `gpt2out_pure_setlevel` (6. Model ladders (known order)). meta.py `models`.
Inputs: SP/run/baselines/gpt2out_200/setlevel.jsonl.
Not available yet (left out): SP/run/baselines/gpt2out_mauve/setlevel.jsonl.

5 texts in 5 systems, known order best first: WebText (human), GPT-2 xl, GPT-2 large, GPT-2 medium, GPT-2 small. Spearman and Kendall (tau-b) are between the oriented system means and the known order (1 = the known order). Bootstrap: 1000 resamples of the texts within each system, seed 20261028; P(perfect) is the share of resamples whose system means are in exactly the known order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.

### self-BLEU (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 1 | 0 | 0.085 [0.085, 0.085] |
| GPT-2 xl | 1 | 0 | 0.074 [0.074, 0.074] |
| GPT-2 large | 1 | 0 | 0.079 [0.079, 0.079] |
| GPT-2 medium | 1 | 0 | 0.058 [0.058, 0.058] |
| GPT-2 small | 1 | 0 | 0.067 [0.067, 0.067] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 0.000 | [0.000, 0.000] | no |
| GPT-2 xl > GPT-2 large | 1.000 | [1.000, 1.000] | yes |
| GPT-2 large > GPT-2 medium | 0.000 | [0.000, 0.000] | no |
| GPT-2 medium > GPT-2 small | 1.000 | [1.000, 1.000] | yes |

Spearman −0.800 [−0.800, −0.800], Kendall −0.600 [−0.600, −0.600], in the known order: no, P(perfect) 0.000.

### |Zipf − human Zipf| (lower is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| WebText (human) | 1 | 0 | 0.000 [0.000, 0.000] |
| GPT-2 xl | 1 | 0 | 0.008 [0.008, 0.008] |
| GPT-2 large | 1 | 0 | 0.001 [0.001, 0.001] |
| GPT-2 medium | 1 | 0 | 0.035 [0.035, 0.035] |
| GPT-2 small | 1 | 0 | 0.011 [0.011, 0.011] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| WebText (human) > GPT-2 xl | 1.000 | [1.000, 1.000] | yes |
| GPT-2 xl > GPT-2 large | 0.000 | [0.000, 0.000] | no |
| GPT-2 large > GPT-2 medium | 1.000 | [1.000, 1.000] | yes |
| GPT-2 medium > GPT-2 small | 0.000 | [0.000, 0.000] | no |

Spearman 0.800 [0.800, 0.800], Kendall 0.600 [0.600, 0.600], in the known order: no, P(perfect) 0.000.

## Summary

| metric | Spearman [95% interval] | Kendall [95% interval] | in order | P(perfect) | WebText (human) > GPT-2 xl | GPT-2 xl > GPT-2 large | GPT-2 large > GPT-2 medium | GPT-2 medium > GPT-2 small |
|---|---|---|---|---|---|---|---|---|
| self-BLEU | −0.800 [−0.800, −0.800] | −0.600 [−0.600, −0.600] | no | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| \|Zipf − human Zipf\| | 0.800 [0.800, 0.800] | 0.600 [0.600, 0.600] | no | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
