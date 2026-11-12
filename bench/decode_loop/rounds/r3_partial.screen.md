# Round table (subset A, 100 prompts; best r2_WF5; human mean ± sd per text)

| metric | human | r2_WF5 | r3_K2 | r3_K4 | r3_R25 |
|---|---|---|---|---|---|
| surp_mean | 8.916 ± 0.798 | 7.745 | 7.929 | 7.935 | 7.069 |
| surp_sd | 5.440 ± 0.372 | 5.318 | 5.393 | 5.385 | 5.098 |
| sl_mean | 20.053 ± 5.180 | 12.663 | 12.703 | 12.844 | 12.186 |
| on_prev | 0.571 ± 0.384 | 0.322 | 0.323 | 0.352 | 0.297 |
| verbless | 0.050 ± 0.147 | 0.215 | 0.215 | 0.206 | 0.244 |
| stuff3 | 0.033 ± 0.042 | 0.029 | 0.033 | 0.031 | 0.026 |
| commit_text | 0.677 ± 0.282 | 0.495 | 0.470 | 0.481 | 0.440 |
| junk | 0.012 ± 0.019 | 0.034 | 0.038 | 0.037 | 0.029 |
| rep3 | 0.009 ± 0.020 | 0.009 | 0.009 | 0.008 | 0.007 |
| splice | 0.216 ± 0.047 | 0.205 | 0.203 | 0.209 | – |
| tail | 0.498 ± 0.069 | 0.151 | 0.171 | 0.202 | – |
| scribe_v2 | | 0.843 | 0.854 | 0.83 | 0.768 |
| scribe_v31 | | 0.72 | 0.746 | 0.739 | 0.596 |
| fact_precision | | 0.938 | 0.935 | 0.918 | 0.938 |
| copy32 | | 0.557 | 0.594 | 0.603 | 0.477 |
| verbatim13_n | | 0 | 0 | 0 | 0 |
| unlike_best_R | | 0 | 43 | 43 | 43 |

## Screen

- r3_K2: pass
- r3_K4: pass
- r3_R25: FAIL (surp_mean +0.85 human sd further from human; surp_sd +0.59 human sd further from human; no probe metrics)

## Ranking by surp_mean, scribe_v2, stuff3, verbless, splice (rank sum, lower is better; ties: lower copy32)

- r3_K2: rank sum 2
- r3_K4: rank sum 3

kept for the readers (at most 2): r3_K2, r3_K4
