# Human against writer, GEN-A dev (cut near 460 characters)

Human: 250 texts. Writer: 10500 texts from 42 dev GEN files. Current default: k5_coherence/L1 (250 texts).

AUC = P(a human text's value > a writer text's value), ties half; 0.5 = no separation. Raw values, not oriented by the registered direction.

| signal | dir | human mean | writer mean | current mean | AUC human vs all writers | AUC human vs current |
|---|---|---:|---:|---:|---:|---:|
| surp_mean | - | 8.923 | 6.832 | 7.396 | 0.949 | 0.929 |
| surp_sd | + | 5.409 | 4.372 | 4.838 | 0.898 | 0.822 |
| surp_ac1 | + | -0.057 | -0.058 | -0.035 | 0.504 | 0.447 |
| surp_cf | + | 6.898 | 5.245 | 6.067 | 0.828 | 0.715 |
| burst_cw | + | 0.143 | 0.063 | 0.067 | 0.789 | 0.771 |
| commit_text | + | 0.668 | 0.262 | 0.279 | 0.825 | 0.806 |
| commit_prompt | + | 0.537 | 0.221 | 0.201 | 0.789 | 0.807 |
| ent_reuse | + | 0.214 | 0.108 | 0.107 | 0.627 | 0.621 |
| sl_mean_h | - | 0.189 | 0.434 | 0.493 | 0.214 | 0.155 |
| sl_cv_h | - | 0.139 | 0.227 | 0.208 | 0.367 | 0.387 |
| fw_js | - | 0.362 | 0.311 | 0.319 | 0.788 | 0.750 |
| fw_share_h | - | 0.050 | 0.097 | 0.079 | 0.231 | 0.297 |
| rep3 | - | 0.009 | 0.004 | 0.004 | 0.575 | 0.573 |
| junk | - | 0.011 | 0.029 | 0.034 | 0.313 | 0.289 |
| oov | - | 0.058 | 0.012 | 0.014 | 0.870 | 0.848 |
| scribe_v2 | + | 0.886 | 0.608 | 0.628 | 0.783 | 0.761 |
| scribe_v31 | + | 0.805 | 0.547 | 0.555 | 0.806 | 0.786 |
| (sl_mean, raw) | | 19.711 | 14.984 | 12.692 | | |
| (sl_cv, raw) | | 0.423 | 0.591 | 0.573 | | |
| (fw_share, raw) | | 0.453 | 0.532 | 0.505 | | |
