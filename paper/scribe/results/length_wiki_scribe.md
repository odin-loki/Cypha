# Held-out human text, standard (human) and short (human_short) buckets and pooled, SCRIBE

Analysis `length_wiki_scribe` (8. Correlation with length). meta.py `length`.
Inputs: SP/run/fresh/wiki_fresh/texts.jsonl, SP/run/fresh/wiki_fresh/scores.jsonl.

1000 texts. Pearson and Spearman between the oriented score (lower-is-better metrics negated, so a positive value means longer texts score as better) and length in bytes (UTF-8) and words (whitespace tokens). Intervals: 1000 resamples of the texts, seed 20261028.

| metric | set | n | missing | Pearson, bytes | Spearman, bytes | Pearson, words | Spearman, words |
|---|---|---|---|---|---|---|---|
| SCRIBE v2 | human | 500 | 0 | 0.009 [−0.081, 0.116] | 0.033 [−0.055, 0.128] | −0.025 [−0.123, 0.090] | 0.035 [−0.055, 0.130] |
| SCRIBE v2 | human_short | 500 | 0 | 0.041 [−0.042, 0.127] | 0.117 [0.031, 0.206] | 0.035 [−0.057, 0.127] | 0.122 [0.035, 0.214] |
| SCRIBE v2 | pooled | 1000 | 0 | −0.124 [−0.182, −0.060] | 0.010 [−0.052, 0.076] | −0.129 [−0.187, −0.062] | 0.012 [−0.049, 0.080] |
| SCRIBE v1 | human | 500 | 0 | 0.034 [−0.060, 0.129] | 0.028 [−0.065, 0.119] | 0.084 [−0.022, 0.187] | 0.095 [0.004, 0.185] |
| SCRIBE v1 | human_short | 500 | 0 | 0.073 [−0.011, 0.157] | 0.084 [−0.008, 0.174] | 0.111 [0.023, 0.194] | 0.114 [0.028, 0.201] |
| SCRIBE v1 | pooled | 1000 | 0 | −0.062 [−0.118, 0.000] | −0.032 [−0.090, 0.027] | −0.046 [−0.103, 0.014] | −0.007 [−0.069, 0.051] |
| surface S | human | 500 | 0 | −0.045 [−0.129, 0.045] | −0.017 [−0.104, 0.070] | −0.024 [−0.132, 0.076] | 0.011 [−0.080, 0.098] |
| surface S | human_short | 500 | 0 | 0.002 [−0.087, 0.080] | 0.027 [−0.062, 0.110] | 0.020 [−0.072, 0.099] | 0.042 [−0.047, 0.131] |
| surface S | pooled | 1000 | 0 | −0.041 [−0.103, 0.018] | −0.055 [−0.115, 0.001] | −0.037 [−0.100, 0.023] | −0.044 [−0.104, 0.013] |
