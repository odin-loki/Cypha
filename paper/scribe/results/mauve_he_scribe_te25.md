# SCRIBE, judgments with an estimated answer time over 25 s only (the MAUVE paper's filter; times from the raw CSV)

Analysis `mauve_he_scribe_te25` (4. MAUVE human evaluation (pairwise)). meta.py `pairs`.
Inputs: SP/data/mauve_he/texts.jsonl, SP/data/mauve_he/pairs.jsonl, SP/dl/mauve_he/mauve_human_eval.csv (answer times), SP/run/human/mauve_he.v2.jsonl, SP/run/human/mauve_he.v2p.jsonl, SP/run/ts/mauve_he.ts.jsonl.
MAUVE paper (Pillutla et al. 2021), published system-level Spearman between MAUVE (GPT-2-large features) and the Bradley-Terry scores over the 8 models, judgments with te > 25 s: q1 0.810, q2 0.857, q3 0.952.

3240 judged pairs over 6116 texts in 9 systems (gpt2 p0.9, gpt2 p1.0, gpt2-large p0.95, gpt2-large p1.0, gpt2-medium p0.9, gpt2-medium p1.0, gpt2-xl p0.95, gpt2-xl p1.0, human). judgments with `te` <= 25.0 s dropped: 1542 of 3240. Codes: 2a definitely A, 1a slightly A or tie, 1b slightly B, 2b definitely B; a 1a or 2a is a win for A (as in the MAUVE paper). Metric agreement: the metric prefers the text with the higher oriented score; exact metric ties and missing scores are dropped; Wilson 95% intervals. Bradley-Terry: penalised maximum likelihood (ridge 0.01), reported as 100 x log-strength centred at 0 (the MAUVE paper's scale). System-level correlations are between the metric's mean oriented score per system (over the unique texts of the pairs) and the BT scores, over the models (human excluded, as in the MAUVE paper) and over all systems. Bootstrap: 1000 resamples of the pairs, seed 20261028; each resample refits BT and recomputes the system means over the texts it contains.

## Question q1

Per-pair agreement with the people's choice:

| metric | n | agreement [95% interval] | n definite | definite only [95% interval] | metric ties | missing |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 1667 | 0.547 [0.523, 0.571] | 611 | 0.573 [0.533, 0.611] | 31 | 0 |
| SCRIBE v2p | 1695 | 0.526 [0.502, 0.549] | 624 | 0.543 [0.504, 0.582] | 3 | 0 |
| SCRIBE v2, two-sided (exploratory) | 1688 | 0.544 [0.520, 0.567] | 622 | 0.561 [0.522, 0.600] | 10 | 0 |
| SCRIBE v1 | 1695 | 0.550 [0.526, 0.573] | 624 | 0.564 [0.525, 0.603] | 3 | 0 |
| surface S | 1695 | 0.534 [0.510, 0.558] | 624 | 0.529 [0.490, 0.568] | 3 | 0 |

Bradley-Terry scores of the people's choices:

| system | wins | comparisons | win rate | BT (x100) [95% interval] |
|---|---|---|---|---|
| human | 209 | 366 | 0.571 | 25.501 [7.370, 44.847] |
| gpt2-xl p0.95 | 203 | 359 | 0.565 | 23.043 [4.765, 41.044] |
| gpt2-xl p1.0 | 198 | 376 | 0.527 | 9.528 [−8.620, 27.925] |
| gpt2-large p0.95 | 199 | 384 | 0.518 | 6.784 [−12.624, 25.819] |
| gpt2 p0.9 | 193 | 387 | 0.499 | −0.697 [−17.828, 17.268] |
| gpt2-large p1.0 | 188 | 377 | 0.499 | −1.532 [−19.398, 18.017] |
| gpt2-medium p0.9 | 176 | 379 | 0.464 | −12.822 [−31.514, 4.291] |
| gpt2 p1.0 | 176 | 384 | 0.458 | −15.486 [−33.332, 1.877] |
| gpt2-medium p1.0 | 156 | 384 | 0.406 | −34.320 [−53.820, −16.976] |

System level, metric means against BT scores (8 models; 9 systems with human):

| metric | Spearman, models | Kendall, models | Spearman, all | Kendall, all |
|---|---|---|---|---|
| SCRIBE v2 | 0.714 [0.071, 0.857] | 0.571 [0.071, 0.714] | 0.567 [0.083, 0.800] | 0.444 [0.056, 0.667] |
| SCRIBE v2p | 0.690 [0.095, 0.857] | 0.500 [0.071, 0.714] | 0.733 [0.267, 0.867] | 0.556 [0.167, 0.722] |
| SCRIBE v2, two-sided (exploratory) | 0.524 [0.024, 0.810] | 0.429 [0.000, 0.643] | 0.500 [0.100, 0.800] | 0.389 [0.056, 0.667] |
| SCRIBE v1 | 0.500 [0.024, 0.786] | 0.429 [0.000, 0.643] | 0.417 [0.017, 0.700] | 0.333 [0.000, 0.556] |
| surface S | 0.476 [−0.048, 0.714] | 0.357 [−0.071, 0.571] | 0.317 [−0.100, 0.567] | 0.222 [−0.111, 0.444] |

## Question q2

Per-pair agreement with the people's choice:

| metric | n | agreement [95% interval] | n definite | definite only [95% interval] | metric ties | missing |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 1667 | 0.534 [0.510, 0.558] | 710 | 0.566 [0.529, 0.602] | 31 | 0 |
| SCRIBE v2p | 1695 | 0.529 [0.505, 0.552] | 725 | 0.568 [0.532, 0.604] | 3 | 0 |
| SCRIBE v2, two-sided (exploratory) | 1688 | 0.539 [0.515, 0.563] | 721 | 0.544 [0.507, 0.580] | 10 | 0 |
| SCRIBE v1 | 1695 | 0.541 [0.517, 0.565] | 725 | 0.577 [0.540, 0.612] | 3 | 0 |
| surface S | 1695 | 0.535 [0.511, 0.558] | 725 | 0.553 [0.517, 0.589] | 3 | 0 |

Bradley-Terry scores of the people's choices:

| system | wins | comparisons | win rate | BT (x100) [95% interval] |
|---|---|---|---|---|
| human | 226 | 366 | 0.617 | 43.225 [25.172, 62.077] |
| gpt2-xl p0.95 | 212 | 359 | 0.591 | 31.884 [13.644, 50.383] |
| gpt2-large p0.95 | 201 | 384 | 0.523 | 8.781 [−10.366, 26.833] |
| gpt2-xl p1.0 | 196 | 376 | 0.521 | 7.752 [−10.234, 25.735] |
| gpt2-large p1.0 | 183 | 377 | 0.485 | −7.105 [−24.560, 11.810] |
| gpt2-medium p0.9 | 182 | 379 | 0.480 | −7.293 [−25.665, 10.283] |
| gpt2 p0.9 | 186 | 387 | 0.481 | −7.441 [−25.004, 11.472] |
| gpt2-medium p1.0 | 159 | 384 | 0.414 | −32.001 [−50.417, −15.384] |
| gpt2 p1.0 | 153 | 384 | 0.398 | −37.802 [−56.509, −20.436] |

System level, metric means against BT scores (8 models; 9 systems with human):

| metric | Spearman, models | Kendall, models | Spearman, all | Kendall, all |
|---|---|---|---|---|
| SCRIBE v2 | 0.762 [0.262, 0.905] | 0.571 [0.143, 0.786] | 0.600 [0.233, 0.833] | 0.444 [0.167, 0.667] |
| SCRIBE v2p | 0.786 [0.357, 0.905] | 0.643 [0.284, 0.786] | 0.817 [0.483, 0.917] | 0.667 [0.333, 0.778] |
| SCRIBE v2, two-sided (exploratory) | 0.643 [0.286, 0.881] | 0.429 [0.143, 0.714] | 0.617 [0.317, 0.867] | 0.389 [0.222, 0.722] |
| SCRIBE v1 | 0.500 [0.238, 0.833] | 0.286 [0.071, 0.645] | 0.417 [0.200, 0.717] | 0.222 [0.111, 0.556] |
| surface S | 0.405 [0.167, 0.762] | 0.214 [0.071, 0.571] | 0.283 [0.083, 0.567] | 0.111 [−0.056, 0.444] |

## Question q3

Per-pair agreement with the people's choice:

| metric | n | agreement [95% interval] | n definite | definite only [95% interval] | metric ties | missing |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 1667 | 0.541 [0.517, 0.565] | 720 | 0.560 [0.523, 0.596] | 31 | 0 |
| SCRIBE v2p | 1695 | 0.542 [0.518, 0.566] | 728 | 0.541 [0.505, 0.577] | 3 | 0 |
| SCRIBE v2, two-sided (exploratory) | 1688 | 0.533 [0.509, 0.556] | 723 | 0.552 [0.515, 0.588] | 10 | 0 |
| SCRIBE v1 | 1695 | 0.535 [0.511, 0.558] | 728 | 0.576 [0.539, 0.611] | 3 | 0 |
| surface S | 1695 | 0.528 [0.504, 0.552] | 728 | 0.536 [0.499, 0.572] | 3 | 0 |

Bradley-Terry scores of the people's choices:

| system | wins | comparisons | win rate | BT (x100) [95% interval] |
|---|---|---|---|---|
| human | 230 | 366 | 0.628 | 47.246 [28.917, 65.896] |
| gpt2-xl p0.95 | 196 | 359 | 0.546 | 15.662 [−1.515, 33.385] |
| gpt2-large p0.95 | 205 | 384 | 0.534 | 12.551 [−6.275, 32.578] |
| gpt2-xl p1.0 | 197 | 376 | 0.524 | 8.965 [−8.145, 27.739] |
| gpt2-medium p0.9 | 186 | 379 | 0.491 | −3.429 [−22.778, 14.661] |
| gpt2-large p1.0 | 183 | 377 | 0.485 | −6.934 [−25.326, 11.210] |
| gpt2 p0.9 | 177 | 387 | 0.457 | −15.782 [−34.456, 0.222] |
| gpt2 p1.0 | 164 | 384 | 0.427 | −27.515 [−45.496, −10.034] |
| gpt2-medium p1.0 | 160 | 384 | 0.417 | −30.766 [−49.227, −12.599] |

System level, metric means against BT scores (8 models; 9 systems with human):

| metric | Spearman, models | Kendall, models | Spearman, all | Kendall, all |
|---|---|---|---|---|
| SCRIBE v2 | 0.857 [0.214, 0.881] | 0.714 [0.143, 0.786] | 0.683 [0.183, 0.800] | 0.556 [0.111, 0.667] |
| SCRIBE v2p | 0.833 [0.286, 0.905] | 0.643 [0.214, 0.786] | 0.850 [0.450, 0.933] | 0.667 [0.333, 0.833] |
| SCRIBE v2, two-sided (exploratory) | 0.738 [0.238, 0.881] | 0.571 [0.143, 0.786] | 0.700 [0.283, 0.850] | 0.500 [0.222, 0.722] |
| SCRIBE v1 | 0.571 [0.143, 0.810] | 0.429 [0.071, 0.643] | 0.483 [0.117, 0.683] | 0.333 [0.056, 0.501] |
| surface S | 0.476 [0.048, 0.762] | 0.357 [0.000, 0.571] | 0.333 [−0.034, 0.534] | 0.222 [−0.056, 0.389] |

## Agreement between workers

No pair of texts was judged more than once (3240 judgments, 214 workers), so agreement between workers cannot be measured on this file.
