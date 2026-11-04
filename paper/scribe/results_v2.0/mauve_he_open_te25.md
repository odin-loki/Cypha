# MAUVE human evaluation: pairwise judgments

3240 judged pairs over 6116 texts in 9 systems (gpt2 p0.9, gpt2 p1.0, gpt2-large p0.95, gpt2-large p1.0, gpt2-medium p0.9, gpt2-medium p1.0, gpt2-xl p0.95, gpt2-xl p1.0, human). judgments with `te` <= 25.0 s dropped: 1542 of 3240. Codes: 2a definitely A, 1a slightly A or tie, 1b slightly B, 2b definitely B; a 1a or 2a is a win for A (as in the MAUVE paper). Metric agreement: the metric prefers the text with the higher oriented score; exact metric ties and missing scores are dropped; Wilson 95% intervals. Bradley-Terry: penalised maximum likelihood (ridge 0.01), reported as 100 x log-strength centred at 0 (the MAUVE paper's scale). System-level correlations are between the metric's mean oriented score per system (over the unique texts of the pairs) and the BT scores, over the models (human excluded, as in the MAUVE paper) and over all systems. Bootstrap: 1000 resamples of the pairs, seed 20261028; each resample refits BT and recomputes the system means over the texts it contains.

## Question q1

Per-pair agreement with the people's choice:

| metric | n | agreement [95% interval] | n definite | definite only [95% interval] | metric ties | missing |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 1666 | 0.543 [0.519, 0.567] | 612 | 0.577 [0.537, 0.615] | 32 | 0 |
| SCRIBE v3 | 1695 | 0.550 [0.527, 0.574] | 624 | 0.583 [0.544, 0.621] | 3 | 0 |
| SCRIBE v3.1 | 1695 | 0.547 [0.524, 0.571] | 624 | 0.569 [0.530, 0.607] | 3 | 0 |
| SURPRISE | 1691 | 0.529 [0.505, 0.552] | 622 | 0.548 [0.509, 0.587] | 7 | 0 |
| MATCH | 1676 | 0.528 [0.504, 0.552] | 621 | 0.572 [0.532, 0.610] | 22 | 0 |
| SURPRISE 3.1 | 1687 | 0.533 [0.510, 0.557] | 622 | 0.543 [0.504, 0.582] | 11 | 0 |
| MATCH 3.1 | 1677 | 0.532 [0.509, 0.556] | 621 | 0.573 [0.534, 0.612] | 21 | 0 |
| min of 3 | 1693 | 0.552 [0.528, 0.576] | 624 | 0.585 [0.546, 0.623] | 5 | 0 |

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
| SCRIBE v2 | 0.762 [0.071, 0.857] | 0.643 [0.071, 0.714] | 0.600 [0.067, 0.783] | 0.500 [0.056, 0.667] |
| SCRIBE v3 | 0.762 [0.262, 0.905] | 0.643 [0.213, 0.786] | 0.833 [0.433, 0.900] | 0.722 [0.333, 0.778] |
| SCRIBE v3.1 | 0.690 [0.238, 0.881] | 0.571 [0.143, 0.786] | 0.783 [0.367, 0.900] | 0.667 [0.278, 0.778] |
| SURPRISE | 0.786 [0.286, 0.905] | 0.643 [0.214, 0.786] | 0.850 [0.467, 0.917] | 0.722 [0.333, 0.833] |
| MATCH | 0.690 [0.262, 0.881] | 0.571 [0.143, 0.786] | 0.783 [0.417, 0.900] | 0.667 [0.278, 0.778] |
| SURPRISE 3.1 | 0.667 [0.238, 0.881] | 0.500 [0.143, 0.786] | 0.767 [0.417, 0.917] | 0.611 [0.278, 0.779] |
| MATCH 3.1 | 0.690 [0.262, 0.881] | 0.571 [0.143, 0.786] | 0.783 [0.433, 0.900] | 0.667 [0.278, 0.778] |
| min of 3 | 0.762 [0.238, 0.881] | 0.643 [0.143, 0.786] | 0.767 [0.333, 0.883] | 0.611 [0.222, 0.778] |

## Question q2

Per-pair agreement with the people's choice:

| metric | n | agreement [95% interval] | n definite | definite only [95% interval] | metric ties | missing |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 1666 | 0.538 [0.514, 0.562] | 709 | 0.584 [0.547, 0.620] | 32 | 0 |
| SCRIBE v3 | 1695 | 0.547 [0.524, 0.571] | 725 | 0.593 [0.557, 0.628] | 3 | 0 |
| SCRIBE v3.1 | 1695 | 0.547 [0.523, 0.570] | 725 | 0.597 [0.561, 0.632] | 3 | 0 |
| SURPRISE | 1691 | 0.535 [0.511, 0.559] | 721 | 0.571 [0.535, 0.607] | 7 | 0 |
| MATCH | 1676 | 0.529 [0.505, 0.552] | 721 | 0.573 [0.536, 0.608] | 22 | 0 |
| SURPRISE 3.1 | 1687 | 0.539 [0.515, 0.562] | 721 | 0.574 [0.538, 0.610] | 11 | 0 |
| MATCH 3.1 | 1677 | 0.530 [0.506, 0.554] | 721 | 0.580 [0.543, 0.615] | 21 | 0 |
| min of 3 | 1693 | 0.545 [0.521, 0.568] | 724 | 0.590 [0.554, 0.625] | 5 | 0 |

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
| SCRIBE v2 | 0.786 [0.262, 0.905] | 0.643 [0.143, 0.786] | 0.617 [0.233, 0.817] | 0.500 [0.111, 0.667] |
| SCRIBE v3 | 0.786 [0.476, 0.929] | 0.643 [0.355, 0.857] | 0.850 [0.617, 0.950] | 0.722 [0.444, 0.889] |
| SCRIBE v3.1 | 0.762 [0.452, 0.929] | 0.571 [0.286, 0.786] | 0.833 [0.566, 0.933] | 0.667 [0.389, 0.833] |
| SURPRISE | 0.905 [0.524, 0.952] | 0.786 [0.357, 0.857] | 0.933 [0.667, 0.967] | 0.833 [0.500, 0.889] |
| MATCH | 0.762 [0.523, 0.929] | 0.571 [0.357, 0.857] | 0.833 [0.633, 0.950] | 0.667 [0.444, 0.889] |
| SURPRISE 3.1 | 0.810 [0.452, 0.952] | 0.643 [0.286, 0.857] | 0.867 [0.616, 0.950] | 0.722 [0.444, 0.889] |
| MATCH 3.1 | 0.762 [0.476, 0.929] | 0.571 [0.357, 0.857] | 0.833 [0.617, 0.950] | 0.667 [0.444, 0.889] |
| min of 3 | 0.786 [0.452, 0.929] | 0.643 [0.286, 0.857] | 0.800 [0.517, 0.933] | 0.611 [0.388, 0.833] |

## Question q3

Per-pair agreement with the people's choice:

| metric | n | agreement [95% interval] | n definite | definite only [95% interval] | metric ties | missing |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 1666 | 0.541 [0.517, 0.565] | 718 | 0.563 [0.526, 0.599] | 32 | 0 |
| SCRIBE v3 | 1695 | 0.559 [0.535, 0.582] | 728 | 0.577 [0.541, 0.612] | 3 | 0 |
| SCRIBE v3.1 | 1695 | 0.556 [0.532, 0.579] | 728 | 0.578 [0.542, 0.614] | 3 | 0 |
| SURPRISE | 1691 | 0.552 [0.529, 0.576] | 726 | 0.570 [0.534, 0.606] | 7 | 0 |
| MATCH | 1676 | 0.550 [0.526, 0.574] | 724 | 0.577 [0.541, 0.613] | 22 | 0 |
| SURPRISE 3.1 | 1687 | 0.557 [0.533, 0.581] | 724 | 0.573 [0.537, 0.609] | 11 | 0 |
| MATCH 3.1 | 1677 | 0.549 [0.525, 0.573] | 724 | 0.581 [0.545, 0.617] | 21 | 0 |
| min of 3 | 1693 | 0.555 [0.531, 0.578] | 727 | 0.579 [0.543, 0.614] | 5 | 0 |

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
| SCRIBE v2 | 0.881 [0.214, 0.881] | 0.786 [0.143, 0.786] | 0.700 [0.183, 0.817] | 0.611 [0.111, 0.667] |
| SCRIBE v3 | 0.881 [0.405, 0.905] | 0.786 [0.286, 0.786] | 0.917 [0.567, 0.933] | 0.833 [0.444, 0.833] |
| SCRIBE v3.1 | 0.833 [0.381, 0.905] | 0.714 [0.286, 0.786] | 0.883 [0.517, 0.933] | 0.778 [0.389, 0.833] |
| SURPRISE | 0.976 [0.476, 0.953] | 0.929 [0.357, 0.859] | 0.983 [0.633, 0.967] | 0.944 [0.500, 0.890] |
| MATCH | 0.833 [0.405, 0.905] | 0.714 [0.286, 0.786] | 0.883 [0.567, 0.933] | 0.778 [0.389, 0.833] |
| SURPRISE 3.1 | 0.857 [0.452, 0.952] | 0.786 [0.286, 0.857] | 0.900 [0.600, 0.967] | 0.833 [0.444, 0.889] |
| MATCH 3.1 | 0.833 [0.381, 0.905] | 0.714 [0.286, 0.786] | 0.883 [0.550, 0.933] | 0.778 [0.389, 0.833] |
| min of 3 | 0.881 [0.404, 0.929] | 0.786 [0.286, 0.857] | 0.867 [0.483, 0.917] | 0.722 [0.333, 0.833] |

## Agreement between workers

No pair of texts was judged more than once (3240 judgments, 214 workers), so agreement between workers cannot be measured on this file.
