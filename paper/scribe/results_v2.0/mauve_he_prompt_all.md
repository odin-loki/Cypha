# MAUVE human evaluation: pairwise judgments

3240 judged pairs over 6116 texts in 9 systems (gpt2 p0.9, gpt2 p1.0, gpt2-large p0.95, gpt2-large p1.0, gpt2-medium p0.9, gpt2-medium p1.0, gpt2-xl p0.95, gpt2-xl p1.0, human). no time filter. Codes: 2a definitely A, 1a slightly A or tie, 1b slightly B, 2b definitely B; a 1a or 2a is a win for A (as in the MAUVE paper). Metric agreement: the metric prefers the text with the higher oriented score; exact metric ties and missing scores are dropped; Wilson 95% intervals. Bradley-Terry: penalised maximum likelihood (ridge 0.01), reported as 100 x log-strength centred at 0 (the MAUVE paper's scale). System-level correlations are between the metric's mean oriented score per system (over the unique texts of the pairs) and the BT scores, over the models (human excluded, as in the MAUVE paper) and over all systems. Bootstrap: 1000 resamples of the pairs, seed 20261028; each resample refits BT and recomputes the system means over the texts it contains.

## Question q1

Per-pair agreement with the people's choice:

| metric | n | agreement [95% interval] | n definite | definite only [95% interval] | metric ties | missing |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 3175 | 0.527 [0.509, 0.544] | 1192 | 0.550 [0.522, 0.578] | 65 | 0 |
| SCRIBE v3 | 3234 | 0.539 [0.522, 0.556] | 1215 | 0.563 [0.535, 0.591] | 6 | 0 |
| SCRIBE v3.1 | 3234 | 0.541 [0.524, 0.558] | 1215 | 0.570 [0.542, 0.597] | 6 | 0 |
| SURPRISE | 3218 | 0.520 [0.503, 0.537] | 1208 | 0.541 [0.512, 0.568] | 22 | 0 |
| MATCH | 3228 | 0.535 [0.518, 0.552] | 1211 | 0.561 [0.533, 0.588] | 12 | 0 |
| SURPRISE 3.1 | 3230 | 0.531 [0.514, 0.548] | 1213 | 0.550 [0.522, 0.578] | 10 | 0 |
| MATCH 3.1 | 3229 | 0.537 [0.520, 0.554] | 1211 | 0.569 [0.541, 0.597] | 11 | 0 |
| min of 3 | 3230 | 0.538 [0.521, 0.555] | 1213 | 0.556 [0.528, 0.584] | 10 | 0 |

Bradley-Terry scores of the people's choices:

| system | wins | comparisons | win rate | BT (x100) [95% interval] |
|---|---|---|---|---|
| human | 396 | 720 | 0.550 | 17.873 [3.919, 31.148] |
| gpt2-xl p0.95 | 376 | 720 | 0.522 | 7.928 [−4.184, 21.918] |
| gpt2-xl p1.0 | 370 | 720 | 0.514 | 4.954 [−8.579, 17.658] |
| gpt2-large p0.95 | 367 | 720 | 0.510 | 3.468 [−9.468, 16.986] |
| gpt2-large p1.0 | 365 | 720 | 0.507 | 2.478 [−9.283, 15.694] |
| gpt2 p0.9 | 357 | 720 | 0.496 | −1.483 [−14.761, 11.203] |
| gpt2 p1.0 | 348 | 720 | 0.483 | −5.941 [−18.362, 6.050] |
| gpt2-medium p0.9 | 341 | 720 | 0.474 | −9.411 [−22.647, 3.173] |
| gpt2-medium p1.0 | 320 | 720 | 0.444 | −19.866 [−32.586, −6.184] |

System level, metric means against BT scores (8 models; 9 systems with human):

| metric | Spearman, models | Kendall, models | Spearman, all | Kendall, all |
|---|---|---|---|---|
| SCRIBE v2 | 0.619 [−0.095, 0.810] | 0.500 [−0.071, 0.643] | 0.467 [−0.100, 0.733] | 0.389 [−0.056, 0.556] |
| SCRIBE v3 | 0.619 [0.000, 0.833] | 0.500 [0.000, 0.714] | 0.733 [0.217, 0.883] | 0.611 [0.167, 0.778] |
| SCRIBE v3.1 | 0.619 [0.000, 0.833] | 0.500 [0.000, 0.714] | 0.733 [0.233, 0.883] | 0.611 [0.167, 0.778] |
| SURPRISE | 0.714 [0.023, 0.881] | 0.643 [0.000, 0.716] | 0.800 [0.266, 0.900] | 0.722 [0.221, 0.778] |
| MATCH | 0.571 [−0.071, 0.810] | 0.429 [−0.071, 0.714] | 0.700 [0.183, 0.867] | 0.556 [0.111, 0.722] |
| SURPRISE 3.1 | 0.714 [0.048, 0.857] | 0.643 [0.000, 0.714] | 0.800 [0.283, 0.900] | 0.722 [0.222, 0.778] |
| MATCH 3.1 | 0.571 [−0.071, 0.810] | 0.429 [−0.071, 0.643] | 0.700 [0.183, 0.867] | 0.556 [0.111, 0.722] |
| min of 3 | 0.619 [0.000, 0.833] | 0.500 [0.000, 0.714] | 0.717 [0.167, 0.867] | 0.556 [0.111, 0.722] |

## Question q2

Per-pair agreement with the people's choice:

| metric | n | agreement [95% interval] | n definite | definite only [95% interval] | metric ties | missing |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 3175 | 0.528 [0.510, 0.545] | 1259 | 0.563 [0.536, 0.590] | 65 | 0 |
| SCRIBE v3 | 3234 | 0.535 [0.518, 0.552] | 1286 | 0.576 [0.549, 0.603] | 6 | 0 |
| SCRIBE v3.1 | 3234 | 0.542 [0.525, 0.559] | 1286 | 0.576 [0.549, 0.603] | 6 | 0 |
| SURPRISE | 3218 | 0.529 [0.512, 0.546] | 1277 | 0.542 [0.514, 0.569] | 22 | 0 |
| MATCH | 3228 | 0.533 [0.515, 0.550] | 1283 | 0.564 [0.537, 0.591] | 12 | 0 |
| SURPRISE 3.1 | 3230 | 0.535 [0.517, 0.552] | 1283 | 0.568 [0.541, 0.595] | 10 | 0 |
| MATCH 3.1 | 3229 | 0.544 [0.526, 0.561] | 1284 | 0.578 [0.551, 0.605] | 11 | 0 |
| min of 3 | 3230 | 0.531 [0.514, 0.548] | 1284 | 0.569 [0.541, 0.595] | 10 | 0 |

Bradley-Terry scores of the people's choices:

| system | wins | comparisons | win rate | BT (x100) [95% interval] |
|---|---|---|---|---|
| human | 420 | 720 | 0.583 | 30.001 [15.759, 43.655] |
| gpt2-xl p0.95 | 378 | 720 | 0.525 | 8.934 [−3.431, 22.355] |
| gpt2-xl p1.0 | 372 | 720 | 0.517 | 5.951 [−7.207, 18.323] |
| gpt2-large p0.95 | 370 | 720 | 0.514 | 4.958 [−8.010, 18.780] |
| gpt2-medium p0.9 | 357 | 720 | 0.496 | −1.498 [−15.306, 10.823] |
| gpt2 p0.9 | 356 | 720 | 0.494 | −1.994 [−14.508, 11.590] |
| gpt2-large p1.0 | 352 | 720 | 0.489 | −3.981 [−16.315, 9.076] |
| gpt2 p1.0 | 318 | 720 | 0.442 | −20.935 [−33.898, −8.624] |
| gpt2-medium p1.0 | 317 | 720 | 0.440 | −21.437 [−33.767, −8.299] |

System level, metric means against BT scores (8 models; 9 systems with human):

| metric | Spearman, models | Kendall, models | Spearman, all | Kendall, all |
|---|---|---|---|---|
| SCRIBE v2 | 0.857 [0.238, 0.929] | 0.786 [0.141, 0.786] | 0.683 [0.183, 0.817] | 0.611 [0.111, 0.667] |
| SCRIBE v3 | 0.857 [0.333, 0.952] | 0.786 [0.214, 0.857] | 0.900 [0.517, 0.967] | 0.833 [0.389, 0.889] |
| SCRIBE v3.1 | 0.857 [0.357, 0.952] | 0.786 [0.214, 0.857] | 0.900 [0.533, 0.967] | 0.833 [0.389, 0.889] |
| SURPRISE | 0.905 [0.357, 0.952] | 0.786 [0.214, 0.857] | 0.933 [0.550, 0.967] | 0.833 [0.389, 0.889] |
| MATCH | 0.833 [0.333, 0.953] | 0.714 [0.214, 0.859] | 0.883 [0.533, 0.967] | 0.778 [0.389, 0.889] |
| SURPRISE 3.1 | 0.905 [0.381, 0.952] | 0.786 [0.214, 0.857] | 0.933 [0.566, 0.967] | 0.833 [0.389, 0.889] |
| MATCH 3.1 | 0.833 [0.333, 0.952] | 0.714 [0.214, 0.857] | 0.883 [0.533, 0.967] | 0.778 [0.389, 0.889] |
| min of 3 | 0.857 [0.357, 0.952] | 0.786 [0.214, 0.857] | 0.883 [0.450, 0.950] | 0.778 [0.333, 0.833] |

## Question q3

Per-pair agreement with the people's choice:

| metric | n | agreement [95% interval] | n definite | definite only [95% interval] | metric ties | missing |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 3175 | 0.529 [0.511, 0.546] | 1327 | 0.540 [0.513, 0.566] | 65 | 0 |
| SCRIBE v3 | 3234 | 0.544 [0.527, 0.561] | 1351 | 0.553 [0.526, 0.579] | 6 | 0 |
| SCRIBE v3.1 | 3234 | 0.551 [0.534, 0.568] | 1351 | 0.559 [0.532, 0.585] | 6 | 0 |
| SURPRISE | 3218 | 0.533 [0.516, 0.550] | 1344 | 0.550 [0.523, 0.576] | 22 | 0 |
| MATCH | 3228 | 0.542 [0.525, 0.560] | 1346 | 0.547 [0.520, 0.573] | 12 | 0 |
| SURPRISE 3.1 | 3230 | 0.547 [0.530, 0.564] | 1348 | 0.556 [0.529, 0.582] | 10 | 0 |
| MATCH 3.1 | 3229 | 0.552 [0.535, 0.569] | 1346 | 0.553 [0.527, 0.580] | 11 | 0 |
| min of 3 | 3230 | 0.539 [0.522, 0.556] | 1348 | 0.541 [0.514, 0.567] | 10 | 0 |

Bradley-Terry scores of the people's choices:

| system | wins | comparisons | win rate | BT (x100) [95% interval] |
|---|---|---|---|---|
| human | 429 | 720 | 0.596 | 34.583 [21.289, 47.456] |
| gpt2-large p0.95 | 375 | 720 | 0.521 | 7.428 [−5.011, 21.157] |
| gpt2-xl p0.95 | 371 | 720 | 0.515 | 5.439 [−7.215, 18.327] |
| gpt2-large p1.0 | 366 | 720 | 0.508 | 2.954 [−10.062, 15.376] |
| gpt2-xl p1.0 | 365 | 720 | 0.507 | 2.458 [−10.311, 15.361] |
| gpt2-medium p0.9 | 350 | 720 | 0.486 | −4.994 [−18.212, 7.403] |
| gpt2 p1.0 | 335 | 720 | 0.465 | −12.458 [−25.825, 0.739] |
| gpt2 p0.9 | 329 | 720 | 0.457 | −15.453 [−28.053, −2.701] |
| gpt2-medium p1.0 | 320 | 720 | 0.444 | −19.957 [−33.026, −6.477] |

System level, metric means against BT scores (8 models; 9 systems with human):

| metric | Spearman, models | Kendall, models | Spearman, all | Kendall, all |
|---|---|---|---|---|
| SCRIBE v2 | 0.690 [−0.071, 0.786] | 0.500 [−0.071, 0.643] | 0.533 [−0.067, 0.700] | 0.389 [−0.001, 0.556] |
| SCRIBE v3 | 0.690 [0.119, 0.833] | 0.500 [0.071, 0.714] | 0.783 [0.367, 0.883] | 0.611 [0.278, 0.778] |
| SCRIBE v3.1 | 0.690 [0.119, 0.834] | 0.500 [0.071, 0.714] | 0.783 [0.383, 0.884] | 0.611 [0.278, 0.778] |
| SURPRISE | 0.857 [0.214, 0.929] | 0.643 [0.143, 0.786] | 0.900 [0.450, 0.950] | 0.722 [0.333, 0.833] |
| MATCH | 0.643 [0.071, 0.834] | 0.429 [0.071, 0.714] | 0.750 [0.350, 0.883] | 0.556 [0.278, 0.778] |
| SURPRISE 3.1 | 0.857 [0.262, 0.929] | 0.643 [0.143, 0.857] | 0.900 [0.483, 0.950] | 0.722 [0.333, 0.889] |
| MATCH 3.1 | 0.643 [0.071, 0.857] | 0.429 [0.071, 0.714] | 0.750 [0.350, 0.900] | 0.556 [0.278, 0.778] |
| min of 3 | 0.690 [0.119, 0.834] | 0.500 [0.071, 0.714] | 0.750 [0.317, 0.867] | 0.556 [0.222, 0.722] |

## Agreement between workers

No pair of texts was judged more than once (3240 judgments, 214 workers), so agreement between workers cannot be measured on this file.
