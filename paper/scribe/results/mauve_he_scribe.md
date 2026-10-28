# SCRIBE on all 3,240 judgments

Analysis `mauve_he_scribe` (4. MAUVE human evaluation (pairwise)). meta.py `pairs`.
Inputs: SP/data/mauve_he/texts.jsonl, SP/data/mauve_he/pairs.jsonl, SP/run/human/mauve_he.v2.jsonl, SP/run/human/mauve_he.v2p.jsonl, SP/run/ts/mauve_he.ts.jsonl.
MAUVE paper (Pillutla et al. 2021), published system-level Spearman between MAUVE (GPT-2-large features) and the Bradley-Terry scores over the 8 models, judgments with te > 25 s: q1 0.810, q2 0.857, q3 0.952.

3240 judged pairs over 6116 texts in 9 systems (gpt2 p0.9, gpt2 p1.0, gpt2-large p0.95, gpt2-large p1.0, gpt2-medium p0.9, gpt2-medium p1.0, gpt2-xl p0.95, gpt2-xl p1.0, human). no time filter. Codes: 2a definitely A, 1a slightly A or tie, 1b slightly B, 2b definitely B; a 1a or 2a is a win for A (as in the MAUVE paper). Metric agreement: the metric prefers the text with the higher oriented score; exact metric ties and missing scores are dropped; Wilson 95% intervals. Bradley-Terry: penalised maximum likelihood (ridge 0.01), reported as 100 x log-strength centred at 0 (the MAUVE paper's scale). System-level correlations are between the metric's mean oriented score per system (over the unique texts of the pairs) and the BT scores, over the models (human excluded, as in the MAUVE paper) and over all systems. Bootstrap: 1000 resamples of the pairs, seed 20261028; each resample refits BT and recomputes the system means over the texts it contains.

## Question q1

Per-pair agreement with the people's choice:

| metric | n | agreement [95% interval] | n definite | definite only [95% interval] | metric ties | missing |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 3176 | 0.531 [0.514, 0.549] | 1189 | 0.550 [0.522, 0.578] | 64 | 0 |
| SCRIBE v2p | 3234 | 0.521 [0.504, 0.538] | 1215 | 0.535 [0.507, 0.563] | 6 | 0 |
| SCRIBE v2, two-sided (exploratory) | 3223 | 0.531 [0.514, 0.548] | 1212 | 0.556 [0.528, 0.584] | 17 | 0 |
| SCRIBE v1 | 3234 | 0.511 [0.494, 0.529] | 1215 | 0.521 [0.493, 0.549] | 6 | 0 |
| surface S | 3234 | 0.531 [0.513, 0.548] | 1215 | 0.535 [0.507, 0.563] | 6 | 0 |

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
| SCRIBE v2 | 0.571 [−0.119, 0.810] | 0.429 [−0.071, 0.643] | 0.433 [−0.100, 0.733] | 0.333 [−0.056, 0.611] |
| SCRIBE v2p | 0.524 [−0.143, 0.810] | 0.357 [−0.143, 0.643] | 0.617 [0.067, 0.834] | 0.444 [0.056, 0.667] |
| SCRIBE v2, two-sided (exploratory) | 0.381 [−0.119, 0.762] | 0.286 [−0.143, 0.643] | 0.383 [−0.050, 0.750] | 0.278 [−0.056, 0.611] |
| SCRIBE v1 | 0.333 [−0.143, 0.786] | 0.286 [−0.143, 0.643] | 0.267 [−0.167, 0.667] | 0.222 [−0.111, 0.500] |
| surface S | 0.286 [−0.190, 0.738] | 0.214 [−0.214, 0.571] | 0.167 [−0.233, 0.533] | 0.111 [−0.222, 0.389] |

## Question q2

Per-pair agreement with the people's choice:

| metric | n | agreement [95% interval] | n definite | definite only [95% interval] | metric ties | missing |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 3176 | 0.527 [0.510, 0.544] | 1261 | 0.553 [0.525, 0.580] | 64 | 0 |
| SCRIBE v2p | 3234 | 0.531 [0.514, 0.548] | 1286 | 0.558 [0.530, 0.584] | 6 | 0 |
| SCRIBE v2, two-sided (exploratory) | 3223 | 0.529 [0.511, 0.546] | 1282 | 0.530 [0.502, 0.557] | 17 | 0 |
| SCRIBE v1 | 3234 | 0.510 [0.493, 0.527] | 1286 | 0.531 [0.504, 0.558] | 6 | 0 |
| surface S | 3234 | 0.530 [0.513, 0.547] | 1286 | 0.552 [0.525, 0.579] | 6 | 0 |

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
| SCRIBE v2 | 0.810 [0.238, 0.929] | 0.714 [0.143, 0.787] | 0.650 [0.183, 0.817] | 0.556 [0.111, 0.667] |
| SCRIBE v2p | 0.786 [0.286, 0.929] | 0.643 [0.143, 0.857] | 0.800 [0.433, 0.950] | 0.667 [0.278, 0.833] |
| SCRIBE v2, two-sided (exploratory) | 0.643 [0.214, 0.929] | 0.571 [0.071, 0.786] | 0.617 [0.250, 0.883] | 0.500 [0.167, 0.722] |
| SCRIBE v1 | 0.524 [0.214, 0.881] | 0.429 [0.071, 0.786] | 0.450 [0.167, 0.767] | 0.333 [0.056, 0.611] |
| surface S | 0.476 [0.143, 0.833] | 0.357 [0.000, 0.714] | 0.317 [0.066, 0.617] | 0.222 [−0.056, 0.500] |

## Question q3

Per-pair agreement with the people's choice:

| metric | n | agreement [95% interval] | n definite | definite only [95% interval] | metric ties | missing |
|---|---|---|---|---|---|---|
| SCRIBE v2 | 3176 | 0.531 [0.514, 0.549] | 1328 | 0.539 [0.512, 0.566] | 64 | 0 |
| SCRIBE v2p | 3234 | 0.531 [0.513, 0.548] | 1351 | 0.528 [0.501, 0.554] | 6 | 0 |
| SCRIBE v2, two-sided (exploratory) | 3223 | 0.521 [0.503, 0.538] | 1345 | 0.527 [0.500, 0.554] | 17 | 0 |
| SCRIBE v1 | 3234 | 0.506 [0.488, 0.523] | 1351 | 0.536 [0.509, 0.562] | 6 | 0 |
| surface S | 3234 | 0.525 [0.508, 0.543] | 1351 | 0.538 [0.511, 0.565] | 6 | 0 |

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
| SCRIBE v2 | 0.714 [−0.071, 0.810] | 0.571 [−0.071, 0.643] | 0.550 [−0.067, 0.717] | 0.444 [−0.056, 0.556] |
| SCRIBE v2p | 0.667 [0.048, 0.810] | 0.500 [0.000, 0.643] | 0.750 [0.300, 0.850] | 0.556 [0.222, 0.722] |
| SCRIBE v2, two-sided (exploratory) | 0.667 [0.024, 0.833] | 0.571 [0.000, 0.714] | 0.633 [0.117, 0.817] | 0.500 [0.111, 0.667] |
| SCRIBE v1 | 0.500 [−0.095, 0.738] | 0.429 [−0.071, 0.643] | 0.400 [−0.084, 0.600] | 0.333 [−0.056, 0.500] |
| surface S | 0.357 [−0.143, 0.690] | 0.357 [−0.143, 0.571] | 0.250 [−0.200, 0.483] | 0.222 [−0.167, 0.389] |

## Agreement between workers

No pair of texts was judged more than once (3240 judgments, 214 workers), so agreement between workers cannot be measured on this file.
