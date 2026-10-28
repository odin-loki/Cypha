# All per-text metrics on S40 (every family and strength of the first 40 sources)

Analysis `dose_S40_all` (3. Held-out corruptions, dose-response). meta.py `dose`.
Inputs: SP/run/baselines/subsets/corrupt_S40.jsonl, SP/run/fresh/corrupt.scribe.jsonl, SP/run/baselines/corrupt/stats.jsonl.
Not available yet (left out): R/corrupt_S40/ppl_gpt2.jsonl, R/corrupt_S40/cola.jsonl, R/corrupt_S40/coherence.jsonl, R/corrupt_S40/detector.jsonl.

2160 texts, families from `meta.family`, strengths from `meta.strength`. A family without its own strength-0 texts uses the shared base texts (strength 0, family ['', 'base', 'none'] or none): 0. Bootstrap: 1000 resamples within each strength, seed 20261028. Spearman uses the oriented score, so −1 means quality falls perfectly with strength; a step moves the right way when the oriented mean falls.

## Family: dropword

### SCRIBE v2 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.877 [0.822, 0.928] |
| 10 | 40 | 0 | 0.803 [0.731, 0.868] |
| 25 | 40 | 0 | 0.487 [0.383, 0.596] |
| 50 | 40 | 0 | 0.206 [0.134, 0.290] |
| 75 | 40 | 0 | 0.027 [0.016, 0.041] |
| 100 | 40 | 0 | 0.009 [0.002, 0.019] |

Spearman, samples: −0.876 [−0.903, −0.847]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.503 [0.432, 0.568] |
| 10 | 40 | 0 | 0.446 [0.377, 0.511] |
| 25 | 40 | 0 | 0.377 [0.315, 0.442] |
| 50 | 40 | 0 | 0.394 [0.329, 0.459] |
| 75 | 40 | 0 | 0.337 [0.266, 0.402] |
| 100 | 40 | 0 | 0.377 [0.294, 0.465] |

Spearman, samples: −0.202 [−0.331, −0.071]. Spearman, means: −0.714. Steps the right way: 3 of 5.

### surface S (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.897 [0.869, 0.920] |
| 10 | 40 | 0 | 0.900 [0.879, 0.923] |
| 25 | 40 | 0 | 0.894 [0.876, 0.912] |
| 50 | 40 | 0 | 0.925 [0.905, 0.942] |
| 75 | 40 | 0 | 0.917 [0.901, 0.932] |
| 100 | 40 | 0 | 0.854 [0.814, 0.892] |

Spearman, samples: −0.028 [−0.154, 0.092]. Spearman, means: −0.086. Steps the right way: 3 of 5.

### distinct-1 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.708, 0.758] |
| 10 | 40 | 0 | 0.746 [0.721, 0.771] |
| 25 | 40 | 0 | 0.779 [0.754, 0.801] |
| 50 | 40 | 0 | 0.820 [0.794, 0.844] |
| 75 | 40 | 0 | 0.877 [0.855, 0.899] |
| 100 | 40 | 0 | 0.988 [0.969, 1.000] |

Spearman, samples: 0.716 [0.641, 0.777]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-2 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.957 [0.940, 0.969] |
| 10 | 40 | 0 | 0.967 [0.955, 0.978] |
| 25 | 40 | 0 | 0.977 [0.968, 0.985] |
| 50 | 40 | 0 | 0.991 [0.985, 0.995] |
| 75 | 40 | 0 | 0.999 [0.997, 1.000] |
| 100 | 40 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.662 [0.592, 0.721]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-3 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.985 [0.976, 0.992] |
| 10 | 40 | 0 | 0.992 [0.987, 0.997] |
| 25 | 40 | 0 | 0.995 [0.991, 0.999] |
| 50 | 40 | 0 | 0.999 [0.997, 1.000] |
| 75 | 40 | 0 | 1.000 [1.000, 1.000] |
| 100 | 40 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.416 [0.333, 0.494]. Spearman, means: 0.986. Steps the right way: 0 of 5.

### distinct-4 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.995 [0.990, 0.998] |
| 10 | 40 | 0 | 0.998 [0.996, 1.000] |
| 25 | 40 | 0 | 0.998 [0.996, 1.000] |
| 50 | 40 | 0 | 1.000 [1.000, 1.000] |
| 75 | 40 | 0 | 1.000 [1.000, 1.000] |
| 100 | 27 | 13 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.301 [0.210, 0.384]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### rep-2 (lower is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.043 [0.031, 0.060] |
| 10 | 40 | 0 | 0.033 [0.022, 0.045] |
| 25 | 40 | 0 | 0.023 [0.015, 0.032] |
| 50 | 40 | 0 | 0.009 [0.005, 0.015] |
| 75 | 40 | 0 | 0.001 [0.000, 0.003] |
| 100 | 40 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.662 [0.592, 0.721]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### rep-3 (lower is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.015 [0.008, 0.024] |
| 10 | 40 | 0 | 0.008 [0.003, 0.013] |
| 25 | 40 | 0 | 0.005 [0.001, 0.009] |
| 50 | 40 | 0 | 0.001 [0.000, 0.003] |
| 75 | 40 | 0 | 0.000 [0.000, 0.000] |
| 100 | 40 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.416 [0.333, 0.494]. Spearman, means: 0.986. Steps the right way: 0 of 5.

### rep-4 (lower is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.010] |
| 10 | 40 | 0 | 0.002 [0.000, 0.004] |
| 25 | 40 | 0 | 0.002 [0.000, 0.004] |
| 50 | 40 | 0 | 0.000 [0.000, 0.000] |
| 75 | 40 | 0 | 0.000 [0.000, 0.000] |
| 100 | 27 | 13 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.301 [0.210, 0.384]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### seq-rep-4 (lower is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.010] |
| 10 | 40 | 0 | 0.002 [0.000, 0.004] |
| 25 | 40 | 0 | 0.002 [0.000, 0.004] |
| 50 | 40 | 0 | 0.000 [0.000, 0.000] |
| 75 | 40 | 0 | 0.000 [0.000, 0.000] |
| 100 | 27 | 13 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.301 [0.210, 0.384]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### diversity (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.939 [0.913, 0.959] |
| 10 | 40 | 0 | 0.959 [0.940, 0.975] |
| 25 | 40 | 0 | 0.972 [0.957, 0.983] |
| 50 | 40 | 0 | 0.989 [0.983, 0.995] |
| 75 | 40 | 0 | 0.999 [0.997, 1.000] |
| 100 | 27 | 13 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.646 [0.573, 0.709]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### type-token ratio (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.708, 0.758] |
| 10 | 40 | 0 | 0.746 [0.721, 0.771] |
| 25 | 40 | 0 | 0.779 [0.754, 0.801] |
| 50 | 40 | 0 | 0.820 [0.794, 0.844] |
| 75 | 40 | 0 | 0.877 [0.855, 0.899] |
| 100 | 40 | 0 | 0.988 [0.969, 1.000] |

Spearman, samples: 0.716 [0.641, 0.777]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### gzip ratio (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.620 [0.605, 0.635] |
| 10 | 40 | 0 | 0.638 [0.624, 0.653] |
| 25 | 40 | 0 | 0.669 [0.654, 0.683] |
| 50 | 40 | 0 | 0.748 [0.734, 0.763] |
| 75 | 40 | 0 | 0.934 [0.920, 0.948] |
| 100 | 40 | 0 | 1.573 [1.517, 1.631] |

Spearman, samples: 0.910 [0.886, 0.930]. Spearman, means: 1.000. Steps the right way: 0 of 5.

## Family: nounswap

### SCRIBE v2 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.877 [0.818, 0.925] |
| 10 | 40 | 0 | 0.867 [0.819, 0.914] |
| 25 | 40 | 0 | 0.792 [0.732, 0.846] |
| 50 | 40 | 0 | 0.481 [0.398, 0.558] |
| 75 | 40 | 0 | 0.138 [0.088, 0.191] |
| 100 | 40 | 0 | 0.055 [0.029, 0.085] |

Spearman, samples: −0.832 [−0.865, −0.798]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.503 [0.434, 0.569] |
| 10 | 40 | 0 | 0.491 [0.424, 0.551] |
| 25 | 40 | 0 | 0.430 [0.373, 0.494] |
| 50 | 40 | 0 | 0.307 [0.261, 0.357] |
| 75 | 40 | 0 | 0.162 [0.142, 0.183] |
| 100 | 40 | 0 | 0.102 [0.091, 0.114] |

Spearman, samples: −0.678 [−0.759, −0.585]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.897 [0.871, 0.920] |
| 10 | 40 | 0 | 0.906 [0.884, 0.925] |
| 25 | 40 | 0 | 0.907 [0.891, 0.923] |
| 50 | 40 | 0 | 0.909 [0.898, 0.922] |
| 75 | 40 | 0 | 0.890 [0.878, 0.900] |
| 100 | 40 | 0 | 0.880 [0.868, 0.890] |

Spearman, samples: −0.256 [−0.384, −0.124]. Spearman, means: −0.429. Steps the right way: 2 of 5.

### distinct-1 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.708, 0.759] |
| 10 | 40 | 0 | 0.747 [0.724, 0.769] |
| 25 | 40 | 0 | 0.759 [0.738, 0.780] |
| 50 | 40 | 0 | 0.777 [0.759, 0.794] |
| 75 | 40 | 0 | 0.792 [0.776, 0.809] |
| 100 | 40 | 0 | 0.794 [0.777, 0.811] |

Spearman, samples: 0.304 [0.185, 0.415]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-2 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.957 [0.941, 0.970] |
| 10 | 40 | 0 | 0.964 [0.951, 0.976] |
| 25 | 40 | 0 | 0.973 [0.963, 0.981] |
| 50 | 40 | 0 | 0.980 [0.973, 0.986] |
| 75 | 40 | 0 | 0.986 [0.980, 0.990] |
| 100 | 40 | 0 | 0.986 [0.981, 0.991] |

Spearman, samples: 0.342 [0.226, 0.455]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-3 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.985 [0.976, 0.992] |
| 10 | 40 | 0 | 0.989 [0.982, 0.994] |
| 25 | 40 | 0 | 0.994 [0.988, 0.997] |
| 50 | 40 | 0 | 0.997 [0.995, 0.999] |
| 75 | 40 | 0 | 0.998 [0.997, 0.999] |
| 100 | 40 | 0 | 0.999 [0.997, 1.000] |

Spearman, samples: 0.310 [0.191, 0.422]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-4 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.995 [0.990, 0.998] |
| 10 | 40 | 0 | 0.997 [0.994, 0.999] |
| 25 | 40 | 0 | 0.998 [0.996, 0.999] |
| 50 | 40 | 0 | 1.000 [1.000, 1.000] |
| 75 | 40 | 0 | 1.000 [1.000, 1.000] |
| 100 | 40 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.304 [0.221, 0.389]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### rep-2 (lower is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.043 [0.030, 0.059] |
| 10 | 40 | 0 | 0.036 [0.024, 0.049] |
| 25 | 40 | 0 | 0.027 [0.019, 0.037] |
| 50 | 40 | 0 | 0.020 [0.014, 0.027] |
| 75 | 40 | 0 | 0.014 [0.010, 0.020] |
| 100 | 40 | 0 | 0.014 [0.009, 0.019] |

Spearman, samples: 0.342 [0.226, 0.455]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### rep-3 (lower is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.015 [0.008, 0.024] |
| 10 | 40 | 0 | 0.011 [0.006, 0.018] |
| 25 | 40 | 0 | 0.006 [0.003, 0.012] |
| 50 | 40 | 0 | 0.003 [0.001, 0.005] |
| 75 | 40 | 0 | 0.002 [0.001, 0.003] |
| 100 | 40 | 0 | 0.001 [0.000, 0.003] |

Spearman, samples: 0.310 [0.191, 0.422]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### rep-4 (lower is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.010] |
| 10 | 40 | 0 | 0.003 [0.001, 0.006] |
| 25 | 40 | 0 | 0.002 [0.001, 0.004] |
| 50 | 40 | 0 | 0.000 [0.000, 0.000] |
| 75 | 40 | 0 | 0.000 [0.000, 0.000] |
| 100 | 40 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.304 [0.221, 0.389]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### seq-rep-4 (lower is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.010] |
| 10 | 40 | 0 | 0.003 [0.001, 0.006] |
| 25 | 40 | 0 | 0.002 [0.001, 0.004] |
| 50 | 40 | 0 | 0.000 [0.000, 0.000] |
| 75 | 40 | 0 | 0.000 [0.000, 0.000] |
| 100 | 40 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.304 [0.221, 0.389]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### diversity (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.939 [0.913, 0.961] |
| 10 | 40 | 0 | 0.951 [0.930, 0.969] |
| 25 | 40 | 0 | 0.965 [0.950, 0.977] |
| 50 | 40 | 0 | 0.977 [0.969, 0.985] |
| 75 | 40 | 0 | 0.984 [0.978, 0.990] |
| 100 | 40 | 0 | 0.985 [0.979, 0.990] |

Spearman, samples: 0.333 [0.213, 0.444]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### type-token ratio (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.708, 0.759] |
| 10 | 40 | 0 | 0.747 [0.724, 0.769] |
| 25 | 40 | 0 | 0.759 [0.738, 0.780] |
| 50 | 40 | 0 | 0.777 [0.759, 0.794] |
| 75 | 40 | 0 | 0.792 [0.776, 0.809] |
| 100 | 40 | 0 | 0.794 [0.777, 0.811] |

Spearman, samples: 0.304 [0.185, 0.415]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### gzip ratio (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.620 [0.604, 0.634] |
| 10 | 40 | 0 | 0.627 [0.614, 0.640] |
| 25 | 40 | 0 | 0.635 [0.623, 0.648] |
| 50 | 40 | 0 | 0.647 [0.637, 0.656] |
| 75 | 40 | 0 | 0.656 [0.647, 0.666] |
| 100 | 40 | 0 | 0.659 [0.651, 0.667] |

Spearman, samples: 0.387 [0.275, 0.497]. Spearman, means: 1.000. Steps the right way: 0 of 5.

## Family: offtopic

### SCRIBE v2 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.877 [0.826, 0.924] |
| 10 | 40 | 0 | 0.875 [0.817, 0.923] |
| 25 | 40 | 0 | 0.729 [0.662, 0.791] |
| 50 | 40 | 0 | 0.449 [0.388, 0.513] |
| 75 | 40 | 0 | 0.359 [0.315, 0.408] |
| 100 | 40 | 0 | 0.346 [0.311, 0.384] |

Spearman, samples: −0.742 [−0.793, −0.683]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.503 [0.439, 0.570] |
| 10 | 40 | 0 | 0.507 [0.438, 0.573] |
| 25 | 40 | 0 | 0.457 [0.397, 0.516] |
| 50 | 40 | 0 | 0.271 [0.232, 0.312] |
| 75 | 40 | 0 | 0.225 [0.191, 0.260] |
| 100 | 40 | 0 | 0.229 [0.199, 0.259] |

Spearman, samples: −0.519 [−0.622, −0.415]. Spearman, means: −0.886. Steps the right way: 3 of 5.

### surface S (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.897 [0.873, 0.920] |
| 10 | 40 | 0 | 0.897 [0.871, 0.921] |
| 25 | 40 | 0 | 0.908 [0.894, 0.921] |
| 50 | 40 | 0 | 0.901 [0.885, 0.915] |
| 75 | 40 | 0 | 0.900 [0.885, 0.914] |
| 100 | 40 | 0 | 0.900 [0.882, 0.918] |

Spearman, samples: −0.066 [−0.195, 0.068]. Spearman, means: 0.429. Steps the right way: 3 of 5.

### distinct-1 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.706, 0.759] |
| 10 | 40 | 0 | 0.734 [0.710, 0.759] |
| 25 | 40 | 0 | 0.763 [0.743, 0.781] |
| 50 | 40 | 0 | 0.793 [0.775, 0.811] |
| 75 | 40 | 0 | 0.793 [0.777, 0.807] |
| 100 | 40 | 0 | 0.779 [0.752, 0.805] |

Spearman, samples: 0.240 [0.106, 0.359]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### distinct-2 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.957 [0.941, 0.969] |
| 10 | 40 | 0 | 0.956 [0.942, 0.970] |
| 25 | 40 | 0 | 0.967 [0.955, 0.977] |
| 50 | 40 | 0 | 0.981 [0.973, 0.987] |
| 75 | 40 | 0 | 0.983 [0.977, 0.988] |
| 100 | 40 | 0 | 0.971 [0.953, 0.983] |

Spearman, samples: 0.246 [0.121, 0.360]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### distinct-3 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.985 [0.976, 0.992] |
| 10 | 40 | 0 | 0.985 [0.977, 0.992] |
| 25 | 40 | 0 | 0.990 [0.984, 0.995] |
| 50 | 40 | 0 | 0.995 [0.992, 0.998] |
| 75 | 40 | 0 | 0.999 [0.997, 1.000] |
| 100 | 40 | 0 | 0.990 [0.974, 0.999] |

Spearman, samples: 0.292 [0.163, 0.414]. Spearman, means: 0.638. Steps the right way: 1 of 5.

### distinct-4 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.995 [0.990, 0.998] |
| 10 | 40 | 0 | 0.995 [0.990, 0.998] |
| 25 | 40 | 0 | 0.997 [0.993, 1.000] |
| 50 | 40 | 0 | 0.998 [0.996, 0.999] |
| 75 | 40 | 0 | 1.000 [0.999, 1.000] |
| 100 | 40 | 0 | 0.992 [0.976, 0.999] |

Spearman, samples: 0.135 [−0.001, 0.265]. Spearman, means: 0.116. Steps the right way: 1 of 5.

### rep-2 (lower is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.043 [0.031, 0.059] |
| 10 | 40 | 0 | 0.044 [0.030, 0.058] |
| 25 | 40 | 0 | 0.033 [0.023, 0.045] |
| 50 | 40 | 0 | 0.019 [0.013, 0.027] |
| 75 | 40 | 0 | 0.017 [0.012, 0.023] |
| 100 | 40 | 0 | 0.029 [0.017, 0.047] |

Spearman, samples: 0.246 [0.121, 0.360]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### rep-3 (lower is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.015 [0.008, 0.024] |
| 10 | 40 | 0 | 0.015 [0.008, 0.023] |
| 25 | 40 | 0 | 0.010 [0.005, 0.016] |
| 50 | 40 | 0 | 0.005 [0.002, 0.008] |
| 75 | 40 | 0 | 0.001 [0.000, 0.003] |
| 100 | 40 | 0 | 0.010 [0.001, 0.026] |

Spearman, samples: 0.292 [0.163, 0.414]. Spearman, means: 0.638. Steps the right way: 1 of 5.

### rep-4 (lower is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.010] |
| 10 | 40 | 0 | 0.005 [0.002, 0.010] |
| 25 | 40 | 0 | 0.003 [0.000, 0.007] |
| 50 | 40 | 0 | 0.002 [0.001, 0.004] |
| 75 | 40 | 0 | 0.000 [0.000, 0.001] |
| 100 | 40 | 0 | 0.008 [0.001, 0.024] |

Spearman, samples: 0.135 [−0.001, 0.265]. Spearman, means: 0.116. Steps the right way: 1 of 5.

### seq-rep-4 (lower is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.010] |
| 10 | 40 | 0 | 0.005 [0.002, 0.010] |
| 25 | 40 | 0 | 0.003 [0.000, 0.007] |
| 50 | 40 | 0 | 0.002 [0.001, 0.004] |
| 75 | 40 | 0 | 0.000 [0.000, 0.001] |
| 100 | 40 | 0 | 0.008 [0.001, 0.024] |

Spearman, samples: 0.135 [−0.001, 0.265]. Spearman, means: 0.116. Steps the right way: 1 of 5.

### diversity (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.939 [0.913, 0.960] |
| 10 | 40 | 0 | 0.939 [0.915, 0.961] |
| 25 | 40 | 0 | 0.956 [0.937, 0.971] |
| 50 | 40 | 0 | 0.974 [0.963, 0.984] |
| 75 | 40 | 0 | 0.981 [0.974, 0.988] |
| 100 | 40 | 0 | 0.958 [0.920, 0.980] |

Spearman, samples: 0.258 [0.130, 0.373]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### type-token ratio (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.706, 0.759] |
| 10 | 40 | 0 | 0.734 [0.710, 0.759] |
| 25 | 40 | 0 | 0.763 [0.743, 0.781] |
| 50 | 40 | 0 | 0.793 [0.775, 0.811] |
| 75 | 40 | 0 | 0.793 [0.777, 0.807] |
| 100 | 40 | 0 | 0.779 [0.752, 0.805] |

Spearman, samples: 0.240 [0.106, 0.359]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### gzip ratio (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.620 [0.606, 0.636] |
| 10 | 40 | 0 | 0.620 [0.606, 0.636] |
| 25 | 40 | 0 | 0.641 [0.630, 0.651] |
| 50 | 40 | 0 | 0.666 [0.652, 0.680] |
| 75 | 40 | 0 | 0.656 [0.643, 0.667] |
| 100 | 40 | 0 | 0.661 [0.642, 0.677] |

Spearman, samples: 0.313 [0.187, 0.430]. Spearman, means: 0.829. Steps the right way: 1 of 5.

## Family: reorder

### SCRIBE v2 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.877 [0.823, 0.928] |
| 10 | 40 | 0 | 0.877 [0.815, 0.925] |
| 25 | 40 | 0 | 0.880 [0.823, 0.928] |
| 50 | 40 | 0 | 0.881 [0.823, 0.930] |
| 75 | 40 | 0 | 0.868 [0.814, 0.917] |
| 100 | 40 | 0 | 0.864 [0.804, 0.914] |

Spearman, samples: −0.037 [−0.178, 0.099]. Spearman, means: −0.464. Steps the right way: 2 of 5.

### SCRIBE v1 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.503 [0.433, 0.568] |
| 10 | 40 | 0 | 0.503 [0.431, 0.568] |
| 25 | 40 | 0 | 0.505 [0.443, 0.571] |
| 50 | 40 | 0 | 0.497 [0.436, 0.557] |
| 75 | 40 | 0 | 0.498 [0.431, 0.561] |
| 100 | 40 | 0 | 0.492 [0.422, 0.558] |

Spearman, samples: −0.024 [−0.147, 0.100]. Spearman, means: −0.754. Steps the right way: 2 of 5.

### surface S (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.897 [0.870, 0.921] |
| 10 | 40 | 0 | 0.897 [0.870, 0.920] |
| 25 | 40 | 0 | 0.897 [0.871, 0.920] |
| 50 | 40 | 0 | 0.897 [0.873, 0.918] |
| 75 | 40 | 0 | 0.897 [0.870, 0.920] |
| 100 | 40 | 0 | 0.897 [0.869, 0.919] |

Spearman, samples: 0.001 [−0.119, 0.121]. Spearman, means: −0.058. Steps the right way: 2 of 5.

### distinct-1 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.709, 0.758] |
| 10 | 40 | 0 | 0.734 [0.708, 0.758] |
| 25 | 40 | 0 | 0.734 [0.709, 0.760] |
| 50 | 40 | 0 | 0.734 [0.711, 0.759] |
| 75 | 40 | 0 | 0.734 [0.707, 0.758] |
| 100 | 40 | 0 | 0.734 [0.709, 0.759] |

Spearman, samples: 0.000 [−0.118, 0.131]. Spearman, means: n/a. Steps the right way: 0 of 5.

### distinct-2 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.957 [0.941, 0.970] |
| 10 | 40 | 0 | 0.957 [0.942, 0.970] |
| 25 | 40 | 0 | 0.957 [0.941, 0.971] |
| 50 | 40 | 0 | 0.957 [0.941, 0.970] |
| 75 | 40 | 0 | 0.957 [0.940, 0.971] |
| 100 | 40 | 0 | 0.957 [0.941, 0.970] |

Spearman, samples: 0.000 [−0.126, 0.132]. Spearman, means: n/a. Steps the right way: 0 of 5.

### distinct-3 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.985 [0.977, 0.992] |
| 10 | 40 | 0 | 0.985 [0.977, 0.992] |
| 25 | 40 | 0 | 0.985 [0.976, 0.993] |
| 50 | 40 | 0 | 0.985 [0.977, 0.992] |
| 75 | 40 | 0 | 0.985 [0.976, 0.992] |
| 100 | 40 | 0 | 0.985 [0.976, 0.992] |

Spearman, samples: 0.000 [−0.130, 0.133]. Spearman, means: n/a. Steps the right way: 0 of 5.

### distinct-4 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.995 [0.990, 0.998] |
| 10 | 40 | 0 | 0.995 [0.990, 0.998] |
| 25 | 40 | 0 | 0.995 [0.990, 0.998] |
| 50 | 40 | 0 | 0.995 [0.990, 0.998] |
| 75 | 40 | 0 | 0.995 [0.990, 0.998] |
| 100 | 40 | 0 | 0.995 [0.990, 0.998] |

Spearman, samples: 0.000 [−0.129, 0.129]. Spearman, means: n/a. Steps the right way: 0 of 5.

### rep-2 (lower is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.043 [0.030, 0.059] |
| 10 | 40 | 0 | 0.043 [0.030, 0.058] |
| 25 | 40 | 0 | 0.043 [0.029, 0.059] |
| 50 | 40 | 0 | 0.043 [0.030, 0.059] |
| 75 | 40 | 0 | 0.043 [0.029, 0.060] |
| 100 | 40 | 0 | 0.043 [0.030, 0.059] |

Spearman, samples: 0.000 [−0.126, 0.132]. Spearman, means: n/a. Steps the right way: 0 of 5.

### rep-3 (lower is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.015 [0.008, 0.023] |
| 10 | 40 | 0 | 0.015 [0.008, 0.023] |
| 25 | 40 | 0 | 0.015 [0.007, 0.024] |
| 50 | 40 | 0 | 0.015 [0.008, 0.023] |
| 75 | 40 | 0 | 0.015 [0.008, 0.024] |
| 100 | 40 | 0 | 0.015 [0.008, 0.024] |

Spearman, samples: 0.000 [−0.130, 0.133]. Spearman, means: n/a. Steps the right way: 0 of 5.

### rep-4 (lower is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.010] |
| 10 | 40 | 0 | 0.005 [0.002, 0.010] |
| 25 | 40 | 0 | 0.005 [0.002, 0.010] |
| 50 | 40 | 0 | 0.005 [0.002, 0.010] |
| 75 | 40 | 0 | 0.005 [0.002, 0.010] |
| 100 | 40 | 0 | 0.005 [0.002, 0.010] |

Spearman, samples: 0.000 [−0.129, 0.129]. Spearman, means: n/a. Steps the right way: 0 of 5.

### seq-rep-4 (lower is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.010] |
| 10 | 40 | 0 | 0.005 [0.002, 0.010] |
| 25 | 40 | 0 | 0.005 [0.002, 0.010] |
| 50 | 40 | 0 | 0.005 [0.002, 0.010] |
| 75 | 40 | 0 | 0.005 [0.002, 0.010] |
| 100 | 40 | 0 | 0.005 [0.002, 0.010] |

Spearman, samples: 0.000 [−0.129, 0.129]. Spearman, means: n/a. Steps the right way: 0 of 5.

### diversity (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.939 [0.914, 0.961] |
| 10 | 40 | 0 | 0.939 [0.915, 0.960] |
| 25 | 40 | 0 | 0.939 [0.914, 0.962] |
| 50 | 40 | 0 | 0.939 [0.915, 0.961] |
| 75 | 40 | 0 | 0.939 [0.913, 0.961] |
| 100 | 40 | 0 | 0.939 [0.914, 0.960] |

Spearman, samples: 0.000 [−0.127, 0.135]. Spearman, means: n/a. Steps the right way: 0 of 5.

### type-token ratio (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.709, 0.758] |
| 10 | 40 | 0 | 0.734 [0.708, 0.758] |
| 25 | 40 | 0 | 0.734 [0.709, 0.760] |
| 50 | 40 | 0 | 0.734 [0.711, 0.759] |
| 75 | 40 | 0 | 0.734 [0.707, 0.758] |
| 100 | 40 | 0 | 0.734 [0.709, 0.759] |

Spearman, samples: 0.000 [−0.118, 0.131]. Spearman, means: n/a. Steps the right way: 0 of 5.

### gzip ratio (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.620 [0.605, 0.635] |
| 10 | 40 | 0 | 0.620 [0.605, 0.634] |
| 25 | 40 | 0 | 0.620 [0.605, 0.635] |
| 50 | 40 | 0 | 0.621 [0.606, 0.636] |
| 75 | 40 | 0 | 0.621 [0.605, 0.636] |
| 100 | 40 | 0 | 0.620 [0.605, 0.636] |

Spearman, samples: 0.004 [−0.126, 0.136]. Spearman, means: 0.638. Steps the right way: 2 of 5.

## Family: repeat

### SCRIBE v2 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.877 [0.820, 0.927] |
| 10 | 40 | 0 | 0.856 [0.785, 0.920] |
| 25 | 40 | 0 | 0.315 [0.186, 0.451] |
| 50 | 40 | 0 | 0.000 [0.000, 0.000] |
| 75 | 40 | 0 | 0.000 [0.000, 0.000] |
| 100 | 40 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: −0.920 [−0.936, −0.902]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.503 [0.436, 0.566] |
| 10 | 40 | 0 | 0.490 [0.420, 0.567] |
| 25 | 40 | 0 | 0.160 [0.089, 0.239] |
| 50 | 40 | 0 | 0.001 [0.001, 0.002] |
| 75 | 40 | 0 | 0.001 [0.001, 0.001] |
| 100 | 40 | 0 | 0.001 [0.001, 0.001] |

Spearman, samples: −0.905 [−0.924, −0.887]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.897 [0.872, 0.920] |
| 10 | 40 | 0 | 0.892 [0.865, 0.918] |
| 25 | 40 | 0 | 0.732 [0.676, 0.789] |
| 50 | 40 | 0 | 0.430 [0.384, 0.475] |
| 75 | 40 | 0 | 0.241 [0.218, 0.264] |
| 100 | 40 | 0 | 0.128 [0.115, 0.140] |

Spearman, samples: −0.903 [−0.922, −0.884]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### distinct-1 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.707, 0.757] |
| 10 | 40 | 0 | 0.731 [0.706, 0.757] |
| 25 | 40 | 0 | 0.644 [0.607, 0.680] |
| 50 | 40 | 0 | 0.473 [0.440, 0.506] |
| 75 | 40 | 0 | 0.342 [0.323, 0.365] |
| 100 | 40 | 0 | 0.232 [0.215, 0.247] |

Spearman, samples: −0.880 [−0.903, −0.857]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### distinct-2 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.957 [0.941, 0.970] |
| 10 | 40 | 0 | 0.953 [0.936, 0.969] |
| 25 | 40 | 0 | 0.825 [0.781, 0.865] |
| 50 | 40 | 0 | 0.582 [0.545, 0.619] |
| 75 | 40 | 0 | 0.407 [0.382, 0.434] |
| 100 | 40 | 0 | 0.260 [0.241, 0.277] |

Spearman, samples: −0.909 [−0.926, −0.890]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### distinct-3 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.985 [0.976, 0.992] |
| 10 | 40 | 0 | 0.981 [0.970, 0.991] |
| 25 | 40 | 0 | 0.851 [0.808, 0.890] |
| 50 | 40 | 0 | 0.604 [0.568, 0.641] |
| 75 | 40 | 0 | 0.422 [0.395, 0.451] |
| 100 | 40 | 0 | 0.265 [0.246, 0.283] |

Spearman, samples: −0.917 [−0.934, −0.899]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### distinct-4 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.995 [0.990, 0.998] |
| 10 | 40 | 0 | 0.991 [0.982, 0.998] |
| 25 | 40 | 0 | 0.864 [0.822, 0.902] |
| 50 | 40 | 0 | 0.620 [0.584, 0.656] |
| 75 | 40 | 0 | 0.434 [0.406, 0.466] |
| 100 | 40 | 0 | 0.269 [0.250, 0.287] |

Spearman, samples: −0.927 [−0.943, −0.912]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### rep-2 (lower is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.043 [0.030, 0.059] |
| 10 | 40 | 0 | 0.047 [0.031, 0.064] |
| 25 | 40 | 0 | 0.175 [0.135, 0.219] |
| 50 | 40 | 0 | 0.418 [0.381, 0.455] |
| 75 | 40 | 0 | 0.593 [0.566, 0.618] |
| 100 | 40 | 0 | 0.740 [0.723, 0.759] |

Spearman, samples: −0.909 [−0.926, −0.890]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### rep-3 (lower is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.015 [0.008, 0.024] |
| 10 | 40 | 0 | 0.019 [0.009, 0.030] |
| 25 | 40 | 0 | 0.149 [0.110, 0.192] |
| 50 | 40 | 0 | 0.396 [0.359, 0.432] |
| 75 | 40 | 0 | 0.578 [0.549, 0.605] |
| 100 | 40 | 0 | 0.735 [0.717, 0.754] |

Spearman, samples: −0.917 [−0.934, −0.899]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### rep-4 (lower is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.010] |
| 10 | 40 | 0 | 0.009 [0.002, 0.018] |
| 25 | 40 | 0 | 0.136 [0.098, 0.178] |
| 50 | 40 | 0 | 0.380 [0.344, 0.416] |
| 75 | 40 | 0 | 0.566 [0.534, 0.594] |
| 100 | 40 | 0 | 0.731 [0.713, 0.750] |

Spearman, samples: −0.927 [−0.943, −0.912]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### seq-rep-4 (lower is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.010] |
| 10 | 40 | 0 | 0.009 [0.002, 0.018] |
| 25 | 40 | 0 | 0.136 [0.098, 0.178] |
| 50 | 40 | 0 | 0.380 [0.344, 0.416] |
| 75 | 40 | 0 | 0.566 [0.534, 0.594] |
| 100 | 40 | 0 | 0.731 [0.713, 0.750] |

Spearman, samples: −0.927 [−0.943, −0.912]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### diversity (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.939 [0.914, 0.960] |
| 10 | 40 | 0 | 0.929 [0.898, 0.956] |
| 25 | 40 | 0 | 0.646 [0.555, 0.726] |
| 50 | 40 | 0 | 0.240 [0.199, 0.283] |
| 75 | 40 | 0 | 0.085 [0.067, 0.108] |
| 100 | 40 | 0 | 0.021 [0.017, 0.025] |

Spearman, samples: −0.912 [−0.929, −0.893]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### type-token ratio (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.707, 0.757] |
| 10 | 40 | 0 | 0.731 [0.706, 0.757] |
| 25 | 40 | 0 | 0.644 [0.607, 0.680] |
| 50 | 40 | 0 | 0.473 [0.440, 0.506] |
| 75 | 40 | 0 | 0.342 [0.323, 0.365] |
| 100 | 40 | 0 | 0.232 [0.215, 0.247] |

Spearman, samples: −0.880 [−0.903, −0.857]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### gzip ratio (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.620 [0.604, 0.634] |
| 10 | 40 | 0 | 0.617 [0.602, 0.635] |
| 25 | 40 | 0 | 0.547 [0.523, 0.572] |
| 50 | 40 | 0 | 0.419 [0.393, 0.444] |
| 75 | 40 | 0 | 0.326 [0.307, 0.348] |
| 100 | 40 | 0 | 0.244 [0.225, 0.261] |

Spearman, samples: −0.879 [−0.902, −0.855]. Spearman, means: −1.000. Steps the right way: 5 of 5.

## Family: subst

### SCRIBE v2 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.877 [0.824, 0.926] |
| 10 | 40 | 0 | 0.789 [0.701, 0.863] |
| 25 | 40 | 0 | 0.098 [0.047, 0.163] |
| 50 | 40 | 0 | 0.003 [0.003, 0.004] |
| 75 | 40 | 0 | 0.001 [0.001, 0.001] |
| 100 | 40 | 0 | 0.001 [0.001, 0.001] |

Spearman, samples: −0.935 [−0.946, −0.923]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.503 [0.439, 0.568] |
| 10 | 40 | 0 | 0.458 [0.395, 0.519] |
| 25 | 40 | 0 | 0.387 [0.331, 0.450] |
| 50 | 40 | 0 | 0.149 [0.123, 0.179] |
| 75 | 40 | 0 | 0.071 [0.065, 0.077] |
| 100 | 40 | 0 | 0.066 [0.060, 0.072] |

Spearman, samples: −0.780 [−0.831, −0.723]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.897 [0.873, 0.920] |
| 10 | 40 | 0 | 0.909 [0.889, 0.926] |
| 25 | 40 | 0 | 0.919 [0.905, 0.934] |
| 50 | 40 | 0 | 0.900 [0.886, 0.913] |
| 75 | 40 | 0 | 0.887 [0.880, 0.894] |
| 100 | 40 | 0 | 0.892 [0.886, 0.897] |

Spearman, samples: −0.267 [−0.380, −0.155]. Spearman, means: −0.543. Steps the right way: 2 of 5.

### distinct-1 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.708, 0.760] |
| 10 | 40 | 0 | 0.758 [0.734, 0.780] |
| 25 | 40 | 0 | 0.766 [0.746, 0.785] |
| 50 | 40 | 0 | 0.803 [0.790, 0.815] |
| 75 | 40 | 0 | 0.818 [0.805, 0.830] |
| 100 | 40 | 0 | 0.813 [0.799, 0.826] |

Spearman, samples: 0.432 [0.330, 0.527]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### distinct-2 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.957 [0.940, 0.970] |
| 10 | 40 | 0 | 0.970 [0.958, 0.980] |
| 25 | 40 | 0 | 0.980 [0.974, 0.986] |
| 50 | 40 | 0 | 0.995 [0.992, 0.997] |
| 75 | 40 | 0 | 0.997 [0.993, 0.999] |
| 100 | 40 | 0 | 0.994 [0.991, 0.997] |

Spearman, samples: 0.555 [0.462, 0.641]. Spearman, means: 0.829. Steps the right way: 1 of 5.

### distinct-3 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.985 [0.976, 0.992] |
| 10 | 40 | 0 | 0.992 [0.987, 0.996] |
| 25 | 40 | 0 | 0.997 [0.995, 0.999] |
| 50 | 40 | 0 | 1.000 [1.000, 1.000] |
| 75 | 40 | 0 | 1.000 [0.999, 1.000] |
| 100 | 40 | 0 | 1.000 [0.999, 1.000] |

Spearman, samples: 0.425 [0.318, 0.518]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### distinct-4 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.995 [0.991, 0.998] |
| 10 | 40 | 0 | 0.998 [0.995, 1.000] |
| 25 | 40 | 0 | 0.999 [0.999, 1.000] |
| 50 | 40 | 0 | 1.000 [1.000, 1.000] |
| 75 | 40 | 0 | 1.000 [1.000, 1.000] |
| 100 | 40 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.299 [0.205, 0.378]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### rep-2 (lower is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.043 [0.030, 0.060] |
| 10 | 40 | 0 | 0.030 [0.020, 0.042] |
| 25 | 40 | 0 | 0.020 [0.014, 0.026] |
| 50 | 40 | 0 | 0.005 [0.003, 0.008] |
| 75 | 40 | 0 | 0.003 [0.001, 0.007] |
| 100 | 40 | 0 | 0.006 [0.003, 0.009] |

Spearman, samples: 0.555 [0.462, 0.641]. Spearman, means: 0.829. Steps the right way: 1 of 5.

### rep-3 (lower is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.015 [0.008, 0.024] |
| 10 | 40 | 0 | 0.008 [0.004, 0.013] |
| 25 | 40 | 0 | 0.003 [0.001, 0.005] |
| 50 | 40 | 0 | 0.000 [0.000, 0.000] |
| 75 | 40 | 0 | 0.000 [0.000, 0.001] |
| 100 | 40 | 0 | 0.000 [0.000, 0.001] |

Spearman, samples: 0.425 [0.318, 0.518]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### rep-4 (lower is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.009] |
| 10 | 40 | 0 | 0.002 [0.000, 0.005] |
| 25 | 40 | 0 | 0.001 [0.000, 0.001] |
| 50 | 40 | 0 | 0.000 [0.000, 0.000] |
| 75 | 40 | 0 | 0.000 [0.000, 0.000] |
| 100 | 40 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.299 [0.205, 0.378]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### seq-rep-4 (lower is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.009] |
| 10 | 40 | 0 | 0.002 [0.000, 0.005] |
| 25 | 40 | 0 | 0.001 [0.000, 0.001] |
| 50 | 40 | 0 | 0.000 [0.000, 0.000] |
| 75 | 40 | 0 | 0.000 [0.000, 0.000] |
| 100 | 40 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.299 [0.205, 0.378]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### diversity (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.939 [0.913, 0.961] |
| 10 | 40 | 0 | 0.960 [0.943, 0.976] |
| 25 | 40 | 0 | 0.977 [0.968, 0.985] |
| 50 | 40 | 0 | 0.995 [0.992, 0.997] |
| 75 | 40 | 0 | 0.997 [0.993, 0.999] |
| 100 | 40 | 0 | 0.994 [0.990, 0.997] |

Spearman, samples: 0.557 [0.463, 0.644]. Spearman, means: 0.829. Steps the right way: 1 of 5.

### type-token ratio (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.708, 0.760] |
| 10 | 40 | 0 | 0.758 [0.734, 0.780] |
| 25 | 40 | 0 | 0.766 [0.746, 0.785] |
| 50 | 40 | 0 | 0.803 [0.790, 0.815] |
| 75 | 40 | 0 | 0.818 [0.805, 0.830] |
| 100 | 40 | 0 | 0.813 [0.799, 0.826] |

Spearman, samples: 0.432 [0.330, 0.527]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### gzip ratio (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.620 [0.605, 0.636] |
| 10 | 40 | 0 | 0.630 [0.616, 0.644] |
| 25 | 40 | 0 | 0.644 [0.633, 0.655] |
| 50 | 40 | 0 | 0.658 [0.648, 0.668] |
| 75 | 40 | 0 | 0.667 [0.659, 0.675] |
| 100 | 40 | 0 | 0.666 [0.659, 0.674] |

Spearman, samples: 0.471 [0.363, 0.572]. Spearman, means: 0.943. Steps the right way: 1 of 5.

## Family: swap

### SCRIBE v2 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.877 [0.825, 0.927] |
| 10 | 40 | 0 | 0.567 [0.443, 0.690] |
| 25 | 40 | 0 | 0.037 [0.021, 0.058] |
| 50 | 40 | 0 | 0.005 [0.005, 0.006] |
| 75 | 40 | 0 | 0.003 [0.003, 0.004] |
| 100 | 40 | 0 | 0.003 [0.003, 0.004] |

Spearman, samples: −0.908 [−0.924, −0.890]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.503 [0.433, 0.568] |
| 10 | 40 | 0 | 0.455 [0.393, 0.515] |
| 25 | 40 | 0 | 0.363 [0.307, 0.415] |
| 50 | 40 | 0 | 0.231 [0.182, 0.279] |
| 75 | 40 | 0 | 0.155 [0.119, 0.195] |
| 100 | 40 | 0 | 0.124 [0.099, 0.152] |

Spearman, samples: −0.617 [−0.694, −0.528]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.897 [0.870, 0.920] |
| 10 | 40 | 0 | 0.904 [0.883, 0.923] |
| 25 | 40 | 0 | 0.910 [0.890, 0.928] |
| 50 | 40 | 0 | 0.910 [0.891, 0.927] |
| 75 | 40 | 0 | 0.911 [0.892, 0.928] |
| 100 | 40 | 0 | 0.907 [0.886, 0.925] |

Spearman, samples: 0.024 [−0.098, 0.153]. Spearman, means: 0.657. Steps the right way: 1 of 5.

### distinct-1 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.709, 0.757] |
| 10 | 40 | 0 | 0.734 [0.708, 0.758] |
| 25 | 40 | 0 | 0.734 [0.707, 0.758] |
| 50 | 40 | 0 | 0.734 [0.709, 0.759] |
| 75 | 40 | 0 | 0.734 [0.707, 0.757] |
| 100 | 40 | 0 | 0.734 [0.707, 0.757] |

Spearman, samples: 0.000 [−0.130, 0.123]. Spearman, means: n/a. Steps the right way: 0 of 5.

### distinct-2 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.957 [0.941, 0.969] |
| 10 | 40 | 0 | 0.971 [0.961, 0.980] |
| 25 | 40 | 0 | 0.982 [0.973, 0.989] |
| 50 | 40 | 0 | 0.984 [0.977, 0.990] |
| 75 | 40 | 0 | 0.983 [0.974, 0.990] |
| 100 | 40 | 0 | 0.981 [0.971, 0.989] |

Spearman, samples: 0.257 [0.147, 0.369]. Spearman, means: 0.600. Steps the right way: 2 of 5.

### distinct-3 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.985 [0.976, 0.992] |
| 10 | 40 | 0 | 0.992 [0.988, 0.996] |
| 25 | 40 | 0 | 0.997 [0.994, 0.999] |
| 50 | 40 | 0 | 0.997 [0.995, 0.999] |
| 75 | 40 | 0 | 0.996 [0.993, 0.999] |
| 100 | 40 | 0 | 0.996 [0.991, 1.000] |

Spearman, samples: 0.279 [0.155, 0.396]. Spearman, means: 0.486. Steps the right way: 2 of 5.

### distinct-4 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.995 [0.990, 0.998] |
| 10 | 40 | 0 | 0.999 [0.998, 1.000] |
| 25 | 40 | 0 | 0.999 [0.999, 1.000] |
| 50 | 40 | 0 | 0.999 [0.998, 1.000] |
| 75 | 40 | 0 | 0.998 [0.997, 0.999] |
| 100 | 40 | 0 | 0.998 [0.995, 1.000] |

Spearman, samples: 0.093 [−0.045, 0.220]. Spearman, means: 0.029. Steps the right way: 3 of 5.

### rep-2 (lower is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.043 [0.031, 0.059] |
| 10 | 40 | 0 | 0.029 [0.020, 0.039] |
| 25 | 40 | 0 | 0.018 [0.011, 0.027] |
| 50 | 40 | 0 | 0.016 [0.010, 0.023] |
| 75 | 40 | 0 | 0.017 [0.010, 0.026] |
| 100 | 40 | 0 | 0.019 [0.011, 0.029] |

Spearman, samples: 0.257 [0.147, 0.369]. Spearman, means: 0.600. Steps the right way: 2 of 5.

### rep-3 (lower is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.015 [0.008, 0.024] |
| 10 | 40 | 0 | 0.008 [0.004, 0.012] |
| 25 | 40 | 0 | 0.003 [0.001, 0.006] |
| 50 | 40 | 0 | 0.003 [0.001, 0.005] |
| 75 | 40 | 0 | 0.004 [0.001, 0.007] |
| 100 | 40 | 0 | 0.004 [0.000, 0.009] |

Spearman, samples: 0.279 [0.155, 0.396]. Spearman, means: 0.486. Steps the right way: 2 of 5.

### rep-4 (lower is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.010] |
| 10 | 40 | 0 | 0.001 [0.000, 0.002] |
| 25 | 40 | 0 | 0.001 [0.000, 0.001] |
| 50 | 40 | 0 | 0.001 [0.000, 0.002] |
| 75 | 40 | 0 | 0.002 [0.001, 0.003] |
| 100 | 40 | 0 | 0.002 [0.000, 0.005] |

Spearman, samples: 0.093 [−0.045, 0.220]. Spearman, means: 0.029. Steps the right way: 3 of 5.

### seq-rep-4 (lower is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.010] |
| 10 | 40 | 0 | 0.001 [0.000, 0.002] |
| 25 | 40 | 0 | 0.001 [0.000, 0.001] |
| 50 | 40 | 0 | 0.001 [0.000, 0.002] |
| 75 | 40 | 0 | 0.002 [0.001, 0.003] |
| 100 | 40 | 0 | 0.002 [0.000, 0.005] |

Spearman, samples: 0.093 [−0.045, 0.220]. Spearman, means: 0.029. Steps the right way: 3 of 5.

### diversity (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.939 [0.913, 0.960] |
| 10 | 40 | 0 | 0.963 [0.949, 0.975] |
| 25 | 40 | 0 | 0.978 [0.968, 0.988] |
| 50 | 40 | 0 | 0.981 [0.971, 0.989] |
| 75 | 40 | 0 | 0.978 [0.965, 0.988] |
| 100 | 40 | 0 | 0.975 [0.959, 0.988] |

Spearman, samples: 0.260 [0.150, 0.371]. Spearman, means: 0.486. Steps the right way: 2 of 5.

### type-token ratio (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.709, 0.757] |
| 10 | 40 | 0 | 0.734 [0.708, 0.758] |
| 25 | 40 | 0 | 0.734 [0.707, 0.758] |
| 50 | 40 | 0 | 0.734 [0.709, 0.759] |
| 75 | 40 | 0 | 0.734 [0.707, 0.757] |
| 100 | 40 | 0 | 0.734 [0.707, 0.757] |

Spearman, samples: 0.000 [−0.130, 0.123]. Spearman, means: n/a. Steps the right way: 0 of 5.

### gzip ratio (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.620 [0.605, 0.635] |
| 10 | 40 | 0 | 0.623 [0.610, 0.637] |
| 25 | 40 | 0 | 0.629 [0.615, 0.642] |
| 50 | 40 | 0 | 0.630 [0.617, 0.643] |
| 75 | 40 | 0 | 0.632 [0.618, 0.646] |
| 100 | 40 | 0 | 0.632 [0.618, 0.647] |

Spearman, samples: 0.119 [−0.008, 0.240]. Spearman, means: 1.000. Steps the right way: 0 of 5.

## Family: truncate

### SCRIBE v2 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.877 [0.813, 0.927] |
| 10 | 40 | 0 | 0.877 [0.823, 0.925] |
| 25 | 40 | 0 | 0.885 [0.837, 0.931] |
| 50 | 40 | 0 | 0.821 [0.760, 0.878] |
| 75 | 40 | 0 | 0.867 [0.819, 0.916] |
| 100 | 40 | 0 | 0.925 [0.897, 0.954] |

Spearman, samples: 0.015 [−0.106, 0.136]. Spearman, means: 0.116. Steps the right way: 1 of 5.

### SCRIBE v1 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.503 [0.436, 0.570] |
| 10 | 40 | 0 | 0.492 [0.425, 0.558] |
| 25 | 40 | 0 | 0.483 [0.420, 0.540] |
| 50 | 40 | 0 | 0.491 [0.437, 0.543] |
| 75 | 40 | 0 | 0.470 [0.423, 0.518] |
| 100 | 40 | 0 | 0.479 [0.414, 0.541] |

Spearman, samples: −0.063 [−0.196, 0.066]. Spearman, means: −0.886. Steps the right way: 3 of 5.

### surface S (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.897 [0.871, 0.920] |
| 10 | 40 | 0 | 0.897 [0.870, 0.920] |
| 25 | 40 | 0 | 0.901 [0.879, 0.921] |
| 50 | 40 | 0 | 0.911 [0.894, 0.926] |
| 75 | 40 | 0 | 0.915 [0.899, 0.929] |
| 100 | 40 | 0 | 0.925 [0.908, 0.941] |

Spearman, samples: 0.106 [−0.021, 0.230]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### distinct-1 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.710, 0.759] |
| 10 | 40 | 0 | 0.735 [0.712, 0.761] |
| 25 | 40 | 0 | 0.753 [0.729, 0.778] |
| 50 | 40 | 0 | 0.775 [0.751, 0.797] |
| 75 | 40 | 0 | 0.787 [0.765, 0.810] |
| 100 | 40 | 0 | 0.812 [0.787, 0.834] |

Spearman, samples: 0.347 [0.232, 0.446]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-2 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.957 [0.941, 0.970] |
| 10 | 40 | 0 | 0.957 [0.942, 0.971] |
| 25 | 40 | 0 | 0.964 [0.951, 0.974] |
| 50 | 40 | 0 | 0.970 [0.957, 0.980] |
| 75 | 40 | 0 | 0.974 [0.963, 0.983] |
| 100 | 40 | 0 | 0.973 [0.959, 0.985] |

Spearman, samples: 0.220 [0.097, 0.337]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### distinct-3 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.985 [0.977, 0.992] |
| 10 | 40 | 0 | 0.985 [0.977, 0.992] |
| 25 | 40 | 0 | 0.990 [0.984, 0.995] |
| 50 | 40 | 0 | 0.991 [0.984, 0.997] |
| 75 | 40 | 0 | 0.993 [0.988, 0.997] |
| 100 | 40 | 0 | 0.992 [0.986, 0.997] |

Spearman, samples: 0.205 [0.085, 0.315]. Spearman, means: 0.928. Steps the right way: 1 of 5.

### distinct-4 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.995 [0.991, 0.998] |
| 10 | 40 | 0 | 0.995 [0.990, 0.998] |
| 25 | 40 | 0 | 0.997 [0.995, 0.999] |
| 50 | 40 | 0 | 0.996 [0.992, 0.999] |
| 75 | 40 | 0 | 0.998 [0.996, 1.000] |
| 100 | 40 | 0 | 0.997 [0.994, 0.999] |

Spearman, samples: 0.117 [−0.008, 0.229]. Spearman, means: 0.870. Steps the right way: 2 of 5.

### rep-2 (lower is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.043 [0.030, 0.059] |
| 10 | 40 | 0 | 0.043 [0.029, 0.058] |
| 25 | 40 | 0 | 0.036 [0.026, 0.049] |
| 50 | 40 | 0 | 0.030 [0.020, 0.043] |
| 75 | 40 | 0 | 0.026 [0.017, 0.037] |
| 100 | 40 | 0 | 0.027 [0.015, 0.041] |

Spearman, samples: 0.220 [0.097, 0.337]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### rep-3 (lower is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.015 [0.008, 0.023] |
| 10 | 40 | 0 | 0.015 [0.008, 0.023] |
| 25 | 40 | 0 | 0.010 [0.005, 0.016] |
| 50 | 40 | 0 | 0.009 [0.003, 0.016] |
| 75 | 40 | 0 | 0.007 [0.003, 0.012] |
| 100 | 40 | 0 | 0.008 [0.003, 0.014] |

Spearman, samples: 0.205 [0.085, 0.315]. Spearman, means: 0.928. Steps the right way: 1 of 5.

### rep-4 (lower is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.009] |
| 10 | 40 | 0 | 0.005 [0.002, 0.010] |
| 25 | 40 | 0 | 0.003 [0.001, 0.005] |
| 50 | 40 | 0 | 0.004 [0.001, 0.008] |
| 75 | 40 | 0 | 0.002 [0.000, 0.004] |
| 100 | 40 | 0 | 0.003 [0.001, 0.006] |

Spearman, samples: 0.117 [−0.008, 0.229]. Spearman, means: 0.870. Steps the right way: 2 of 5.

### seq-rep-4 (lower is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.009] |
| 10 | 40 | 0 | 0.005 [0.002, 0.010] |
| 25 | 40 | 0 | 0.003 [0.001, 0.005] |
| 50 | 40 | 0 | 0.004 [0.001, 0.008] |
| 75 | 40 | 0 | 0.002 [0.000, 0.004] |
| 100 | 40 | 0 | 0.003 [0.001, 0.006] |

Spearman, samples: 0.117 [−0.008, 0.229]. Spearman, means: 0.870. Steps the right way: 2 of 5.

### diversity (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.939 [0.916, 0.960] |
| 10 | 40 | 0 | 0.939 [0.915, 0.962] |
| 25 | 40 | 0 | 0.952 [0.933, 0.968] |
| 50 | 40 | 0 | 0.959 [0.937, 0.976] |
| 75 | 40 | 0 | 0.966 [0.950, 0.980] |
| 100 | 40 | 0 | 0.964 [0.943, 0.981] |

Spearman, samples: 0.236 [0.115, 0.352]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### type-token ratio (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.710, 0.759] |
| 10 | 40 | 0 | 0.735 [0.712, 0.761] |
| 25 | 40 | 0 | 0.753 [0.729, 0.778] |
| 50 | 40 | 0 | 0.775 [0.751, 0.797] |
| 75 | 40 | 0 | 0.787 [0.765, 0.810] |
| 100 | 40 | 0 | 0.812 [0.787, 0.834] |

Spearman, samples: 0.347 [0.232, 0.446]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### gzip ratio (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.620 [0.605, 0.635] |
| 10 | 40 | 0 | 0.620 [0.606, 0.634] |
| 25 | 40 | 0 | 0.643 [0.629, 0.657] |
| 50 | 40 | 0 | 0.672 [0.657, 0.687] |
| 75 | 40 | 0 | 0.700 [0.686, 0.716] |
| 100 | 40 | 0 | 0.744 [0.731, 0.757] |

Spearman, samples: 0.683 [0.614, 0.739]. Spearman, means: 1.000. Steps the right way: 0 of 5.

## Family: typo

### SCRIBE v2 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.877 [0.820, 0.925] |
| 10 | 40 | 0 | 0.844 [0.780, 0.904] |
| 25 | 40 | 0 | 0.372 [0.288, 0.467] |
| 50 | 40 | 0 | 0.015 [0.009, 0.022] |
| 75 | 40 | 0 | 0.002 [0.000, 0.005] |
| 100 | 40 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: −0.930 [−0.944, −0.916]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.503 [0.436, 0.565] |
| 10 | 40 | 0 | 0.331 [0.269, 0.392] |
| 25 | 40 | 0 | 0.091 [0.060, 0.127] |
| 50 | 40 | 0 | 0.012 [0.008, 0.018] |
| 75 | 40 | 0 | 0.006 [0.005, 0.007] |
| 100 | 40 | 0 | 0.005 [0.004, 0.005] |

Spearman, samples: −0.886 [−0.907, −0.864]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.897 [0.872, 0.919] |
| 10 | 40 | 0 | 0.826 [0.806, 0.847] |
| 25 | 40 | 0 | 0.712 [0.696, 0.727] |
| 50 | 40 | 0 | 0.519 [0.507, 0.530] |
| 75 | 40 | 0 | 0.321 [0.308, 0.334] |
| 100 | 40 | 0 | 0.117 [0.108, 0.127] |

Spearman, samples: −0.963 [−0.974, −0.951]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### distinct-1 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.709, 0.759] |
| 10 | 40 | 0 | 0.759 [0.736, 0.784] |
| 25 | 40 | 0 | 0.796 [0.774, 0.818] |
| 50 | 40 | 0 | 0.848 [0.828, 0.865] |
| 75 | 40 | 0 | 0.882 [0.864, 0.898] |
| 100 | 40 | 0 | 0.894 [0.876, 0.910] |

Spearman, samples: 0.689 [0.620, 0.751]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-2 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.957 [0.943, 0.970] |
| 10 | 40 | 0 | 0.968 [0.956, 0.978] |
| 25 | 40 | 0 | 0.977 [0.969, 0.986] |
| 50 | 40 | 0 | 0.991 [0.986, 0.994] |
| 75 | 40 | 0 | 0.997 [0.995, 0.999] |
| 100 | 40 | 0 | 0.998 [0.997, 0.999] |

Spearman, samples: 0.613 [0.529, 0.685]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-3 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.985 [0.977, 0.992] |
| 10 | 40 | 0 | 0.989 [0.982, 0.995] |
| 25 | 40 | 0 | 0.994 [0.991, 0.998] |
| 50 | 40 | 0 | 0.999 [0.997, 1.000] |
| 75 | 40 | 0 | 1.000 [1.000, 1.000] |
| 100 | 40 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.430 [0.341, 0.511]. Spearman, means: 0.986. Steps the right way: 0 of 5.

### distinct-4 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.995 [0.991, 0.998] |
| 10 | 40 | 0 | 0.997 [0.993, 0.999] |
| 25 | 40 | 0 | 0.999 [0.998, 1.000] |
| 50 | 40 | 0 | 1.000 [1.000, 1.000] |
| 75 | 40 | 0 | 1.000 [1.000, 1.000] |
| 100 | 40 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.306 [0.216, 0.387]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### rep-2 (lower is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.043 [0.030, 0.057] |
| 10 | 40 | 0 | 0.032 [0.022, 0.044] |
| 25 | 40 | 0 | 0.023 [0.014, 0.031] |
| 50 | 40 | 0 | 0.009 [0.006, 0.014] |
| 75 | 40 | 0 | 0.003 [0.001, 0.005] |
| 100 | 40 | 0 | 0.002 [0.001, 0.003] |

Spearman, samples: 0.613 [0.529, 0.685]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### rep-3 (lower is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.015 [0.008, 0.023] |
| 10 | 40 | 0 | 0.011 [0.005, 0.018] |
| 25 | 40 | 0 | 0.006 [0.002, 0.009] |
| 50 | 40 | 0 | 0.001 [0.000, 0.003] |
| 75 | 40 | 0 | 0.000 [0.000, 0.000] |
| 100 | 40 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.430 [0.341, 0.511]. Spearman, means: 0.986. Steps the right way: 0 of 5.

### rep-4 (lower is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.009] |
| 10 | 40 | 0 | 0.003 [0.001, 0.007] |
| 25 | 40 | 0 | 0.001 [0.000, 0.002] |
| 50 | 40 | 0 | 0.000 [0.000, 0.000] |
| 75 | 40 | 0 | 0.000 [0.000, 0.000] |
| 100 | 40 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.306 [0.216, 0.387]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### seq-rep-4 (lower is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.005 [0.002, 0.009] |
| 10 | 40 | 0 | 0.003 [0.001, 0.007] |
| 25 | 40 | 0 | 0.001 [0.000, 0.002] |
| 50 | 40 | 0 | 0.000 [0.000, 0.000] |
| 75 | 40 | 0 | 0.000 [0.000, 0.000] |
| 100 | 40 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.306 [0.216, 0.387]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### diversity (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.939 [0.917, 0.961] |
| 10 | 40 | 0 | 0.955 [0.935, 0.973] |
| 25 | 40 | 0 | 0.971 [0.959, 0.983] |
| 50 | 40 | 0 | 0.989 [0.984, 0.993] |
| 75 | 40 | 0 | 0.997 [0.995, 0.999] |
| 100 | 40 | 0 | 0.998 [0.997, 0.999] |

Spearman, samples: 0.610 [0.528, 0.684]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### type-token ratio (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.734 [0.709, 0.759] |
| 10 | 40 | 0 | 0.759 [0.736, 0.784] |
| 25 | 40 | 0 | 0.796 [0.774, 0.818] |
| 50 | 40 | 0 | 0.848 [0.828, 0.865] |
| 75 | 40 | 0 | 0.882 [0.864, 0.898] |
| 100 | 40 | 0 | 0.894 [0.876, 0.910] |

Spearman, samples: 0.689 [0.620, 0.751]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### gzip ratio (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 40 | 0 | 0.620 [0.606, 0.635] |
| 10 | 40 | 0 | 0.628 [0.614, 0.642] |
| 25 | 40 | 0 | 0.638 [0.625, 0.651] |
| 50 | 40 | 0 | 0.652 [0.641, 0.664] |
| 75 | 40 | 0 | 0.666 [0.656, 0.676] |
| 100 | 40 | 0 | 0.674 [0.665, 0.683] |

Spearman, samples: 0.472 [0.361, 0.565]. Spearman, means: 1.000. Steps the right way: 0 of 5.

## Summary

Spearman of the oriented means with strength (steps the right way). Last column: families with Spearman on the means <= −0.900.

| metric | dropword | nounswap | offtopic | reorder | repeat | subst | swap | truncate | typo | families hit |
|---|---|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | −1.000 (5/5) | −1.000 (5/5) | −1.000 (5/5) | −0.464 (2/5) | −1.000 (5/5) | −1.000 (5/5) | −1.000 (5/5) | 0.116 (1/5) | −1.000 (5/5) | 7 of 9 |
| SCRIBE v1 | −0.714 (3/5) | −1.000 (5/5) | −0.886 (3/5) | −0.754 (2/5) | −1.000 (5/5) | −1.000 (5/5) | −1.000 (5/5) | −0.886 (3/5) | −1.000 (5/5) | 5 of 9 |
| surface S | −0.086 (3/5) | −0.429 (2/5) | 0.429 (3/5) | −0.058 (2/5) | −1.000 (5/5) | −0.543 (2/5) | 0.657 (1/5) | 0.943 (1/5) | −1.000 (5/5) | 2 of 9 |
| distinct-1 | 1.000 (0/5) | 1.000 (0/5) | 0.771 (2/5) | n/a (0/5) | −1.000 (5/5) | 0.943 (1/5) | n/a (0/5) | 1.000 (0/5) | 1.000 (0/5) | 1 of 9 |
| distinct-2 | 1.000 (0/5) | 1.000 (0/5) | 0.771 (2/5) | n/a (0/5) | −1.000 (5/5) | 0.829 (1/5) | 0.600 (2/5) | 0.943 (1/5) | 1.000 (0/5) | 1 of 9 |
| distinct-3 | 0.986 (0/5) | 1.000 (0/5) | 0.638 (1/5) | n/a (0/5) | −1.000 (5/5) | 0.771 (2/5) | 0.486 (2/5) | 0.928 (1/5) | 0.986 (0/5) | 1 of 9 |
| distinct-4 | 0.941 (0/5) | 0.941 (0/5) | 0.116 (1/5) | n/a (0/5) | −1.000 (5/5) | 0.941 (0/5) | 0.029 (3/5) | 0.870 (2/5) | 0.941 (0/5) | 1 of 9 |
| rep-2 | 1.000 (0/5) | 1.000 (0/5) | 0.771 (2/5) | n/a (0/5) | −1.000 (5/5) | 0.829 (1/5) | 0.600 (2/5) | 0.943 (1/5) | 1.000 (0/5) | 1 of 9 |
| rep-3 | 0.986 (0/5) | 1.000 (0/5) | 0.638 (1/5) | n/a (0/5) | −1.000 (5/5) | 0.771 (2/5) | 0.486 (2/5) | 0.928 (1/5) | 0.986 (0/5) | 1 of 9 |
| rep-4 | 0.941 (0/5) | 0.941 (0/5) | 0.116 (1/5) | n/a (0/5) | −1.000 (5/5) | 0.941 (0/5) | 0.029 (3/5) | 0.870 (2/5) | 0.941 (0/5) | 1 of 9 |
| seq-rep-4 | 0.941 (0/5) | 0.941 (0/5) | 0.116 (1/5) | n/a (0/5) | −1.000 (5/5) | 0.941 (0/5) | 0.029 (3/5) | 0.870 (2/5) | 0.941 (0/5) | 1 of 9 |
| diversity | 1.000 (0/5) | 1.000 (0/5) | 0.771 (2/5) | n/a (0/5) | −1.000 (5/5) | 0.829 (1/5) | 0.486 (2/5) | 0.943 (1/5) | 1.000 (0/5) | 1 of 9 |
| type-token ratio | 1.000 (0/5) | 1.000 (0/5) | 0.771 (2/5) | n/a (0/5) | −1.000 (5/5) | 0.943 (1/5) | n/a (0/5) | 1.000 (0/5) | 1.000 (0/5) | 1 of 9 |
| gzip ratio | 1.000 (0/5) | 1.000 (0/5) | 0.829 (1/5) | 0.638 (2/5) | −1.000 (5/5) | 0.943 (1/5) | 1.000 (0/5) | 1.000 (0/5) | 1.000 (0/5) | 1 of 9 |

Spearman at sample level [95% interval]:

| metric | dropword | nounswap | offtopic | reorder | repeat | subst | swap | truncate | typo |
|---|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | −0.876 [−0.903, −0.847] | −0.832 [−0.865, −0.798] | −0.742 [−0.793, −0.683] | −0.037 [−0.178, 0.099] | −0.920 [−0.936, −0.902] | −0.935 [−0.946, −0.923] | −0.908 [−0.924, −0.890] | 0.015 [−0.106, 0.136] | −0.930 [−0.944, −0.916] |
| SCRIBE v1 | −0.202 [−0.331, −0.071] | −0.678 [−0.759, −0.585] | −0.519 [−0.622, −0.415] | −0.024 [−0.147, 0.100] | −0.905 [−0.924, −0.887] | −0.780 [−0.831, −0.723] | −0.617 [−0.694, −0.528] | −0.063 [−0.196, 0.066] | −0.886 [−0.907, −0.864] |
| surface S | −0.028 [−0.154, 0.092] | −0.256 [−0.384, −0.124] | −0.066 [−0.195, 0.068] | 0.001 [−0.119, 0.121] | −0.903 [−0.922, −0.884] | −0.267 [−0.380, −0.155] | 0.024 [−0.098, 0.153] | 0.106 [−0.021, 0.230] | −0.963 [−0.974, −0.951] |
| distinct-1 | 0.716 [0.641, 0.777] | 0.304 [0.185, 0.415] | 0.240 [0.106, 0.359] | 0.000 [−0.118, 0.131] | −0.880 [−0.903, −0.857] | 0.432 [0.330, 0.527] | 0.000 [−0.130, 0.123] | 0.347 [0.232, 0.446] | 0.689 [0.620, 0.751] |
| distinct-2 | 0.662 [0.592, 0.721] | 0.342 [0.226, 0.455] | 0.246 [0.121, 0.360] | 0.000 [−0.126, 0.132] | −0.909 [−0.926, −0.890] | 0.555 [0.462, 0.641] | 0.257 [0.147, 0.369] | 0.220 [0.097, 0.337] | 0.613 [0.529, 0.685] |
| distinct-3 | 0.416 [0.333, 0.494] | 0.310 [0.191, 0.422] | 0.292 [0.163, 0.414] | 0.000 [−0.130, 0.133] | −0.917 [−0.934, −0.899] | 0.425 [0.318, 0.518] | 0.279 [0.155, 0.396] | 0.205 [0.085, 0.315] | 0.430 [0.341, 0.511] |
| distinct-4 | 0.301 [0.210, 0.384] | 0.304 [0.221, 0.389] | 0.135 [−0.001, 0.265] | 0.000 [−0.129, 0.129] | −0.927 [−0.943, −0.912] | 0.299 [0.205, 0.378] | 0.093 [−0.045, 0.220] | 0.117 [−0.008, 0.229] | 0.306 [0.216, 0.387] |
| rep-2 | 0.662 [0.592, 0.721] | 0.342 [0.226, 0.455] | 0.246 [0.121, 0.360] | 0.000 [−0.126, 0.132] | −0.909 [−0.926, −0.890] | 0.555 [0.462, 0.641] | 0.257 [0.147, 0.369] | 0.220 [0.097, 0.337] | 0.613 [0.529, 0.685] |
| rep-3 | 0.416 [0.333, 0.494] | 0.310 [0.191, 0.422] | 0.292 [0.163, 0.414] | 0.000 [−0.130, 0.133] | −0.917 [−0.934, −0.899] | 0.425 [0.318, 0.518] | 0.279 [0.155, 0.396] | 0.205 [0.085, 0.315] | 0.430 [0.341, 0.511] |
| rep-4 | 0.301 [0.210, 0.384] | 0.304 [0.221, 0.389] | 0.135 [−0.001, 0.265] | 0.000 [−0.129, 0.129] | −0.927 [−0.943, −0.912] | 0.299 [0.205, 0.378] | 0.093 [−0.045, 0.220] | 0.117 [−0.008, 0.229] | 0.306 [0.216, 0.387] |
| seq-rep-4 | 0.301 [0.210, 0.384] | 0.304 [0.221, 0.389] | 0.135 [−0.001, 0.265] | 0.000 [−0.129, 0.129] | −0.927 [−0.943, −0.912] | 0.299 [0.205, 0.378] | 0.093 [−0.045, 0.220] | 0.117 [−0.008, 0.229] | 0.306 [0.216, 0.387] |
| diversity | 0.646 [0.573, 0.709] | 0.333 [0.213, 0.444] | 0.258 [0.130, 0.373] | 0.000 [−0.127, 0.135] | −0.912 [−0.929, −0.893] | 0.557 [0.463, 0.644] | 0.260 [0.150, 0.371] | 0.236 [0.115, 0.352] | 0.610 [0.528, 0.684] |
| type-token ratio | 0.716 [0.641, 0.777] | 0.304 [0.185, 0.415] | 0.240 [0.106, 0.359] | 0.000 [−0.118, 0.131] | −0.880 [−0.903, −0.857] | 0.432 [0.330, 0.527] | 0.000 [−0.130, 0.123] | 0.347 [0.232, 0.446] | 0.689 [0.620, 0.751] |
| gzip ratio | 0.910 [0.886, 0.930] | 0.387 [0.275, 0.497] | 0.313 [0.187, 0.430] | 0.004 [−0.126, 0.136] | −0.879 [−0.902, −0.855] | 0.471 [0.363, 0.572] | 0.119 [−0.008, 0.240] | 0.683 [0.614, 0.739] | 0.472 [0.361, 0.565] |
