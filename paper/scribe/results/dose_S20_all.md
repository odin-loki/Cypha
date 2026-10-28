# All per-text metrics on S20 (every family and strength of the first 20 sources)

Analysis `dose_S20_all` (3. Held-out corruptions, dose-response). meta.py `dose`.
Inputs: SP/run/baselines/subsets/corrupt_S20.jsonl, SP/run/fresh/corrupt.scribe.jsonl, SP/run/baselines/corrupt_S20/stats.jsonl, SP/run/baselines/corrupt_S20/ppl_gpt2.jsonl, SP/run/baselines/corrupt_S20/cola.jsonl, SP/run/baselines/corrupt_S20/coherence.jsonl, SP/run/baselines/corrupt_S20/detector.jsonl.

1080 texts, families from `meta.family`, strengths from `meta.strength`. A family without its own strength-0 texts uses the shared base texts (strength 0, family ['', 'base', 'none'] or none): 0. Bootstrap: 1000 resamples within each strength, seed 20261028. Spearman uses the oriented score, so −1 means quality falls perfectly with strength; a step moves the right way when the oriented mean falls.

## Family: dropword

### SCRIBE v2 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.911 [0.863, 0.956] |
| 10 | 20 | 0 | 0.856 [0.746, 0.935] |
| 25 | 20 | 0 | 0.616 [0.470, 0.758] |
| 50 | 20 | 0 | 0.202 [0.092, 0.332] |
| 75 | 20 | 0 | 0.022 [0.010, 0.045] |
| 100 | 20 | 0 | 0.015 [0.003, 0.037] |

Spearman, samples: −0.857 [−0.899, −0.813]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.555 [0.461, 0.627] |
| 10 | 20 | 0 | 0.499 [0.408, 0.580] |
| 25 | 20 | 0 | 0.420 [0.336, 0.505] |
| 50 | 20 | 0 | 0.426 [0.326, 0.519] |
| 75 | 20 | 0 | 0.385 [0.293, 0.476] |
| 100 | 20 | 0 | 0.521 [0.398, 0.633] |

Spearman, samples: −0.091 [−0.285, 0.086]. Spearman, means: −0.371. Steps the right way: 3 of 5.

### surface S (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.907 [0.873, 0.934] |
| 10 | 20 | 0 | 0.909 [0.879, 0.937] |
| 25 | 20 | 0 | 0.910 [0.892, 0.926] |
| 50 | 20 | 0 | 0.938 [0.912, 0.961] |
| 75 | 20 | 0 | 0.935 [0.918, 0.952] |
| 100 | 20 | 0 | 0.922 [0.912, 0.936] |

Spearman, samples: 0.085 [−0.080, 0.264]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### distinct-1 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.697, 0.770] |
| 10 | 20 | 0 | 0.748 [0.714, 0.783] |
| 25 | 20 | 0 | 0.773 [0.737, 0.805] |
| 50 | 20 | 0 | 0.808 [0.771, 0.842] |
| 75 | 20 | 0 | 0.879 [0.846, 0.914] |
| 100 | 20 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.724 [0.629, 0.806]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-2 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.959 [0.938, 0.974] |
| 10 | 20 | 0 | 0.968 [0.951, 0.982] |
| 25 | 20 | 0 | 0.978 [0.965, 0.988] |
| 50 | 20 | 0 | 0.989 [0.981, 0.995] |
| 75 | 20 | 0 | 1.000 [1.000, 1.000] |
| 100 | 20 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.702 [0.628, 0.768]. Spearman, means: 0.986. Steps the right way: 0 of 5.

### distinct-3 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.988 [0.978, 0.996] |
| 10 | 20 | 0 | 0.993 [0.984, 0.999] |
| 25 | 20 | 0 | 0.996 [0.991, 1.000] |
| 50 | 20 | 0 | 0.999 [0.997, 1.000] |
| 75 | 20 | 0 | 1.000 [1.000, 1.000] |
| 100 | 20 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.384 [0.260, 0.501]. Spearman, means: 0.986. Steps the right way: 0 of 5.

### distinct-4 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.997 [0.993, 1.000] |
| 10 | 20 | 0 | 0.998 [0.994, 1.000] |
| 25 | 20 | 0 | 0.999 [0.997, 1.000] |
| 50 | 20 | 0 | 1.000 [1.000, 1.000] |
| 75 | 20 | 0 | 1.000 [1.000, 1.000] |
| 100 | 14 | 6 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.244 [0.104, 0.356]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### rep-2 (lower is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.041 [0.026, 0.062] |
| 10 | 20 | 0 | 0.032 [0.018, 0.049] |
| 25 | 20 | 0 | 0.022 [0.012, 0.035] |
| 50 | 20 | 0 | 0.011 [0.005, 0.019] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 20 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.702 [0.628, 0.768]. Spearman, means: 0.986. Steps the right way: 0 of 5.

### rep-3 (lower is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.012 [0.004, 0.022] |
| 10 | 20 | 0 | 0.007 [0.001, 0.016] |
| 25 | 20 | 0 | 0.004 [0.000, 0.009] |
| 50 | 20 | 0 | 0.001 [0.000, 0.003] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 20 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.384 [0.260, 0.501]. Spearman, means: 0.986. Steps the right way: 0 of 5.

### rep-4 (lower is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.002 [0.000, 0.006] |
| 25 | 20 | 0 | 0.001 [0.000, 0.003] |
| 50 | 20 | 0 | 0.000 [0.000, 0.000] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 14 | 6 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.244 [0.104, 0.356]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### seq-rep-4 (lower is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.002 [0.000, 0.006] |
| 25 | 20 | 0 | 0.001 [0.000, 0.003] |
| 50 | 20 | 0 | 0.000 [0.000, 0.000] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 14 | 6 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.244 [0.104, 0.356]. Spearman, means: 0.941. Steps the right way: 0 of 5.

### diversity (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.946 [0.914, 0.970] |
| 10 | 20 | 0 | 0.960 [0.934, 0.981] |
| 25 | 20 | 0 | 0.974 [0.957, 0.987] |
| 50 | 20 | 0 | 0.988 [0.979, 0.995] |
| 75 | 20 | 0 | 1.000 [1.000, 1.000] |
| 100 | 14 | 6 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.684 [0.601, 0.757]. Spearman, means: 0.986. Steps the right way: 0 of 5.

### type-token ratio (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.697, 0.770] |
| 10 | 20 | 0 | 0.748 [0.714, 0.783] |
| 25 | 20 | 0 | 0.773 [0.737, 0.805] |
| 50 | 20 | 0 | 0.808 [0.771, 0.842] |
| 75 | 20 | 0 | 0.879 [0.846, 0.914] |
| 100 | 20 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.724 [0.629, 0.806]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### gzip ratio (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.625 [0.603, 0.646] |
| 10 | 20 | 0 | 0.644 [0.622, 0.665] |
| 25 | 20 | 0 | 0.673 [0.654, 0.692] |
| 50 | 20 | 0 | 0.748 [0.729, 0.768] |
| 75 | 20 | 0 | 0.945 [0.922, 0.965] |
| 100 | 20 | 0 | 1.573 [1.505, 1.641] |

Spearman, samples: 0.903 [0.867, 0.936]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### GPT-2 PPL (lower is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 36.105 [25.422, 49.405] |
| 10 | 20 | 0 | 68.348 [50.837, 89.533] |
| 25 | 20 | 0 | 148.971 [111.240, 194.355] |
| 50 | 20 | 0 | 410.429 [316.968, 515.613] |
| 75 | 20 | 0 | 997.195 [673.564, 1552.491] |
| 100 | 20 | 0 | 470.860 [202.937, 1136.907] |

Spearman, samples: −0.755 [−0.806, −0.694]. Spearman, means: −0.943. Steps the right way: 4 of 5.

### CoLA acceptability, mean (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.882 [0.849, 0.910] |
| 10 | 20 | 0 | 0.536 [0.462, 0.611] |
| 25 | 20 | 0 | 0.305 [0.247, 0.361] |
| 50 | 20 | 0 | 0.173 [0.131, 0.212] |
| 75 | 20 | 0 | 0.139 [0.105, 0.179] |
| 100 | 20 | 0 | 0.686 [0.634, 0.733] |

Spearman, samples: −0.410 [−0.461, −0.359]. Spearman, means: −0.429. Steps the right way: 4 of 5.

### CoLA acceptability, min (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.742 [0.638, 0.827] |
| 10 | 20 | 0 | 0.280 [0.209, 0.349] |
| 25 | 20 | 0 | 0.125 [0.101, 0.156] |
| 50 | 20 | 0 | 0.061 [0.054, 0.069] |
| 75 | 20 | 0 | 0.047 [0.041, 0.056] |
| 100 | 20 | 0 | 0.359 [0.268, 0.440] |

Spearman, samples: −0.457 [−0.538, −0.375]. Spearman, means: −0.429. Steps the right way: 4 of 5.

### MiniLM coherence, adjacent (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.357 [0.312, 0.410] |
| 10 | 20 | 0 | 0.336 [0.293, 0.383] |
| 25 | 20 | 0 | 0.294 [0.262, 0.330] |
| 50 | 20 | 0 | 0.293 [0.241, 0.350] |
| 75 | 20 | 0 | 0.233 [0.197, 0.271] |
| 100 | 20 | 0 | 0.308 [0.265, 0.352] |

Spearman, samples: −0.238 [−0.396, −0.078]. Spearman, means: −0.657. Steps the right way: 4 of 5.

### MiniLM coherence, all pairs (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.332 [0.287, 0.382] |
| 10 | 20 | 0 | 0.316 [0.274, 0.363] |
| 25 | 20 | 0 | 0.284 [0.251, 0.325] |
| 50 | 20 | 0 | 0.278 [0.228, 0.333] |
| 75 | 20 | 0 | 0.222 [0.194, 0.251] |
| 100 | 20 | 0 | 0.325 [0.293, 0.357] |

Spearman, samples: −0.122 [−0.275, 0.034]. Spearman, means: −0.429. Steps the right way: 4 of 5.

### GPT-2 detector P(human) (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.807 [0.659, 0.935] |
| 10 | 20 | 0 | 0.643 [0.470, 0.810] |
| 25 | 20 | 0 | 0.278 [0.156, 0.414] |
| 50 | 20 | 0 | 0.334 [0.165, 0.512] |
| 75 | 20 | 0 | 0.321 [0.185, 0.465] |
| 100 | 20 | 0 | 0.260 [0.178, 0.345] |

Spearman, samples: −0.410 [−0.542, −0.267]. Spearman, means: −0.829. Steps the right way: 4 of 5.

## Family: nounswap

### SCRIBE v2 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.911 [0.862, 0.953] |
| 10 | 20 | 0 | 0.902 [0.854, 0.948] |
| 25 | 20 | 0 | 0.826 [0.733, 0.910] |
| 50 | 20 | 0 | 0.567 [0.467, 0.667] |
| 75 | 20 | 0 | 0.177 [0.091, 0.264] |
| 100 | 20 | 0 | 0.085 [0.038, 0.142] |

Spearman, samples: −0.826 [−0.869, −0.783]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.555 [0.466, 0.632] |
| 10 | 20 | 0 | 0.534 [0.445, 0.608] |
| 25 | 20 | 0 | 0.485 [0.408, 0.560] |
| 50 | 20 | 0 | 0.354 [0.287, 0.422] |
| 75 | 20 | 0 | 0.164 [0.138, 0.189] |
| 100 | 20 | 0 | 0.107 [0.091, 0.125] |

Spearman, samples: −0.723 [−0.827, −0.605]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.907 [0.875, 0.935] |
| 10 | 20 | 0 | 0.914 [0.887, 0.938] |
| 25 | 20 | 0 | 0.914 [0.892, 0.933] |
| 50 | 20 | 0 | 0.919 [0.906, 0.933] |
| 75 | 20 | 0 | 0.888 [0.872, 0.901] |
| 100 | 20 | 0 | 0.874 [0.859, 0.890] |

Spearman, samples: −0.332 [−0.497, −0.146]. Spearman, means: −0.486. Steps the right way: 3 of 5.

### distinct-1 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.699, 0.772] |
| 10 | 20 | 0 | 0.746 [0.712, 0.779] |
| 25 | 20 | 0 | 0.757 [0.726, 0.785] |
| 50 | 20 | 0 | 0.777 [0.753, 0.804] |
| 75 | 20 | 0 | 0.790 [0.763, 0.816] |
| 100 | 20 | 0 | 0.791 [0.763, 0.815] |

Spearman, samples: 0.292 [0.133, 0.453]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-2 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.959 [0.938, 0.975] |
| 10 | 20 | 0 | 0.964 [0.943, 0.980] |
| 25 | 20 | 0 | 0.974 [0.962, 0.983] |
| 50 | 20 | 0 | 0.978 [0.969, 0.987] |
| 75 | 20 | 0 | 0.985 [0.977, 0.991] |
| 100 | 20 | 0 | 0.985 [0.977, 0.991] |

Spearman, samples: 0.312 [0.152, 0.463]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-3 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.988 [0.978, 0.996] |
| 10 | 20 | 0 | 0.990 [0.980, 0.997] |
| 25 | 20 | 0 | 0.997 [0.992, 1.000] |
| 50 | 20 | 0 | 0.998 [0.996, 0.999] |
| 75 | 20 | 0 | 0.999 [0.998, 1.000] |
| 100 | 20 | 0 | 0.999 [0.998, 1.000] |

Spearman, samples: 0.346 [0.193, 0.495]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### distinct-4 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.997 [0.993, 1.000] |
| 10 | 20 | 0 | 0.997 [0.993, 1.000] |
| 25 | 20 | 0 | 0.999 [0.997, 1.000] |
| 50 | 20 | 0 | 1.000 [1.000, 1.000] |
| 75 | 20 | 0 | 1.000 [1.000, 1.000] |
| 100 | 20 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.260 [0.137, 0.378]. Spearman, means: 0.926. Steps the right way: 0 of 5.

### rep-2 (lower is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.041 [0.025, 0.062] |
| 10 | 20 | 0 | 0.036 [0.020, 0.057] |
| 25 | 20 | 0 | 0.026 [0.017, 0.038] |
| 50 | 20 | 0 | 0.022 [0.013, 0.031] |
| 75 | 20 | 0 | 0.015 [0.009, 0.023] |
| 100 | 20 | 0 | 0.015 [0.009, 0.023] |

Spearman, samples: 0.312 [0.152, 0.463]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### rep-3 (lower is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.012 [0.004, 0.022] |
| 10 | 20 | 0 | 0.010 [0.003, 0.020] |
| 25 | 20 | 0 | 0.003 [0.000, 0.008] |
| 50 | 20 | 0 | 0.002 [0.001, 0.004] |
| 75 | 20 | 0 | 0.001 [0.000, 0.002] |
| 100 | 20 | 0 | 0.001 [0.000, 0.002] |

Spearman, samples: 0.346 [0.193, 0.495]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### rep-4 (lower is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.003 [0.000, 0.007] |
| 25 | 20 | 0 | 0.001 [0.000, 0.003] |
| 50 | 20 | 0 | 0.000 [0.000, 0.000] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 20 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.260 [0.137, 0.378]. Spearman, means: 0.926. Steps the right way: 0 of 5.

### seq-rep-4 (lower is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.003 [0.000, 0.007] |
| 25 | 20 | 0 | 0.001 [0.000, 0.003] |
| 50 | 20 | 0 | 0.000 [0.000, 0.000] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 20 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.260 [0.137, 0.378]. Spearman, means: 0.926. Steps the right way: 0 of 5.

### diversity (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.946 [0.915, 0.970] |
| 10 | 20 | 0 | 0.952 [0.921, 0.977] |
| 25 | 20 | 0 | 0.970 [0.953, 0.982] |
| 50 | 20 | 0 | 0.976 [0.967, 0.986] |
| 75 | 20 | 0 | 0.984 [0.977, 0.991] |
| 100 | 20 | 0 | 0.984 [0.976, 0.990] |

Spearman, samples: 0.310 [0.143, 0.456]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### type-token ratio (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.699, 0.772] |
| 10 | 20 | 0 | 0.746 [0.712, 0.779] |
| 25 | 20 | 0 | 0.757 [0.726, 0.785] |
| 50 | 20 | 0 | 0.777 [0.753, 0.804] |
| 75 | 20 | 0 | 0.790 [0.763, 0.816] |
| 100 | 20 | 0 | 0.791 [0.763, 0.815] |

Spearman, samples: 0.292 [0.133, 0.453]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### gzip ratio (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.625 [0.603, 0.647] |
| 10 | 20 | 0 | 0.631 [0.613, 0.649] |
| 25 | 20 | 0 | 0.641 [0.624, 0.658] |
| 50 | 20 | 0 | 0.650 [0.634, 0.667] |
| 75 | 20 | 0 | 0.659 [0.644, 0.675] |
| 100 | 20 | 0 | 0.663 [0.652, 0.677] |

Spearman, samples: 0.376 [0.212, 0.529]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### GPT-2 PPL (lower is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 36.105 [25.443, 49.615] |
| 10 | 20 | 0 | 62.329 [45.383, 81.953] |
| 25 | 20 | 0 | 141.218 [96.201, 203.031] |
| 50 | 20 | 0 | 345.948 [283.005, 420.822] |
| 75 | 20 | 0 | 684.137 [541.705, 860.545] |
| 100 | 20 | 0 | 1016.673 [798.387, 1249.774] |

Spearman, samples: −0.914 [−0.939, −0.884]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### CoLA acceptability, mean (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.882 [0.852, 0.912] |
| 10 | 20 | 0 | 0.755 [0.721, 0.786] |
| 25 | 20 | 0 | 0.559 [0.519, 0.601] |
| 50 | 20 | 0 | 0.310 [0.268, 0.350] |
| 75 | 20 | 0 | 0.204 [0.163, 0.248] |
| 100 | 20 | 0 | 0.136 [0.109, 0.168] |

Spearman, samples: −0.937 [−0.954, −0.916]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### CoLA acceptability, min (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.742 [0.643, 0.832] |
| 10 | 20 | 0 | 0.459 [0.379, 0.539] |
| 25 | 20 | 0 | 0.232 [0.181, 0.281] |
| 50 | 20 | 0 | 0.114 [0.088, 0.139] |
| 75 | 20 | 0 | 0.078 [0.063, 0.096] |
| 100 | 20 | 0 | 0.065 [0.052, 0.083] |

Spearman, samples: −0.856 [−0.898, −0.806]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### MiniLM coherence, adjacent (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.357 [0.311, 0.404] |
| 10 | 20 | 0 | 0.319 [0.270, 0.372] |
| 25 | 20 | 0 | 0.273 [0.228, 0.322] |
| 50 | 20 | 0 | 0.209 [0.180, 0.240] |
| 75 | 20 | 0 | 0.158 [0.139, 0.175] |
| 100 | 20 | 0 | 0.149 [0.132, 0.167] |

Spearman, samples: −0.713 [−0.787, −0.636]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### MiniLM coherence, all pairs (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.332 [0.288, 0.381] |
| 10 | 20 | 0 | 0.302 [0.258, 0.351] |
| 25 | 20 | 0 | 0.259 [0.219, 0.300] |
| 50 | 20 | 0 | 0.202 [0.175, 0.229] |
| 75 | 20 | 0 | 0.157 [0.136, 0.176] |
| 100 | 20 | 0 | 0.147 [0.131, 0.164] |

Spearman, samples: −0.685 [−0.764, −0.602]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### GPT-2 detector P(human) (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.807 [0.659, 0.948] |
| 10 | 20 | 0 | 0.626 [0.470, 0.800] |
| 25 | 20 | 0 | 0.108 [0.019, 0.231] |
| 50 | 20 | 0 | 0.051 [0.001, 0.128] |
| 75 | 20 | 0 | 0.002 [0.000, 0.006] |
| 100 | 20 | 0 | 0.002 [0.001, 0.003] |

Spearman, samples: −0.699 [−0.754, −0.640]. Spearman, means: −1.000. Steps the right way: 5 of 5.

## Family: offtopic

### SCRIBE v2 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.911 [0.861, 0.953] |
| 10 | 20 | 0 | 0.912 [0.864, 0.956] |
| 25 | 20 | 0 | 0.753 [0.657, 0.849] |
| 50 | 20 | 0 | 0.498 [0.404, 0.596] |
| 75 | 20 | 0 | 0.372 [0.302, 0.448] |
| 100 | 20 | 0 | 0.336 [0.275, 0.398] |

Spearman, samples: −0.778 [−0.829, −0.729]. Spearman, means: −0.943. Steps the right way: 4 of 5.

### SCRIBE v1 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.555 [0.466, 0.637] |
| 10 | 20 | 0 | 0.558 [0.470, 0.634] |
| 25 | 20 | 0 | 0.465 [0.372, 0.554] |
| 50 | 20 | 0 | 0.311 [0.246, 0.380] |
| 75 | 20 | 0 | 0.242 [0.190, 0.303] |
| 100 | 20 | 0 | 0.219 [0.175, 0.273] |

Spearman, samples: −0.589 [−0.721, −0.446]. Spearman, means: −0.943. Steps the right way: 4 of 5.

### surface S (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.907 [0.875, 0.935] |
| 10 | 20 | 0 | 0.909 [0.876, 0.935] |
| 25 | 20 | 0 | 0.917 [0.895, 0.934] |
| 50 | 20 | 0 | 0.903 [0.883, 0.920] |
| 75 | 20 | 0 | 0.900 [0.880, 0.921] |
| 100 | 20 | 0 | 0.886 [0.855, 0.915] |

Spearman, samples: −0.171 [−0.346, 0.015]. Spearman, means: −0.771. Steps the right way: 3 of 5.

### distinct-1 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.702, 0.771] |
| 10 | 20 | 0 | 0.736 [0.699, 0.772] |
| 25 | 20 | 0 | 0.759 [0.731, 0.784] |
| 50 | 20 | 0 | 0.786 [0.755, 0.815] |
| 75 | 20 | 0 | 0.789 [0.761, 0.817] |
| 100 | 20 | 0 | 0.793 [0.752, 0.833] |

Spearman, samples: 0.269 [0.079, 0.441]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-2 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.959 [0.940, 0.975] |
| 10 | 20 | 0 | 0.958 [0.939, 0.976] |
| 25 | 20 | 0 | 0.967 [0.949, 0.982] |
| 50 | 20 | 0 | 0.977 [0.964, 0.987] |
| 75 | 20 | 0 | 0.981 [0.968, 0.990] |
| 100 | 20 | 0 | 0.971 [0.943, 0.988] |

Spearman, samples: 0.278 [0.124, 0.422]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### distinct-3 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.988 [0.978, 0.996] |
| 10 | 20 | 0 | 0.988 [0.978, 0.997] |
| 25 | 20 | 0 | 0.991 [0.981, 0.997] |
| 50 | 20 | 0 | 0.995 [0.989, 0.999] |
| 75 | 20 | 0 | 0.999 [0.996, 1.000] |
| 100 | 20 | 0 | 0.987 [0.961, 1.000] |

Spearman, samples: 0.283 [0.115, 0.450]. Spearman, means: 0.116. Steps the right way: 1 of 5.

### distinct-4 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.997 [0.993, 1.000] |
| 10 | 20 | 0 | 0.997 [0.993, 1.000] |
| 25 | 20 | 0 | 0.998 [0.993, 1.000] |
| 50 | 20 | 0 | 0.998 [0.995, 1.000] |
| 75 | 20 | 0 | 1.000 [1.000, 1.000] |
| 100 | 20 | 0 | 0.988 [0.963, 1.000] |

Spearman, samples: 0.105 [−0.094, 0.277]. Spearman, means: 0.116. Steps the right way: 1 of 5.

### rep-2 (lower is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.041 [0.025, 0.060] |
| 10 | 20 | 0 | 0.042 [0.024, 0.061] |
| 25 | 20 | 0 | 0.033 [0.018, 0.051] |
| 50 | 20 | 0 | 0.023 [0.013, 0.036] |
| 75 | 20 | 0 | 0.019 [0.010, 0.032] |
| 100 | 20 | 0 | 0.029 [0.012, 0.057] |

Spearman, samples: 0.278 [0.124, 0.422]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### rep-3 (lower is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.012 [0.004, 0.022] |
| 10 | 20 | 0 | 0.012 [0.003, 0.022] |
| 25 | 20 | 0 | 0.009 [0.003, 0.019] |
| 50 | 20 | 0 | 0.005 [0.001, 0.011] |
| 75 | 20 | 0 | 0.001 [0.000, 0.004] |
| 100 | 20 | 0 | 0.013 [0.000, 0.039] |

Spearman, samples: 0.283 [0.115, 0.450]. Spearman, means: 0.116. Steps the right way: 1 of 5.

### rep-4 (lower is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.003 [0.000, 0.007] |
| 25 | 20 | 0 | 0.002 [0.000, 0.007] |
| 50 | 20 | 0 | 0.002 [0.000, 0.005] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 20 | 0 | 0.012 [0.000, 0.037] |

Spearman, samples: 0.105 [−0.094, 0.277]. Spearman, means: 0.116. Steps the right way: 1 of 5.

### seq-rep-4 (lower is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.003 [0.000, 0.007] |
| 25 | 20 | 0 | 0.002 [0.000, 0.007] |
| 50 | 20 | 0 | 0.002 [0.000, 0.005] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 20 | 0 | 0.012 [0.000, 0.037] |

Spearman, samples: 0.105 [−0.094, 0.277]. Spearman, means: 0.116. Steps the right way: 1 of 5.

### diversity (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.946 [0.916, 0.970] |
| 10 | 20 | 0 | 0.946 [0.914, 0.971] |
| 25 | 20 | 0 | 0.957 [0.928, 0.979] |
| 50 | 20 | 0 | 0.970 [0.951, 0.986] |
| 75 | 20 | 0 | 0.980 [0.964, 0.990] |
| 100 | 20 | 0 | 0.954 [0.894, 0.987] |

Spearman, samples: 0.284 [0.126, 0.429]. Spearman, means: 0.600. Steps the right way: 2 of 5.

### type-token ratio (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.702, 0.771] |
| 10 | 20 | 0 | 0.736 [0.699, 0.772] |
| 25 | 20 | 0 | 0.759 [0.731, 0.784] |
| 50 | 20 | 0 | 0.786 [0.755, 0.815] |
| 75 | 20 | 0 | 0.789 [0.761, 0.817] |
| 100 | 20 | 0 | 0.793 [0.752, 0.833] |

Spearman, samples: 0.269 [0.079, 0.441]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### gzip ratio (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.625 [0.604, 0.647] |
| 10 | 20 | 0 | 0.624 [0.602, 0.646] |
| 25 | 20 | 0 | 0.638 [0.622, 0.652] |
| 50 | 20 | 0 | 0.673 [0.651, 0.695] |
| 75 | 20 | 0 | 0.661 [0.639, 0.682] |
| 100 | 20 | 0 | 0.668 [0.640, 0.695] |

Spearman, samples: 0.346 [0.165, 0.520]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### GPT-2 PPL (lower is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 36.105 [25.955, 49.573] |
| 10 | 20 | 0 | 35.255 [24.600, 48.344] |
| 25 | 20 | 0 | 47.630 [36.867, 61.023] |
| 50 | 20 | 0 | 60.958 [52.117, 70.108] |
| 75 | 20 | 0 | 69.372 [57.343, 82.806] |
| 100 | 20 | 0 | 68.948 [58.995, 80.435] |

Spearman, samples: −0.567 [−0.697, −0.419]. Spearman, means: −0.886. Steps the right way: 3 of 5.

### CoLA acceptability, mean (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.882 [0.849, 0.910] |
| 10 | 20 | 0 | 0.882 [0.851, 0.911] |
| 25 | 20 | 0 | 0.878 [0.845, 0.908] |
| 50 | 20 | 0 | 0.881 [0.854, 0.908] |
| 75 | 20 | 0 | 0.889 [0.865, 0.910] |
| 100 | 20 | 0 | 0.885 [0.857, 0.911] |

Spearman, samples: 0.009 [−0.152, 0.201]. Spearman, means: 0.486. Steps the right way: 2 of 5.

### CoLA acceptability, min (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.742 [0.649, 0.829] |
| 10 | 20 | 0 | 0.742 [0.633, 0.832] |
| 25 | 20 | 0 | 0.737 [0.641, 0.825] |
| 50 | 20 | 0 | 0.724 [0.636, 0.802] |
| 75 | 20 | 0 | 0.722 [0.637, 0.797] |
| 100 | 20 | 0 | 0.724 [0.631, 0.804] |

Spearman, samples: −0.074 [−0.240, 0.110]. Spearman, means: −0.812. Steps the right way: 3 of 5.

### MiniLM coherence, adjacent (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.357 [0.312, 0.409] |
| 10 | 20 | 0 | 0.354 [0.307, 0.403] |
| 25 | 20 | 0 | 0.196 [0.150, 0.249] |
| 50 | 20 | 0 | 0.073 [0.034, 0.115] |
| 75 | 20 | 0 | 0.050 [0.021, 0.087] |
| 100 | 20 | 0 | 0.027 [0.007, 0.050] |

Spearman, samples: −0.790 [−0.835, −0.740]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### MiniLM coherence, all pairs (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.332 [0.286, 0.381] |
| 10 | 20 | 0 | 0.329 [0.281, 0.380] |
| 25 | 20 | 0 | 0.168 [0.134, 0.207] |
| 50 | 20 | 0 | 0.071 [0.048, 0.094] |
| 75 | 20 | 0 | 0.047 [0.029, 0.068] |
| 100 | 20 | 0 | 0.030 [0.010, 0.054] |

Spearman, samples: −0.837 [−0.879, −0.792]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### GPT-2 detector P(human) (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.807 [0.644, 0.940] |
| 10 | 20 | 0 | 0.807 [0.664, 0.937] |
| 25 | 20 | 0 | 0.794 [0.631, 0.926] |
| 50 | 20 | 0 | 0.593 [0.428, 0.755] |
| 75 | 20 | 0 | 0.701 [0.566, 0.826] |
| 100 | 20 | 0 | 0.691 [0.538, 0.844] |

Spearman, samples: −0.281 [−0.439, −0.112]. Spearman, means: −0.829. Steps the right way: 4 of 5.

## Family: reorder

### SCRIBE v2 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.911 [0.861, 0.958] |
| 10 | 20 | 0 | 0.911 [0.862, 0.955] |
| 25 | 20 | 0 | 0.917 [0.869, 0.960] |
| 50 | 20 | 0 | 0.918 [0.860, 0.963] |
| 75 | 20 | 0 | 0.892 [0.837, 0.942] |
| 100 | 20 | 0 | 0.899 [0.840, 0.952] |

Spearman, samples: −0.045 [−0.229, 0.139]. Spearman, means: −0.406. Steps the right way: 1 of 5.

### SCRIBE v1 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.555 [0.462, 0.638] |
| 10 | 20 | 0 | 0.555 [0.460, 0.638] |
| 25 | 20 | 0 | 0.560 [0.472, 0.639] |
| 50 | 20 | 0 | 0.554 [0.467, 0.642] |
| 75 | 20 | 0 | 0.538 [0.443, 0.620] |
| 100 | 20 | 0 | 0.547 [0.459, 0.626] |

Spearman, samples: −0.025 [−0.201, 0.151]. Spearman, means: −0.754. Steps the right way: 2 of 5.

### surface S (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.907 [0.874, 0.935] |
| 10 | 20 | 0 | 0.907 [0.873, 0.937] |
| 25 | 20 | 0 | 0.907 [0.876, 0.934] |
| 50 | 20 | 0 | 0.907 [0.874, 0.934] |
| 75 | 20 | 0 | 0.907 [0.874, 0.935] |
| 100 | 20 | 0 | 0.908 [0.874, 0.934] |

Spearman, samples: 0.007 [−0.167, 0.182]. Spearman, means: −0.030. Steps the right way: 2 of 5.

### distinct-1 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.696, 0.771] |
| 10 | 20 | 0 | 0.736 [0.701, 0.769] |
| 25 | 20 | 0 | 0.736 [0.699, 0.771] |
| 50 | 20 | 0 | 0.736 [0.698, 0.769] |
| 75 | 20 | 0 | 0.736 [0.699, 0.772] |
| 100 | 20 | 0 | 0.736 [0.698, 0.770] |

Spearman, samples: 0.000 [−0.171, 0.192]. Spearman, means: n/a. Steps the right way: 0 of 5.

### distinct-2 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.959 [0.937, 0.975] |
| 10 | 20 | 0 | 0.959 [0.937, 0.975] |
| 25 | 20 | 0 | 0.959 [0.939, 0.976] |
| 50 | 20 | 0 | 0.959 [0.938, 0.976] |
| 75 | 20 | 0 | 0.959 [0.939, 0.975] |
| 100 | 20 | 0 | 0.959 [0.938, 0.974] |

Spearman, samples: 0.000 [−0.169, 0.194]. Spearman, means: n/a. Steps the right way: 0 of 5.

### distinct-3 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.988 [0.978, 0.996] |
| 10 | 20 | 0 | 0.988 [0.977, 0.996] |
| 25 | 20 | 0 | 0.988 [0.978, 0.996] |
| 50 | 20 | 0 | 0.988 [0.978, 0.996] |
| 75 | 20 | 0 | 0.988 [0.978, 0.996] |
| 100 | 20 | 0 | 0.988 [0.978, 0.996] |

Spearman, samples: 0.000 [−0.174, 0.185]. Spearman, means: n/a. Steps the right way: 0 of 5.

### distinct-4 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.997 [0.993, 1.000] |
| 10 | 20 | 0 | 0.997 [0.993, 1.000] |
| 25 | 20 | 0 | 0.997 [0.993, 1.000] |
| 50 | 20 | 0 | 0.997 [0.993, 1.000] |
| 75 | 20 | 0 | 0.997 [0.993, 1.000] |
| 100 | 20 | 0 | 0.997 [0.993, 1.000] |

Spearman, samples: 0.000 [−0.190, 0.182]. Spearman, means: n/a. Steps the right way: 0 of 5.

### rep-2 (lower is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.041 [0.025, 0.063] |
| 10 | 20 | 0 | 0.041 [0.025, 0.063] |
| 25 | 20 | 0 | 0.041 [0.024, 0.061] |
| 50 | 20 | 0 | 0.041 [0.024, 0.062] |
| 75 | 20 | 0 | 0.041 [0.025, 0.061] |
| 100 | 20 | 0 | 0.041 [0.026, 0.062] |

Spearman, samples: 0.000 [−0.169, 0.194]. Spearman, means: n/a. Steps the right way: 0 of 5.

### rep-3 (lower is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.012 [0.004, 0.022] |
| 10 | 20 | 0 | 0.012 [0.004, 0.023] |
| 25 | 20 | 0 | 0.012 [0.004, 0.022] |
| 50 | 20 | 0 | 0.012 [0.004, 0.022] |
| 75 | 20 | 0 | 0.012 [0.004, 0.022] |
| 100 | 20 | 0 | 0.012 [0.004, 0.022] |

Spearman, samples: 0.000 [−0.174, 0.185]. Spearman, means: n/a. Steps the right way: 0 of 5.

### rep-4 (lower is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.003 [0.000, 0.007] |
| 25 | 20 | 0 | 0.003 [0.000, 0.007] |
| 50 | 20 | 0 | 0.003 [0.000, 0.007] |
| 75 | 20 | 0 | 0.003 [0.000, 0.007] |
| 100 | 20 | 0 | 0.003 [0.000, 0.007] |

Spearman, samples: 0.000 [−0.190, 0.182]. Spearman, means: n/a. Steps the right way: 0 of 5.

### seq-rep-4 (lower is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.003 [0.000, 0.007] |
| 25 | 20 | 0 | 0.003 [0.000, 0.007] |
| 50 | 20 | 0 | 0.003 [0.000, 0.007] |
| 75 | 20 | 0 | 0.003 [0.000, 0.007] |
| 100 | 20 | 0 | 0.003 [0.000, 0.007] |

Spearman, samples: 0.000 [−0.190, 0.182]. Spearman, means: n/a. Steps the right way: 0 of 5.

### diversity (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.946 [0.913, 0.971] |
| 10 | 20 | 0 | 0.946 [0.912, 0.971] |
| 25 | 20 | 0 | 0.946 [0.916, 0.971] |
| 50 | 20 | 0 | 0.946 [0.915, 0.971] |
| 75 | 20 | 0 | 0.946 [0.916, 0.971] |
| 100 | 20 | 0 | 0.946 [0.913, 0.969] |

Spearman, samples: 0.000 [−0.176, 0.192]. Spearman, means: n/a. Steps the right way: 0 of 5.

### type-token ratio (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.696, 0.771] |
| 10 | 20 | 0 | 0.736 [0.701, 0.769] |
| 25 | 20 | 0 | 0.736 [0.699, 0.771] |
| 50 | 20 | 0 | 0.736 [0.698, 0.769] |
| 75 | 20 | 0 | 0.736 [0.699, 0.772] |
| 100 | 20 | 0 | 0.736 [0.698, 0.770] |

Spearman, samples: 0.000 [−0.171, 0.192]. Spearman, means: n/a. Steps the right way: 0 of 5.

### gzip ratio (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.625 [0.603, 0.645] |
| 10 | 20 | 0 | 0.625 [0.604, 0.647] |
| 25 | 20 | 0 | 0.625 [0.604, 0.648] |
| 50 | 20 | 0 | 0.627 [0.607, 0.649] |
| 75 | 20 | 0 | 0.626 [0.607, 0.649] |
| 100 | 20 | 0 | 0.626 [0.605, 0.648] |

Spearman, samples: 0.013 [−0.169, 0.203]. Spearman, means: 0.754. Steps the right way: 2 of 5.

### GPT-2 PPL (lower is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 36.105 [25.226, 49.494] |
| 10 | 20 | 0 | 36.105 [25.967, 48.539] |
| 25 | 20 | 0 | 38.250 [25.562, 52.302] |
| 50 | 20 | 0 | 37.706 [26.946, 52.161] |
| 75 | 20 | 0 | 40.378 [26.813, 57.404] |
| 100 | 20 | 0 | 39.949 [28.061, 54.306] |

Spearman, samples: −0.112 [−0.290, 0.065]. Spearman, means: −0.870. Steps the right way: 2 of 5.

### CoLA acceptability, mean (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.882 [0.850, 0.913] |
| 10 | 20 | 0 | 0.882 [0.848, 0.909] |
| 25 | 20 | 0 | 0.882 [0.850, 0.914] |
| 50 | 20 | 0 | 0.882 [0.851, 0.910] |
| 75 | 20 | 0 | 0.882 [0.850, 0.912] |
| 100 | 20 | 0 | 0.882 [0.852, 0.911] |

Spearman, samples: 0.001 [−0.179, 0.190]. Spearman, means: 0.912. Steps the right way: 1 of 5.

### CoLA acceptability, min (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.742 [0.636, 0.828] |
| 10 | 20 | 0 | 0.742 [0.637, 0.824] |
| 25 | 20 | 0 | 0.742 [0.637, 0.837] |
| 50 | 20 | 0 | 0.742 [0.638, 0.827] |
| 75 | 20 | 0 | 0.742 [0.638, 0.830] |
| 100 | 20 | 0 | 0.742 [0.640, 0.832] |

Spearman, samples: 0.003 [−0.173, 0.189]. Spearman, means: 0.956. Steps the right way: 0 of 5.

### MiniLM coherence, adjacent (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.357 [0.308, 0.402] |
| 10 | 20 | 0 | 0.357 [0.309, 0.406] |
| 25 | 20 | 0 | 0.355 [0.310, 0.405] |
| 50 | 20 | 0 | 0.326 [0.270, 0.377] |
| 75 | 20 | 0 | 0.360 [0.298, 0.420] |
| 100 | 20 | 0 | 0.342 [0.281, 0.407] |

Spearman, samples: −0.064 [−0.243, 0.120]. Spearman, means: −0.290. Steps the right way: 3 of 5.

### MiniLM coherence, all pairs (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.332 [0.285, 0.379] |
| 10 | 20 | 0 | 0.332 [0.287, 0.381] |
| 25 | 20 | 0 | 0.332 [0.286, 0.380] |
| 50 | 20 | 0 | 0.332 [0.283, 0.379] |
| 75 | 20 | 0 | 0.332 [0.288, 0.384] |
| 100 | 20 | 0 | 0.332 [0.287, 0.381] |

Spearman, samples: 0.001 [−0.185, 0.174]. Spearman, means: −0.759. Steps the right way: 2 of 5.

### GPT-2 detector P(human) (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.807 [0.661, 0.941] |
| 10 | 20 | 0 | 0.807 [0.662, 0.930] |
| 25 | 20 | 0 | 0.795 [0.634, 0.925] |
| 50 | 20 | 0 | 0.831 [0.677, 0.944] |
| 75 | 20 | 0 | 0.813 [0.672, 0.919] |
| 100 | 20 | 0 | 0.787 [0.622, 0.918] |

Spearman, samples: −0.065 [−0.256, 0.111]. Spearman, means: −0.116. Steps the right way: 3 of 5.

## Family: repeat

### SCRIBE v2 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.911 [0.861, 0.956] |
| 10 | 20 | 0 | 0.869 [0.761, 0.949] |
| 25 | 20 | 0 | 0.274 [0.114, 0.460] |
| 50 | 20 | 0 | 0.000 [0.000, 0.000] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 20 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: −0.927 [−0.947, −0.904]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.555 [0.467, 0.631] |
| 10 | 20 | 0 | 0.529 [0.428, 0.620] |
| 25 | 20 | 0 | 0.175 [0.062, 0.303] |
| 50 | 20 | 0 | 0.001 [0.001, 0.001] |
| 75 | 20 | 0 | 0.001 [0.001, 0.001] |
| 100 | 20 | 0 | 0.001 [0.001, 0.001] |

Spearman, samples: −0.895 [−0.920, −0.863]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.907 [0.876, 0.934] |
| 10 | 20 | 0 | 0.897 [0.864, 0.929] |
| 25 | 20 | 0 | 0.717 [0.630, 0.807] |
| 50 | 20 | 0 | 0.393 [0.337, 0.446] |
| 75 | 20 | 0 | 0.231 [0.203, 0.264] |
| 100 | 20 | 0 | 0.129 [0.110, 0.149] |

Spearman, samples: −0.899 [−0.923, −0.867]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### distinct-1 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.699, 0.767] |
| 10 | 20 | 0 | 0.729 [0.691, 0.765] |
| 25 | 20 | 0 | 0.631 [0.575, 0.687] |
| 50 | 20 | 0 | 0.447 [0.402, 0.489] |
| 75 | 20 | 0 | 0.332 [0.309, 0.358] |
| 100 | 20 | 0 | 0.229 [0.203, 0.255] |

Spearman, samples: −0.884 [−0.912, −0.853]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### distinct-2 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.959 [0.939, 0.974] |
| 10 | 20 | 0 | 0.951 [0.927, 0.970] |
| 25 | 20 | 0 | 0.810 [0.746, 0.873] |
| 50 | 20 | 0 | 0.552 [0.504, 0.598] |
| 75 | 20 | 0 | 0.389 [0.361, 0.422] |
| 100 | 20 | 0 | 0.256 [0.229, 0.281] |

Spearman, samples: −0.906 [−0.931, −0.877]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### distinct-3 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.988 [0.979, 0.996] |
| 10 | 20 | 0 | 0.981 [0.961, 0.993] |
| 25 | 20 | 0 | 0.834 [0.774, 0.896] |
| 50 | 20 | 0 | 0.574 [0.526, 0.618] |
| 75 | 20 | 0 | 0.404 [0.374, 0.438] |
| 100 | 20 | 0 | 0.261 [0.233, 0.286] |

Spearman, samples: −0.915 [−0.938, −0.889]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### distinct-4 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.997 [0.993, 1.000] |
| 10 | 20 | 0 | 0.989 [0.973, 0.999] |
| 25 | 20 | 0 | 0.846 [0.788, 0.905] |
| 50 | 20 | 0 | 0.590 [0.543, 0.633] |
| 75 | 20 | 0 | 0.416 [0.383, 0.451] |
| 100 | 20 | 0 | 0.265 [0.236, 0.291] |

Spearman, samples: −0.932 [−0.949, −0.911]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### rep-2 (lower is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.041 [0.026, 0.061] |
| 10 | 20 | 0 | 0.049 [0.030, 0.073] |
| 25 | 20 | 0 | 0.190 [0.127, 0.254] |
| 50 | 20 | 0 | 0.448 [0.402, 0.496] |
| 75 | 20 | 0 | 0.611 [0.578, 0.639] |
| 100 | 20 | 0 | 0.744 [0.719, 0.771] |

Spearman, samples: −0.906 [−0.931, −0.877]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### rep-3 (lower is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.012 [0.004, 0.021] |
| 10 | 20 | 0 | 0.019 [0.007, 0.039] |
| 25 | 20 | 0 | 0.166 [0.104, 0.226] |
| 50 | 20 | 0 | 0.426 [0.382, 0.474] |
| 75 | 20 | 0 | 0.596 [0.562, 0.626] |
| 100 | 20 | 0 | 0.739 [0.714, 0.767] |

Spearman, samples: −0.915 [−0.938, −0.889]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### rep-4 (lower is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.011 [0.001, 0.027] |
| 25 | 20 | 0 | 0.154 [0.095, 0.212] |
| 50 | 20 | 0 | 0.410 [0.367, 0.457] |
| 75 | 20 | 0 | 0.584 [0.549, 0.617] |
| 100 | 20 | 0 | 0.735 [0.709, 0.764] |

Spearman, samples: −0.932 [−0.949, −0.911]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### seq-rep-4 (lower is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.011 [0.001, 0.027] |
| 25 | 20 | 0 | 0.154 [0.095, 0.212] |
| 50 | 20 | 0 | 0.410 [0.367, 0.457] |
| 75 | 20 | 0 | 0.584 [0.549, 0.617] |
| 100 | 20 | 0 | 0.735 [0.709, 0.764] |

Spearman, samples: −0.932 [−0.949, −0.911]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### diversity (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.946 [0.916, 0.969] |
| 10 | 20 | 0 | 0.926 [0.875, 0.963] |
| 25 | 20 | 0 | 0.621 [0.500, 0.749] |
| 50 | 20 | 0 | 0.206 [0.163, 0.254] |
| 75 | 20 | 0 | 0.072 [0.056, 0.091] |
| 100 | 20 | 0 | 0.021 [0.015, 0.026] |

Spearman, samples: −0.909 [−0.934, −0.880]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### type-token ratio (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.699, 0.767] |
| 10 | 20 | 0 | 0.729 [0.691, 0.765] |
| 25 | 20 | 0 | 0.631 [0.575, 0.687] |
| 50 | 20 | 0 | 0.447 [0.402, 0.489] |
| 75 | 20 | 0 | 0.332 [0.309, 0.358] |
| 100 | 20 | 0 | 0.229 [0.203, 0.255] |

Spearman, samples: −0.884 [−0.912, −0.853]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### gzip ratio (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.625 [0.603, 0.647] |
| 10 | 20 | 0 | 0.619 [0.596, 0.642] |
| 25 | 20 | 0 | 0.535 [0.495, 0.574] |
| 50 | 20 | 0 | 0.395 [0.359, 0.429] |
| 75 | 20 | 0 | 0.307 [0.286, 0.331] |
| 100 | 20 | 0 | 0.230 [0.206, 0.254] |

Spearman, samples: −0.885 [−0.912, −0.853]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### GPT-2 PPL (lower is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 36.105 [26.018, 48.767] |
| 10 | 20 | 0 | 35.252 [25.316, 48.986] |
| 25 | 20 | 0 | 24.660 [16.238, 37.115] |
| 50 | 20 | 0 | 10.267 [7.787, 13.006] |
| 75 | 20 | 0 | 5.583 [4.663, 6.606] |
| 100 | 20 | 0 | 3.407 [2.885, 3.973] |

Spearman, samples: 0.847 [0.809, 0.882]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### CoLA acceptability, mean (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.882 [0.849, 0.909] |
| 10 | 20 | 0 | 0.883 [0.854, 0.911] |
| 25 | 20 | 0 | 0.878 [0.846, 0.909] |
| 50 | 20 | 0 | 0.882 [0.840, 0.918] |
| 75 | 20 | 0 | 0.871 [0.820, 0.916] |
| 100 | 20 | 0 | 0.869 [0.807, 0.920] |

Spearman, samples: 0.055 [−0.134, 0.244]. Spearman, means: −0.714. Steps the right way: 3 of 5.

### CoLA acceptability, min (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.742 [0.639, 0.825] |
| 10 | 20 | 0 | 0.742 [0.644, 0.831] |
| 25 | 20 | 0 | 0.747 [0.647, 0.832] |
| 50 | 20 | 0 | 0.819 [0.741, 0.883] |
| 75 | 20 | 0 | 0.826 [0.731, 0.905] |
| 100 | 20 | 0 | 0.868 [0.806, 0.920] |

Spearman, samples: 0.295 [0.124, 0.453]. Spearman, means: 0.986. Steps the right way: 0 of 5.

### MiniLM coherence, adjacent (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.357 [0.313, 0.409] |
| 10 | 20 | 0 | 0.361 [0.317, 0.410] |
| 25 | 20 | 0 | 0.457 [0.399, 0.513] |
| 50 | 20 | 0 | 0.553 [0.464, 0.637] |
| 75 | 20 | 0 | 0.800 [0.735, 0.869] |
| 100 | 20 | 0 | 0.986 [0.958, 1.000] |

Spearman, samples: 0.797 [0.736, 0.849]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### MiniLM coherence, all pairs (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.332 [0.286, 0.383] |
| 10 | 20 | 0 | 0.336 [0.292, 0.383] |
| 25 | 20 | 0 | 0.407 [0.357, 0.458] |
| 50 | 20 | 0 | 0.583 [0.524, 0.642] |
| 75 | 20 | 0 | 0.764 [0.691, 0.840] |
| 100 | 20 | 0 | 0.992 [0.976, 1.000] |

Spearman, samples: 0.851 [0.802, 0.889]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### GPT-2 detector P(human) (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.807 [0.656, 0.944] |
| 10 | 20 | 0 | 0.807 [0.654, 0.941] |
| 25 | 20 | 0 | 0.638 [0.437, 0.819] |
| 50 | 20 | 0 | 0.520 [0.336, 0.726] |
| 75 | 20 | 0 | 0.335 [0.166, 0.523] |
| 100 | 20 | 0 | 0.163 [0.041, 0.317] |

Spearman, samples: −0.520 [−0.648, −0.366]. Spearman, means: −1.000. Steps the right way: 5 of 5.

## Family: subst

### SCRIBE v2 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.911 [0.859, 0.957] |
| 10 | 20 | 0 | 0.806 [0.685, 0.903] |
| 25 | 20 | 0 | 0.159 [0.062, 0.280] |
| 50 | 20 | 0 | 0.003 [0.003, 0.004] |
| 75 | 20 | 0 | 0.001 [0.001, 0.001] |
| 100 | 20 | 0 | 0.001 [0.001, 0.001] |

Spearman, samples: −0.936 [−0.952, −0.919]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.555 [0.465, 0.634] |
| 10 | 20 | 0 | 0.518 [0.449, 0.592] |
| 25 | 20 | 0 | 0.452 [0.364, 0.535] |
| 50 | 20 | 0 | 0.164 [0.129, 0.199] |
| 75 | 20 | 0 | 0.072 [0.064, 0.081] |
| 100 | 20 | 0 | 0.068 [0.062, 0.076] |

Spearman, samples: −0.811 [−0.869, −0.739]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.907 [0.874, 0.934] |
| 10 | 20 | 0 | 0.914 [0.893, 0.933] |
| 25 | 20 | 0 | 0.929 [0.910, 0.947] |
| 50 | 20 | 0 | 0.903 [0.888, 0.917] |
| 75 | 20 | 0 | 0.889 [0.879, 0.900] |
| 100 | 20 | 0 | 0.895 [0.887, 0.902] |

Spearman, samples: −0.276 [−0.430, −0.102]. Spearman, means: −0.714. Steps the right way: 2 of 5.

### distinct-1 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.701, 0.771] |
| 10 | 20 | 0 | 0.759 [0.728, 0.788] |
| 25 | 20 | 0 | 0.762 [0.732, 0.792] |
| 50 | 20 | 0 | 0.806 [0.789, 0.824] |
| 75 | 20 | 0 | 0.821 [0.799, 0.841] |
| 100 | 20 | 0 | 0.803 [0.788, 0.818] |

Spearman, samples: 0.405 [0.253, 0.551]. Spearman, means: 0.829. Steps the right way: 1 of 5.

### distinct-2 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.959 [0.939, 0.976] |
| 10 | 20 | 0 | 0.973 [0.959, 0.984] |
| 25 | 20 | 0 | 0.982 [0.974, 0.989] |
| 50 | 20 | 0 | 0.995 [0.991, 0.998] |
| 75 | 20 | 0 | 0.997 [0.991, 1.000] |
| 100 | 20 | 0 | 0.994 [0.990, 0.997] |

Spearman, samples: 0.559 [0.428, 0.676]. Spearman, means: 0.829. Steps the right way: 1 of 5.

### distinct-3 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.988 [0.978, 0.996] |
| 10 | 20 | 0 | 0.995 [0.990, 0.998] |
| 25 | 20 | 0 | 0.998 [0.995, 1.000] |
| 50 | 20 | 0 | 1.000 [1.000, 1.000] |
| 75 | 20 | 0 | 0.999 [0.998, 1.000] |
| 100 | 20 | 0 | 0.999 [0.998, 1.000] |

Spearman, samples: 0.359 [0.192, 0.497]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### distinct-4 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.997 [0.993, 1.000] |
| 10 | 20 | 0 | 1.000 [0.999, 1.000] |
| 25 | 20 | 0 | 1.000 [1.000, 1.000] |
| 50 | 20 | 0 | 1.000 [1.000, 1.000] |
| 75 | 20 | 0 | 1.000 [1.000, 1.000] |
| 100 | 20 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.245 [0.134, 0.355]. Spearman, means: 0.845. Steps the right way: 0 of 5.

### rep-2 (lower is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.041 [0.024, 0.061] |
| 10 | 20 | 0 | 0.027 [0.016, 0.041] |
| 25 | 20 | 0 | 0.018 [0.011, 0.026] |
| 50 | 20 | 0 | 0.005 [0.002, 0.009] |
| 75 | 20 | 0 | 0.003 [0.000, 0.009] |
| 100 | 20 | 0 | 0.006 [0.003, 0.010] |

Spearman, samples: 0.559 [0.428, 0.676]. Spearman, means: 0.829. Steps the right way: 1 of 5.

### rep-3 (lower is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.012 [0.004, 0.022] |
| 10 | 20 | 0 | 0.005 [0.002, 0.010] |
| 25 | 20 | 0 | 0.002 [0.000, 0.005] |
| 50 | 20 | 0 | 0.000 [0.000, 0.000] |
| 75 | 20 | 0 | 0.001 [0.000, 0.002] |
| 100 | 20 | 0 | 0.001 [0.000, 0.002] |

Spearman, samples: 0.359 [0.192, 0.497]. Spearman, means: 0.771. Steps the right way: 2 of 5.

### rep-4 (lower is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.000 [0.000, 0.001] |
| 25 | 20 | 0 | 0.000 [0.000, 0.000] |
| 50 | 20 | 0 | 0.000 [0.000, 0.000] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 20 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.245 [0.134, 0.355]. Spearman, means: 0.845. Steps the right way: 0 of 5.

### seq-rep-4 (lower is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.000 [0.000, 0.001] |
| 25 | 20 | 0 | 0.000 [0.000, 0.000] |
| 50 | 20 | 0 | 0.000 [0.000, 0.000] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 20 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.245 [0.134, 0.355]. Spearman, means: 0.845. Steps the right way: 0 of 5.

### diversity (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.946 [0.915, 0.972] |
| 10 | 20 | 0 | 0.968 [0.949, 0.982] |
| 25 | 20 | 0 | 0.980 [0.969, 0.989] |
| 50 | 20 | 0 | 0.995 [0.991, 0.998] |
| 75 | 20 | 0 | 0.996 [0.990, 1.000] |
| 100 | 20 | 0 | 0.993 [0.988, 0.997] |

Spearman, samples: 0.558 [0.423, 0.676]. Spearman, means: 0.829. Steps the right way: 1 of 5.

### type-token ratio (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.701, 0.771] |
| 10 | 20 | 0 | 0.759 [0.728, 0.788] |
| 25 | 20 | 0 | 0.762 [0.732, 0.792] |
| 50 | 20 | 0 | 0.806 [0.789, 0.824] |
| 75 | 20 | 0 | 0.821 [0.799, 0.841] |
| 100 | 20 | 0 | 0.803 [0.788, 0.818] |

Spearman, samples: 0.405 [0.253, 0.551]. Spearman, means: 0.829. Steps the right way: 1 of 5.

### gzip ratio (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.625 [0.604, 0.646] |
| 10 | 20 | 0 | 0.635 [0.615, 0.656] |
| 25 | 20 | 0 | 0.650 [0.634, 0.667] |
| 50 | 20 | 0 | 0.661 [0.647, 0.677] |
| 75 | 20 | 0 | 0.667 [0.654, 0.680] |
| 100 | 20 | 0 | 0.669 [0.656, 0.682] |

Spearman, samples: 0.431 [0.256, 0.585]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### GPT-2 PPL (lower is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 36.105 [26.200, 49.944] |
| 10 | 20 | 0 | 109.950 [84.526, 144.711] |
| 25 | 20 | 0 | 309.742 [252.572, 373.449] |
| 50 | 20 | 0 | 1144.950 [973.954, 1338.004] |
| 75 | 20 | 0 | 1812.688 [1525.165, 2113.418] |
| 100 | 20 | 0 | 2338.263 [2021.680, 2645.262] |

Spearman, samples: −0.943 [−0.959, −0.924]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### CoLA acceptability, mean (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.882 [0.850, 0.912] |
| 10 | 20 | 0 | 0.516 [0.467, 0.565] |
| 25 | 20 | 0 | 0.237 [0.184, 0.293] |
| 50 | 20 | 0 | 0.098 [0.075, 0.126] |
| 75 | 20 | 0 | 0.074 [0.059, 0.097] |
| 100 | 20 | 0 | 0.060 [0.050, 0.076] |

Spearman, samples: −0.919 [−0.945, −0.887]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### CoLA acceptability, min (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.742 [0.641, 0.832] |
| 10 | 20 | 0 | 0.185 [0.136, 0.236] |
| 25 | 20 | 0 | 0.068 [0.061, 0.074] |
| 50 | 20 | 0 | 0.044 [0.042, 0.047] |
| 75 | 20 | 0 | 0.043 [0.041, 0.046] |
| 100 | 20 | 0 | 0.040 [0.038, 0.042] |

Spearman, samples: −0.861 [−0.914, −0.793]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### MiniLM coherence, adjacent (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.357 [0.311, 0.406] |
| 10 | 20 | 0 | 0.302 [0.250, 0.356] |
| 25 | 20 | 0 | 0.264 [0.228, 0.302] |
| 50 | 20 | 0 | 0.218 [0.183, 0.253] |
| 75 | 20 | 0 | 0.146 [0.124, 0.167] |
| 100 | 20 | 0 | 0.160 [0.137, 0.184] |

Spearman, samples: −0.658 [−0.733, −0.571]. Spearman, means: −0.943. Steps the right way: 4 of 5.

### MiniLM coherence, all pairs (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.332 [0.285, 0.380] |
| 10 | 20 | 0 | 0.292 [0.244, 0.343] |
| 25 | 20 | 0 | 0.247 [0.215, 0.280] |
| 50 | 20 | 0 | 0.207 [0.175, 0.239] |
| 75 | 20 | 0 | 0.143 [0.125, 0.162] |
| 100 | 20 | 0 | 0.151 [0.134, 0.168] |

Spearman, samples: −0.685 [−0.750, −0.605]. Spearman, means: −0.943. Steps the right way: 4 of 5.

### GPT-2 detector P(human) (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.807 [0.657, 0.936] |
| 10 | 20 | 0 | 0.315 [0.147, 0.504] |
| 25 | 20 | 0 | 0.172 [0.050, 0.318] |
| 50 | 20 | 0 | 0.131 [0.058, 0.231] |
| 75 | 20 | 0 | 0.037 [0.009, 0.083] |
| 100 | 20 | 0 | 0.041 [0.016, 0.072] |

Spearman, samples: −0.414 [−0.522, −0.294]. Spearman, means: −0.943. Steps the right way: 4 of 5.

## Family: swap

### SCRIBE v2 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.911 [0.860, 0.954] |
| 10 | 20 | 0 | 0.687 [0.519, 0.837] |
| 25 | 20 | 0 | 0.039 [0.017, 0.068] |
| 50 | 20 | 0 | 0.006 [0.005, 0.007] |
| 75 | 20 | 0 | 0.003 [0.003, 0.004] |
| 100 | 20 | 0 | 0.003 [0.003, 0.003] |

Spearman, samples: −0.916 [−0.936, −0.893]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.555 [0.467, 0.634] |
| 10 | 20 | 0 | 0.521 [0.431, 0.592] |
| 25 | 20 | 0 | 0.409 [0.338, 0.478] |
| 50 | 20 | 0 | 0.270 [0.200, 0.336] |
| 75 | 20 | 0 | 0.182 [0.127, 0.244] |
| 100 | 20 | 0 | 0.132 [0.101, 0.166] |

Spearman, samples: −0.693 [−0.788, −0.585]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.907 [0.877, 0.933] |
| 10 | 20 | 0 | 0.913 [0.888, 0.937] |
| 25 | 20 | 0 | 0.919 [0.897, 0.940] |
| 50 | 20 | 0 | 0.919 [0.896, 0.938] |
| 75 | 20 | 0 | 0.922 [0.900, 0.944] |
| 100 | 20 | 0 | 0.919 [0.896, 0.939] |

Spearman, samples: 0.046 [−0.127, 0.229]. Spearman, means: 0.714. Steps the right way: 2 of 5.

### distinct-1 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.699, 0.770] |
| 10 | 20 | 0 | 0.736 [0.699, 0.771] |
| 25 | 20 | 0 | 0.736 [0.700, 0.768] |
| 50 | 20 | 0 | 0.736 [0.698, 0.771] |
| 75 | 20 | 0 | 0.736 [0.701, 0.770] |
| 100 | 20 | 0 | 0.736 [0.699, 0.770] |

Spearman, samples: 0.000 [−0.172, 0.171]. Spearman, means: n/a. Steps the right way: 0 of 5.

### distinct-2 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.959 [0.938, 0.975] |
| 10 | 20 | 0 | 0.972 [0.959, 0.984] |
| 25 | 20 | 0 | 0.982 [0.970, 0.992] |
| 50 | 20 | 0 | 0.985 [0.973, 0.993] |
| 75 | 20 | 0 | 0.988 [0.979, 0.996] |
| 100 | 20 | 0 | 0.979 [0.966, 0.989] |

Spearman, samples: 0.282 [0.137, 0.423]. Spearman, means: 0.657. Steps the right way: 1 of 5.

### distinct-3 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.988 [0.978, 0.996] |
| 10 | 20 | 0 | 0.994 [0.989, 0.998] |
| 25 | 20 | 0 | 0.998 [0.994, 1.000] |
| 50 | 20 | 0 | 0.997 [0.993, 1.000] |
| 75 | 20 | 0 | 0.998 [0.994, 1.000] |
| 100 | 20 | 0 | 0.996 [0.989, 1.000] |

Spearman, samples: 0.328 [0.177, 0.479]. Spearman, means: 0.600. Steps the right way: 2 of 5.

### distinct-4 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.997 [0.993, 1.000] |
| 10 | 20 | 0 | 1.000 [0.999, 1.000] |
| 25 | 20 | 0 | 0.999 [0.998, 1.000] |
| 50 | 20 | 0 | 0.999 [0.997, 1.000] |
| 75 | 20 | 0 | 0.999 [0.997, 1.000] |
| 100 | 20 | 0 | 0.998 [0.994, 1.000] |

Spearman, samples: 0.093 [−0.103, 0.275]. Spearman, means: −0.086. Steps the right way: 3 of 5.

### rep-2 (lower is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.041 [0.025, 0.062] |
| 10 | 20 | 0 | 0.028 [0.016, 0.041] |
| 25 | 20 | 0 | 0.018 [0.008, 0.030] |
| 50 | 20 | 0 | 0.015 [0.007, 0.027] |
| 75 | 20 | 0 | 0.012 [0.004, 0.021] |
| 100 | 20 | 0 | 0.021 [0.011, 0.034] |

Spearman, samples: 0.282 [0.137, 0.423]. Spearman, means: 0.657. Steps the right way: 1 of 5.

### rep-3 (lower is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.012 [0.004, 0.022] |
| 10 | 20 | 0 | 0.006 [0.002, 0.011] |
| 25 | 20 | 0 | 0.002 [0.000, 0.006] |
| 50 | 20 | 0 | 0.003 [0.000, 0.007] |
| 75 | 20 | 0 | 0.002 [0.000, 0.006] |
| 100 | 20 | 0 | 0.004 [0.000, 0.011] |

Spearman, samples: 0.328 [0.177, 0.479]. Spearman, means: 0.600. Steps the right way: 2 of 5.

### rep-4 (lower is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.000 [0.000, 0.001] |
| 25 | 20 | 0 | 0.001 [0.000, 0.002] |
| 50 | 20 | 0 | 0.001 [0.000, 0.003] |
| 75 | 20 | 0 | 0.001 [0.000, 0.003] |
| 100 | 20 | 0 | 0.002 [0.000, 0.006] |

Spearman, samples: 0.093 [−0.103, 0.275]. Spearman, means: −0.086. Steps the right way: 3 of 5.

### seq-rep-4 (lower is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.000 [0.000, 0.001] |
| 25 | 20 | 0 | 0.001 [0.000, 0.002] |
| 50 | 20 | 0 | 0.001 [0.000, 0.003] |
| 75 | 20 | 0 | 0.001 [0.000, 0.003] |
| 100 | 20 | 0 | 0.002 [0.000, 0.006] |

Spearman, samples: 0.093 [−0.103, 0.275]. Spearman, means: −0.086. Steps the right way: 3 of 5.

### diversity (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.946 [0.914, 0.970] |
| 10 | 20 | 0 | 0.966 [0.947, 0.982] |
| 25 | 20 | 0 | 0.980 [0.963, 0.992] |
| 50 | 20 | 0 | 0.981 [0.964, 0.993] |
| 75 | 20 | 0 | 0.986 [0.971, 0.996] |
| 100 | 20 | 0 | 0.975 [0.951, 0.989] |

Spearman, samples: 0.287 [0.141, 0.430]. Spearman, means: 0.657. Steps the right way: 1 of 5.

### type-token ratio (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.699, 0.770] |
| 10 | 20 | 0 | 0.736 [0.699, 0.771] |
| 25 | 20 | 0 | 0.736 [0.700, 0.768] |
| 50 | 20 | 0 | 0.736 [0.698, 0.771] |
| 75 | 20 | 0 | 0.736 [0.701, 0.770] |
| 100 | 20 | 0 | 0.736 [0.699, 0.770] |

Spearman, samples: 0.000 [−0.172, 0.171]. Spearman, means: n/a. Steps the right way: 0 of 5.

### gzip ratio (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.625 [0.603, 0.645] |
| 10 | 20 | 0 | 0.628 [0.608, 0.649] |
| 25 | 20 | 0 | 0.633 [0.614, 0.652] |
| 50 | 20 | 0 | 0.637 [0.620, 0.656] |
| 75 | 20 | 0 | 0.637 [0.619, 0.656] |
| 100 | 20 | 0 | 0.637 [0.619, 0.656] |

Spearman, samples: 0.124 [−0.048, 0.302]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### GPT-2 PPL (lower is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 36.105 [25.869, 49.048] |
| 10 | 20 | 0 | 95.931 [67.554, 144.856] |
| 25 | 20 | 0 | 238.767 [184.208, 300.980] |
| 50 | 20 | 0 | 424.241 [314.577, 581.929] |
| 75 | 20 | 0 | 614.487 [437.830, 861.376] |
| 100 | 20 | 0 | 725.632 [528.592, 976.264] |

Spearman, samples: −0.853 [−0.906, −0.793]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### CoLA acceptability, mean (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.882 [0.847, 0.910] |
| 10 | 20 | 0 | 0.425 [0.367, 0.481] |
| 25 | 20 | 0 | 0.206 [0.164, 0.252] |
| 50 | 20 | 0 | 0.114 [0.092, 0.144] |
| 75 | 20 | 0 | 0.095 [0.075, 0.123] |
| 100 | 20 | 0 | 0.094 [0.072, 0.127] |

Spearman, samples: −0.847 [−0.892, −0.794]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### CoLA acceptability, min (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.742 [0.628, 0.831] |
| 10 | 20 | 0 | 0.129 [0.101, 0.161] |
| 25 | 20 | 0 | 0.071 [0.058, 0.087] |
| 50 | 20 | 0 | 0.055 [0.049, 0.061] |
| 75 | 20 | 0 | 0.048 [0.044, 0.053] |
| 100 | 20 | 0 | 0.045 [0.042, 0.048] |

Spearman, samples: −0.766 [−0.832, −0.695]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### MiniLM coherence, adjacent (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.357 [0.314, 0.407] |
| 10 | 20 | 0 | 0.354 [0.310, 0.404] |
| 25 | 20 | 0 | 0.347 [0.304, 0.392] |
| 50 | 20 | 0 | 0.352 [0.307, 0.395] |
| 75 | 20 | 0 | 0.356 [0.317, 0.399] |
| 100 | 20 | 0 | 0.349 [0.307, 0.392] |

Spearman, samples: −0.005 [−0.188, 0.175]. Spearman, means: −0.429. Steps the right way: 3 of 5.

### MiniLM coherence, all pairs (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.332 [0.288, 0.382] |
| 10 | 20 | 0 | 0.329 [0.288, 0.377] |
| 25 | 20 | 0 | 0.324 [0.280, 0.368] |
| 50 | 20 | 0 | 0.328 [0.284, 0.372] |
| 75 | 20 | 0 | 0.333 [0.293, 0.375] |
| 100 | 20 | 0 | 0.327 [0.286, 0.371] |

Spearman, samples: 0.010 [−0.163, 0.173]. Spearman, means: −0.200. Steps the right way: 3 of 5.

### GPT-2 detector P(human) (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.807 [0.661, 0.936] |
| 10 | 20 | 0 | 0.686 [0.514, 0.826] |
| 25 | 20 | 0 | 0.524 [0.357, 0.692] |
| 50 | 20 | 0 | 0.631 [0.480, 0.780] |
| 75 | 20 | 0 | 0.770 [0.621, 0.913] |
| 100 | 20 | 0 | 0.761 [0.602, 0.888] |

Spearman, samples: 0.002 [−0.169, 0.162]. Spearman, means: −0.086. Steps the right way: 3 of 5.

## Family: truncate

### SCRIBE v2 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.911 [0.864, 0.957] |
| 10 | 20 | 0 | 0.911 [0.864, 0.954] |
| 25 | 20 | 0 | 0.910 [0.846, 0.966] |
| 50 | 20 | 0 | 0.807 [0.723, 0.896] |
| 75 | 20 | 0 | 0.861 [0.787, 0.927] |
| 100 | 20 | 0 | 0.936 [0.895, 0.971] |

Spearman, samples: −0.029 [−0.192, 0.136]. Spearman, means: −0.058. Steps the right way: 2 of 5.

### SCRIBE v1 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.555 [0.460, 0.632] |
| 10 | 20 | 0 | 0.554 [0.456, 0.630] |
| 25 | 20 | 0 | 0.509 [0.424, 0.590] |
| 50 | 20 | 0 | 0.496 [0.438, 0.557] |
| 75 | 20 | 0 | 0.478 [0.409, 0.542] |
| 100 | 20 | 0 | 0.476 [0.394, 0.554] |

Spearman, samples: −0.211 [−0.375, −0.027]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.907 [0.876, 0.937] |
| 10 | 20 | 0 | 0.907 [0.873, 0.933] |
| 25 | 20 | 0 | 0.907 [0.879, 0.931] |
| 50 | 20 | 0 | 0.905 [0.885, 0.925] |
| 75 | 20 | 0 | 0.912 [0.892, 0.930] |
| 100 | 20 | 0 | 0.923 [0.893, 0.949] |

Spearman, samples: 0.070 [−0.123, 0.257]. Spearman, means: 0.429. Steps the right way: 3 of 5.

### distinct-1 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.704, 0.768] |
| 10 | 20 | 0 | 0.736 [0.696, 0.770] |
| 25 | 20 | 0 | 0.748 [0.712, 0.782] |
| 50 | 20 | 0 | 0.774 [0.738, 0.805] |
| 75 | 20 | 0 | 0.788 [0.750, 0.818] |
| 100 | 20 | 0 | 0.804 [0.763, 0.844] |

Spearman, samples: 0.319 [0.150, 0.483]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-2 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.959 [0.939, 0.974] |
| 10 | 20 | 0 | 0.959 [0.938, 0.975] |
| 25 | 20 | 0 | 0.961 [0.944, 0.976] |
| 50 | 20 | 0 | 0.970 [0.951, 0.984] |
| 75 | 20 | 0 | 0.970 [0.950, 0.986] |
| 100 | 20 | 0 | 0.971 [0.948, 0.991] |

Spearman, samples: 0.246 [0.070, 0.424]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### distinct-3 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.988 [0.978, 0.996] |
| 10 | 20 | 0 | 0.988 [0.978, 0.996] |
| 25 | 20 | 0 | 0.990 [0.982, 0.997] |
| 50 | 20 | 0 | 0.993 [0.986, 0.998] |
| 75 | 20 | 0 | 0.992 [0.983, 0.998] |
| 100 | 20 | 0 | 0.992 [0.982, 1.000] |

Spearman, samples: 0.159 [−0.010, 0.335]. Spearman, means: 0.812. Steps the right way: 1 of 5.

### distinct-4 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.997 [0.993, 1.000] |
| 10 | 20 | 0 | 0.997 [0.993, 1.000] |
| 25 | 20 | 0 | 0.998 [0.996, 1.000] |
| 50 | 20 | 0 | 0.998 [0.996, 1.000] |
| 75 | 20 | 0 | 0.998 [0.996, 1.000] |
| 100 | 20 | 0 | 0.998 [0.995, 1.000] |

Spearman, samples: 0.047 [−0.130, 0.215]. Spearman, means: 0.696. Steps the right way: 2 of 5.

### rep-2 (lower is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.041 [0.026, 0.061] |
| 10 | 20 | 0 | 0.041 [0.025, 0.062] |
| 25 | 20 | 0 | 0.039 [0.024, 0.056] |
| 50 | 20 | 0 | 0.030 [0.016, 0.049] |
| 75 | 20 | 0 | 0.030 [0.014, 0.050] |
| 100 | 20 | 0 | 0.029 [0.009, 0.052] |

Spearman, samples: 0.246 [0.070, 0.424]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### rep-3 (lower is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.012 [0.004, 0.022] |
| 10 | 20 | 0 | 0.012 [0.004, 0.022] |
| 25 | 20 | 0 | 0.010 [0.003, 0.018] |
| 50 | 20 | 0 | 0.007 [0.002, 0.014] |
| 75 | 20 | 0 | 0.008 [0.002, 0.017] |
| 100 | 20 | 0 | 0.008 [0.000, 0.018] |

Spearman, samples: 0.159 [−0.010, 0.335]. Spearman, means: 0.812. Steps the right way: 1 of 5.

### rep-4 (lower is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.003 [0.000, 0.007] |
| 25 | 20 | 0 | 0.002 [0.000, 0.004] |
| 50 | 20 | 0 | 0.002 [0.000, 0.004] |
| 75 | 20 | 0 | 0.002 [0.000, 0.004] |
| 100 | 20 | 0 | 0.002 [0.000, 0.005] |

Spearman, samples: 0.047 [−0.130, 0.215]. Spearman, means: 0.696. Steps the right way: 2 of 5.

### seq-rep-4 (lower is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.003 [0.000, 0.007] |
| 25 | 20 | 0 | 0.002 [0.000, 0.004] |
| 50 | 20 | 0 | 0.002 [0.000, 0.004] |
| 75 | 20 | 0 | 0.002 [0.000, 0.004] |
| 100 | 20 | 0 | 0.002 [0.000, 0.005] |

Spearman, samples: 0.047 [−0.130, 0.215]. Spearman, means: 0.696. Steps the right way: 2 of 5.

### diversity (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.946 [0.916, 0.969] |
| 10 | 20 | 0 | 0.946 [0.914, 0.971] |
| 25 | 20 | 0 | 0.951 [0.926, 0.972] |
| 50 | 20 | 0 | 0.962 [0.935, 0.982] |
| 75 | 20 | 0 | 0.961 [0.932, 0.984] |
| 100 | 20 | 0 | 0.963 [0.929, 0.990] |

Spearman, samples: 0.253 [0.079, 0.424]. Spearman, means: 0.886. Steps the right way: 2 of 5.

### type-token ratio (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.704, 0.768] |
| 10 | 20 | 0 | 0.736 [0.696, 0.770] |
| 25 | 20 | 0 | 0.748 [0.712, 0.782] |
| 50 | 20 | 0 | 0.774 [0.738, 0.805] |
| 75 | 20 | 0 | 0.788 [0.750, 0.818] |
| 100 | 20 | 0 | 0.804 [0.763, 0.844] |

Spearman, samples: 0.319 [0.150, 0.483]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### gzip ratio (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.625 [0.604, 0.648] |
| 10 | 20 | 0 | 0.625 [0.602, 0.648] |
| 25 | 20 | 0 | 0.644 [0.624, 0.664] |
| 50 | 20 | 0 | 0.674 [0.653, 0.695] |
| 75 | 20 | 0 | 0.705 [0.683, 0.732] |
| 100 | 20 | 0 | 0.749 [0.728, 0.768] |

Spearman, samples: 0.654 [0.546, 0.754]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### GPT-2 PPL (lower is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 36.105 [25.268, 49.498] |
| 10 | 20 | 0 | 35.960 [25.456, 49.088] |
| 25 | 20 | 0 | 41.837 [29.950, 56.792] |
| 50 | 20 | 0 | 45.384 [32.909, 57.984] |
| 75 | 20 | 0 | 56.636 [36.654, 82.029] |
| 100 | 20 | 0 | 75.370 [51.339, 107.164] |

Spearman, samples: −0.370 [−0.511, −0.208]. Spearman, means: −0.943. Steps the right way: 4 of 5.

### CoLA acceptability, mean (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.882 [0.850, 0.912] |
| 10 | 20 | 0 | 0.877 [0.837, 0.912] |
| 25 | 20 | 0 | 0.829 [0.781, 0.874] |
| 50 | 20 | 0 | 0.707 [0.663, 0.757] |
| 75 | 20 | 0 | 0.688 [0.634, 0.746] |
| 100 | 20 | 0 | 0.594 [0.538, 0.646] |

Spearman, samples: −0.702 [−0.775, −0.616]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### CoLA acceptability, min (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.742 [0.634, 0.833] |
| 10 | 20 | 0 | 0.742 [0.639, 0.832] |
| 25 | 20 | 0 | 0.614 [0.485, 0.736] |
| 50 | 20 | 0 | 0.398 [0.307, 0.495] |
| 75 | 20 | 0 | 0.345 [0.239, 0.457] |
| 100 | 20 | 0 | 0.230 [0.169, 0.294] |

Spearman, samples: −0.632 [−0.729, −0.525]. Spearman, means: −0.986. Steps the right way: 4 of 5.

### MiniLM coherence, adjacent (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.357 [0.314, 0.408] |
| 10 | 20 | 0 | 0.356 [0.309, 0.406] |
| 25 | 20 | 0 | 0.330 [0.283, 0.382] |
| 50 | 20 | 0 | 0.283 [0.239, 0.339] |
| 75 | 20 | 0 | 0.285 [0.237, 0.334] |
| 100 | 20 | 0 | 0.283 [0.235, 0.340] |

Spearman, samples: −0.271 [−0.435, −0.097]. Spearman, means: −0.943. Steps the right way: 4 of 5.

### MiniLM coherence, all pairs (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.332 [0.287, 0.381] |
| 10 | 20 | 0 | 0.332 [0.284, 0.381] |
| 25 | 20 | 0 | 0.303 [0.258, 0.353] |
| 50 | 20 | 0 | 0.271 [0.222, 0.324] |
| 75 | 20 | 0 | 0.274 [0.223, 0.321] |
| 100 | 20 | 0 | 0.267 [0.218, 0.319] |

Spearman, samples: −0.195 [−0.365, −0.010]. Spearman, means: −0.943. Steps the right way: 4 of 5.

### GPT-2 detector P(human) (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.807 [0.644, 0.934] |
| 10 | 20 | 0 | 0.807 [0.661, 0.942] |
| 25 | 20 | 0 | 0.752 [0.593, 0.901] |
| 50 | 20 | 0 | 0.722 [0.563, 0.863] |
| 75 | 20 | 0 | 0.592 [0.428, 0.741] |
| 100 | 20 | 0 | 0.627 [0.512, 0.755] |

Spearman, samples: −0.339 [−0.496, −0.168]. Spearman, means: −0.943. Steps the right way: 4 of 5.

## Family: typo

### SCRIBE v2 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.911 [0.861, 0.955] |
| 10 | 20 | 0 | 0.886 [0.813, 0.949] |
| 25 | 20 | 0 | 0.457 [0.353, 0.573] |
| 50 | 20 | 0 | 0.016 [0.008, 0.025] |
| 75 | 20 | 0 | 0.003 [0.000, 0.009] |
| 100 | 20 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: −0.938 [−0.956, −0.920]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.555 [0.458, 0.633] |
| 10 | 20 | 0 | 0.373 [0.288, 0.458] |
| 25 | 20 | 0 | 0.115 [0.070, 0.166] |
| 50 | 20 | 0 | 0.015 [0.007, 0.026] |
| 75 | 20 | 0 | 0.006 [0.005, 0.008] |
| 100 | 20 | 0 | 0.004 [0.003, 0.005] |

Spearman, samples: −0.892 [−0.921, −0.864]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.907 [0.874, 0.935] |
| 10 | 20 | 0 | 0.838 [0.814, 0.862] |
| 25 | 20 | 0 | 0.719 [0.700, 0.736] |
| 50 | 20 | 0 | 0.523 [0.510, 0.535] |
| 75 | 20 | 0 | 0.326 [0.309, 0.346] |
| 100 | 20 | 0 | 0.116 [0.103, 0.130] |

Spearman, samples: −0.970 [−0.981, −0.955]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### distinct-1 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.699, 0.772] |
| 10 | 20 | 0 | 0.759 [0.723, 0.792] |
| 25 | 20 | 0 | 0.797 [0.766, 0.829] |
| 50 | 20 | 0 | 0.850 [0.824, 0.871] |
| 75 | 20 | 0 | 0.877 [0.853, 0.900] |
| 100 | 20 | 0 | 0.888 [0.862, 0.911] |

Spearman, samples: 0.662 [0.555, 0.752]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### distinct-2 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.959 [0.938, 0.975] |
| 10 | 20 | 0 | 0.965 [0.945, 0.981] |
| 25 | 20 | 0 | 0.978 [0.963, 0.990] |
| 50 | 20 | 0 | 0.992 [0.986, 0.996] |
| 75 | 20 | 0 | 0.997 [0.994, 1.000] |
| 100 | 20 | 0 | 0.997 [0.994, 1.000] |

Spearman, samples: 0.645 [0.538, 0.737]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### distinct-3 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.988 [0.978, 0.996] |
| 10 | 20 | 0 | 0.990 [0.979, 0.998] |
| 25 | 20 | 0 | 0.996 [0.990, 0.999] |
| 50 | 20 | 0 | 0.998 [0.996, 1.000] |
| 75 | 20 | 0 | 1.000 [1.000, 1.000] |
| 100 | 20 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.387 [0.267, 0.509]. Spearman, means: 0.986. Steps the right way: 0 of 5.

### distinct-4 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.997 [0.993, 1.000] |
| 10 | 20 | 0 | 0.997 [0.993, 1.000] |
| 25 | 20 | 0 | 0.999 [0.997, 1.000] |
| 50 | 20 | 0 | 1.000 [1.000, 1.000] |
| 75 | 20 | 0 | 1.000 [1.000, 1.000] |
| 100 | 20 | 0 | 1.000 [1.000, 1.000] |

Spearman, samples: 0.260 [0.135, 0.365]. Spearman, means: 0.926. Steps the right way: 0 of 5.

### rep-2 (lower is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.041 [0.025, 0.062] |
| 10 | 20 | 0 | 0.035 [0.019, 0.055] |
| 25 | 20 | 0 | 0.022 [0.010, 0.037] |
| 50 | 20 | 0 | 0.008 [0.004, 0.014] |
| 75 | 20 | 0 | 0.003 [0.000, 0.006] |
| 100 | 20 | 0 | 0.003 [0.000, 0.006] |

Spearman, samples: 0.645 [0.538, 0.737]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### rep-3 (lower is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.012 [0.004, 0.022] |
| 10 | 20 | 0 | 0.010 [0.002, 0.021] |
| 25 | 20 | 0 | 0.004 [0.001, 0.010] |
| 50 | 20 | 0 | 0.002 [0.000, 0.004] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 20 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.387 [0.267, 0.509]. Spearman, means: 0.986. Steps the right way: 0 of 5.

### rep-4 (lower is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.003 [0.000, 0.007] |
| 25 | 20 | 0 | 0.001 [0.000, 0.003] |
| 50 | 20 | 0 | 0.000 [0.000, 0.000] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 20 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.260 [0.135, 0.365]. Spearman, means: 0.926. Steps the right way: 0 of 5.

### seq-rep-4 (lower is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.003 [0.000, 0.007] |
| 10 | 20 | 0 | 0.003 [0.000, 0.007] |
| 25 | 20 | 0 | 0.001 [0.000, 0.003] |
| 50 | 20 | 0 | 0.000 [0.000, 0.000] |
| 75 | 20 | 0 | 0.000 [0.000, 0.000] |
| 100 | 20 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: 0.260 [0.135, 0.365]. Spearman, means: 0.926. Steps the right way: 0 of 5.

### diversity (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.946 [0.915, 0.970] |
| 10 | 20 | 0 | 0.954 [0.921, 0.978] |
| 25 | 20 | 0 | 0.973 [0.952, 0.989] |
| 50 | 20 | 0 | 0.990 [0.983, 0.996] |
| 75 | 20 | 0 | 0.997 [0.994, 1.000] |
| 100 | 20 | 0 | 0.997 [0.994, 1.000] |

Spearman, samples: 0.641 [0.533, 0.729]. Spearman, means: 0.943. Steps the right way: 1 of 5.

### type-token ratio (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.736 [0.699, 0.772] |
| 10 | 20 | 0 | 0.759 [0.723, 0.792] |
| 25 | 20 | 0 | 0.797 [0.766, 0.829] |
| 50 | 20 | 0 | 0.850 [0.824, 0.871] |
| 75 | 20 | 0 | 0.877 [0.853, 0.900] |
| 100 | 20 | 0 | 0.888 [0.862, 0.911] |

Spearman, samples: 0.662 [0.555, 0.752]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### gzip ratio (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.625 [0.604, 0.646] |
| 10 | 20 | 0 | 0.631 [0.610, 0.654] |
| 25 | 20 | 0 | 0.642 [0.623, 0.663] |
| 50 | 20 | 0 | 0.657 [0.641, 0.675] |
| 75 | 20 | 0 | 0.670 [0.654, 0.685] |
| 100 | 20 | 0 | 0.677 [0.664, 0.692] |

Spearman, samples: 0.458 [0.305, 0.599]. Spearman, means: 1.000. Steps the right way: 0 of 5.

### GPT-2 PPL (lower is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 36.105 [25.404, 49.732] |
| 10 | 20 | 0 | 92.835 [71.052, 119.822] |
| 25 | 20 | 0 | 240.448 [186.600, 306.372] |
| 50 | 20 | 0 | 551.396 [469.272, 624.946] |
| 75 | 20 | 0 | 825.557 [706.432, 946.635] |
| 100 | 20 | 0 | 936.262 [852.016, 1025.203] |

Spearman, samples: −0.925 [−0.946, −0.901]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### CoLA acceptability, mean (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.882 [0.849, 0.911] |
| 10 | 20 | 0 | 0.706 [0.668, 0.743] |
| 25 | 20 | 0 | 0.529 [0.474, 0.583] |
| 50 | 20 | 0 | 0.399 [0.359, 0.443] |
| 75 | 20 | 0 | 0.298 [0.269, 0.333] |
| 100 | 20 | 0 | 0.261 [0.238, 0.289] |

Spearman, samples: −0.916 [−0.938, −0.887]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### CoLA acceptability, min (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.742 [0.633, 0.830] |
| 10 | 20 | 0 | 0.460 [0.391, 0.532] |
| 25 | 20 | 0 | 0.330 [0.273, 0.384] |
| 50 | 20 | 0 | 0.196 [0.174, 0.220] |
| 75 | 20 | 0 | 0.185 [0.158, 0.216] |
| 100 | 20 | 0 | 0.167 [0.150, 0.186] |

Spearman, samples: −0.759 [−0.835, −0.668]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### MiniLM coherence, adjacent (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.357 [0.312, 0.406] |
| 10 | 20 | 0 | 0.326 [0.282, 0.377] |
| 25 | 20 | 0 | 0.317 [0.273, 0.364] |
| 50 | 20 | 0 | 0.319 [0.286, 0.356] |
| 75 | 20 | 0 | 0.336 [0.301, 0.369] |
| 100 | 20 | 0 | 0.368 [0.338, 0.397] |

Spearman, samples: 0.110 [−0.063, 0.281]. Spearman, means: 0.257. Steps the right way: 2 of 5.

### MiniLM coherence, all pairs (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.332 [0.286, 0.381] |
| 10 | 20 | 0 | 0.312 [0.272, 0.357] |
| 25 | 20 | 0 | 0.294 [0.249, 0.341] |
| 50 | 20 | 0 | 0.310 [0.276, 0.346] |
| 75 | 20 | 0 | 0.337 [0.307, 0.365] |
| 100 | 20 | 0 | 0.370 [0.336, 0.401] |

Spearman, samples: 0.197 [0.018, 0.373]. Spearman, means: 0.486. Steps the right way: 2 of 5.

### GPT-2 detector P(human) (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 20 | 0 | 0.807 [0.658, 0.939] |
| 10 | 20 | 0 | 0.972 [0.920, 0.999] |
| 25 | 20 | 0 | 0.989 [0.971, 1.000] |
| 50 | 20 | 0 | 0.975 [0.925, 1.000] |
| 75 | 20 | 0 | 0.993 [0.987, 0.999] |
| 100 | 20 | 0 | 0.994 [0.988, 0.998] |

Spearman, samples: 0.225 [0.084, 0.356]. Spearman, means: 0.943. Steps the right way: 1 of 5.

## Summary

Spearman of the oriented means with strength (steps the right way). Last column: families with Spearman on the means <= −0.900.

| metric | dropword | nounswap | offtopic | reorder | repeat | subst | swap | truncate | typo | families hit |
|---|---|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | −1.000 (5/5) | −1.000 (5/5) | −0.943 (4/5) | −0.406 (1/5) | −1.000 (5/5) | −1.000 (5/5) | −1.000 (5/5) | −0.058 (2/5) | −1.000 (5/5) | 7 of 9 |
| SCRIBE v1 | −0.371 (3/5) | −1.000 (5/5) | −0.943 (4/5) | −0.754 (2/5) | −1.000 (5/5) | −1.000 (5/5) | −1.000 (5/5) | −1.000 (5/5) | −1.000 (5/5) | 7 of 9 |
| surface S | 0.771 (2/5) | −0.486 (3/5) | −0.771 (3/5) | −0.030 (2/5) | −1.000 (5/5) | −0.714 (2/5) | 0.714 (2/5) | 0.429 (3/5) | −1.000 (5/5) | 2 of 9 |
| distinct-1 | 1.000 (0/5) | 1.000 (0/5) | 1.000 (0/5) | n/a (0/5) | −1.000 (5/5) | 0.829 (1/5) | n/a (0/5) | 1.000 (0/5) | 1.000 (0/5) | 1 of 9 |
| distinct-2 | 0.986 (0/5) | 1.000 (0/5) | 0.771 (2/5) | n/a (0/5) | −1.000 (5/5) | 0.829 (1/5) | 0.657 (1/5) | 0.943 (1/5) | 0.943 (1/5) | 1 of 9 |
| distinct-3 | 0.986 (0/5) | 0.943 (1/5) | 0.116 (1/5) | n/a (0/5) | −1.000 (5/5) | 0.771 (2/5) | 0.600 (2/5) | 0.812 (1/5) | 0.986 (0/5) | 1 of 9 |
| distinct-4 | 0.941 (0/5) | 0.926 (0/5) | 0.116 (1/5) | n/a (0/5) | −1.000 (5/5) | 0.845 (0/5) | −0.086 (3/5) | 0.696 (2/5) | 0.926 (0/5) | 1 of 9 |
| rep-2 | 0.986 (0/5) | 1.000 (0/5) | 0.771 (2/5) | n/a (0/5) | −1.000 (5/5) | 0.829 (1/5) | 0.657 (1/5) | 0.943 (1/5) | 0.943 (1/5) | 1 of 9 |
| rep-3 | 0.986 (0/5) | 0.943 (1/5) | 0.116 (1/5) | n/a (0/5) | −1.000 (5/5) | 0.771 (2/5) | 0.600 (2/5) | 0.812 (1/5) | 0.986 (0/5) | 1 of 9 |
| rep-4 | 0.941 (0/5) | 0.926 (0/5) | 0.116 (1/5) | n/a (0/5) | −1.000 (5/5) | 0.845 (0/5) | −0.086 (3/5) | 0.696 (2/5) | 0.926 (0/5) | 1 of 9 |
| seq-rep-4 | 0.941 (0/5) | 0.926 (0/5) | 0.116 (1/5) | n/a (0/5) | −1.000 (5/5) | 0.845 (0/5) | −0.086 (3/5) | 0.696 (2/5) | 0.926 (0/5) | 1 of 9 |
| diversity | 0.986 (0/5) | 0.943 (1/5) | 0.600 (2/5) | n/a (0/5) | −1.000 (5/5) | 0.829 (1/5) | 0.657 (1/5) | 0.886 (2/5) | 0.943 (1/5) | 1 of 9 |
| type-token ratio | 1.000 (0/5) | 1.000 (0/5) | 1.000 (0/5) | n/a (0/5) | −1.000 (5/5) | 0.829 (1/5) | n/a (0/5) | 1.000 (0/5) | 1.000 (0/5) | 1 of 9 |
| gzip ratio | 1.000 (0/5) | 1.000 (0/5) | 0.771 (2/5) | 0.754 (2/5) | −1.000 (5/5) | 1.000 (0/5) | 1.000 (0/5) | 1.000 (0/5) | 1.000 (0/5) | 1 of 9 |
| GPT-2 PPL | −0.943 (4/5) | −1.000 (5/5) | −0.886 (3/5) | −0.870 (2/5) | 1.000 (0/5) | −1.000 (5/5) | −1.000 (5/5) | −0.943 (4/5) | −1.000 (5/5) | 6 of 9 |
| CoLA acceptability, mean | −0.429 (4/5) | −1.000 (5/5) | 0.486 (2/5) | 0.912 (1/5) | −0.714 (3/5) | −1.000 (5/5) | −1.000 (5/5) | −1.000 (5/5) | −1.000 (5/5) | 5 of 9 |
| CoLA acceptability, min | −0.429 (4/5) | −1.000 (5/5) | −0.812 (3/5) | 0.956 (0/5) | 0.986 (0/5) | −1.000 (5/5) | −1.000 (5/5) | −0.986 (4/5) | −1.000 (5/5) | 5 of 9 |
| MiniLM coherence, adjacent | −0.657 (4/5) | −1.000 (5/5) | −1.000 (5/5) | −0.290 (3/5) | 1.000 (0/5) | −0.943 (4/5) | −0.429 (3/5) | −0.943 (4/5) | 0.257 (2/5) | 4 of 9 |
| MiniLM coherence, all pairs | −0.429 (4/5) | −1.000 (5/5) | −1.000 (5/5) | −0.759 (2/5) | 1.000 (0/5) | −0.943 (4/5) | −0.200 (3/5) | −0.943 (4/5) | 0.486 (2/5) | 4 of 9 |
| GPT-2 detector P(human) | −0.829 (4/5) | −1.000 (5/5) | −0.829 (4/5) | −0.116 (3/5) | −1.000 (5/5) | −0.943 (4/5) | −0.086 (3/5) | −0.943 (4/5) | 0.943 (1/5) | 4 of 9 |

Spearman at sample level [95% interval]:

| metric | dropword | nounswap | offtopic | reorder | repeat | subst | swap | truncate | typo |
|---|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | −0.857 [−0.899, −0.813] | −0.826 [−0.869, −0.783] | −0.778 [−0.829, −0.729] | −0.045 [−0.229, 0.139] | −0.927 [−0.947, −0.904] | −0.936 [−0.952, −0.919] | −0.916 [−0.936, −0.893] | −0.029 [−0.192, 0.136] | −0.938 [−0.956, −0.920] |
| SCRIBE v1 | −0.091 [−0.285, 0.086] | −0.723 [−0.827, −0.605] | −0.589 [−0.721, −0.446] | −0.025 [−0.201, 0.151] | −0.895 [−0.920, −0.863] | −0.811 [−0.869, −0.739] | −0.693 [−0.788, −0.585] | −0.211 [−0.375, −0.027] | −0.892 [−0.921, −0.864] |
| surface S | 0.085 [−0.080, 0.264] | −0.332 [−0.497, −0.146] | −0.171 [−0.346, 0.015] | 0.007 [−0.167, 0.182] | −0.899 [−0.923, −0.867] | −0.276 [−0.430, −0.102] | 0.046 [−0.127, 0.229] | 0.070 [−0.123, 0.257] | −0.970 [−0.981, −0.955] |
| distinct-1 | 0.724 [0.629, 0.806] | 0.292 [0.133, 0.453] | 0.269 [0.079, 0.441] | 0.000 [−0.171, 0.192] | −0.884 [−0.912, −0.853] | 0.405 [0.253, 0.551] | 0.000 [−0.172, 0.171] | 0.319 [0.150, 0.483] | 0.662 [0.555, 0.752] |
| distinct-2 | 0.702 [0.628, 0.768] | 0.312 [0.152, 0.463] | 0.278 [0.124, 0.422] | 0.000 [−0.169, 0.194] | −0.906 [−0.931, −0.877] | 0.559 [0.428, 0.676] | 0.282 [0.137, 0.423] | 0.246 [0.070, 0.424] | 0.645 [0.538, 0.737] |
| distinct-3 | 0.384 [0.260, 0.501] | 0.346 [0.193, 0.495] | 0.283 [0.115, 0.450] | 0.000 [−0.174, 0.185] | −0.915 [−0.938, −0.889] | 0.359 [0.192, 0.497] | 0.328 [0.177, 0.479] | 0.159 [−0.010, 0.335] | 0.387 [0.267, 0.509] |
| distinct-4 | 0.244 [0.104, 0.356] | 0.260 [0.137, 0.378] | 0.105 [−0.094, 0.277] | 0.000 [−0.190, 0.182] | −0.932 [−0.949, −0.911] | 0.245 [0.134, 0.355] | 0.093 [−0.103, 0.275] | 0.047 [−0.130, 0.215] | 0.260 [0.135, 0.365] |
| rep-2 | 0.702 [0.628, 0.768] | 0.312 [0.152, 0.463] | 0.278 [0.124, 0.422] | 0.000 [−0.169, 0.194] | −0.906 [−0.931, −0.877] | 0.559 [0.428, 0.676] | 0.282 [0.137, 0.423] | 0.246 [0.070, 0.424] | 0.645 [0.538, 0.737] |
| rep-3 | 0.384 [0.260, 0.501] | 0.346 [0.193, 0.495] | 0.283 [0.115, 0.450] | 0.000 [−0.174, 0.185] | −0.915 [−0.938, −0.889] | 0.359 [0.192, 0.497] | 0.328 [0.177, 0.479] | 0.159 [−0.010, 0.335] | 0.387 [0.267, 0.509] |
| rep-4 | 0.244 [0.104, 0.356] | 0.260 [0.137, 0.378] | 0.105 [−0.094, 0.277] | 0.000 [−0.190, 0.182] | −0.932 [−0.949, −0.911] | 0.245 [0.134, 0.355] | 0.093 [−0.103, 0.275] | 0.047 [−0.130, 0.215] | 0.260 [0.135, 0.365] |
| seq-rep-4 | 0.244 [0.104, 0.356] | 0.260 [0.137, 0.378] | 0.105 [−0.094, 0.277] | 0.000 [−0.190, 0.182] | −0.932 [−0.949, −0.911] | 0.245 [0.134, 0.355] | 0.093 [−0.103, 0.275] | 0.047 [−0.130, 0.215] | 0.260 [0.135, 0.365] |
| diversity | 0.684 [0.601, 0.757] | 0.310 [0.143, 0.456] | 0.284 [0.126, 0.429] | 0.000 [−0.176, 0.192] | −0.909 [−0.934, −0.880] | 0.558 [0.423, 0.676] | 0.287 [0.141, 0.430] | 0.253 [0.079, 0.424] | 0.641 [0.533, 0.729] |
| type-token ratio | 0.724 [0.629, 0.806] | 0.292 [0.133, 0.453] | 0.269 [0.079, 0.441] | 0.000 [−0.171, 0.192] | −0.884 [−0.912, −0.853] | 0.405 [0.253, 0.551] | 0.000 [−0.172, 0.171] | 0.319 [0.150, 0.483] | 0.662 [0.555, 0.752] |
| gzip ratio | 0.903 [0.867, 0.936] | 0.376 [0.212, 0.529] | 0.346 [0.165, 0.520] | 0.013 [−0.169, 0.203] | −0.885 [−0.912, −0.853] | 0.431 [0.256, 0.585] | 0.124 [−0.048, 0.302] | 0.654 [0.546, 0.754] | 0.458 [0.305, 0.599] |
| GPT-2 PPL | −0.755 [−0.806, −0.694] | −0.914 [−0.939, −0.884] | −0.567 [−0.697, −0.419] | −0.112 [−0.290, 0.065] | 0.847 [0.809, 0.882] | −0.943 [−0.959, −0.924] | −0.853 [−0.906, −0.793] | −0.370 [−0.511, −0.208] | −0.925 [−0.946, −0.901] |
| CoLA acceptability, mean | −0.410 [−0.461, −0.359] | −0.937 [−0.954, −0.916] | 0.009 [−0.152, 0.201] | 0.001 [−0.179, 0.190] | 0.055 [−0.134, 0.244] | −0.919 [−0.945, −0.887] | −0.847 [−0.892, −0.794] | −0.702 [−0.775, −0.616] | −0.916 [−0.938, −0.887] |
| CoLA acceptability, min | −0.457 [−0.538, −0.375] | −0.856 [−0.898, −0.806] | −0.074 [−0.240, 0.110] | 0.003 [−0.173, 0.189] | 0.295 [0.124, 0.453] | −0.861 [−0.914, −0.793] | −0.766 [−0.832, −0.695] | −0.632 [−0.729, −0.525] | −0.759 [−0.835, −0.668] |
| MiniLM coherence, adjacent | −0.238 [−0.396, −0.078] | −0.713 [−0.787, −0.636] | −0.790 [−0.835, −0.740] | −0.064 [−0.243, 0.120] | 0.797 [0.736, 0.849] | −0.658 [−0.733, −0.571] | −0.005 [−0.188, 0.175] | −0.271 [−0.435, −0.097] | 0.110 [−0.063, 0.281] |
| MiniLM coherence, all pairs | −0.122 [−0.275, 0.034] | −0.685 [−0.764, −0.602] | −0.837 [−0.879, −0.792] | 0.001 [−0.185, 0.174] | 0.851 [0.802, 0.889] | −0.685 [−0.750, −0.605] | 0.010 [−0.163, 0.173] | −0.195 [−0.365, −0.010] | 0.197 [0.018, 0.373] |
| GPT-2 detector P(human) | −0.410 [−0.542, −0.267] | −0.699 [−0.754, −0.640] | −0.281 [−0.439, −0.112] | −0.065 [−0.256, 0.111] | −0.520 [−0.648, −0.366] | −0.414 [−0.522, −0.294] | 0.002 [−0.169, 0.162] | −0.339 [−0.496, −0.168] | 0.225 [0.084, 0.356] |
