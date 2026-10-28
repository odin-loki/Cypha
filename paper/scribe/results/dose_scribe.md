# SCRIBE on all corruptions (9 families x 6 strengths x 200 texts)

Analysis `dose_scribe` (3. Held-out corruptions, dose-response). meta.py `dose`.
Inputs: SP/run/fresh/corrupt.jsonl, SP/run/fresh/corrupt.scribe.jsonl.

10800 texts, families from `meta.family`, strengths from `meta.strength`. A family without its own strength-0 texts uses the shared base texts (strength 0, family ['', 'base', 'none'] or none): 0. Bootstrap: 1000 resamples within each strength, seed 20261028. Spearman uses the oriented score, so −1 means quality falls perfectly with strength; a step moves the right way when the oriented mean falls.

## Family: dropword

### SCRIBE v2 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.905 [0.882, 0.927] |
| 10 | 200 | 0 | 0.842 [0.810, 0.872] |
| 25 | 200 | 0 | 0.447 [0.395, 0.501] |
| 50 | 200 | 0 | 0.195 [0.157, 0.230] |
| 75 | 200 | 0 | 0.053 [0.035, 0.075] |
| 100 | 200 | 0 | 0.013 [0.006, 0.022] |

Spearman, samples: −0.858 [−0.873, −0.844]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.551 [0.526, 0.579] |
| 10 | 200 | 0 | 0.502 [0.471, 0.531] |
| 25 | 200 | 0 | 0.408 [0.379, 0.438] |
| 50 | 200 | 0 | 0.380 [0.349, 0.411] |
| 75 | 200 | 0 | 0.346 [0.316, 0.373] |
| 100 | 200 | 0 | 0.370 [0.335, 0.406] |

Spearman, samples: −0.293 [−0.347, −0.244]. Spearman, means: −0.943. Steps the right way: 4 of 5.

### surface S (higher is better), dropword

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.912 [0.903, 0.920] |
| 10 | 200 | 0 | 0.914 [0.905, 0.922] |
| 25 | 200 | 0 | 0.906 [0.898, 0.913] |
| 50 | 200 | 0 | 0.926 [0.919, 0.933] |
| 75 | 200 | 0 | 0.916 [0.908, 0.923] |
| 100 | 200 | 0 | 0.852 [0.834, 0.869] |

Spearman, samples: −0.106 [−0.160, −0.052]. Spearman, means: −0.086. Steps the right way: 3 of 5.

## Family: nounswap

### SCRIBE v2 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.905 [0.884, 0.927] |
| 10 | 200 | 0 | 0.872 [0.848, 0.898] |
| 25 | 200 | 0 | 0.795 [0.765, 0.824] |
| 50 | 200 | 0 | 0.534 [0.496, 0.574] |
| 75 | 200 | 0 | 0.191 [0.167, 0.218] |
| 100 | 200 | 0 | 0.054 [0.044, 0.066] |

Spearman, samples: −0.819 [−0.835, −0.803]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.551 [0.521, 0.580] |
| 10 | 200 | 0 | 0.540 [0.513, 0.571] |
| 25 | 200 | 0 | 0.485 [0.456, 0.513] |
| 50 | 200 | 0 | 0.327 [0.302, 0.349] |
| 75 | 200 | 0 | 0.175 [0.164, 0.188] |
| 100 | 200 | 0 | 0.108 [0.103, 0.115] |

Spearman, samples: −0.707 [−0.737, −0.677]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), nounswap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.912 [0.902, 0.920] |
| 10 | 200 | 0 | 0.916 [0.908, 0.924] |
| 25 | 200 | 0 | 0.914 [0.908, 0.921] |
| 50 | 200 | 0 | 0.907 [0.902, 0.913] |
| 75 | 200 | 0 | 0.893 [0.888, 0.897] |
| 100 | 200 | 0 | 0.886 [0.882, 0.891] |

Spearman, samples: −0.298 [−0.349, −0.244]. Spearman, means: −0.829. Steps the right way: 4 of 5.

## Family: offtopic

### SCRIBE v2 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.905 [0.883, 0.927] |
| 10 | 200 | 0 | 0.893 [0.869, 0.916] |
| 25 | 200 | 0 | 0.737 [0.707, 0.768] |
| 50 | 200 | 0 | 0.484 [0.456, 0.512] |
| 75 | 200 | 0 | 0.356 [0.337, 0.377] |
| 100 | 200 | 0 | 0.364 [0.345, 0.384] |

Spearman, samples: −0.754 [−0.778, −0.729]. Spearman, means: −0.943. Steps the right way: 4 of 5.

### SCRIBE v1 (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.551 [0.522, 0.579] |
| 10 | 200 | 0 | 0.546 [0.518, 0.572] |
| 25 | 200 | 0 | 0.491 [0.464, 0.518] |
| 50 | 200 | 0 | 0.312 [0.290, 0.334] |
| 75 | 200 | 0 | 0.234 [0.218, 0.251] |
| 100 | 200 | 0 | 0.235 [0.220, 0.250] |

Spearman, samples: −0.592 [−0.629, −0.553]. Spearman, means: −0.943. Steps the right way: 4 of 5.

### surface S (higher is better), offtopic

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.912 [0.902, 0.921] |
| 10 | 200 | 0 | 0.913 [0.904, 0.921] |
| 25 | 200 | 0 | 0.916 [0.908, 0.922] |
| 50 | 200 | 0 | 0.905 [0.899, 0.911] |
| 75 | 200 | 0 | 0.906 [0.900, 0.911] |
| 100 | 200 | 0 | 0.894 [0.886, 0.902] |

Spearman, samples: −0.187 [−0.245, −0.128]. Spearman, means: −0.714. Steps the right way: 2 of 5.

## Family: reorder

### SCRIBE v2 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.905 [0.883, 0.926] |
| 10 | 200 | 0 | 0.906 [0.881, 0.927] |
| 25 | 200 | 0 | 0.905 [0.883, 0.926] |
| 50 | 200 | 0 | 0.905 [0.881, 0.926] |
| 75 | 200 | 0 | 0.907 [0.884, 0.929] |
| 100 | 200 | 0 | 0.906 [0.881, 0.928] |

Spearman, samples: 0.013 [−0.042, 0.064]. Spearman, means: 0.429. Steps the right way: 3 of 5.

### SCRIBE v1 (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.551 [0.523, 0.578] |
| 10 | 200 | 0 | 0.548 [0.518, 0.579] |
| 25 | 200 | 0 | 0.548 [0.519, 0.576] |
| 50 | 200 | 0 | 0.553 [0.523, 0.582] |
| 75 | 200 | 0 | 0.551 [0.521, 0.581] |
| 100 | 200 | 0 | 0.547 [0.516, 0.574] |

Spearman, samples: −0.002 [−0.058, 0.055]. Spearman, means: −0.371. Steps the right way: 4 of 5.

### surface S (higher is better), reorder

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.912 [0.903, 0.920] |
| 10 | 200 | 0 | 0.912 [0.902, 0.921] |
| 25 | 200 | 0 | 0.912 [0.903, 0.921] |
| 50 | 200 | 0 | 0.912 [0.903, 0.921] |
| 75 | 200 | 0 | 0.912 [0.903, 0.921] |
| 100 | 200 | 0 | 0.912 [0.903, 0.921] |

Spearman, samples: 0.000 [−0.056, 0.056]. Spearman, means: −0.143. Steps the right way: 4 of 5.

## Family: repeat

### SCRIBE v2 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.905 [0.883, 0.927] |
| 10 | 200 | 0 | 0.887 [0.857, 0.914] |
| 25 | 200 | 0 | 0.310 [0.252, 0.374] |
| 50 | 200 | 0 | 0.009 [0.000, 0.023] |
| 75 | 200 | 0 | 0.005 [0.000, 0.015] |
| 100 | 200 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: −0.917 [−0.926, −0.907]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.551 [0.522, 0.579] |
| 10 | 200 | 0 | 0.537 [0.505, 0.569] |
| 25 | 200 | 0 | 0.175 [0.141, 0.214] |
| 50 | 200 | 0 | 0.007 [0.002, 0.014] |
| 75 | 200 | 0 | 0.002 [0.001, 0.006] |
| 100 | 200 | 0 | 0.001 [0.001, 0.001] |

Spearman, samples: −0.898 [−0.907, −0.888]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), repeat

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.912 [0.903, 0.921] |
| 10 | 200 | 0 | 0.907 [0.896, 0.917] |
| 25 | 200 | 0 | 0.742 [0.721, 0.765] |
| 50 | 200 | 0 | 0.457 [0.437, 0.475] |
| 75 | 200 | 0 | 0.252 [0.238, 0.268] |
| 100 | 200 | 0 | 0.123 [0.117, 0.130] |

Spearman, samples: −0.903 [−0.912, −0.892]. Spearman, means: −1.000. Steps the right way: 5 of 5.

## Family: subst

### SCRIBE v2 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.905 [0.884, 0.928] |
| 10 | 200 | 0 | 0.770 [0.728, 0.806] |
| 25 | 200 | 0 | 0.111 [0.085, 0.141] |
| 50 | 200 | 0 | 0.004 [0.003, 0.004] |
| 75 | 200 | 0 | 0.001 [0.001, 0.001] |
| 100 | 200 | 0 | 0.001 [0.001, 0.001] |

Spearman, samples: −0.939 [−0.944, −0.934]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.551 [0.525, 0.581] |
| 10 | 200 | 0 | 0.507 [0.480, 0.532] |
| 25 | 200 | 0 | 0.403 [0.379, 0.430] |
| 50 | 200 | 0 | 0.172 [0.158, 0.186] |
| 75 | 200 | 0 | 0.082 [0.078, 0.087] |
| 100 | 200 | 0 | 0.063 [0.061, 0.066] |

Spearman, samples: −0.817 [−0.837, −0.796]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), subst

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.912 [0.903, 0.921] |
| 10 | 200 | 0 | 0.919 [0.911, 0.926] |
| 25 | 200 | 0 | 0.920 [0.914, 0.926] |
| 50 | 200 | 0 | 0.901 [0.896, 0.907] |
| 75 | 200 | 0 | 0.890 [0.886, 0.894] |
| 100 | 200 | 0 | 0.888 [0.885, 0.891] |

Spearman, samples: −0.347 [−0.396, −0.295]. Spearman, means: −0.771. Steps the right way: 3 of 5.

## Family: swap

### SCRIBE v2 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.905 [0.881, 0.927] |
| 10 | 200 | 0 | 0.620 [0.567, 0.674] |
| 25 | 200 | 0 | 0.067 [0.047, 0.090] |
| 50 | 200 | 0 | 0.005 [0.005, 0.006] |
| 75 | 200 | 0 | 0.003 [0.003, 0.004] |
| 100 | 200 | 0 | 0.003 [0.003, 0.003] |

Spearman, samples: −0.902 [−0.910, −0.893]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.551 [0.521, 0.579] |
| 10 | 200 | 0 | 0.496 [0.466, 0.524] |
| 25 | 200 | 0 | 0.388 [0.360, 0.415] |
| 50 | 200 | 0 | 0.229 [0.207, 0.253] |
| 75 | 200 | 0 | 0.147 [0.131, 0.162] |
| 100 | 200 | 0 | 0.117 [0.105, 0.130] |

Spearman, samples: −0.677 [−0.708, −0.644]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), swap

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.912 [0.903, 0.921] |
| 10 | 200 | 0 | 0.917 [0.909, 0.925] |
| 25 | 200 | 0 | 0.921 [0.913, 0.927] |
| 50 | 200 | 0 | 0.921 [0.914, 0.928] |
| 75 | 200 | 0 | 0.921 [0.913, 0.928] |
| 100 | 200 | 0 | 0.920 [0.912, 0.928] |

Spearman, samples: 0.025 [−0.037, 0.084]. Spearman, means: 0.600. Steps the right way: 2 of 5.

## Family: truncate

### SCRIBE v2 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.905 [0.882, 0.928] |
| 10 | 200 | 0 | 0.904 [0.880, 0.925] |
| 25 | 200 | 0 | 0.899 [0.877, 0.920] |
| 50 | 200 | 0 | 0.864 [0.839, 0.888] |
| 75 | 200 | 0 | 0.857 [0.830, 0.884] |
| 100 | 200 | 0 | 0.922 [0.902, 0.940] |

Spearman, samples: −0.050 [−0.104, 0.006]. Spearman, means: −0.143. Steps the right way: 4 of 5.

### SCRIBE v1 (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.551 [0.524, 0.580] |
| 10 | 200 | 0 | 0.545 [0.515, 0.573] |
| 25 | 200 | 0 | 0.532 [0.504, 0.561] |
| 50 | 200 | 0 | 0.521 [0.494, 0.550] |
| 75 | 200 | 0 | 0.493 [0.465, 0.521] |
| 100 | 200 | 0 | 0.536 [0.507, 0.564] |

Spearman, samples: −0.060 [−0.120, −0.005]. Spearman, means: −0.657. Steps the right way: 4 of 5.

### surface S (higher is better), truncate

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.912 [0.903, 0.921] |
| 10 | 200 | 0 | 0.912 [0.903, 0.920] |
| 25 | 200 | 0 | 0.911 [0.903, 0.919] |
| 50 | 200 | 0 | 0.912 [0.905, 0.919] |
| 75 | 200 | 0 | 0.907 [0.900, 0.915] |
| 100 | 200 | 0 | 0.920 [0.912, 0.928] |

Spearman, samples: 0.003 [−0.057, 0.057]. Spearman, means: 0.257. Steps the right way: 2 of 5.

## Family: typo

### SCRIBE v2 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.905 [0.879, 0.928] |
| 10 | 200 | 0 | 0.861 [0.832, 0.889] |
| 25 | 200 | 0 | 0.417 [0.381, 0.455] |
| 50 | 200 | 0 | 0.018 [0.015, 0.021] |
| 75 | 200 | 0 | 0.001 [0.001, 0.002] |
| 100 | 200 | 0 | 0.000 [0.000, 0.000] |

Spearman, samples: −0.937 [−0.944, −0.931]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### SCRIBE v1 (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.551 [0.521, 0.579] |
| 10 | 200 | 0 | 0.358 [0.332, 0.387] |
| 25 | 200 | 0 | 0.090 [0.078, 0.103] |
| 50 | 200 | 0 | 0.012 [0.011, 0.014] |
| 75 | 200 | 0 | 0.005 [0.005, 0.006] |
| 100 | 200 | 0 | 0.004 [0.004, 0.005] |

Spearman, samples: −0.909 [−0.917, −0.900]. Spearman, means: −1.000. Steps the right way: 5 of 5.

### surface S (higher is better), typo

| strength | n | missing | mean [95% interval] |
|---|---|---|---|
| 0 | 200 | 0 | 0.912 [0.903, 0.921] |
| 10 | 200 | 0 | 0.840 [0.832, 0.848] |
| 25 | 200 | 0 | 0.723 [0.716, 0.730] |
| 50 | 200 | 0 | 0.524 [0.519, 0.529] |
| 75 | 200 | 0 | 0.323 [0.317, 0.328] |
| 100 | 200 | 0 | 0.119 [0.114, 0.124] |

Spearman, samples: −0.970 [−0.974, −0.966]. Spearman, means: −1.000. Steps the right way: 5 of 5.

## Summary

Spearman of the oriented means with strength (steps the right way). Last column: families with Spearman on the means <= −0.900.

| metric | dropword | nounswap | offtopic | reorder | repeat | subst | swap | truncate | typo | families hit |
|---|---|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | −1.000 (5/5) | −1.000 (5/5) | −0.943 (4/5) | 0.429 (3/5) | −1.000 (5/5) | −1.000 (5/5) | −1.000 (5/5) | −0.143 (4/5) | −1.000 (5/5) | 7 of 9 |
| SCRIBE v1 | −0.943 (4/5) | −1.000 (5/5) | −0.943 (4/5) | −0.371 (4/5) | −1.000 (5/5) | −1.000 (5/5) | −1.000 (5/5) | −0.657 (4/5) | −1.000 (5/5) | 7 of 9 |
| surface S | −0.086 (3/5) | −0.829 (4/5) | −0.714 (2/5) | −0.143 (4/5) | −1.000 (5/5) | −0.771 (3/5) | 0.600 (2/5) | 0.257 (2/5) | −1.000 (5/5) | 2 of 9 |

Spearman at sample level [95% interval]:

| metric | dropword | nounswap | offtopic | reorder | repeat | subst | swap | truncate | typo |
|---|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | −0.858 [−0.873, −0.844] | −0.819 [−0.835, −0.803] | −0.754 [−0.778, −0.729] | 0.013 [−0.042, 0.064] | −0.917 [−0.926, −0.907] | −0.939 [−0.944, −0.934] | −0.902 [−0.910, −0.893] | −0.050 [−0.104, 0.006] | −0.937 [−0.944, −0.931] |
| SCRIBE v1 | −0.293 [−0.347, −0.244] | −0.707 [−0.737, −0.677] | −0.592 [−0.629, −0.553] | −0.002 [−0.058, 0.055] | −0.898 [−0.907, −0.888] | −0.817 [−0.837, −0.796] | −0.677 [−0.708, −0.644] | −0.060 [−0.120, −0.005] | −0.909 [−0.917, −0.900] |
| surface S | −0.106 [−0.160, −0.052] | −0.298 [−0.349, −0.244] | −0.187 [−0.245, −0.128] | 0.000 [−0.056, 0.056] | −0.903 [−0.912, −0.892] | −0.347 [−0.396, −0.295] | 0.025 [−0.037, 0.084] | 0.003 [−0.057, 0.057] | −0.970 [−0.974, −0.966] |
