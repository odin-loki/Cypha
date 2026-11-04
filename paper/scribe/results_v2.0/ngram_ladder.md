# Model ladder

1488 texts in 6 systems, known order best first: human, ngram5, ngram4, ngram3, ngram2, ngram1. Spearman and Kendall (tau-b) are between the oriented system means and the known order (1 = the known order). Bootstrap: 1000 resamples of the texts within each system, seed 20261028; P(perfect) is the share of resamples whose system means are in exactly the known order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.

### SCRIBE v2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 250 | 0 | 0.906 [0.883, 0.926] |
| ngram5 | 248 | 0 | 0.085 [0.071, 0.100] |
| ngram4 | 246 | 0 | 0.100 [0.082, 0.119] |
| ngram3 | 249 | 0 | 0.100 [0.083, 0.119] |
| ngram2 | 246 | 0 | 0.034 [0.024, 0.046] |
| ngram1 | 249 | 0 | 0.001 [0.001, 0.001] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > ngram5 | 0.991 | [0.981, 0.998] | yes |
| ngram5 > ngram4 | 0.480 | [0.428, 0.530] | no |
| ngram4 > ngram3 | 0.505 | [0.454, 0.554] | no |
| ngram3 > ngram2 | 0.726 | [0.680, 0.768] | yes |
| ngram2 > ngram1 | 0.982 | [0.972, 0.991] | yes |

Spearman 0.829 [0.771, 0.943], Kendall 0.733 [0.600, 0.867], in the known order: no, P(perfect) 0.015.

### SCRIBE v3 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 250 | 0 | 0.900 [0.883, 0.915] |
| ngram5 | 248 | 0 | 0.456 [0.444, 0.467] |
| ngram4 | 246 | 0 | 0.463 [0.450, 0.476] |
| ngram3 | 249 | 0 | 0.470 [0.457, 0.482] |
| ngram2 | 246 | 0 | 0.407 [0.393, 0.422] |
| ngram1 | 249 | 0 | 0.148 [0.142, 0.155] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > ngram5 | 0.979 | [0.964, 0.992] | yes |
| ngram5 > ngram4 | 0.487 | [0.437, 0.537] | no |
| ngram4 > ngram3 | 0.501 | [0.450, 0.552] | no |
| ngram3 > ngram2 | 0.672 | [0.624, 0.719] | yes |
| ngram2 > ngram1 | 0.978 | [0.965, 0.990] | yes |

Spearman 0.771 [0.771, 0.943], Kendall 0.600 [0.600, 0.867], in the known order: no, P(perfect) 0.009.

### SCRIBE v3.1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 250 | 0 | 0.892 [0.870, 0.911] |
| ngram5 | 248 | 0 | 0.267 [0.247, 0.287] |
| ngram4 | 246 | 0 | 0.281 [0.258, 0.304] |
| ngram3 | 249 | 0 | 0.279 [0.258, 0.302] |
| ngram2 | 246 | 0 | 0.159 [0.144, 0.177] |
| ngram1 | 249 | 0 | 0.019 [0.018, 0.021] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > ngram5 | 0.975 | [0.957, 0.990] | yes |
| ngram5 > ngram4 | 0.489 | [0.440, 0.542] | no |
| ngram4 > ngram3 | 0.503 | [0.452, 0.553] | no |
| ngram3 > ngram2 | 0.732 | [0.688, 0.776] | yes |
| ngram2 > ngram1 | 0.991 | [0.985, 0.996] | yes |

Spearman 0.829 [0.771, 1.000], Kendall 0.733 [0.600, 1.000], in the known order: no, P(perfect) 0.046.

### SURPRISE (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 250 | 0 | 0.900 [0.883, 0.916] |
| ngram5 | 248 | 0 | 0.636 [0.622, 0.651] |
| ngram4 | 246 | 0 | 0.644 [0.629, 0.659] |
| ngram3 | 249 | 0 | 0.656 [0.641, 0.670] |
| ngram2 | 246 | 0 | 0.588 [0.568, 0.610] |
| ngram1 | 249 | 0 | 0.174 [0.168, 0.180] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > ngram5 | 0.930 | [0.904, 0.953] | yes |
| ngram5 > ngram4 | 0.465 | [0.414, 0.518] | no |
| ngram4 > ngram3 | 0.527 | [0.473, 0.577] | no |
| ngram3 > ngram2 | 0.610 | [0.558, 0.660] | yes |
| ngram2 > ngram1 | 0.989 | [0.979, 0.996] | yes |

Spearman 0.771 [0.771, 0.943], Kendall 0.600 [0.600, 0.867], in the known order: no, P(perfect) 0.005.

### MATCH (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 250 | 0 | 0.894 [0.872, 0.913] |
| ngram5 | 248 | 0 | 0.646 [0.632, 0.661] |
| ngram4 | 246 | 0 | 0.645 [0.630, 0.660] |
| ngram3 | 249 | 0 | 0.654 [0.639, 0.667] |
| ngram2 | 246 | 0 | 0.597 [0.576, 0.618] |
| ngram1 | 249 | 0 | 0.269 [0.255, 0.285] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > ngram5 | 0.937 | [0.911, 0.959] | yes |
| ngram5 > ngram4 | 0.538 | [0.484, 0.590] | no |
| ngram4 > ngram3 | 0.476 | [0.424, 0.529] | no |
| ngram3 > ngram2 | 0.565 | [0.509, 0.612] | yes |
| ngram2 > ngram1 | 0.924 | [0.896, 0.951] | yes |

Spearman 0.829 [0.771, 1.000], Kendall 0.733 [0.600, 1.000], in the known order: no, P(perfect) 0.054.

### SURPRISE 3.1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 250 | 0 | 0.916 [0.897, 0.932] |
| ngram5 | 248 | 0 | 0.587 [0.557, 0.619] |
| ngram4 | 246 | 0 | 0.571 [0.539, 0.602] |
| ngram3 | 249 | 0 | 0.580 [0.549, 0.611] |
| ngram2 | 246 | 0 | 0.407 [0.374, 0.444] |
| ngram1 | 249 | 0 | 0.030 [0.029, 0.031] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > ngram5 | 0.941 | [0.919, 0.961] | yes |
| ngram5 > ngram4 | 0.498 | [0.446, 0.549] | no |
| ngram4 > ngram3 | 0.513 | [0.464, 0.563] | no |
| ngram3 > ngram2 | 0.667 | [0.619, 0.716] | yes |
| ngram2 > ngram1 | 0.996 | [0.990, 0.999] | yes |

Spearman 0.943 [0.771, 1.000], Kendall 0.867 [0.600, 1.000], in the known order: no, P(perfect) 0.215.

### MATCH 3.1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 250 | 0 | 0.894 [0.871, 0.913] |
| ngram5 | 248 | 0 | 0.663 [0.650, 0.678] |
| ngram4 | 246 | 0 | 0.662 [0.647, 0.677] |
| ngram3 | 249 | 0 | 0.671 [0.656, 0.684] |
| ngram2 | 246 | 0 | 0.615 [0.594, 0.636] |
| ngram1 | 249 | 0 | 0.273 [0.258, 0.291] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > ngram5 | 0.929 | [0.899, 0.952] | yes |
| ngram5 > ngram4 | 0.538 | [0.484, 0.590] | no |
| ngram4 > ngram3 | 0.478 | [0.425, 0.531] | no |
| ngram3 > ngram2 | 0.563 | [0.508, 0.611] | yes |
| ngram2 > ngram1 | 0.931 | [0.905, 0.956] | yes |

Spearman 0.829 [0.771, 1.000], Kendall 0.733 [0.600, 1.000], in the known order: no, P(perfect) 0.056.

### min of 3 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 250 | 0 | 0.822 [0.795, 0.846] |
| ngram5 | 248 | 0 | 0.085 [0.071, 0.100] |
| ngram4 | 246 | 0 | 0.100 [0.082, 0.119] |
| ngram3 | 249 | 0 | 0.099 [0.082, 0.117] |
| ngram2 | 246 | 0 | 0.034 [0.024, 0.046] |
| ngram1 | 249 | 0 | 0.001 [0.001, 0.001] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > ngram5 | 0.975 | [0.956, 0.990] | yes |
| ngram5 > ngram4 | 0.480 | [0.428, 0.530] | no |
| ngram4 > ngram3 | 0.505 | [0.454, 0.554] | no |
| ngram3 > ngram2 | 0.726 | [0.680, 0.768] | yes |
| ngram2 > ngram1 | 0.982 | [0.972, 0.991] | yes |

Spearman 0.829 [0.771, 0.943], Kendall 0.733 [0.600, 0.867], in the known order: no, P(perfect) 0.016.

## Summary

| metric | Spearman [95% interval] | Kendall [95% interval] | in order | P(perfect) | human > ngram5 | ngram5 > ngram4 | ngram4 > ngram3 | ngram3 > ngram2 | ngram2 > ngram1 |
|---|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.829 [0.771, 0.943] | 0.733 [0.600, 0.867] | no | 0.015 | 0.991 | 0.480 | 0.505 | 0.726 | 0.982 |
| SCRIBE v3 | 0.771 [0.771, 0.943] | 0.600 [0.600, 0.867] | no | 0.009 | 0.979 | 0.487 | 0.501 | 0.672 | 0.978 |
| SCRIBE v3.1 | 0.829 [0.771, 1.000] | 0.733 [0.600, 1.000] | no | 0.046 | 0.975 | 0.489 | 0.503 | 0.732 | 0.991 |
| SURPRISE | 0.771 [0.771, 0.943] | 0.600 [0.600, 0.867] | no | 0.005 | 0.930 | 0.465 | 0.527 | 0.610 | 0.989 |
| MATCH | 0.829 [0.771, 1.000] | 0.733 [0.600, 1.000] | no | 0.054 | 0.937 | 0.538 | 0.476 | 0.565 | 0.924 |
| SURPRISE 3.1 | 0.943 [0.771, 1.000] | 0.867 [0.600, 1.000] | no | 0.215 | 0.941 | 0.498 | 0.513 | 0.667 | 0.996 |
| MATCH 3.1 | 0.829 [0.771, 1.000] | 0.733 [0.600, 1.000] | no | 0.056 | 0.929 | 0.538 | 0.478 | 0.563 | 0.931 |
| min of 3 | 0.829 [0.771, 0.943] | 0.733 [0.600, 0.867] | no | 0.016 | 0.975 | 0.480 | 0.505 | 0.726 | 0.982 |
