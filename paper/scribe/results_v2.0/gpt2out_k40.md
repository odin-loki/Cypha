# Model ladder

5000 texts in 5 systems, known order best first: human, xl-1542M-k40, large-762M-k40, medium-345M-k40, small-117M-k40. Spearman and Kendall (tau-b) are between the oriented system means and the known order (1 = the known order). Bootstrap: 1000 resamples of the texts within each system, seed 20261028; P(perfect) is the share of resamples whose system means are in exactly the known order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.

### SCRIBE v2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.917 [0.906, 0.926] |
| xl-1542M-k40 | 1000 | 0 | 0.971 [0.964, 0.978] |
| large-762M-k40 | 1000 | 0 | 0.964 [0.955, 0.971] |
| medium-345M-k40 | 1000 | 0 | 0.967 [0.960, 0.973] |
| small-117M-k40 | 1000 | 0 | 0.975 [0.967, 0.982] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M-k40 | 0.376 | [0.358, 0.397] | no |
| xl-1542M-k40 > large-762M-k40 | 0.505 | [0.489, 0.523] | no |
| large-762M-k40 > medium-345M-k40 | 0.510 | [0.492, 0.527] | no |
| medium-345M-k40 > small-117M-k40 | 0.466 | [0.449, 0.484] | no |

Spearman −0.700 [−0.900, −0.300], Kendall −0.600 [−0.800, −0.200], in the known order: no, P(perfect) 0.000.

### SCRIBE v3 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.912 [0.904, 0.919] |
| xl-1542M-k40 | 1000 | 0 | 0.903 [0.894, 0.910] |
| large-762M-k40 | 1000 | 0 | 0.884 [0.875, 0.892] |
| medium-345M-k40 | 1000 | 0 | 0.884 [0.876, 0.892] |
| small-117M-k40 | 1000 | 0 | 0.841 [0.832, 0.850] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M-k40 | 0.539 | [0.514, 0.564] | no |
| xl-1542M-k40 > large-762M-k40 | 0.543 | [0.517, 0.565] | yes |
| large-762M-k40 > medium-345M-k40 | 0.515 | [0.491, 0.541] | no |
| medium-345M-k40 > small-117M-k40 | 0.588 | [0.563, 0.613] | yes |

Spearman 0.900 [0.900, 1.000], Kendall 0.800 [0.800, 1.000], in the known order: no, P(perfect) 0.435.

### SCRIBE v3.1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.907 [0.898, 0.916] |
| xl-1542M-k40 | 1000 | 0 | 0.886 [0.875, 0.896] |
| large-762M-k40 | 1000 | 0 | 0.857 [0.845, 0.869] |
| medium-345M-k40 | 1000 | 0 | 0.872 [0.861, 0.882] |
| small-117M-k40 | 1000 | 0 | 0.800 [0.786, 0.813] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M-k40 | 0.511 | [0.486, 0.536] | yes |
| xl-1542M-k40 > large-762M-k40 | 0.551 | [0.526, 0.574] | yes |
| large-762M-k40 > medium-345M-k40 | 0.495 | [0.471, 0.521] | no |
| medium-345M-k40 > small-117M-k40 | 0.588 | [0.562, 0.612] | yes |

Spearman 0.900 [0.700, 1.000], Kendall 0.800 [0.600, 1.000], in the known order: no, P(perfect) 0.029.

### SURPRISE (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.914 [0.906, 0.921] |
| xl-1542M-k40 | 1000 | 0 | 0.847 [0.835, 0.857] |
| large-762M-k40 | 1000 | 0 | 0.815 [0.804, 0.826] |
| medium-345M-k40 | 1000 | 0 | 0.814 [0.802, 0.825] |
| small-117M-k40 | 1000 | 0 | 0.720 [0.706, 0.734] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M-k40 | 0.640 | [0.615, 0.663] | yes |
| xl-1542M-k40 > large-762M-k40 | 0.543 | [0.518, 0.567] | yes |
| large-762M-k40 > medium-345M-k40 | 0.512 | [0.488, 0.537] | no |
| medium-345M-k40 > small-117M-k40 | 0.614 | [0.590, 0.639] | yes |

Spearman 1.000 [0.900, 1.000], Kendall 1.000 [0.800, 1.000], in the known order: yes, P(perfect) 0.602.

### MATCH (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.905 [0.895, 0.914] |
| xl-1542M-k40 | 1000 | 0 | 0.891 [0.878, 0.901] |
| large-762M-k40 | 1000 | 0 | 0.872 [0.860, 0.884] |
| medium-345M-k40 | 1000 | 0 | 0.873 [0.861, 0.884] |
| small-117M-k40 | 1000 | 0 | 0.828 [0.813, 0.841] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M-k40 | 0.502 | [0.478, 0.526] | no |
| xl-1542M-k40 > large-762M-k40 | 0.531 | [0.504, 0.555] | no |
| large-762M-k40 > medium-345M-k40 | 0.517 | [0.493, 0.543] | no |
| medium-345M-k40 > small-117M-k40 | 0.542 | [0.518, 0.566] | yes |

Spearman 0.900 [0.800, 1.000], Kendall 0.800 [0.600, 1.000], in the known order: no, P(perfect) 0.444.

### SURPRISE 3.1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.926 [0.918, 0.934] |
| xl-1542M-k40 | 1000 | 0 | 0.841 [0.826, 0.856] |
| large-762M-k40 | 1000 | 0 | 0.799 [0.782, 0.814] |
| medium-345M-k40 | 1000 | 0 | 0.823 [0.808, 0.837] |
| small-117M-k40 | 1000 | 0 | 0.701 [0.683, 0.720] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M-k40 | 0.584 | [0.559, 0.607] | yes |
| xl-1542M-k40 > large-762M-k40 | 0.551 | [0.527, 0.574] | yes |
| large-762M-k40 > medium-345M-k40 | 0.492 | [0.466, 0.517] | no |
| medium-345M-k40 > small-117M-k40 | 0.601 | [0.576, 0.627] | yes |

Spearman 0.900 [0.700, 0.900], Kendall 0.800 [0.600, 0.800], in the known order: no, P(perfect) 0.013.

### MATCH 3.1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.908 [0.898, 0.917] |
| xl-1542M-k40 | 1000 | 0 | 0.894 [0.882, 0.904] |
| large-762M-k40 | 1000 | 0 | 0.877 [0.864, 0.889] |
| medium-345M-k40 | 1000 | 0 | 0.878 [0.867, 0.889] |
| small-117M-k40 | 1000 | 0 | 0.835 [0.820, 0.848] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M-k40 | 0.503 | [0.479, 0.527] | no |
| xl-1542M-k40 > large-762M-k40 | 0.531 | [0.504, 0.555] | no |
| large-762M-k40 > medium-345M-k40 | 0.517 | [0.493, 0.543] | no |
| medium-345M-k40 > small-117M-k40 | 0.540 | [0.517, 0.565] | yes |

Spearman 0.900 [0.800, 1.000], Kendall 0.800 [0.600, 1.000], in the known order: no, P(perfect) 0.412.

### min of 3 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.843 [0.831, 0.853] |
| xl-1542M-k40 | 1000 | 0 | 0.809 [0.795, 0.820] |
| large-762M-k40 | 1000 | 0 | 0.775 [0.761, 0.789] |
| medium-345M-k40 | 1000 | 0 | 0.775 [0.761, 0.787] |
| small-117M-k40 | 1000 | 0 | 0.680 [0.664, 0.694] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M-k40 | 0.556 | [0.531, 0.581] | yes |
| xl-1542M-k40 > large-762M-k40 | 0.542 | [0.517, 0.565] | yes |
| large-762M-k40 > medium-345M-k40 | 0.513 | [0.487, 0.538] | no |
| medium-345M-k40 > small-117M-k40 | 0.610 | [0.585, 0.634] | yes |

Spearman 1.000 [0.900, 1.000], Kendall 1.000 [0.800, 1.000], in the known order: yes, P(perfect) 0.526.

## Summary

| metric | Spearman [95% interval] | Kendall [95% interval] | in order | P(perfect) | human > xl-1542M-k40 | xl-1542M-k40 > large-762M-k40 | large-762M-k40 > medium-345M-k40 | medium-345M-k40 > small-117M-k40 |
|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | −0.700 [−0.900, −0.300] | −0.600 [−0.800, −0.200] | no | 0.000 | 0.376 | 0.505 | 0.510 | 0.466 |
| SCRIBE v3 | 0.900 [0.900, 1.000] | 0.800 [0.800, 1.000] | no | 0.435 | 0.539 | 0.543 | 0.515 | 0.588 |
| SCRIBE v3.1 | 0.900 [0.700, 1.000] | 0.800 [0.600, 1.000] | no | 0.029 | 0.511 | 0.551 | 0.495 | 0.588 |
| SURPRISE | 1.000 [0.900, 1.000] | 1.000 [0.800, 1.000] | yes | 0.602 | 0.640 | 0.543 | 0.512 | 0.614 |
| MATCH | 0.900 [0.800, 1.000] | 0.800 [0.600, 1.000] | no | 0.444 | 0.502 | 0.531 | 0.517 | 0.542 |
| SURPRISE 3.1 | 0.900 [0.700, 0.900] | 0.800 [0.600, 0.800] | no | 0.013 | 0.584 | 0.551 | 0.492 | 0.601 |
| MATCH 3.1 | 0.900 [0.800, 1.000] | 0.800 [0.600, 1.000] | no | 0.412 | 0.503 | 0.531 | 0.517 | 0.540 |
| min of 3 | 1.000 [0.900, 1.000] | 1.000 [0.800, 1.000] | yes | 0.526 | 0.556 | 0.542 | 0.513 | 0.610 |
