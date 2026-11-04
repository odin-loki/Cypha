# Model ladder

5000 texts in 5 systems, known order best first: human, xl-1542M, large-762M, medium-345M, small-117M. Spearman and Kendall (tau-b) are between the oriented system means and the known order (1 = the known order). Bootstrap: 1000 resamples of the texts within each system, seed 20261028; P(perfect) is the share of resamples whose system means are in exactly the known order. Adjacent accuracy is P(upper > lower) over text pairs, ties one half.

### SCRIBE v2 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.917 [0.906, 0.926] |
| xl-1542M | 1000 | 0 | 0.836 [0.822, 0.848] |
| large-762M | 1000 | 0 | 0.815 [0.801, 0.829] |
| medium-345M | 1000 | 0 | 0.715 [0.696, 0.733] |
| small-117M | 1000 | 0 | 0.753 [0.737, 0.767] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M | 0.639 | [0.617, 0.660] | yes |
| xl-1542M > large-762M | 0.517 | [0.491, 0.541] | no |
| large-762M > medium-345M | 0.610 | [0.584, 0.633] | yes |
| medium-345M > small-117M | 0.470 | [0.445, 0.495] | no |

Spearman 0.900 [0.900, 0.900], Kendall 0.800 [0.800, 0.800], in the known order: no, P(perfect) 0.001.

### SCRIBE v3 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.912 [0.904, 0.919] |
| xl-1542M | 1000 | 0 | 0.845 [0.835, 0.854] |
| large-762M | 1000 | 0 | 0.827 [0.816, 0.836] |
| medium-345M | 1000 | 0 | 0.753 [0.740, 0.765] |
| small-117M | 1000 | 0 | 0.775 [0.764, 0.785] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M | 0.665 | [0.641, 0.688] | yes |
| xl-1542M > large-762M | 0.535 | [0.509, 0.560] | no |
| large-762M > medium-345M | 0.616 | [0.590, 0.639] | yes |
| medium-345M > small-117M | 0.474 | [0.446, 0.498] | no |

Spearman 0.900 [0.900, 0.900], Kendall 0.800 [0.800, 0.800], in the known order: no, P(perfect) 0.003.

### SCRIBE v3.1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.907 [0.898, 0.916] |
| xl-1542M | 1000 | 0 | 0.842 [0.831, 0.852] |
| large-762M | 1000 | 0 | 0.824 [0.812, 0.835] |
| medium-345M | 1000 | 0 | 0.740 [0.725, 0.755] |
| small-117M | 1000 | 0 | 0.771 [0.759, 0.782] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M | 0.664 | [0.640, 0.686] | yes |
| xl-1542M > large-762M | 0.532 | [0.506, 0.556] | no |
| large-762M > medium-345M | 0.617 | [0.591, 0.639] | yes |
| medium-345M > small-117M | 0.471 | [0.444, 0.495] | no |

Spearman 0.900 [0.900, 0.900], Kendall 0.800 [0.800, 0.800], in the known order: no, P(perfect) 0.001.

### SURPRISE (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.914 [0.906, 0.921] |
| xl-1542M | 1000 | 0 | 0.855 [0.845, 0.863] |
| large-762M | 1000 | 0 | 0.834 [0.824, 0.843] |
| medium-345M | 1000 | 0 | 0.778 [0.767, 0.790] |
| small-117M | 1000 | 0 | 0.795 [0.786, 0.805] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M | 0.632 | [0.609, 0.657] | yes |
| xl-1542M > large-762M | 0.545 | [0.520, 0.570] | yes |
| large-762M > medium-345M | 0.589 | [0.561, 0.611] | yes |
| medium-345M > small-117M | 0.482 | [0.455, 0.505] | no |

Spearman 0.900 [0.900, 0.900], Kendall 0.800 [0.800, 0.800], in the known order: no, P(perfect) 0.010.

### MATCH (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.905 [0.895, 0.914] |
| xl-1542M | 1000 | 0 | 0.843 [0.832, 0.853] |
| large-762M | 1000 | 0 | 0.830 [0.820, 0.841] |
| medium-345M | 1000 | 0 | 0.765 [0.753, 0.777] |
| small-117M | 1000 | 0 | 0.777 [0.766, 0.787] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M | 0.646 | [0.625, 0.671] | yes |
| xl-1542M > large-762M | 0.527 | [0.500, 0.550] | no |
| large-762M > medium-345M | 0.612 | [0.587, 0.636] | yes |
| medium-345M > small-117M | 0.483 | [0.458, 0.509] | no |

Spearman 0.900 [0.800, 1.000], Kendall 0.800 [0.600, 1.000], in the known order: no, P(perfect) 0.060.

### SURPRISE 3.1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.926 [0.918, 0.934] |
| xl-1542M | 1000 | 0 | 0.872 [0.862, 0.882] |
| large-762M | 1000 | 0 | 0.858 [0.848, 0.868] |
| medium-345M | 1000 | 0 | 0.787 [0.773, 0.802] |
| small-117M | 1000 | 0 | 0.815 [0.803, 0.827] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M | 0.638 | [0.615, 0.662] | yes |
| xl-1542M > large-762M | 0.539 | [0.513, 0.563] | no |
| large-762M > medium-345M | 0.599 | [0.573, 0.623] | yes |
| medium-345M > small-117M | 0.475 | [0.449, 0.500] | no |

Spearman 0.900 [0.800, 0.900], Kendall 0.800 [0.600, 0.800], in the known order: no, P(perfect) 0.000.

### MATCH 3.1 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.908 [0.898, 0.917] |
| xl-1542M | 1000 | 0 | 0.848 [0.837, 0.857] |
| large-762M | 1000 | 0 | 0.835 [0.825, 0.845] |
| medium-345M | 1000 | 0 | 0.772 [0.759, 0.783] |
| small-117M | 1000 | 0 | 0.785 [0.774, 0.796] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M | 0.648 | [0.627, 0.672] | yes |
| xl-1542M > large-762M | 0.527 | [0.500, 0.551] | no |
| large-762M > medium-345M | 0.611 | [0.586, 0.636] | yes |
| medium-345M > small-117M | 0.481 | [0.456, 0.506] | no |

Spearman 0.900 [0.800, 1.000], Kendall 0.800 [0.600, 1.000], in the known order: no, P(perfect) 0.034.

### min of 3 (higher is better)

| system | n | missing | mean [95% interval] |
|---|---|---|---|
| human | 1000 | 0 | 0.843 [0.831, 0.853] |
| xl-1542M | 1000 | 0 | 0.758 [0.745, 0.770] |
| large-762M | 1000 | 0 | 0.733 [0.720, 0.747] |
| medium-345M | 1000 | 0 | 0.642 [0.625, 0.657] |
| small-117M | 1000 | 0 | 0.671 [0.657, 0.684] |

| upper > lower | accuracy | 95% interval | intervals disjoint |
|---|---|---|---|
| human > xl-1542M | 0.651 | [0.628, 0.675] | yes |
| xl-1542M > large-762M | 0.534 | [0.508, 0.559] | no |
| large-762M > medium-345M | 0.607 | [0.581, 0.630] | yes |
| medium-345M > small-117M | 0.474 | [0.447, 0.498] | no |

Spearman 0.900 [0.900, 0.900], Kendall 0.800 [0.800, 0.800], in the known order: no, P(perfect) 0.002.

## Summary

| metric | Spearman [95% interval] | Kendall [95% interval] | in order | P(perfect) | human > xl-1542M | xl-1542M > large-762M | large-762M > medium-345M | medium-345M > small-117M |
|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.900 [0.900, 0.900] | 0.800 [0.800, 0.800] | no | 0.001 | 0.639 | 0.517 | 0.610 | 0.470 |
| SCRIBE v3 | 0.900 [0.900, 0.900] | 0.800 [0.800, 0.800] | no | 0.003 | 0.665 | 0.535 | 0.616 | 0.474 |
| SCRIBE v3.1 | 0.900 [0.900, 0.900] | 0.800 [0.800, 0.800] | no | 0.001 | 0.664 | 0.532 | 0.617 | 0.471 |
| SURPRISE | 0.900 [0.900, 0.900] | 0.800 [0.800, 0.800] | no | 0.010 | 0.632 | 0.545 | 0.589 | 0.482 |
| MATCH | 0.900 [0.800, 1.000] | 0.800 [0.600, 1.000] | no | 0.060 | 0.646 | 0.527 | 0.612 | 0.483 |
| SURPRISE 3.1 | 0.900 [0.800, 0.900] | 0.800 [0.600, 0.800] | no | 0.000 | 0.638 | 0.539 | 0.599 | 0.475 |
| MATCH 3.1 | 0.900 [0.800, 1.000] | 0.800 [0.600, 1.000] | no | 0.034 | 0.648 | 0.527 | 0.611 | 0.481 |
| min of 3 | 0.900 [0.900, 0.900] | 0.800 [0.800, 0.800] | no | 0.002 | 0.651 | 0.534 | 0.607 | 0.474 |
