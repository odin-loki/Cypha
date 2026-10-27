# The same on attacks_50 (first 50 prompts) with every baseline

Analysis `attacks_50_all` (7. Attacks). meta.py `ladder`.
Inputs: SP/run/baselines/subsets/attacks_50.jsonl, SP/run/attacks/attacks.v2.jsonl, SP/run/attacks/attacks.v2p.jsonl, SP/run/attacks/attacks.v2_noi1.jsonl, SP/run/baselines/attacks_50/stats.jsonl.
Not available yet (left out): R/attacks_50/ppl_gpt2.jsonl, R/attacks_50/detector.jsonl.

500 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.900 [0.843, 0.943] |
| copy_pack | 50 | 0 | 0.097 [0.085, 0.108] |
| offprompt | 50 | 0 | 0.921 [0.880, 0.957] |
| echo | 50 | 0 | 0.000 [0.000, 0.000] |
| stuffing | 50 | 0 | 0.016 [0.011, 0.022] |
| generic | 50 | 0 | 0.835 [0.835, 0.835] |
| generic_topic | 50 | 0 | 0.912 [0.900, 0.925] |
| numbers | 50 | 0 | 0.900 [0.846, 0.942] |
| negation | 50 | 0 | 0.894 [0.835, 0.941] |
| fwloop | 50 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.980 | [0.940, 1.000] | yes | **PASS** |
| copy_pack > offprompt | 0.000 | [0.000, 0.000] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic > generic_topic | 0.040 | [0.000, 0.100] | no | FAIL |
| generic_topic > numbers | 0.324 | [0.211, 0.454] | no | FAIL |
| numbers > negation | 0.504 | [0.400, 0.600] | no | FAIL |
| negation > fwloop | 0.999 | [0.995, 1.000] | yes | **PASS** |

Passed 3 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.980 | 0.480 | 1.000 | 0.980 | 0.720 | 0.676 | 0.500 | 0.504 | 0.998 |
| copy_pack |  | 0.000 | 1.000 | 0.925 | 0.000 | 0.000 | 0.020 | 0.020 | 1.000 |
| offprompt |  |  | 1.000 | 1.000 | 0.820 | 0.707 | 0.520 | 0.525 | 1.000 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.000 | 0.000 | 0.020 | 0.020 | 0.995 |
| generic |  |  |  |  |  | 0.040 | 0.280 | 0.280 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.324 | 0.324 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.504 | 0.998 |
| negation |  |  |  |  |  |  |  |  | 0.999 |

### SCRIBE v2p (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.508 [0.410, 0.600] |
| copy_pack | 50 | 0 | 0.002 [0.000, 0.004] |
| offprompt | 50 | 0 | 0.003 [0.002, 0.005] |
| echo | 50 | 0 | 0.000 [0.000, 0.000] |
| stuffing | 50 | 0 | 0.000 [0.000, 0.000] |
| generic | 50 | 0 | 0.009 [0.003, 0.019] |
| generic_topic | 50 | 0 | 0.144 [0.115, 0.176] |
| numbers | 50 | 0 | 0.508 [0.410, 0.609] |
| negation | 50 | 0 | 0.504 [0.408, 0.604] |
| fwloop | 50 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.983 | [0.951, 1.000] | yes | **PASS** |
| copy_pack > offprompt | 0.139 | [0.066, 0.220] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.022 | [0.006, 0.047] | no | FAIL |
| generic > generic_topic | 0.040 | [0.009, 0.086] | no | FAIL |
| generic_topic > numbers | 0.213 | [0.119, 0.315] | no | FAIL |
| numbers > negation | 0.505 | [0.388, 0.614] | no | FAIL |
| negation > fwloop | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.983 | 0.973 | 1.000 | 0.994 | 0.966 | 0.787 | 0.500 | 0.505 | 1.000 |
| copy_pack |  | 0.139 | 1.000 | 0.745 | 0.157 | 0.009 | 0.017 | 0.014 | 1.000 |
| offprompt |  |  | 1.000 | 0.998 | 0.535 | 0.018 | 0.027 | 0.025 | 1.000 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.022 | 0.000 | 0.006 | 0.006 | 0.996 |
| generic |  |  |  |  |  | 0.040 | 0.034 | 0.033 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.213 | 0.214 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.505 | 1.000 |
| negation |  |  |  |  |  |  |  |  | 1.000 |

### SCRIBE v2 without I1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.900 [0.843, 0.943] |
| copy_pack | 50 | 0 | 0.939 [0.908, 0.968] |
| offprompt | 50 | 0 | 0.919 [0.874, 0.956] |
| echo | 50 | 0 | 0.000 [0.000, 0.000] |
| stuffing | 50 | 0 | 0.016 [0.011, 0.022] |
| generic | 50 | 0 | 0.835 [0.835, 0.835] |
| generic_topic | 50 | 0 | 0.912 [0.900, 0.925] |
| numbers | 50 | 0 | 0.900 [0.846, 0.942] |
| negation | 50 | 0 | 0.894 [0.835, 0.941] |
| fwloop | 50 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.443 | [0.342, 0.538] | no | FAIL |
| copy_pack > offprompt | 0.536 | [0.440, 0.631] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic > generic_topic | 0.040 | [0.000, 0.100] | no | FAIL |
| generic_topic > numbers | 0.324 | [0.211, 0.454] | no | FAIL |
| numbers > negation | 0.504 | [0.400, 0.600] | no | FAIL |
| negation > fwloop | 0.999 | [0.995, 1.000] | yes | **PASS** |

Passed 2 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.443 | 0.480 | 1.000 | 0.980 | 0.720 | 0.676 | 0.500 | 0.504 | 0.998 |
| copy_pack |  | 0.536 | 1.000 | 1.000 | 0.780 | 0.751 | 0.557 | 0.563 | 1.000 |
| offprompt |  |  | 1.000 | 1.000 | 0.820 | 0.707 | 0.520 | 0.525 | 1.000 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.000 | 0.000 | 0.020 | 0.020 | 0.995 |
| generic |  |  |  |  |  | 0.040 | 0.280 | 0.280 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.324 | 0.324 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.504 | 0.998 |
| negation |  |  |  |  |  |  |  |  | 0.999 |

### SCRIBE v1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.515 [0.464, 0.563] |
| copy_pack | 50 | 0 | 0.556 [0.496, 0.612] |
| offprompt | 50 | 0 | 0.516 [0.456, 0.565] |
| echo | 50 | 0 | 0.001 [0.001, 0.001] |
| stuffing | 50 | 0 | 0.022 [0.018, 0.027] |
| generic | 50 | 0 | 0.749 [0.749, 0.749] |
| generic_topic | 50 | 0 | 0.713 [0.696, 0.730] |
| numbers | 50 | 0 | 0.515 [0.457, 0.564] |
| negation | 50 | 0 | 0.513 [0.454, 0.569] |
| fwloop | 50 | 0 | 0.133 [0.122, 0.146] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.415 | [0.303, 0.530] | no | FAIL |
| copy_pack > offprompt | 0.574 | [0.466, 0.684] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic > generic_topic | 0.660 | [0.540, 0.800] | yes | FAIL |
| generic_topic > numbers | 0.846 | [0.757, 0.921] | yes | FAIL |
| numbers > negation | 0.494 | [0.382, 0.613] | no | FAIL |
| negation > fwloop | 0.946 | [0.880, 0.993] | yes | FAIL |

Passed 1 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.415 | 0.483 | 1.000 | 0.987 | 0.100 | 0.157 | 0.500 | 0.494 | 0.952 |
| copy_pack |  | 0.574 | 1.000 | 1.000 | 0.140 | 0.248 | 0.582 | 0.579 | 0.930 |
| offprompt |  |  | 1.000 | 0.998 | 0.100 | 0.162 | 0.517 | 0.510 | 0.934 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.000 | 0.000 | 0.013 | 0.013 | 0.000 |
| generic |  |  |  |  |  | 0.660 | 0.900 | 0.880 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.846 | 0.827 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.494 | 0.952 |
| negation |  |  |  |  |  |  |  |  | 0.946 |

### surface S (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.902 [0.883, 0.920] |
| copy_pack | 50 | 0 | 0.934 [0.916, 0.951] |
| offprompt | 50 | 0 | 0.912 [0.894, 0.928] |
| echo | 50 | 0 | 0.149 [0.134, 0.165] |
| stuffing | 50 | 0 | 0.582 [0.545, 0.616] |
| generic | 50 | 0 | 0.943 [0.943, 0.943] |
| generic_topic | 50 | 0 | 0.924 [0.914, 0.933] |
| numbers | 50 | 0 | 0.901 [0.881, 0.917] |
| negation | 50 | 0 | 0.904 [0.884, 0.922] |
| fwloop | 50 | 0 | 0.624 [0.614, 0.633] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.303 | [0.198, 0.408] | no | FAIL |
| copy_pack > offprompt | 0.664 | [0.555, 0.774] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic > generic_topic | 0.660 | [0.520, 0.800] | yes | FAIL |
| generic_topic > numbers | 0.594 | [0.480, 0.711] | no | FAIL |
| numbers > negation | 0.479 | [0.367, 0.599] | no | FAIL |
| negation > fwloop | 0.987 | [0.956, 1.000] | yes | **PASS** |

Passed 2 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.303 | 0.445 | 1.000 | 0.982 | 0.260 | 0.414 | 0.510 | 0.486 | 0.987 |
| copy_pack |  | 0.664 | 1.000 | 0.991 | 0.620 | 0.674 | 0.704 | 0.680 | 1.000 |
| offprompt |  |  | 1.000 | 0.989 | 0.300 | 0.464 | 0.564 | 0.535 | 0.998 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.000 | 0.003 | 0.018 | 0.020 | 0.380 |
| generic |  |  |  |  |  | 0.660 | 0.740 | 0.640 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.594 | 0.562 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.479 | 0.987 |
| negation |  |  |  |  |  |  |  |  | 0.987 |

### distinct-1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.715 [0.697, 0.731] |
| copy_pack | 50 | 0 | 0.712 [0.689, 0.733] |
| offprompt | 50 | 0 | 0.717 [0.699, 0.733] |
| echo | 50 | 0 | 0.242 [0.228, 0.257] |
| stuffing | 50 | 0 | 0.537 [0.524, 0.550] |
| generic | 50 | 0 | 0.797 [0.797, 0.797] |
| generic_topic | 50 | 0 | 0.765 [0.763, 0.766] |
| numbers | 50 | 0 | 0.716 [0.698, 0.733] |
| negation | 50 | 0 | 0.701 [0.684, 0.717] |
| fwloop | 50 | 0 | 0.101 [0.098, 0.104] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.498 | [0.392, 0.600] | no | FAIL |
| copy_pack > offprompt | 0.492 | [0.375, 0.598] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic > generic_topic | 1.000 | [1.000, 1.000] | yes | **PASS** |
| generic_topic > numbers | 0.771 | [0.665, 0.875] | yes | FAIL |
| numbers > negation | 0.576 | [0.471, 0.681] | no | FAIL |
| negation > fwloop | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.498 | 0.493 | 1.000 | 0.976 | 0.060 | 0.226 | 0.492 | 0.569 | 1.000 |
| copy_pack |  | 0.492 | 1.000 | 0.955 | 0.180 | 0.229 | 0.496 | 0.557 | 1.000 |
| offprompt |  |  | 1.000 | 0.987 | 0.060 | 0.246 | 0.499 | 0.576 | 1.000 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| stuffing |  |  |  |  | 0.000 | 0.000 | 0.022 | 0.030 | 1.000 |
| generic |  |  |  |  |  | 1.000 | 0.940 | 0.960 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.771 | 0.852 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.576 | 1.000 |
| negation |  |  |  |  |  |  |  |  | 1.000 |

### distinct-2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.955 [0.938, 0.966] |
| copy_pack | 50 | 0 | 0.958 [0.946, 0.969] |
| offprompt | 50 | 0 | 0.958 [0.948, 0.968] |
| echo | 50 | 0 | 0.283 [0.264, 0.302] |
| stuffing | 50 | 0 | 0.906 [0.895, 0.916] |
| generic | 50 | 0 | 0.966 [0.966, 0.966] |
| generic_topic | 50 | 0 | 0.997 [0.995, 0.999] |
| numbers | 50 | 0 | 0.955 [0.938, 0.967] |
| negation | 50 | 0 | 0.945 [0.930, 0.957] |
| fwloop | 50 | 0 | 0.736 [0.724, 0.748] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.460 | [0.349, 0.570] | no | FAIL |
| copy_pack > offprompt | 0.543 | [0.426, 0.654] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.020 | [0.000, 0.060] | no | FAIL |
| generic > generic_topic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic_topic > numbers | 0.896 | [0.834, 0.952] | yes | FAIL |
| numbers > negation | 0.586 | [0.469, 0.692] | no | FAIL |
| negation > fwloop | 0.981 | [0.941, 1.000] | yes | **PASS** |

Passed 2 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.460 | 0.498 | 1.000 | 0.861 | 0.490 | 0.104 | 0.493 | 0.581 | 0.981 |
| copy_pack |  | 0.543 | 1.000 | 0.834 | 0.570 | 0.138 | 0.536 | 0.609 | 1.000 |
| offprompt |  |  | 1.000 | 0.876 | 0.480 | 0.082 | 0.494 | 0.589 | 1.000 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.020 | 0.000 | 0.138 | 0.196 | 0.996 |
| generic |  |  |  |  |  | 0.000 | 0.490 | 0.620 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.896 | 0.951 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.586 | 0.981 |
| negation |  |  |  |  |  |  |  |  | 0.981 |

### distinct-3 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.984 [0.970, 0.992] |
| copy_pack | 50 | 0 | 0.990 [0.986, 0.994] |
| offprompt | 50 | 0 | 0.989 [0.983, 0.994] |
| echo | 50 | 0 | 0.289 [0.269, 0.309] |
| stuffing | 50 | 0 | 0.989 [0.984, 0.994] |
| generic | 50 | 0 | 1.000 [1.000, 1.000] |
| generic_topic | 50 | 0 | 1.000 [1.000, 1.000] |
| numbers | 50 | 0 | 0.984 [0.972, 0.992] |
| negation | 50 | 0 | 0.982 [0.970, 0.990] |
| fwloop | 50 | 0 | 0.977 [0.974, 0.981] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.455 | [0.361, 0.563] | no | FAIL |
| copy_pack > offprompt | 0.502 | [0.398, 0.592] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.300 | [0.230, 0.370] | no | FAIL |
| generic > generic_topic | 0.500 | [0.500, 0.500] | no | FAIL |
| generic_topic > numbers | 0.720 | [0.650, 0.790] | yes | FAIL |
| numbers > negation | 0.552 | [0.443, 0.659] | no | FAIL |
| negation > fwloop | 0.699 | [0.591, 0.802] | no | FAIL |

Passed 1 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.455 | 0.448 | 1.000 | 0.463 | 0.260 | 0.260 | 0.484 | 0.536 | 0.738 |
| copy_pack |  | 0.502 | 1.000 | 0.513 | 0.300 | 0.300 | 0.528 | 0.584 | 0.795 |
| offprompt |  |  | 1.000 | 0.512 | 0.310 | 0.310 | 0.536 | 0.588 | 0.793 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.300 | 0.300 | 0.521 | 0.574 | 0.774 |
| generic |  |  |  |  |  | 0.500 | 0.720 | 0.780 | 0.980 |
| generic_topic |  |  |  |  |  |  | 0.720 | 0.780 | 0.980 |
| numbers |  |  |  |  |  |  |  | 0.552 | 0.747 |
| negation |  |  |  |  |  |  |  |  | 0.699 |

### distinct-4 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.992 [0.982, 0.998] |
| copy_pack | 50 | 0 | 0.998 [0.995, 1.000] |
| offprompt | 50 | 0 | 0.996 [0.993, 0.999] |
| echo | 50 | 0 | 0.292 [0.272, 0.312] |
| stuffing | 50 | 0 | 0.996 [0.993, 0.999] |
| generic | 50 | 0 | 1.000 [1.000, 1.000] |
| generic_topic | 50 | 0 | 1.000 [1.000, 1.000] |
| numbers | 50 | 0 | 0.992 [0.983, 0.998] |
| negation | 50 | 0 | 0.992 [0.982, 0.998] |
| fwloop | 50 | 0 | 0.998 [0.997, 0.999] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.425 | [0.361, 0.492] | no | FAIL |
| copy_pack > offprompt | 0.538 | [0.478, 0.600] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.410 | [0.360, 0.460] | no | FAIL |
| generic > generic_topic | 0.500 | [0.500, 0.500] | no | FAIL |
| generic_topic > numbers | 0.620 | [0.560, 0.680] | yes | FAIL |
| numbers > negation | 0.513 | [0.427, 0.599] | no | FAIL |
| negation > fwloop | 0.447 | [0.366, 0.528] | no | FAIL |

Passed 1 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.425 | 0.460 | 1.000 | 0.473 | 0.380 | 0.380 | 0.500 | 0.513 | 0.459 |
| copy_pack |  | 0.538 | 1.000 | 0.547 | 0.460 | 0.460 | 0.575 | 0.585 | 0.552 |
| offprompt |  |  | 1.000 | 0.511 | 0.420 | 0.420 | 0.540 | 0.550 | 0.506 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.410 | 0.410 | 0.527 | 0.540 | 0.494 |
| generic |  |  |  |  |  | 0.500 | 0.620 | 0.630 | 0.600 |
| generic_topic |  |  |  |  |  |  | 0.620 | 0.630 | 0.600 |
| numbers |  |  |  |  |  |  |  | 0.513 | 0.459 |
| negation |  |  |  |  |  |  |  |  | 0.447 |

### rep-2 (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.045 [0.034, 0.062] |
| copy_pack | 50 | 0 | 0.042 [0.031, 0.054] |
| offprompt | 50 | 0 | 0.042 [0.032, 0.052] |
| echo | 50 | 0 | 0.717 [0.698, 0.736] |
| stuffing | 50 | 0 | 0.094 [0.084, 0.105] |
| generic | 50 | 0 | 0.034 [0.034, 0.034] |
| generic_topic | 50 | 0 | 0.003 [0.001, 0.005] |
| numbers | 50 | 0 | 0.045 [0.033, 0.062] |
| negation | 50 | 0 | 0.055 [0.043, 0.070] |
| fwloop | 50 | 0 | 0.264 [0.252, 0.276] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.460 | [0.349, 0.570] | no | FAIL |
| copy_pack > offprompt | 0.543 | [0.426, 0.654] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.020 | [0.000, 0.060] | no | FAIL |
| generic > generic_topic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic_topic > numbers | 0.896 | [0.834, 0.952] | yes | FAIL |
| numbers > negation | 0.586 | [0.469, 0.692] | no | FAIL |
| negation > fwloop | 0.981 | [0.941, 1.000] | yes | **PASS** |

Passed 2 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.460 | 0.498 | 1.000 | 0.861 | 0.490 | 0.104 | 0.493 | 0.581 | 0.981 |
| copy_pack |  | 0.543 | 1.000 | 0.834 | 0.570 | 0.138 | 0.536 | 0.609 | 1.000 |
| offprompt |  |  | 1.000 | 0.876 | 0.480 | 0.082 | 0.494 | 0.589 | 1.000 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.020 | 0.000 | 0.138 | 0.196 | 0.996 |
| generic |  |  |  |  |  | 0.000 | 0.490 | 0.620 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.896 | 0.951 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.586 | 0.981 |
| negation |  |  |  |  |  |  |  |  | 0.981 |

### rep-3 (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.016 [0.008, 0.030] |
| copy_pack | 50 | 0 | 0.010 [0.006, 0.014] |
| offprompt | 50 | 0 | 0.011 [0.006, 0.017] |
| echo | 50 | 0 | 0.711 [0.691, 0.731] |
| stuffing | 50 | 0 | 0.011 [0.006, 0.016] |
| generic | 50 | 0 | 0.000 [0.000, 0.000] |
| generic_topic | 50 | 0 | 0.000 [0.000, 0.000] |
| numbers | 50 | 0 | 0.016 [0.008, 0.028] |
| negation | 50 | 0 | 0.018 [0.010, 0.030] |
| fwloop | 50 | 0 | 0.023 [0.019, 0.026] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.455 | [0.361, 0.563] | no | FAIL |
| copy_pack > offprompt | 0.502 | [0.398, 0.592] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.300 | [0.230, 0.370] | no | FAIL |
| generic > generic_topic | 0.500 | [0.500, 0.500] | no | FAIL |
| generic_topic > numbers | 0.720 | [0.650, 0.790] | yes | FAIL |
| numbers > negation | 0.552 | [0.443, 0.659] | no | FAIL |
| negation > fwloop | 0.699 | [0.591, 0.802] | no | FAIL |

Passed 1 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.455 | 0.448 | 1.000 | 0.463 | 0.260 | 0.260 | 0.484 | 0.536 | 0.738 |
| copy_pack |  | 0.502 | 1.000 | 0.513 | 0.300 | 0.300 | 0.528 | 0.584 | 0.795 |
| offprompt |  |  | 1.000 | 0.512 | 0.310 | 0.310 | 0.536 | 0.588 | 0.793 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.300 | 0.300 | 0.521 | 0.574 | 0.774 |
| generic |  |  |  |  |  | 0.500 | 0.720 | 0.780 | 0.980 |
| generic_topic |  |  |  |  |  |  | 0.720 | 0.780 | 0.980 |
| numbers |  |  |  |  |  |  |  | 0.552 | 0.747 |
| negation |  |  |  |  |  |  |  |  | 0.699 |

### rep-4 (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.008 [0.002, 0.018] |
| copy_pack | 50 | 0 | 0.002 [0.000, 0.005] |
| offprompt | 50 | 0 | 0.004 [0.001, 0.007] |
| echo | 50 | 0 | 0.708 [0.688, 0.728] |
| stuffing | 50 | 0 | 0.004 [0.001, 0.007] |
| generic | 50 | 0 | 0.000 [0.000, 0.000] |
| generic_topic | 50 | 0 | 0.000 [0.000, 0.000] |
| numbers | 50 | 0 | 0.008 [0.002, 0.017] |
| negation | 50 | 0 | 0.008 [0.002, 0.018] |
| fwloop | 50 | 0 | 0.002 [0.001, 0.003] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.425 | [0.361, 0.492] | no | FAIL |
| copy_pack > offprompt | 0.538 | [0.478, 0.600] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.410 | [0.360, 0.460] | no | FAIL |
| generic > generic_topic | 0.500 | [0.500, 0.500] | no | FAIL |
| generic_topic > numbers | 0.620 | [0.560, 0.680] | yes | FAIL |
| numbers > negation | 0.513 | [0.427, 0.599] | no | FAIL |
| negation > fwloop | 0.447 | [0.366, 0.528] | no | FAIL |

Passed 1 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.425 | 0.460 | 1.000 | 0.473 | 0.380 | 0.380 | 0.500 | 0.513 | 0.459 |
| copy_pack |  | 0.538 | 1.000 | 0.547 | 0.460 | 0.460 | 0.575 | 0.585 | 0.552 |
| offprompt |  |  | 1.000 | 0.511 | 0.420 | 0.420 | 0.540 | 0.550 | 0.506 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.410 | 0.410 | 0.527 | 0.540 | 0.494 |
| generic |  |  |  |  |  | 0.500 | 0.620 | 0.630 | 0.600 |
| generic_topic |  |  |  |  |  |  | 0.620 | 0.630 | 0.600 |
| numbers |  |  |  |  |  |  |  | 0.513 | 0.459 |
| negation |  |  |  |  |  |  |  |  | 0.447 |

### seq-rep-4 (lower is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.008 [0.002, 0.018] |
| copy_pack | 50 | 0 | 0.002 [0.000, 0.005] |
| offprompt | 50 | 0 | 0.004 [0.001, 0.007] |
| echo | 50 | 0 | 0.708 [0.688, 0.728] |
| stuffing | 50 | 0 | 0.004 [0.001, 0.007] |
| generic | 50 | 0 | 0.000 [0.000, 0.000] |
| generic_topic | 50 | 0 | 0.000 [0.000, 0.000] |
| numbers | 50 | 0 | 0.008 [0.002, 0.017] |
| negation | 50 | 0 | 0.008 [0.002, 0.018] |
| fwloop | 50 | 0 | 0.002 [0.001, 0.003] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.425 | [0.361, 0.492] | no | FAIL |
| copy_pack > offprompt | 0.538 | [0.478, 0.600] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.410 | [0.360, 0.460] | no | FAIL |
| generic > generic_topic | 0.500 | [0.500, 0.500] | no | FAIL |
| generic_topic > numbers | 0.620 | [0.560, 0.680] | yes | FAIL |
| numbers > negation | 0.513 | [0.427, 0.599] | no | FAIL |
| negation > fwloop | 0.447 | [0.366, 0.528] | no | FAIL |

Passed 1 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.425 | 0.460 | 1.000 | 0.473 | 0.380 | 0.380 | 0.500 | 0.513 | 0.459 |
| copy_pack |  | 0.538 | 1.000 | 0.547 | 0.460 | 0.460 | 0.575 | 0.585 | 0.552 |
| offprompt |  |  | 1.000 | 0.511 | 0.420 | 0.420 | 0.540 | 0.550 | 0.506 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.410 | 0.410 | 0.527 | 0.540 | 0.494 |
| generic |  |  |  |  |  | 0.500 | 0.620 | 0.630 | 0.600 |
| generic_topic |  |  |  |  |  |  | 0.620 | 0.630 | 0.600 |
| numbers |  |  |  |  |  |  |  | 0.513 | 0.459 |
| negation |  |  |  |  |  |  |  |  | 0.447 |

### diversity (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.936 [0.906, 0.956] |
| copy_pack | 50 | 0 | 0.947 [0.930, 0.963] |
| offprompt | 50 | 0 | 0.945 [0.927, 0.960] |
| echo | 50 | 0 | 0.029 [0.023, 0.036] |
| stuffing | 50 | 0 | 0.894 [0.878, 0.908] |
| generic | 50 | 0 | 0.966 [0.966, 0.966] |
| generic_topic | 50 | 0 | 0.997 [0.995, 0.999] |
| numbers | 50 | 0 | 0.937 [0.907, 0.957] |
| negation | 50 | 0 | 0.924 [0.896, 0.944] |
| fwloop | 50 | 0 | 0.718 [0.705, 0.732] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.441 | [0.332, 0.552] | no | FAIL |
| copy_pack > offprompt | 0.549 | [0.434, 0.656] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.020 | [0.000, 0.060] | no | FAIL |
| generic > generic_topic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic_topic > numbers | 0.896 | [0.834, 0.952] | yes | FAIL |
| numbers > negation | 0.576 | [0.461, 0.685] | no | FAIL |
| negation > fwloop | 0.979 | [0.938, 1.000] | yes | **PASS** |

Passed 2 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.441 | 0.479 | 1.000 | 0.778 | 0.440 | 0.104 | 0.492 | 0.569 | 0.979 |
| copy_pack |  | 0.549 | 1.000 | 0.802 | 0.520 | 0.138 | 0.553 | 0.618 | 0.994 |
| offprompt |  |  | 1.000 | 0.817 | 0.480 | 0.082 | 0.511 | 0.595 | 0.983 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.020 | 0.000 | 0.219 | 0.287 | 0.975 |
| generic |  |  |  |  |  | 0.000 | 0.520 | 0.620 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.896 | 0.951 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.576 | 0.979 |
| negation |  |  |  |  |  |  |  |  | 0.979 |

### type-token ratio (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.715 [0.697, 0.731] |
| copy_pack | 50 | 0 | 0.712 [0.689, 0.733] |
| offprompt | 50 | 0 | 0.717 [0.699, 0.733] |
| echo | 50 | 0 | 0.242 [0.228, 0.257] |
| stuffing | 50 | 0 | 0.537 [0.524, 0.550] |
| generic | 50 | 0 | 0.797 [0.797, 0.797] |
| generic_topic | 50 | 0 | 0.765 [0.763, 0.766] |
| numbers | 50 | 0 | 0.716 [0.698, 0.733] |
| negation | 50 | 0 | 0.701 [0.684, 0.717] |
| fwloop | 50 | 0 | 0.101 [0.098, 0.104] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.498 | [0.392, 0.600] | no | FAIL |
| copy_pack > offprompt | 0.492 | [0.375, 0.598] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic > generic_topic | 1.000 | [1.000, 1.000] | yes | **PASS** |
| generic_topic > numbers | 0.771 | [0.665, 0.875] | yes | FAIL |
| numbers > negation | 0.576 | [0.471, 0.681] | no | FAIL |
| negation > fwloop | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.498 | 0.493 | 1.000 | 0.976 | 0.060 | 0.226 | 0.492 | 0.569 | 1.000 |
| copy_pack |  | 0.492 | 1.000 | 0.955 | 0.180 | 0.229 | 0.496 | 0.557 | 1.000 |
| offprompt |  |  | 1.000 | 0.987 | 0.060 | 0.246 | 0.499 | 0.576 | 1.000 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| stuffing |  |  |  |  | 0.000 | 0.000 | 0.022 | 0.030 | 1.000 |
| generic |  |  |  |  |  | 1.000 | 0.940 | 0.960 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.771 | 0.852 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.576 | 1.000 |
| negation |  |  |  |  |  |  |  |  | 1.000 |

### gzip ratio (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 50 | 0 | 0.620 [0.609, 0.632] |
| copy_pack | 50 | 0 | 0.620 [0.609, 0.632] |
| offprompt | 50 | 0 | 0.622 [0.612, 0.633] |
| echo | 50 | 0 | 0.249 [0.237, 0.260] |
| stuffing | 50 | 0 | 0.471 [0.456, 0.486] |
| generic | 50 | 0 | 0.666 [0.666, 0.666] |
| generic_topic | 50 | 0 | 0.663 [0.660, 0.665] |
| numbers | 50 | 0 | 0.622 [0.610, 0.633] |
| negation | 50 | 0 | 0.615 [0.603, 0.625] |
| fwloop | 50 | 0 | 0.453 [0.449, 0.458] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.510 | [0.385, 0.622] | no | FAIL |
| copy_pack > offprompt | 0.481 | [0.364, 0.598] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.002 | [0.000, 0.006] | no | FAIL |
| stuffing > generic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic > generic_topic | 0.700 | [0.580, 0.820] | yes | FAIL |
| generic_topic > numbers | 0.831 | [0.738, 0.912] | yes | FAIL |
| numbers > negation | 0.557 | [0.445, 0.673] | no | FAIL |
| negation > fwloop | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 2 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.510 | 0.493 | 1.000 | 0.982 | 0.120 | 0.142 | 0.487 | 0.549 | 1.000 |
| copy_pack |  | 0.481 | 1.000 | 0.984 | 0.120 | 0.153 | 0.482 | 0.534 | 1.000 |
| offprompt |  |  | 1.000 | 0.986 | 0.140 | 0.147 | 0.494 | 0.546 | 1.000 |
| echo |  |  |  | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.000 | 0.000 | 0.018 | 0.023 | 0.562 |
| generic |  |  |  |  |  | 0.700 | 0.860 | 0.900 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.831 | 0.900 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.557 | 1.000 |
| negation |  |  |  |  |  |  |  |  | 1.000 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | held-out human > copy_pack | copy_pack > offprompt | offprompt > echo | echo > stuffing | stuffing > generic | generic > generic_topic | generic_topic > numbers | numbers > negation | negation > fwloop | passed |
|---|---|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.980 **PASS** | 0.000 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.000 FAIL | 0.040 FAIL | 0.324 FAIL | 0.504 FAIL | 0.999 **PASS** | 3 of 9 |
| SCRIBE v2p | 0.983 **PASS** | 0.139 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.022 FAIL | 0.040 FAIL | 0.213 FAIL | 0.505 FAIL | 1.000 **PASS** | 3 of 9 |
| SCRIBE v2 without I1 | 0.443 FAIL | 0.536 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.000 FAIL | 0.040 FAIL | 0.324 FAIL | 0.504 FAIL | 0.999 **PASS** | 2 of 9 |
| SCRIBE v1 | 0.415 FAIL | 0.574 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.000 FAIL | 0.660 FAIL | 0.846 FAIL | 0.494 FAIL | 0.946 FAIL | 1 of 9 |
| surface S | 0.303 FAIL | 0.664 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.000 FAIL | 0.660 FAIL | 0.594 FAIL | 0.479 FAIL | 0.987 **PASS** | 2 of 9 |
| distinct-1 | 0.498 FAIL | 0.492 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.000 FAIL | 1.000 **PASS** | 0.771 FAIL | 0.576 FAIL | 1.000 **PASS** | 3 of 9 |
| distinct-2 | 0.460 FAIL | 0.543 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.020 FAIL | 0.000 FAIL | 0.896 FAIL | 0.586 FAIL | 0.981 **PASS** | 2 of 9 |
| distinct-3 | 0.455 FAIL | 0.502 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.300 FAIL | 0.500 FAIL | 0.720 FAIL | 0.552 FAIL | 0.699 FAIL | 1 of 9 |
| distinct-4 | 0.425 FAIL | 0.538 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.410 FAIL | 0.500 FAIL | 0.620 FAIL | 0.513 FAIL | 0.447 FAIL | 1 of 9 |
| rep-2 | 0.460 FAIL | 0.543 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.020 FAIL | 0.000 FAIL | 0.896 FAIL | 0.586 FAIL | 0.981 **PASS** | 2 of 9 |
| rep-3 | 0.455 FAIL | 0.502 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.300 FAIL | 0.500 FAIL | 0.720 FAIL | 0.552 FAIL | 0.699 FAIL | 1 of 9 |
| rep-4 | 0.425 FAIL | 0.538 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.410 FAIL | 0.500 FAIL | 0.620 FAIL | 0.513 FAIL | 0.447 FAIL | 1 of 9 |
| seq-rep-4 | 0.425 FAIL | 0.538 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.410 FAIL | 0.500 FAIL | 0.620 FAIL | 0.513 FAIL | 0.447 FAIL | 1 of 9 |
| diversity | 0.441 FAIL | 0.549 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.020 FAIL | 0.000 FAIL | 0.896 FAIL | 0.576 FAIL | 0.979 **PASS** | 2 of 9 |
| type-token ratio | 0.498 FAIL | 0.492 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.000 FAIL | 1.000 **PASS** | 0.771 FAIL | 0.576 FAIL | 1.000 **PASS** | 3 of 9 |
| gzip ratio | 0.510 FAIL | 0.481 FAIL | 1.000 **PASS** | 0.002 FAIL | 0.000 FAIL | 0.700 FAIL | 0.831 FAIL | 0.557 FAIL | 1.000 **PASS** | 2 of 9 |
