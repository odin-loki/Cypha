# Mean per attack and parity with human text (mean / mean of human); P(human > attack)

Analysis `attacks` (7. Attacks). meta.py `ladder`.
Inputs: SP/run/attacks/attacks.jsonl, SP/run/attacks/attacks.v2.jsonl, SP/run/attacks/attacks.v2p.jsonl, SP/run/attacks/attacks.v2_noi1.jsonl.

2500 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 250 | 0 | 0.911 [0.891, 0.930] |
| copy_pack | 250 | 0 | 0.102 [0.097, 0.106] |
| offprompt | 250 | 0 | 0.913 [0.893, 0.930] |
| echo | 250 | 0 | 0.000 [0.000, 0.000] |
| stuffing | 250 | 0 | 0.021 [0.016, 0.027] |
| generic | 250 | 0 | 0.835 [0.835, 0.835] |
| generic_topic | 250 | 0 | 0.910 [0.904, 0.916] |
| numbers | 250 | 0 | 0.911 [0.891, 0.930] |
| negation | 250 | 0 | 0.907 [0.885, 0.926] |
| fwloop | 250 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.994 | [0.984, 1.000] | yes | **PASS** |
| copy_pack > offprompt | 0.006 | [0.000, 0.016] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic > generic_topic | 0.068 | [0.040, 0.104] | no | FAIL |
| generic_topic > numbers | 0.311 | [0.259, 0.363] | no | FAIL |
| numbers > negation | 0.508 | [0.462, 0.551] | no | FAIL |
| negation > fwloop | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.994 | 0.499 | 1.000 | 0.996 | 0.788 | 0.689 | 0.500 | 0.508 | 1.000 |
| copy_pack |  | 0.006 | 1.000 | 0.942 | 0.000 | 0.000 | 0.006 | 0.004 | 1.000 |
| offprompt |  |  | 1.000 | 0.996 | 0.792 | 0.692 | 0.501 | 0.509 | 1.000 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.000 | 0.000 | 0.004 | 0.004 | 0.978 |
| generic |  |  |  |  |  | 0.068 | 0.212 | 0.224 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.311 | 0.324 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.508 | 1.000 |
| negation |  |  |  |  |  |  |  |  | 1.000 |

### SCRIBE v2p (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 250 | 0 | 0.481 [0.436, 0.526] |
| copy_pack | 250 | 0 | 0.001 [0.001, 0.003] |
| offprompt | 250 | 0 | 0.010 [0.003, 0.021] |
| echo | 250 | 0 | 0.000 [0.000, 0.000] |
| stuffing | 250 | 0 | 0.000 [0.000, 0.000] |
| generic | 250 | 0 | 0.012 [0.005, 0.021] |
| generic_topic | 250 | 0 | 0.130 [0.116, 0.147] |
| numbers | 250 | 0 | 0.484 [0.437, 0.527] |
| negation | 250 | 0 | 0.482 [0.439, 0.525] |
| fwloop | 250 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.990 | [0.981, 0.997] | yes | **PASS** |
| copy_pack > offprompt | 0.114 | [0.086, 0.148] | no | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.029 | [0.017, 0.044] | no | FAIL |
| generic > generic_topic | 0.043 | [0.026, 0.062] | no | FAIL |
| generic_topic > numbers | 0.216 | [0.176, 0.260] | no | FAIL |
| numbers > negation | 0.502 | [0.455, 0.547] | no | FAIL |
| negation > fwloop | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 3 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.990 | 0.969 | 1.000 | 0.997 | 0.966 | 0.780 | 0.498 | 0.499 | 1.000 |
| copy_pack |  | 0.114 | 1.000 | 0.766 | 0.132 | 0.007 | 0.010 | 0.010 | 1.000 |
| offprompt |  |  | 1.000 | 0.978 | 0.526 | 0.030 | 0.031 | 0.030 | 0.998 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.029 | 0.000 | 0.003 | 0.003 | 0.970 |
| generic |  |  |  |  |  | 0.043 | 0.034 | 0.033 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.216 | 0.217 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.502 | 1.000 |
| negation |  |  |  |  |  |  |  |  | 1.000 |

### SCRIBE v2 without I1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 250 | 0 | 0.911 [0.891, 0.930] |
| copy_pack | 250 | 0 | 0.950 [0.936, 0.963] |
| offprompt | 250 | 0 | 0.913 [0.893, 0.930] |
| echo | 250 | 0 | 0.000 [0.000, 0.000] |
| stuffing | 250 | 0 | 0.020 [0.015, 0.026] |
| generic | 250 | 0 | 0.835 [0.835, 0.835] |
| generic_topic | 250 | 0 | 0.910 [0.904, 0.916] |
| numbers | 250 | 0 | 0.911 [0.891, 0.929] |
| negation | 250 | 0 | 0.905 [0.883, 0.925] |
| fwloop | 250 | 0 | 0.000 [0.000, 0.000] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.421 | [0.377, 0.462] | no | FAIL |
| copy_pack > offprompt | 0.578 | [0.535, 0.623] | yes | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.000 | [0.000, 0.000] | no | FAIL |
| stuffing > generic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic > generic_topic | 0.068 | [0.040, 0.104] | no | FAIL |
| generic_topic > numbers | 0.315 | [0.262, 0.367] | no | FAIL |
| numbers > negation | 0.506 | [0.461, 0.550] | no | FAIL |
| negation > fwloop | 1.000 | [1.000, 1.000] | yes | **PASS** |

Passed 2 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.421 | 0.499 | 1.000 | 0.996 | 0.788 | 0.685 | 0.500 | 0.506 | 1.000 |
| copy_pack |  | 0.578 | 1.000 | 1.000 | 0.852 | 0.786 | 0.579 | 0.585 | 1.000 |
| offprompt |  |  | 1.000 | 0.996 | 0.792 | 0.689 | 0.501 | 0.507 | 1.000 |
| echo |  |  |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.000 | 0.000 | 0.004 | 0.004 | 0.978 |
| generic |  |  |  |  |  | 0.068 | 0.212 | 0.224 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.315 | 0.324 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.506 | 1.000 |
| negation |  |  |  |  |  |  |  |  | 1.000 |

### SCRIBE v1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 250 | 0 | 0.522 [0.497, 0.547] |
| copy_pack | 250 | 0 | 0.578 [0.551, 0.602] |
| offprompt | 250 | 0 | 0.520 [0.496, 0.544] |
| echo | 250 | 0 | 0.001 [0.001, 0.001] |
| stuffing | 250 | 0 | 0.023 [0.021, 0.025] |
| generic | 250 | 0 | 0.749 [0.749, 0.749] |
| generic_topic | 250 | 0 | 0.709 [0.701, 0.716] |
| numbers | 250 | 0 | 0.522 [0.498, 0.547] |
| negation | 250 | 0 | 0.513 [0.488, 0.537] |
| fwloop | 250 | 0 | 0.129 [0.125, 0.133] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.418 | [0.369, 0.468] | no | FAIL |
| copy_pack > offprompt | 0.584 | [0.533, 0.633] | yes | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.002 | [0.000, 0.006] | no | FAIL |
| stuffing > generic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic > generic_topic | 0.756 | [0.700, 0.804] | yes | FAIL |
| generic_topic > numbers | 0.809 | [0.769, 0.848] | yes | FAIL |
| numbers > negation | 0.513 | [0.463, 0.563] | no | FAIL |
| negation > fwloop | 0.950 | [0.927, 0.972] | yes | **PASS** |

Passed 2 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.418 | 0.502 | 1.000 | 0.997 | 0.108 | 0.192 | 0.499 | 0.513 | 0.961 |
| copy_pack |  | 0.584 | 1.000 | 1.000 | 0.200 | 0.290 | 0.582 | 0.593 | 0.969 |
| offprompt |  |  | 1.000 | 0.997 | 0.108 | 0.192 | 0.497 | 0.511 | 0.959 |
| echo |  |  |  | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.000 | 0.000 | 0.003 | 0.003 | 0.000 |
| generic |  |  |  |  |  | 0.756 | 0.896 | 0.884 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.809 | 0.810 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.513 | 0.961 |
| negation |  |  |  |  |  |  |  |  | 0.950 |

### surface S (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 250 | 0 | 0.911 [0.903, 0.919] |
| copy_pack | 250 | 0 | 0.938 [0.930, 0.944] |
| offprompt | 250 | 0 | 0.910 [0.902, 0.919] |
| echo | 250 | 0 | 0.147 [0.139, 0.155] |
| stuffing | 250 | 0 | 0.578 [0.563, 0.595] |
| generic | 250 | 0 | 0.943 [0.943, 0.943] |
| generic_topic | 250 | 0 | 0.924 [0.920, 0.929] |
| numbers | 250 | 0 | 0.910 [0.903, 0.918] |
| negation | 250 | 0 | 0.912 [0.904, 0.919] |
| fwloop | 250 | 0 | 0.620 [0.616, 0.624] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > copy_pack | 0.342 | [0.298, 0.391] | no | FAIL |
| copy_pack > offprompt | 0.657 | [0.608, 0.701] | yes | FAIL |
| offprompt > echo | 1.000 | [1.000, 1.000] | yes | **PASS** |
| echo > stuffing | 0.004 | [0.001, 0.009] | no | FAIL |
| stuffing > generic | 0.000 | [0.000, 0.000] | no | FAIL |
| generic > generic_topic | 0.584 | [0.520, 0.640] | yes | FAIL |
| generic_topic > numbers | 0.534 | [0.487, 0.583] | yes | FAIL |
| numbers > negation | 0.488 | [0.439, 0.537] | no | FAIL |
| negation > fwloop | 0.995 | [0.988, 1.000] | yes | **PASS** |

Passed 2 of 9 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | copy_pack | offprompt | echo | stuffing | generic | generic_topic | numbers | negation | fwloop |
|---|---|---|---|---|---|---|---|---|---|
| held-out human | 0.342 | 0.500 | 1.000 | 0.985 | 0.364 | 0.473 | 0.505 | 0.493 | 0.995 |
| copy_pack |  | 0.657 | 1.000 | 0.993 | 0.604 | 0.656 | 0.662 | 0.650 | 1.000 |
| offprompt |  |  | 1.000 | 0.984 | 0.364 | 0.474 | 0.505 | 0.494 | 0.994 |
| echo |  |  |  | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuffing |  |  |  |  | 0.000 | 0.006 | 0.015 | 0.016 | 0.376 |
| generic |  |  |  |  |  | 0.584 | 0.648 | 0.612 | 1.000 |
| generic_topic |  |  |  |  |  |  | 0.534 | 0.517 | 1.000 |
| numbers |  |  |  |  |  |  |  | 0.488 | 0.995 |
| negation |  |  |  |  |  |  |  |  | 0.995 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | held-out human > copy_pack | copy_pack > offprompt | offprompt > echo | echo > stuffing | stuffing > generic | generic > generic_topic | generic_topic > numbers | numbers > negation | negation > fwloop | passed |
|---|---|---|---|---|---|---|---|---|---|---|
| SCRIBE v2 | 0.994 **PASS** | 0.006 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.000 FAIL | 0.068 FAIL | 0.311 FAIL | 0.508 FAIL | 1.000 **PASS** | 3 of 9 |
| SCRIBE v2p | 0.990 **PASS** | 0.114 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.029 FAIL | 0.043 FAIL | 0.216 FAIL | 0.502 FAIL | 1.000 **PASS** | 3 of 9 |
| SCRIBE v2 without I1 | 0.421 FAIL | 0.578 FAIL | 1.000 **PASS** | 0.000 FAIL | 0.000 FAIL | 0.068 FAIL | 0.315 FAIL | 0.506 FAIL | 1.000 **PASS** | 2 of 9 |
| SCRIBE v1 | 0.418 FAIL | 0.584 FAIL | 1.000 **PASS** | 0.002 FAIL | 0.000 FAIL | 0.756 FAIL | 0.809 FAIL | 0.513 FAIL | 0.950 **PASS** | 2 of 9 |
| surface S | 0.342 FAIL | 0.657 FAIL | 1.000 **PASS** | 0.004 FAIL | 0.000 FAIL | 0.584 FAIL | 0.534 FAIL | 0.488 FAIL | 0.995 **PASS** | 2 of 9 |
