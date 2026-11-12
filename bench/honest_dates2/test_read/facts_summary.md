| Arm | samples | statements (S, C, U) | per sample [95%] | contradicted [95%] | supported [95%] | precision S/(S+C) [95%] | supported only in part | invalid dates |
|---|---|---|---|---|---|---|---|---|
| HLT_TEST | 250 | 299 (158, 13, 128) | 1.20 [0.94, 1.48] | 0.043 [0.021, 0.069] | 0.528 [0.446, 0.598] | 0.924 [0.883, 0.959] | 0.01 [0.00, 0.02] | 0.000 [0.000, 0.000] |
| L1_TEST | 250 | 389 (100, 135, 154) | 1.56 [1.22, 1.92] | 0.347 [0.277, 0.414] | 0.257 [0.220, 0.293] | 0.426 [0.364, 0.494] | 0.45 [0.36, 0.54] | 0.010 [0.000, 0.023] |

By sentence position (C = share contradicted, S = share supported, n = statements):

| Arm | sentence 1 | sentence 2 | sentence 3 | sentence 4+ |
|---|---|---|---|---|
| HLT_TEST | C 0.00 [0.00, 0.00]; S 0.22 [0.09, 0.36]; n 36 | C 0.03 [0.00, 0.11]; S 0.39 [0.21, 0.58]; n 31 | C 0.00 [0.00, 0.00]; S 0.29 [0.11, 0.46]; n 38 | C 0.06 [0.03, 0.10]; S 0.65 [0.56, 0.73]; n 194 |
| L1_TEST | C 0.29 [0.16, 0.42]; S 0.12 [0.05, 0.21]; n 48 | C 0.35 [0.21, 0.49]; S 0.24 [0.12, 0.36]; n 51 | C 0.28 [0.13, 0.42]; S 0.19 [0.09, 0.30]; n 47 | C 0.37 [0.29, 0.45]; S 0.30 [0.24, 0.35]; n 243 |

By relation (supported / contradicted / unknown):

- HLT_TEST: birth_date 84/5/3; birth_place 12/7/31; death_date 41/1/3; death_place 0/0/4; is_a 11/0/86; located_in 2/0/1; nationality 8/0/0
- L1_TEST: birth_date 66/58/16; birth_place 9/4/18; death_date 7/58/33; death_place 0/0/5; director 0/0/1; is_a 4/0/76; located_in 8/6/0; nationality 6/9/5
