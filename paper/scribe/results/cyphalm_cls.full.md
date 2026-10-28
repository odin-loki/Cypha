# CyphaLM: human, trie, trie + class expert (no known order: means and pairwise accuracies)

Analysis `cyphalm_cls` (6. Model ladders (known order)). meta.py `ladder`.
Inputs: SP/run/cyphalm/cls_prepared.jsonl, SP/run/cyphalm/cls.scribe.jsonl, SP/run/cyphalm/cls.scribe.ts.jsonl.

493 texts. Bootstrap: 1000 resamples, seed 20261028, percentile 95% intervals. Accuracy is P(upper > lower) over all (upper, lower) pairs with ties one half (the AUROC), computed on oriented scores (lower-is-better metrics negated). "Intervals disjoint" means the rung-mean intervals do not overlap and are in the right order. A pair passes when accuracy >= 0.95 and the intervals are disjoint.

### SCRIBE v2 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 169 | 0 | 0.770 [0.718, 0.819] |
| trie+cls:shape | 162 | 0 | 0.002 [0.002, 0.003] |
| trie samples (pooled) | 162 | 0 | 0.014 [0.006, 0.027] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > trie+cls:shape | 1.000 | [0.999, 1.000] | yes | **PASS** |
| trie+cls:shape > trie samples (pooled) | 0.399 | [0.336, 0.462] | no | FAIL |

Passed 1 of 2 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | trie+cls:shape | trie samples (pooled) |
|---|---|---|
| held-out human | 1.000 | 0.993 |
| trie+cls:shape |  | 0.399 |

### SCRIBE v2, two-sided (exploratory) (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 169 | 0 | 0.756 [0.709, 0.802] |
| trie+cls:shape | 162 | 0 | 0.003 [0.002, 0.003] |
| trie samples (pooled) | 162 | 0 | 0.019 [0.008, 0.033] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > trie+cls:shape | 0.999 | [0.998, 1.000] | yes | **PASS** |
| trie+cls:shape > trie samples (pooled) | 0.395 | [0.333, 0.457] | no | FAIL |

Passed 1 of 2 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | trie+cls:shape | trie samples (pooled) |
|---|---|---|
| held-out human | 0.999 | 0.991 |
| trie+cls:shape |  | 0.395 |

### SCRIBE v1 (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 169 | 0 | 0.456 [0.417, 0.496] |
| trie+cls:shape | 162 | 0 | 0.067 [0.054, 0.081] |
| trie samples (pooled) | 162 | 0 | 0.027 [0.020, 0.035] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > trie+cls:shape | 0.903 | [0.862, 0.937] | yes | FAIL |
| trie+cls:shape > trie samples (pooled) | 0.792 | [0.740, 0.843] | yes | FAIL |

Passed 0 of 2 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | trie+cls:shape | trie samples (pooled) |
|---|---|---|
| held-out human | 0.903 | 0.958 |
| trie+cls:shape |  | 0.792 |

### surface S (higher is better)

| rung | n | missing | mean [95% interval] |
|---|---|---|---|
| held-out human | 169 | 0 | 0.904 [0.891, 0.917] |
| trie+cls:shape | 162 | 0 | 0.901 [0.892, 0.909] |
| trie samples (pooled) | 162 | 0 | 0.749 [0.737, 0.762] |

| upper > lower | accuracy | 95% interval | intervals disjoint | verdict |
|---|---|---|---|---|
| held-out human > trie+cls:shape | 0.572 | [0.505, 0.633] | no | FAIL |
| trie+cls:shape > trie samples (pooled) | 0.944 | [0.918, 0.964] | yes | FAIL |

Passed 0 of 2 adjacent pairs.

All pairs (accuracy of row > column):

| upper \ lower | trie+cls:shape | trie samples (pooled) |
|---|---|---|
| held-out human | 0.572 | 0.917 |
| trie+cls:shape |  | 0.944 |

## Summary

Adjacent pairs, accuracy and verdict (pass: accuracy >= 0.95 and disjoint intervals in the right order).

| metric | held-out human > trie+cls:shape | trie+cls:shape > trie samples (pooled) | passed |
|---|---|---|---|
| SCRIBE v2 | 1.000 **PASS** | 0.399 FAIL | 1 of 2 |
| SCRIBE v2, two-sided (exploratory) | 0.999 **PASS** | 0.395 FAIL | 1 of 2 |
| SCRIBE v1 | 0.903 FAIL | 0.792 FAIL | 0 of 2 |
| surface S | 0.572 FAIL | 0.944 FAIL | 0 of 2 |
