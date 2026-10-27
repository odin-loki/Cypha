# CyphaLM: human, trie, trie + class expert (no known order: means and pairwise accuracies)

Analysis `cyphalm_cls` (6. Model ladders (known order)). meta.py `ladder`.
Inputs: SP/run/cyphalm/cls_prepared.jsonl, SP/run/cyphalm/cls.scribe.jsonl, SP/run/cyphalm/cls.scribe.ts.jsonl.

Means per system [95% interval]; no order is assumed.

| system | n | SCRIBE v2 | SCRIBE v2, two-sided (exploratory) | SCRIBE v1 | surface S |
|---|---|---|---|---|---|
| held-out human | 169 | 0.770 [0.718, 0.819] | 0.756 [0.709, 0.802] | 0.456 [0.417, 0.496] | 0.904 [0.891, 0.917] |
| trie+cls:shape | 162 | 0.002 [0.002, 0.003] | 0.003 [0.002, 0.003] | 0.067 [0.054, 0.081] | 0.901 [0.892, 0.909] |
| trie samples (pooled) | 162 | 0.014 [0.006, 0.027] | 0.019 [0.008, 0.033] | 0.027 [0.020, 0.035] | 0.749 [0.737, 0.762] |

Pairwise accuracy, P(row > column) over text pairs with ties one half, on the oriented score (lower-is-better metrics negated).

### SCRIBE v2

| row > column | held-out human | trie+cls:shape | trie samples (pooled) |
|---|---|---|---|
| held-out human |  | 1.000 | 0.993 |
| trie+cls:shape | 0.000 |  | 0.399 |
| trie samples (pooled) | 0.007 | 0.601 |  |

### SCRIBE v2, two-sided (exploratory)

| row > column | held-out human | trie+cls:shape | trie samples (pooled) |
|---|---|---|---|
| held-out human |  | 0.999 | 0.991 |
| trie+cls:shape | 0.001 |  | 0.395 |
| trie samples (pooled) | 0.009 | 0.605 |  |

### SCRIBE v1

| row > column | held-out human | trie+cls:shape | trie samples (pooled) |
|---|---|---|---|
| held-out human |  | 0.903 | 0.958 |
| trie+cls:shape | 0.097 |  | 0.792 |
| trie samples (pooled) | 0.042 | 0.208 |  |

### surface S

| row > column | held-out human | trie+cls:shape | trie samples (pooled) |
|---|---|---|---|
| held-out human |  | 0.572 | 0.917 |
| trie+cls:shape | 0.428 |  | 0.944 |
| trie samples (pooled) | 0.083 | 0.056 |  |

Full meta.py output: [cyphalm_cls.full.md](cyphalm_cls.full.md).
