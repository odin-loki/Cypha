# CyphaLM winner v3

Serving manifests for the best CyphaLM configurations measured
(`docs/reports/CYPHALM_LM_QUALITY_REPORT.md`, "Winner v3"). The checkpoints are
gigabytes, so they are not in git: rebuild them with

```bash
scripts/build_cyphalm_winner.sh /path/to/enwik8 /path/to/winner native/build 4
native/build/cyphalm_generate --load /path/to/winner/winner.json --prompt "..."
```

| manifest | what | held-out wiki / Alice / lcet10 (bits/byte) | RAM (private + mapped) |
|---|---|---|---|
| `winner.json` (default) | 4 lean shards (upstream mixer settings, occupancy fold 0.8, sentst_ dropped) + ∞-gram index + byte LSTM + byte Transformer (output layers adapting) | 1.5927 / 1.9992 / 1.4984 | 1.0 GB + index |
| `winner_full.json` | all 11 such shards + the same experts | 1.5920 / 1.9929 / 1.4889 | 2.7 GB + index |
| `winner_light.json` | one slim model on 95 MB + the same experts | 1.5997 / 2.0192 / 1.5186 | 0.46 GB + index |

Paths (`shard_N.json`, `slim95.json`, `enwik8`, `lstm.blm`, `gpt.bgt`) are relative to
the manifest. The ∞-gram index is not stored: loading indexes the first 95 MB
of the corpus (`infinigram_bytes`) in memory with libsais, about 6 s on one
core. A stored index (`cyphalm_infinigram_build`, "IGR2") still works in
`infinigram`. The measured `slim95` was a lean 95 MB model cut to
slim after training; the script trains slim directly. An ∞-gram weights file
from before 256 buckets (`INDEX.weights.json`) next to the index makes loading
fail: delete it.
