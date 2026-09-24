# CyphaLM winner v2

Serving manifests for the best CyphaLM configurations measured
(`docs/reports/CYPHALM_LM_QUALITY_REPORT.md`, "Winner v2"). The checkpoints are
gigabytes, so they are not in git: rebuild them with

```bash
scripts/build_cyphalm_winner.sh /path/to/enwik8 /path/to/winner native/build 4
native/build/cyphalm_generate --load /path/to/winner/winner.json --prompt "..."
```

| manifest | what | held-out wiki / Alice / lcet10 (bits/byte) | RAM (private + mapped) |
|---|---|---|---|
| `winner.json` (default) | 4 lean shards (upstream mixer settings, occupancy fold 0.8, sentst_ dropped) + ∞-gram index + byte LSTM | 1.6192 / 2.0191 / 1.5160 | 1.0 GB + 0.4 GB |
| `winner_full.json` | all 11 such shards + index + LSTM | 1.6158 / 2.0113 / 1.5038 | 2.7 GB + 0.4 GB |
| `winner_light.json` | one slim model on 95 MB + index + LSTM | 1.6393 / 2.0466 / 1.5388 | 0.46 GB + 0.44 GB |

Paths (`shard_N.json`, `slim95.json`, `enwik8`, `lstm.blm`) are relative to
the manifest. The ∞-gram index is not stored: loading indexes the first 95 MB
of the corpus (`infinigram_bytes`) in memory with libsais, about 6 s on one
core. A stored index (`cyphalm_infinigram_build`, "IGR2") still works in
`infinigram`. The measured `slim95` was a lean 95 MB model cut to
slim after training; the script trains slim directly. An ∞-gram weights file
from before 256 buckets (`INDEX.weights.json`) next to the index makes loading
fail: delete it.
