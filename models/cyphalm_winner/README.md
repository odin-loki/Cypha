# CyphaLM winner

Serving manifests for the best CyphaLM configurations measured
(`docs/reports/CYPHALM_LM_QUALITY_REPORT.md`, "Speed round and the winner"). The checkpoints are
gigabytes, so they are not in git: rebuild them with

```bash
scripts/build_cyphalm_winner.sh /path/to/enwik8 /path/to/winner native/build 4
native/build/cyphalm_generate --load /path/to/winner/winner.json --prompt "..."
```

| manifest | what | held-out wiki / Alice / lcet10 (bits/byte) | RAM (private + shared) |
|---|---|---|---|
| `winner.json` | 11 slim shard models (enwik8 95 MB split, table bits 20) + ∞-gram index | 1.6516 / 2.0258 / 1.5124 | 3.3 GB + 0.6 GB |
| `winner_light.json` | one slim model on 95 MB + ∞-gram index | 1.6909 / 2.0639 / 1.5566 | 0.44 GB + 0.45 GB |

Both serve with the current defaults: frozen scoring, bit-tree pruning at
1e-4 (~2 ms per next-byte distribution), half-rate serve mixer, learned
ensemble and ∞-gram weights, and word lookahead (K 8) for generation.
Shard paths in `winner.json` (`shard_0.json` … `shard_10.json`, the
`cyphalm_shard_train` names) and `enwik8_95m.igr` are relative to the
manifest.

Measured on 16 KiB held-out slices (wiki = enwik8 @ 96,000,000, Alice =
`alice29.txt` @ 20,000, lcet10 @ 50,000) with `cyphalm_lm_quality --load
MANIFEST` at its defaults: bit-tree pruning 1e-4, learning on (in-context),
trained mixer rate (the harness does not apply the 0.5 serve rate that
generation uses). Raw JSON: `docs/reports/lm_quality/win_*.json`.

Differences from the measured files:

- The measured `slim95` checkpoint was a lean 95 MB model cut to slim after
  training (its config says tier `lean`). The script trains `--tier slim`
  from scratch. The two routes measured equal at 8 MiB (1.8341 trained slim vs 1.8327
  converted), not at 95 MB.
- No ∞-gram starting weights ship. If an `enwik8_95m.igr.weights.json` from
  before commit `2d30e6a` (384 values) sits next to the index, loading
  fails: delete it.

Manifest keys, checkpoint and index formats, flags and env vars: the
report's "Reference" section.
