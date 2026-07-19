# CyphaLM B3 — n-gram position weights (2026-07-17)

Bill of Work §4 / `docs/CYPHALM_UPGRADE_V2.md` Track B option **B3**: learnable scalars `w_0..w_{ngram}` scale each history embed before `W_e`, opt-in vs `ngram_fuse_split` sum baseline.

## Implementation

- Config: `ngram_position_weights` (default **false**); CLI `--ngram-position-weights`.
- Forward: already scaled embeds when `pos_weights_` non-empty (`NgramFusion`).
- Train: `NgramFusion::update_position_weights` — sum-mode CE grad through `W_embed` → per-position SGD at `gria_lr`.
- Checkpoint: `ngram_pos_weights` already serialized.

Build dir: `native/build_b3` (Ninja Release, MinGW).

## Measurement protocol

```text
cyphalm_bench_native --profile d17 --mode hybrid --n-train N --n-eval 256 --threads 1 --bench-seed 42
# vs same + --ngram-position-weights
```

Corpus: WikiText-2. Success criterion in BoW (≥0.03 BPC @ 300k) **not** run here; report honest short-budget deltas only.

## Results

| Condition | n_train | BPC | ΔBPC (B3 − sum) |
|-----------|---------|-----|-----------------|
| sum (`ngram_fuse_split`) | 5000 | **4.039555743981927** | — |
| B3 position weights ON | 5000 | **4.039501326937773** | **−0.000054** |

20k runs were interrupted / not completed in this pass.

## Decision

**Keep default OFF.** At 5k the gain is ~0.00005 BPC (noise floor), far below the 0.03 BPC@300k bar. No evidence to flip the production default.
