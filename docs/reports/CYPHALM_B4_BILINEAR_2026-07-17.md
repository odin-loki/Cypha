# CyphaLM B4 — bilinear fusion (2026-07-17)

Bill of Work §4 / `docs/CYPHALM_UPGRADE_V2.md` Track B option **B4**: low-rank bilinear term on top of sum fusion — `v = field_part + embed_part + U @ ((V_f @ field_x) ⊙ (V_e @ embeds))`, rank = `gria_rank` (32 @ d17).

## Implementation

- Config: `ngram_bilinear_fusion` (default **false**); CLI `--ngram-bilinear-fusion`.
- Sum mode only; gated/MLP paths unchanged.
- Weights: `W_b_u` [field_dim × rank], `W_b_vf` [rank × field_in], `W_b_ve` [rank × embed_in] (~16k params @ d17).
- Truncated BPTT: bilinear Jacobian w.r.t. `field_x` chained in `NgramFusion::grad_field_x`.
- Checkpoint: `ngram_W_b_u`, `ngram_W_b_vf`, `ngram_W_b_ve` in NPZ when enabled.

Build dir: `native/build_b4` (Ninja Release, MinGW).

## Measurement protocol

```text
cyphalm_bench_native --profile d17 --mode hybrid --n-train 5000 --n-eval 256 --threads 1 --bench-seed 42
# vs same + --ngram-bilinear-fusion
```

Corpus: WikiText-2. Success criterion in BoW (≥0.03 BPC @ 300k) **not** run here; report honest short-budget deltas only.

## Results

| Condition | n_train | BPC | ΔBPC (B4 − sum) |
|-----------|---------|-----|-----------------|
| sum (`ngram_fuse_split`) | 5000 | **4.039555743981927** | — |
| B4 bilinear ON | 5000 | **4.039555738583003** | **−5.4×10⁻⁹** |

Delta is at floating-point noise; no measurable BPC gain at 5k.

## Decision

**Keep default OFF.** At 5k the gain is ~0 BPC (noise floor), far below the 0.03 BPC@300k bar. Fixed random bilinear weights (same as sum-fusion `W_field`/`W_embed`) add capacity but no online fusion-weight updates — unlikely to reach ≥0.03 without a 300k sweep and fusion SGD.
