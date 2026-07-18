# PGM (H23) vs Hybrid / CharLSTM — BPC Compare (2026-07-18)

WikiText-2 char LM, profile `d17`, `threads=1`, seed 42, binary
`native/build-pgm/cyphalm_bench_native.exe` (includes H23).

## Matched default recipe

| n_train | CharLSTM BPC | Hybrid BPC | H23 (PGM) BPC | Δ(H23−hybrid) | hybrid GRIA weight |
|--------:|-------------:|-----------:|--------------:|--------------:|-------------------:|
| 2,000 | 4.6821 | 4.7035 | 4.7035 | ~0 | 0.0175 |
| 10,000 | 3.8827 | 3.8870 | 3.8870 | ~0 | 0.0035 |
| 40,000 | 3.3070 | 3.3081 | 3.3081 | ~0 | 0.0009 |

Pin reference (not re-run here): hybrid **2.873** @ 300k.

## Note on `--hybrid-blend-logit 0`

Flag is overridden by learnable blend training (logit still drifts to ≈ −5.66 @ 10k).
Cannot force a lasting 50/50 mix without freezing `hybrid_blend_learnable`.

## Interpretation

1. **H23 ≈ hybrid to ~1e-6 BPC** under the default D17 recipe. PGM only blends into the
   **field / GRIA** path (`0.6·field + 0.4·pgm`). That path’s mixture weight collapses toward
   zero as training proceeds (`hybrid_gria_weight → 10⁻³`), so LSTM dominates BPC.
2. **CharLSTM alone is slightly better** than hybrid/H23 at these budgets — same historical
   pattern: early training, GRIA dilutes the LSTM.
3. PGM is **not yet a BPC competitor** as wired: no BPTT into the graph, no replacement of
   the LSTM head, field path nearly muted.

## What would make a real BPC contest

- Replace or co-train the CharLSTM head with PGM (primary logits from PGM context), **or**
- Keep a non-vanishing field blend and train PGM / binders with gradients, **or**
- Run an ablation mode: `ContextMode` where PGM context → `Wy` directly.

Artifacts: `bench/results/pgm_bpc_compare/*.json`.
