# Quantile-style DIF train replay

- `before.cypha` — fresh classifier snapshot before any `train_step`.
- `f_field.json` — **`memory.world.F_field`** at that time (same M1 pattern as main parity).
- `sidecar.json` — permuted quantile labels, per-step losses, row-major **X**, expected field-conditioned **LLR** (`use_field=True`), hyperparameters.

**Replay:** sidecar sets **`replay_ratio: 0`** so priority replay never runs. **`enc_lr: 0`** freezes the encoder.

**Frozen fixture.** Regenerate only via native **`quantile_dif_train_parity`** when the train contract changes.

Native: **`quantile_dif_train_parity`** replays **`dif_train_step_vector`** then **`batch_llr_from_x`**.

CTest: **`native_quantile_dif_train`**. Override: **`CYPHA_QUANTILE_DIF_TRAIN_PARITY_BIN`**.

For **`replay_ratio > 0`**, see **`fixtures/dif_train_replay/`** (CTest **`native_dif_train_replay`**).
