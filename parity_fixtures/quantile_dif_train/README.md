# Quantile-style DIF train replay

- `before.cypha` — fresh **`CyphaDIF`** snapshot before any `train_step`.
- `f_field.json` — **`memory.world.F_field`** at that time (same M1 pattern as main parity).
- `sidecar.json` — permuted quantile labels (`_ts_*`), per-step losses, row-major **X**, expected field-conditioned **LLR** (`use_field=True`), hyperparameters.

**Replay:** sidecar sets **`replay_ratio: 0`** on both Python and native so priority replay never runs — the buffer may still grow past 10 entries, but no extra `memory.train` from replay and no Python **`np.random.random()`** draw (see **`CyphaDIF`** `replay_ratio` and short-circuit in `train_step`). **`enc_lr: 0`** freezes the encoder.

Avoid **`total_steps` reaching 20** in the fixture without native OOD hooks: the generator keeps the step count below that so Python OOD EMA and native parity (optional null OOD pointers) stay aligned.

Regenerate:

```bash
python scripts/generate_quantile_dif_train_fixture.py
```

Also runs at the end of **`python scripts/generate_parity_fixtures.py`** (with **`PYTHONPATH`** set to the repo root).

Native: **`quantile_dif_train_parity`** replays **`dif_train_step_vector`** then **`batch_llr_from_x`**.

CTest: **`native_quantile_dif_train`**. Pytest: **`tests/test_quantile_dif_train_native_parity.py`** (`CYPHA_QUANTILE_DIF_TRAIN_PARITY_BIN`).

For **`replay_ratio > 0`**, see **`parity_fixtures/dif_train_replay/`** (recorded **`replay_u01`**, CTest **`native_dif_train_replay`**).
