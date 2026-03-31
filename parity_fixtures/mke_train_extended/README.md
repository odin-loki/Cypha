# `mke_train_extended`

Fixture for `native/tools/mke_train_step_parity.cpp` extended mode (CTest: `native_mke_train_extended`).

Multi-step `MKERegressor.train_step` with **`enc_lr > 0`**, **`replay_ratio > 0`**, sidecar **`replay_u01`** (recorded replay draws), and **`replay_warmup`** (buffer repush after `before.cypha`). `fixture_schema` 2 and `steps[]`.

Regenerate:

```bash
python scripts/generate_mke_train_extended_fixture.py
```
