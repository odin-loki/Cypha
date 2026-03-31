# `mke_train_step`

Fixture for `native/tools/mke_train_step_parity.cpp` (CTest: `native_mke_train_step`).

One `MKERegressor.train_step`: RFF φ, routing LLRs from `CyphaDIF.score_matrix(φ)` (same as Python `_route`), expert RLS, then `CyphaDIF.train_step(x, pred)` via native `dif_train_step_vector` with φ as the feature buffer (`enc_lr=0`, `replay_ratio=0`).

Regenerate:

```bash
python scripts/generate_mke_train_step_fixture.py
```
