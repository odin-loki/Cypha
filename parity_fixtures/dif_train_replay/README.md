# DIF train with priority replay (`replay_ratio > 0`)

Same harness as **`quantile_dif_train/`** (`quantile_dif_train_parity`), but the sidecar includes a recorded **`replay_u01`** array: one U(0,1) for each replay **gate** evaluation when the buffer has ≥10 samples, plus **one draw per replay slot** when a replay batch runs (same order as `PriorityReplayBuffer.sample` / native `ReplayBuffer::sample`).

NumPy **`MT19937(seed)`** is **not** bit-identical to **`std::mt19937(seed)`**, so cross-language parity uses this shared stream instead of matching integer seeds.

Regenerate:

```bash
python scripts/generate_dif_train_replay_fixture.py
```

CTest: **`native_dif_train_replay`**. Pytest: **`tests/test_dif_train_replay_native_parity.py`** (override with **`CYPHA_DIF_TRAIN_REPLAY_PARITY_BIN`**; defaults to the same **`quantile_dif_train_parity`** binary).
