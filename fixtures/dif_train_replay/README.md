# DIF train with priority replay (`replay_ratio > 0`)

Same harness as **`quantile_dif_train/`** (`quantile_dif_train_parity`), but the sidecar includes a recorded **`replay_u01`** array for replay gate and sample draws.

**Frozen fixture.** Update only when the replay contract changes; native tool: **`quantile_dif_train_parity`** / replay sidecar workflow.

CTest: **`native_dif_train_replay`**. Override: **`CYPHA_DIF_TRAIN_REPLAY_PARITY_BIN`**.
