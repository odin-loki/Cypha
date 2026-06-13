# MKE train step parity

One scalar **`MKERegressor.train_step`** slice vs native **`mke_train_step_parity`**.

**Frozen fixture.** Gate: **`ctest -R native_mke_train_step`**. Override: **`CYPHA_MKE_TRAIN_STEP_PARITY_BIN`**.
