# Batch LLR from raw X

- `sidecar.json` — rows of `x_input` and matching `llr` from **`fixtures/expected.npz`** (same batch as **`native_parity.bin`**).

**Frozen fixture.** Update only when the inference contract changes; gate with native **`batch_llr_parity`**.

Native checks **`cypha::batch_llr_from_x`** (`batch_encode` + `score_matrix_use_field`) against **`reference.cypha`** + **`f_field.json`**.

CTest: **`native_batch_llr`**. Override binary: **`CYPHA_BATCH_LLR_PARITY_BIN`**.
