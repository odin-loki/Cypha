# Batch LLR from raw X

- `sidecar.json` — rows of `x_input` and matching `llr` from **`parity_fixtures/expected.npz`** (same batch as **`native_parity.bin`**).

Regenerate (after `expected.npz` exists):

```bash
python scripts/generate_batch_llr_fixture.py
```

Native checks **`cypha::batch_llr_from_x`** (`batch_encode` + `score_matrix_use_field`) against **`reference.cypha`** + **`f_field.json`**.

CTest: **`native_batch_llr`**. Pytest: **`tests/test_batch_llr_native_parity.py`** (`CYPHA_BATCH_LLR_PARITY_BIN`).
