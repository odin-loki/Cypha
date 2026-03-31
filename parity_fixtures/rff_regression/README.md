# RFF / ridge / MKE-dot parity fixture

- **`sidecar.json`** — RFF batch features (same layout as **`RFFEncoder.batch_encode`**), ridge-with-bias coefficients on normalized targets (`RFFRegressor`-style closed form), and per-expert linear dots φ·w_k (`MKERegressor` forward slice). CTest **`native_regression_rff`** runs **`regression_rff_parity`**.

Regenerate:

```bash
python scripts/generate_rff_regression_fixture.py
```

For batched LLR from raw **X**, see **`parity_fixtures/batch_llr/`** + **`cypha::batch_llr_from_x`**. Two-stage ridge **fit** / batched predict: **`parity_fixtures/two_stage_ridge_fit/`** and **`two_stage_dif_predict_batch`**.
