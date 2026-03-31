# Two-stage ridge fit (LLR given)

- `sidecar.json` — random LLR (**n×K**), **X**, **y**, RFF stage-2 weights; golden **`w1`**, **`b1`**, **`w2`**, **`b2`** and training **normalized** predictions from NumPy (same algebra as **`TwoStageDIFRegressor.fit`** after LLR is fixed).

Regenerate:

```bash
python scripts/generate_two_stage_ridge_fit_fixture.py
```

CTest: `native_regression_two_stage_ridge_fit` (requires **`k_native_regression_milestone` ≥ 7** — includes **`two_stage_dif_predict_batch`** check). Pytest: `tests/test_two_stage_ridge_fit_native_parity.py` (`CYPHA_TWO_STAGE_RIDGE_FIT_PARITY_BIN`).
