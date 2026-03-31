# Two-stage ridge on **real** quantile-DIF LLR

- `sidecar.json` — training **X**, **y**, **LLR** from a full Python **`TwoStageDIFRegressor.fit`** (y-quantile **`CyphaDIF`** + same ridge/RFF steps as production), plus stage-2 RFF weights and expected **w1/b1/w2/b2** / training **ŷ** (normalized).

Native **`regression_two_stage_ridge_fit_parity`** runs on this file (same binary as synthetic **`two_stage_ridge_fit/`**).

Regenerate (repo root on `PYTHONPATH`):

```bash
python scripts/generate_two_stage_e2e_ridge_fixture.py
```

CTest: **`native_regression_two_stage_e2e_ridge`**.
