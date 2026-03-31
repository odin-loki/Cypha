# Two-stage DIF regression pipeline (native LLR)

- `sidecar.json` — one sample `x` from `parity_fixtures/expected.npz`, **`reference.cypha`** + **`f_field.json`** for native `batch_encode` + `score_matrix`, plus random stage-1/stage-2 weights and RFF params. Native tool recomputes LLR and full **`TwoStageDIFRegressor.predict`**-style output.

Regenerate (repo root on `PYTHONPATH`):

```bash
python scripts/generate_two_stage_pipeline_fixture.py
```

CTest: `native_regression_two_stage_pipeline`. Pytest: `tests/test_two_stage_pipeline_native_parity.py` (`CYPHA_TWO_STAGE_PIPELINE_PARITY_BIN` override).
