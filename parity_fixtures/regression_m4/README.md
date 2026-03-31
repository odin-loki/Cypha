# Regression M4 parity

- `sidecar.json` — golden vectors for native `regression_m4_parity` (CTest `native_regression_m4`):
  - `batch` + `ema` + `ema_init` — `predict_mixture_batch` + `expert_target_ema_step`
  - `rff_rls` — `rff_rls_train_step` vs **`RFFRegressor.train_step`**
  - `mke_rls` — `mke_expert_rls_scalar_step` (incl. forgetting + low-π no-op) vs **`MKERegressor.train_step`** inner loop
  - `two_stage` — `two_stage_dif_predict` vs **`TwoStageDIFRegressor.predict`** combine
  - `mke_route` — `router_softmax_from_llr`, `mke_scalar_predict_from_llr`, `mke_routing_entropy` (K≤8 and K>8 softmax branches) vs **`MKERegressor.predict`** routing

Regenerate:

```bash
python scripts/generate_regression_m4_fixture.py
```

On Windows, if `python` has no NumPy: `py -3 scripts/generate_regression_m4_fixture.py`.
