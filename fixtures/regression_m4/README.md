# Regression M4 parity

- `sidecar.json` — golden vectors for native **`regression_m4_parity`** (CTest **`native_regression_m4`**):
  - `batch` + `ema` + `ema_init` — mixture batch + EMA step
  - `rff_rls`, `mke_rls`, `two_stage`, `mke_route` — RLS, two-stage combine, MKE routing blocks

**Frozen fixture.** Update only when regression kernel contract changes.

Gate: **`ctest -R native_regression_m4`**. Override: **`CYPHA_REGRESSION_M4_PARITY_BIN`**.
