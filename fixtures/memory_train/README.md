# Memory train parity

**Frozen fixture** for one `DIFMemory.train` step.

- `before.cypha` — copy of `fixtures/reference.cypha`
- `after.cypha` — state after one memory train call (includes `field_a_eff` when `field_W_T` is present)
- `sidecar.json` — `h`, `h_field`, hyperparameters, `context_prior`, `f_field`

Gate: **`ctest -R native_memory_train`**. Native tool: **`memory_train_parity`**.
