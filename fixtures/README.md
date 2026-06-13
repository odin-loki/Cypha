# Parity fixtures

**Frozen regression goldens** for native CTest. Committed artifacts are authoritative; Python fixture generators were removed with the Python decommission.

## Root artifacts

- `reference.cypha` — v3 snapshot (Tier-1 `ctx_*`, `field_W_T`, `w_inject` when feat_dim≠field_dim)
- `expected.npz` — numeric targets for `score_matrix` / `infer` / `batch_infer`
- `native_parity.bin` — `F_field` + tensors for **`cypha_parity`** (v2: + `batch_infer_full` entropy & confidence tail)
- `manifest.json` — `fixture_schema`, seeds, geometry, label order
- `train_hparams.json` — training LRs + `align_every` / `temp_recalib_every` for `cypha_rest` /update
- `regression_head.json` — optional scalar MoE targets per class label

## Subdirectories

Each subdirectory has a **`sidecar.json`** (and often `before.cypha`, `f_field.json`) checked by a matching native parity binary. Gate: **`ctest -R native_<name>`**. Inventory: [`docs/verify/VERIFICATION_STATUS.md`](../docs/verify/VERIFICATION_STATUS.md).

Examples: `train_step_vector/`, `preprocessor/`, `preprocessor_fit/`, `csv_ingest/`, `memory_train/`, `batch_llr/`, `quantile_dif_train/`, `dif_train_replay/`, `studio_trainer_*`, `mke_train_*`, `regression_m4/`, `rff_regression/`, `two_stage_*`, `generation/`, `registry_register/`.

## Regenerating fixtures

Only when the **frozen contract** intentionally changes (`.cypha` layout, inference math, sidecar schema):

1. Rebuild native tools: `cmake --build native/build_intel --parallel`
2. Regenerate sidecars with **`cypha_fixture_gen`** (replaces removed Python `scripts/generate_*.py`):

   ```bash
   native/build_intel/cypha_fixture_gen --list
   native/build_intel/cypha_fixture_gen --fixture batch_llr --out fixtures/batch_llr
   native/build_intel/cypha_fixture_gen --fixture memory_train --out fixtures/memory_train
   native/build_intel/cypha_fixture_gen --fixture preprocessor --out fixtures/preprocessor
   native/build_intel/cypha_fixture_gen --fixture train_step_vector --out fixtures/train_step_vector
   native/build_intel/cypha_fixture_gen --fixture quantile_dif_train --out fixtures/quantile_dif_train
   native/build_intel/cypha_fixture_gen --fixture regression_m4 --out fixtures/regression_m4
   ```

   **`cypha_fixture_gen --list`** must include every generator above (CTest **`native_fixture_gen_list`**). Generators read committed design inputs from **`fixtures/`** (and root artifacts where noted), recompute native goldens, and write **`sidecar.json`** plus companion files (`before.cypha`, `f_field.json`, `preprocessor.json`, …).

   Root artifacts (`reference.cypha`, `native_parity.bin`, `f_field.json`, …) are still updated manually or via parity tools when the contract changes.

3. Gate: `ctest --test-dir native/build_intel -R native_ --output-on-failure`

See [`docs/verify/MAINTENANCE.md`](../docs/verify/MAINTENANCE.md).
