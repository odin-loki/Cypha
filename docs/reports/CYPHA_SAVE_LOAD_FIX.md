# Cypha save and load: what was lost, and the fix

**Odin Loch** · 2026-10-29 · core fix · found by [C0_C7_BASELINES_AND_ROUTING.md](C0_C7_BASELINES_AND_ROUTING.md)

**Short version.** `Cypha::save` and `Cypha::load` did not give back the model that was saved. A reloaded model scored inputs differently and learned differently. C0 saw a reloaded two-class model change 45 of 100 labels. Every saved Cypha model was affected. The fix is save format 2. It writes and reads every piece of state that `predict` and `update` use. A new CTest checks it to the bit. Old files still load as before.

## What was lost

Format 1 is the format before this fix. It has no version key. `save` copied the `.cypha` it was loaded from, then patched the world, the classes, `enc_W`, `field_h`, `temperature`, `llr_ema` and `total_steps`. Everything else kept the value from the loaded file. For a fresh model those are zeros.

**Written stale** (the value from the loaded file, not the trained one):

| State | Used by |
|---|---|
| `mahal_ema`, `mahal_std_ema` | GH gate baseline in `predict`; OOD tracking in `update` |
| `llr_scale_ema`, `llr_scale_n`, `llr_scale_baseline`, `base_temp` | temperature recalibration in `update` |
| tiered label context: `mid_n`, `mid_freq`, `mid_trans`, `ctx_hist_packed`, `ctx_cooccur`, `ctx_cooccur_tot`, `ctx_last_label` | the context prior in every score |
| field dynamics: `field_W_T`, `field_a_eff`, `field_sr_vec`, `field_step` | the NIG field in `update`, and so `field_h` |
| `total_correct`, `deliberation_lo`, `deliberation_hi` | settings callers set through `infer()` |

**Written but ignored by `load`**: the GH gate session (`gh_chi_session`, `gh_psi_session`, `gh_R_base`, `gh_inv_v_clean`) and `ood_sigma`. `load` reset them.

**Never written**:

- running totals the loader recomputes as plain sums (`mid_freq_total`, the `mid_trans` row totals); the sums differ in the last bits from the running values (the Tier-1 counts it rebuilds exactly from the window);
- training-memory scalars: the Welford `world_M2` (used for the first 20 rows), the cached `world_log_norm` and its refresh counter, `world_v_mean`, `task_prefix_protect`;
- hyperparameters set in code: `world_lr`, `delta_lr`, `enc_lr`, `replay_ratio`, `replay_cap`, `align_every`, `temp_recalib_every`;
- the encoder update count, the replay buffer, the RNG, the learned self-correction threshold;
- EWC: the anchors, the Fisher weights and `ewc_lambda`;
- the kernel reservoir RNG, `gamma_scale` and landmark mode (the basis and weights were written);
- the regression head after training: `update` with a target trains the MKE experts, but they lived only in `regression_head.json`, which `save` never wrote;
- the last active-query score.

### Measured before the fix

The new CTest (below) on the old code. 160 predictions per check (40 probes × 4 predict modes). Losses over the learning steps after the reload.

| Configuration | Predictions that differ after load | Labels that differ | Losses that differ while learning on |
|---|---|---|---|
| plain | 160 / 160 | 0 | 75 / 80 |
| GH gate | 160 / 160 | 0 | 79 / 80 |
| field dynamics | 160 / 160 | 0 | 79 / 80 |
| hyperparameters + replay | 160 / 160 | 0 | 79 / 80 |
| EWC | 160 / 160 | 0 | 80 / 80 |
| kernel LLR | 160 / 160 | 0 | 75 / 80 |
| regression head | 160 / 160 | 116 | 80 / 80 |
| early save (12 rows) | 160 / 160 | 0 | 39 / 40 |

The toy classes here are well apart, so most labels held. The scores did not. The regression head lost its trained experts and changed 116 of 160 labels. C0's closer two-class toy changed 45 of 100 labels.

## The fix: save format 2

- **Version.** `save` writes `cypha_save_version = 2` at the top level. A file without it is format 1. (`cypha_format` is a different key. It marks the class layout, 3 or 4, and does not change.)
- **Known keys, live values.** `save` now writes the live value of every top-level key the loaders already read: the stale list above, plus `ood_sigma` and the GH session keys. Older readers that open a format-2 file get the trained values.
- **New block.** Everything else goes in one map, `cypha_save_state`:
  - `infer`: the running totals (`mid_freq_total`, `mid_trans_tot`, `t1_total`, `t1_counts`);
  - `memory`: `world_M2`, `world_log_norm`, `world_log_n_ctr`, `world_v_mean`, `task_prefix_protect`;
  - `session`: the hyperparameters, `enc_updates`, `ewc_lambda`, the last active-query score, the RNG, the self-correction threshold;
  - `replay`: the buffer rows, weights and labels;
  - `ewc`, `kernel`, `regression`: present only when used.
- **Exact numbers.** Every number is written as a 64-bit float or integer. The RNG is written in the C++ standard's text form for `std::mt19937`. Hash-map contents are written in sorted key order, so two models with the same state write the same bytes.
- **Load order.**
  - Hyperparameters: defaults, then the saved values, then a `train_hparams.json` (passed in, or next to the file). The JSON file still wins, as before.
  - Regression head: a path passed to `load` wins; then the saved head; then a `regression_head.json` next to the file. The saved head beats the file next to it because that file may predate the training.
  - EWC: a format-2 file without an `ewc` block clears any EWC state the object had.
- **Public API.** Unchanged. `Cypha` gained private helpers. `ReplayBuffer`, `EwcRegularizer`, `NigStatisticState`, `EpistemicThreshold` and `KernelMemory` gained small inline accessors for their state.

### Old files

- A format-1 file loads exactly as before. The new fields get today's behaviour: the GH session, replay buffer, RNG and running totals start fresh.
- When a format-1 file has classes, `load` prints one line to stderr: `note: <path> uses Cypha save format 1; GH gate session, replay buffer, RNG and running totals start fresh (re-save to keep them)`. A fresh, empty model prints nothing.
- Re-saving a format-1 file writes format 2.

### File size

The replay buffer is the only large addition: 8·d bytes a row (plus a label index and a weight), at most `replay_cap` rows (default 10,000). At d = 10 that is under 1 MB. At d = 384 a full buffer adds about 31 MB. The other new state is small. The buffer's `f` rows are not kept, since nothing reads them back. A save option to leave the buffer out is a possible follow-up if size matters.

## Tests

**New CTest `native_cypha_save_load_test`** (`native/tools/cypha_save_load_test.cpp`). It trains model A (d = 10, 3 overlapping classes, sticky label order so the context prior matters), saves it, and loads the file into a fresh model B. Then it requires:

1. every piece of state equal to the bit (128 checks through the public accessors);
2. byte-identical `predict` output on 40 fixed probes (6 of them far out) under four modes: GH gate, plain with field, plain without field, self-correcting. It compares label, confidence, scores, probabilities, y, uncertainty, anomaly, OOD flag, `r_eff`, the self-correction fields, the active-query score and the drift score;
3. identical losses while A and B learn on the same 80 more rows (40 for the early save);
4. the state and predictions equal again after that;
5. byte-identical files when A and B are saved again;
6. a third model loaded from B's second save equal to A;
7. a file with the format-2 keys removed still loads and predicts.

Eight configurations: plain; GH gate; field dynamics (non-zero injection, `F_field` and `W_T`); custom hyperparameters with replay, task-sticky learning and non-default `infer()` settings; EWC; kernel LLR (reservoir of 24, so the RNG is used); regression head (MKE experts trained through `update`); early save after 12 rows (`world_n` ≤ 20).

**Results.**

- Before the fix: all 8 configurations fail (table above). The test was committed failing first. That first version had checks 2 to 7; check 1 came with the fix, because it reads state through the new accessors.
- After the fix: all 8 pass. Every check is exact to the bit. The test takes under a second.

**Planted bugs.** Each row drops one piece of state from `save` (or from `load`, for state format 1 already wrote), rebuilds, and runs the test.

Results (50 planted bugs, one at a time):

| Dropped | Configurations that fail |
|---|---|
| `mahal_ema`, `mahal_std_ema`, `llr_scale_ema`, `llr_scale_n`, `field_step`, `mid_n`, `mid_freq`, `mid_trans`, `ctx_hist_packed`, `ctx_cooccur`, `ctx_cooccur_tot`, `ctx_last_label`, `gh_R_base`, `gh_inv_v_clean`, `ood_sigma`, `mid_freq_total`, `world_M2`, `world_v_mean`, the RNG, the replay buffer, the last active-query score | all 8 |
| `enc_W`, `world_log_norm`, `world_log_n_ctr`, `enc_updates`, `mid_trans_tot`, the self-correction threshold | 7 of 8 |
| `gh_chi_session` | the 3 that train with the GH gate |
| `gh_psi_session`, `field_h`, `field_W_T`, `field_a_eff`, `field_sr_vec` | field dynamics |
| `llr_scale_baseline`, `base_temp`, `total_correct`, `deliberation_lo`, `task_prefix_protect`, `world_lr`, `delta_lr`, `enc_lr`, `replay_ratio`, `align_every`, `temp_recalib_every` | hyperparameters + replay |
| `ewc_lambda`, the EWC block | EWC |
| the kernel RNG | kernel LLR |
| the regression block | regression head |
| `t1_total`, `t1_counts` | none |

48 of 50 are caught. The two that are not lose nothing: the loader rebuilds both exactly from the saved Tier-1 window (`ctx_hist_packed`, whose drop is caught). They are saved anyway, so a zero count survives too.

Some settings never change during training (`base_temp`, `total_correct`, the deliberation band, ψ of the GH session). The test sets them off their defaults so that a drop shows. It uses d = 10 so running means differ in the last bits from recomputed ones (with d = 8, a power of two, they agree).

## Callers checked

Every caller still builds. Results:

- **Full CI set** (`ctest -LE cypha_slow`, 141 tests): 139 pass, and 2 are skipped as in CI (federated TLS, CUDA bench). The first pass ran on a heavily loaded machine (about 16 agents on 4 cores): three hp tests hit their 120 s and 180 s limits, and 16 tests needed binaries not yet built. All of them pass on a re-run.
- **`cypha_one_smoke`** loads `fixtures/reference.cypha` with its regression head, learns, saves, reloads and predicts. It passes. The reload now carries the regression head from the file.
- **REST** (`cypha_rest`, `dif_rest_routes`) only loads models. It builds, and `native_rest_schema_contract` passes. A registry load passes a `regression_head.json` found next to the model as an explicit path, so that file wins over a saved head.
- **Branch A router** is in `cypha_core` and builds. It has its own save (see below).
- **Forecast node estimator** keeps a `Cypha` in memory and never saves it. It builds; `native_forecast_drift_backtest` and `native_forecast_gdelt_poll` pass.
- **Thought bindings** (`bindings_cypha.hpp`) write a fresh root and load it. `native_thought_bindings_test` passes.
- **Benches**: `cypha_bench_native` and `cypha_bench_run` build; `native_d03_xor_kernel_basis_default_smoke` and `_rff_smoke` pass.
- **Other savers and loaders**: `native_kernel_cypha_roundtrip`, `native_memory_train_roundtrip`, `native_class_gmm_p3_smoke`, `native_create_model`, `native_topic_bench_selftest`, `native_class_context_test` and the EWC smokes pass.
- **Not built here**: the Qt shell (no Qt in this container) and `cypha_word_classes` (not a target in this configuration). `cypha_word_classes` only loads fresh roots. The Qt shell calls `Cypha::save` and also has its own writer (below).
- **Portability**: standard C++ only (`std::filesystem` temporary directory, classic locale for the RNG text). No POSIX calls.

## Committed model files

- The repository holds 20 `.cypha` files, all under `fixtures/`. There are none under `data/`, `bench/` or `models/`.
- None was written by the old `Cypha::save`. That save always wrote `gh_chi_session`, and no fixture has it. They come from the Python saver or from `memory_train_roundtrip`, and their `mahal_ema` and context keys hold real values.
- They stay format 1 and load exactly as before. Those with classes (12 of 20) print the one-line note when loaded through `Cypha::load`.
- Nothing needs to be regenerated.

## Not covered, and follow-ups

- **Branch A router.** `BranchARouter::save_checkpoint` (`native/src/branch_a_router.cpp`) writes only the world and the classes through `merge_state_into_root_for_save`. Its encoder, temperature, `mahal_ema`, context and field keep the template's values. Same class of bug; not changed here.
- **Qt shell.** `shell_main.cpp` has its own writer for the studio model (next to its call to `Cypha::save`). Not changed or tested here.
- **Kernel in RFF mode.** `patch_kernel_into_root` does not write the RFF projection. `Cypha` never builds an RFF kernel itself; only callers that set `kernel_mem_owned()` can.
- **Separate files.** The preprocessor and the sequence model have their own files and are unchanged.
