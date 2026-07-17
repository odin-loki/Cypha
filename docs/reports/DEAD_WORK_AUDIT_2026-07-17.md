# Dead-work re-audit: Phase 2 MoE EM, Phase 3 class_gmm, Phase 9 CriticalityVector (2026-07-17)

**Scope:** Re-audit `train_step` / `predict_next` / DIF infer hot paths after Parts 4+6 skipped dead
DIF/BPTT-slow on the locked D17 default. Confirm new features are true no-ops when flags are
default-OFF; fix accidental always-on cost; ship Part-4-style free skips if found.

**Did not touch:** `native/build_math`, `native/build_deff`, `BASELINE_*`, overnight scripts.

**HEAD at audit:** `27dcd07`. Build verification: `native/build_perf_dead` (MinGW Release).

## Summary

| Feature | Default | D17 hot path cost when OFF | Verdict |
|---------|---------|----------------------------|---------|
| **Phase 3 `use_class_gmm`** | `false` | Legacy `K×d` indexing via `class_gmm_d_offset(..., max_m=1)`; GMM loops/EM/responsibilities only in `if (use_class_gmm)` branches | **Clean no-op** |
| **Phase 2 MoE EM (`em_step`)** | N/A on D17 | `responsibilities()` only in `mke_scalar_train_step` (REST/Qt/regression) and `class_gmm_component_responsibilities` (GMM ON) | **Not on D17 path** |
| **Phase 9 CriticalityVector** | REST/report only | `criticality_vector()` / `hot_criticality_input()` not called from `CyphaLMModel::train_step` unless explicit profiler/monitor/REST export | **Clean no-op** |
| **Part 4 DIF skip** | active at D17 | `dif_subsystem_affects_forward()` + `skip_dif` still gate predict/train | **Still active** |
| **Part 6 BPTT slow skip** | active at D17 | `bptt_slow_for_ewc` gates slow-tier outer product when `ewc_lambda=0` | **Still active** |

**Shipped (Part 7b):** skip redundant `last_dif_out_.mean.assign(field_dim, 0)` on the D17 dead-DIF
`predict_next` branch (`skip_dif_subsystem` true). Mean is unreachable there (GRIA gets `nullptr`
DIF); zero-fill was leftover from pre-Part-4 path.

## Phase 3 — `use_class_gmm` (default OFF)

**Hot paths audited:**

- `CyphaDifMemoryState::memory_train` — scoring/update/meta blocks branch on `!use_class_gmm` vs GMM;
  `gmm_view()` is a stack struct fill only used in GMM branches (no extra allocations when OFF).
- `score_matrix_use_field` / `classify_at_h` — legacy batched LLR when OFF; GMM log-sum-exp only when ON.
- `dif_train_step_vector` — `TrainStepExtras::use_class_gmm` defaults `false`; only toggles memory when
  extras pointer sets it (bench smokes opt-in).
- `sync_infer_model_from_memory` — copies empty `class_pi` / `class_n_comp` when OFF (cheap).

**Storage:** `gmm_max_m()` returns `1` when OFF, so `class_gmm_d_offset(k,0,d,1) == k*d` (legacy layout).

**Tests:** `native_class_gmm_p3_smoke` (OFF path bit-identical inference).

## Phase 2 — MoE EM in MKE

**Hot paths audited:**

- `mke_scalar_train_step_from_phi` — always runs EM when MKE is invoked (intentional). Not called from
  CyphaLM D17 (`n_experts: 0`, no MKE head).
- `class_gmm_component_responsibilities` — calls `responsibilities()` only when `use_class_gmm` and
  `M > 1`.

D17 online regressor (`online_reg_train_step`) uses `dif_train_step_vector` without MKE EM.

## Phase 9 — CriticalityVector

**Hot paths audited:**

- `IntelligenceProfiler::criticality_vector()` — REST (`GET /intelligence/criticality`), profile JSON
  export, smoke tools only.
- `LmIntelligenceMonitor::hot_criticality_input()` — defined but **not referenced** from
  `train_step` / `predict_next`; no accidental per-step build.
- Default D17 bench (`cyphalm_bench_native --profile d17`) passes `nullptr` profiler unless
  `--math-integration` / `--intelligence-profile`.

Mid-tier fields remain behind `CriticalityBuildOptions::enable_mid` (default `false`).

**Tests:** `native_criticality_vector_p9_smoke` (infer byte-identical with monitor reads).

## Part-4-style skip shipped (Part 7b)

**Issue:** After Part 4 gates `CyphaDIF::predict` off on D17, the `else` branch still zero-filled
`last_dif_out_.mean` every `predict_next` (~160 doubles/step at `field_dim=160`).

**Fix:** Gate `mean.assign` on `!skip_dif_subsystem` so dead-DIF configs leave `mean` empty.
Conservative: configs that still pass `&last_dif_out_` into GRIA (non-ngram-fusion DIF paths) keep
the zero-fill for `gria_controller` fallback behavior.

**Determinism:** Bit-identical BPC expected on D17 default (mean unread when skipped).

## No further free skips found

At D17 default, remaining per-step work is live: SSM + fast BPTT, GRIA + n-gram fusion, LSTM
forward/backward, memory retrieve/store, Laplace prior refresh. No accidental always-on cost from
Phase 2/3/9 beyond the trivial Part 7b zero-fill.

See also **Part 7b** appendix in `docs/reports/PERFORMANCE_PROFILE_2026-07-12.md`.
