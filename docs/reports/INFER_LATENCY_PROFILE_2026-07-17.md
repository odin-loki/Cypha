# Infer latency profile — product surface (2026-07-17)

**Scope:** Profile `cypha_rest` `/predict` (DIF `gh_infer_at_h` / `score_matrix_use_field`) and CyphaLM `/generate` (`predict_next`). Infer-only; not `train_step`.

**OFF-LIMITS:** `build_math`, `build_deff`, `BASELINE_*`, overnight.

**Build:** `native/build_perf_infer` (Ninja, Release, MinGW 13.2.0). Tool: `infer_latency_bench`.

**Parity:** `gh_infer_deliberation_golden` → OK (bit-identical labels/confidence on fixture).

## Hot paths

| Surface | Call chain |
|---------|------------|
| REST `POST /predict` (default GH) | `batch_encode` → `gh_infer_at_h` → `classify_at_h` |
| REST batch / uncertainty-rank | `score_matrix_use_field` → `rpsm_score_matrix_batched` (default) |
| CyphaLM `POST /generate` | `generate_decode` → `predict_next` (D17 hybrid already scratch-reused) |

## Measurements (`infer_latency_bench`)

Fixture: `fixtures/gh_infer_deliberation` (d=8, K=3). CyphaLM: D17 hybrid synthetic (short train then decode).

Host load was noisy (CyphaLM wall times vary ~2× across back-to-back runs). Prefer DIF `score_matrix_*` for A/B; report medians / quieter second pass where noted.

### Before (pre-scratch, same session earlier; 5k iters)

| Metric | µs |
|--------|-----|
| `gh_infer_at_h` single | 10.89 |
| encode + `gh_infer_at_h` | 14.62 |
| `score_matrix` n=1 | 6.95 |
| `score_matrix` n=32 | 9.00 batch (**0.28**/row) |
| `predict_next` | 774.69 |
| `generate` 1 tok | 1858.56 |

### After (scratch reuse; 20k iters DIF / 2k LM; quieter of two runs)

| Metric | µs |
|--------|-----|
| `gh_infer_at_h` single | 16.06 |
| encode + `gh_infer_at_h` | 18.58 |
| `score_matrix` n=1 | **3.60** |
| `score_matrix` n=32 | **5.25** batch (**0.16**/row) |
| `predict_next` | 688.46 |
| `generate` 1 tok | 1785.07 |

**Batch vs single-row:** Already strong — n=32 is ~**22×** cheaper per row than n=1 after fix (0.16 vs 3.60 µs). Confirms batch path amortization; single-row REST `/predict` remains allocation-sensitive on RPSM score.

`gh_infer_at_h` did not improve (uses `classify_at_h`, not RPSM scratch). Absolute GH numbers worsened vs the early quiet baseline under load; treat as floor noise, not regression from this change (golden parity OK).

## Fix shipped

**Allocation churn on RPSM infer score path:**

1. `rpsm_score_matrix_batched` — `thread_local` `PsiMatrices` + ctx via `build_psi_from_model_into`.
2. `batched_llr_gemm` — `thread_local` `b_row` / `bias`.
3. `context_prior_for_labels` — skip per-call `unordered_map` for `mid_freq` (linear scan; K and mid_freq small on product fixtures).

**Not changed:** CyphaLM `predict_next` (already scratch-reused Parts 1–7); ngram_fusion / B3 files left alone.

**Win:** `score_matrix` n=1 **~6.95 → 3.60 µs** (~48%); n=32 per-row **0.28 → 0.16 µs** (~43%). Safe: same numerics, golden OK.

## Floor

- REST single-row GH classify: compute-bound + small-K overhead; further wins need classify_at_h scratch (deferred — A/B noisy).
- CyphaLM generate: dominated by `predict_next` (~0.7–1.2 ms/tok on this host); D17 train micro-opts exhausted; infer already clean relative to train.

## Reproduce

```text
cmake -S native -B native/build_perf_infer -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_perf_infer -j8 --target infer_latency_bench gh_infer_deliberation_golden
native/build_perf_infer/gh_infer_deliberation_golden fixtures/gh_infer_deliberation
native/build_perf_infer/infer_latency_bench
```
