# Parallel score_matrix / batch_llr (2026-07-17)

**Scope:** Row-parallel RPSM `batched_llr_gemm` (default `score_matrix_use_field` / `batch_llr_from_x`) when batch work is large enough.

**OFF-LIMITS:** `build_math`, `build_deff`, `BASELINE_*`, overnight.

**Build:** `native/build_perf_omp` (Ninja, Release, MinGW 13.2.0, OpenMP 4.5 on `cypha_core`).

**Opt-out:** `CYPHA_SCORE_PARALLEL_ROWS=0` forces serial (parity / A/B).

## Change

1. `native/include/cypha/parallel_rows.hpp` — `parallel_for_score_rows` (OpenMP when `_OPENMP`, else `std::thread`).
2. Gates: `n >= 16` **and** `n*d*K >= 1e6` (avoids OpenMP fork overhead on tiny product fixtures).
3. `rpsm::batched_llr_gemm` + GMM branch in `score_matrix_use_field` use the helper.
4. CTest `native_score_matrix_parallel_parity` — serial vs parallel on synth d=256 K=32 at n=128/256, tol `1e-12`.

## Parity

`score_matrix_parallel_parity` → **OK** (synth d=256 K=32, n=128/256, tol=1e-12).

## Bench (`infer_latency_bench`, `OMP_NUM_THREADS=4`)

### Product fixture (gh_infer_deliberation d=8 K=3) — stays serial under work gate

| Metric | µs |
|--------|-----|
| `score_matrix` n=1 | 10.12 |
| `score_matrix` n=32 | 11.44 batch (0.36/row) |
| `score_matrix` n=256 | 16.85 batch (0.07/row) |

No parallel path engaged (`n*d*K` ≪ 1e6). Numbers are floor noise vs prior scratch-reuse profile; no intentional regression.

### Synth RPSM GEMM (d=256 K=32) — OpenMP engaged at n=256

| n | Serial µs | Parallel µs | Speedup |
|---|-----------|-------------|---------|
| 32 | work-gated serial | same | 1.0× (gate) |
| 256 | **2231** | **649** | **3.44×** |

Per-row at n=256: 8.72 → 2.53 µs.

## Reproduce

```text
cmake -S native -B native/build_perf_omp -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_perf_omp -j8 --target infer_latency_bench score_matrix_parallel_parity
$env:OMP_NUM_THREADS="4"
native/build_perf_omp/score_matrix_parallel_parity
native/build_perf_omp/infer_latency_bench
```
