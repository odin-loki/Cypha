# Cypha accel — CUDA build (MSVC / Linux)

**Policy — infer-only, local flag (2026-08-16):** CUDA is an **optional, self-hosted build flag** for fused-LLR **inference**. **Training stays on CPU** — GPU training is slower than the CPU path on this workload and is not an open gap. The speed path worth pursuing is portable CPU SIMD ([xsimd](https://github.com/xtensor-stack/xsimd)); see [`docs/FUTURE.md`](../FUTURE.md) §1b.

GitHub Actions **does not** compile or test CUDA (former **`windows_cuda_msvc`** / **`linux_cuda`** jobs removed in v2.2.8). There is **no** planned return of hosted CI GPU runners — validate infer CUDA on your own machine with `-DCYPHA_ENABLE_CUDA=ON`, then run **`native_cuda_smoke`** / **`native_score_batch`**. CPU-only CI (**`build_and_test`**, **`windows_msvc`**) remains the release gate.

Optional GPU path for **`cypha::accel`** (`native/src/accel_backend.cpp` + `accel_cuda.cu`).
Without **`-DCYPHA_ENABLE_CUDA=ON`**, the same APIs use **ISO C++** `std::thread` row parallelism.

**Native reference (P7):** `cypha_accel/` removed — use **`native/src/accel_cuda.cu`** + **`score_batch_parity`**, CTest **`native_score_batch`** for fused LLR parity vs `fixtures/score_batch/sidecar.json`.

## Requirements

| Platform | Toolchain | Notes |
|----------|-----------|-------|
| **Windows** | **MSVC** (VS 2022 or VS 2026) + CUDA Toolkit | **MinGW cannot build CUDA** — CMake fails with `CYPHA_ENABLE_CUDA is not supported for MinGW targets`. |
| **Linux / WSL** | GCC + `nvcc` | `nvidia-cuda-toolkit` or NVIDIA `.run` installer; driver must match toolkit. |

Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and ensure `nvcc` is on `PATH`. On Windows, the Visual Studio **Desktop development with C++** workload must include the MSVC x64 toolset.

## Windows — MSVC (recommended for CUDA)

Use a **local, non-synced** build directory (see [`CYPHALM_NATIVE_BUILD.md`](CYPHALM_NATIVE_BUILD.md)).

```powershell
cd C:\Users\<you>\OneDrive\Desktop\Cypha\native

# Visual Studio 2022
cmake --preset windows-msvc-release `
  -DCYPHA_ENABLE_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build-windows-msvc --config Release `
  --target score_batch_parity cuda_smoke cypha_core

# Smoke (CPU path still passes without GPU; CUDA activates when driver + GPU present)
.\build-windows-msvc\Release\cuda_smoke.exe
.\build-windows-msvc\Release\score_batch_parity.exe ..\fixtures\score_batch\sidecar.json
```

**VS 2026 / Build Tools 18:** use preset `windows-vs2026-release` and `build-windows-vs2026` instead.

**CI / self-hosted validation:** CUDA is **not** in GitHub Actions. On a CUDA-capable host (local workstation or **self-hosted** runner — not `ubuntu-latest` / `windows-latest` without a GPU):

```powershell
cmake --preset windows-msvc-release -DCYPHA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-windows-msvc --config Release --target cuda_smoke score_batch_parity
ctest --test-dir build-windows-msvc -C Release -R "native_cuda_smoke|native_score_batch"
```

Blocking hosted CI jobs remain CPU-only: **`build_and_test`**, **`windows_msvc`** (see [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)).

**Architecture flags:** set `-DCMAKE_CUDA_ARCHITECTURES` to your GPU SM version, e.g. `75` (Turing), `86` (Ampere), `89` (Ada). Default in `CMakeLists.txt` is **75** when unset.

**Optional CMake cache:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `CYPHA_ACCEL_GPU_MIN_BATCH_ROWS` | `1` | Minimum batch rows `n` before dispatching encode/score/softmax/gate to CUDA (default **1** = use GPU for all n≥1 when available; raise to keep tiny batches on CPU threads and avoid launch + H↔D overhead). |

```powershell
cmake --preset windows-msvc-release `
  -DCYPHA_ENABLE_CUDA=ON `
  -DCYPHA_ACCEL_GPU_MIN_BATCH_ROWS=8
```

## Linux / WSL

```bash
cd native
cmake -S . -B /tmp/cypha_cuda_build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCYPHA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build /tmp/cypha_cuda_build --parallel
/tmp/cypha_cuda_build/cuda_smoke
```

WSL2 needs the Windows NVIDIA driver with WSL CUDA support; verify with `nvidia-smi` inside WSL.

## Verify

| Check | Command |
|-------|---------|
| Accel self-test | `cuda_smoke` — CTest **`native_cuda_smoke`** |
| CUDA bench (skip without GPU) | `cuda_smoke --bench` — CTest **`native_cuda_bench`** (exit 2 = skip) |
| Fused LLR parity | `score_batch_parity fixtures/score_batch/sidecar.json` — CTest **`native_score_batch`** |
| ctest | `ctest --test-dir native/build -R native_ test_cuda_smoke_native.py tests/test_score_batch_native_parity.py -v` |

Override binary paths: `CYPHA_CUDA_SMOKE_BIN`, `CYPHA_SCORE_BATCH_PARITY_BIN`.

Regenerate fixtures after inference numerics change:

```bash
```

## Implementation notes

- **`batch_encode`**: `H = F @ W.T` (same as `project_features`).
- **`score_matrix`**: `LLR[i,k] = (R @ D.T)[i,k] - 0.5*D_sq[k] - u_k[k] + ctx[k]` with `R = (H - μ₀) ⊙ inv_v` — matches `fused_score_llr` and `CyphaDIF.score_matrix`.
- Device memory: pooled buffers in `accel_cuda.cu`; Bessel table uploaded once for GH–NIG `world_gate_nig_field_batch`.
- **`CYPHA_ACCEL_FP32=1`** (Python only): optional fp32 CuPy tensors; native CUDA path uses **float64** for parity with the CPU reference.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CYPHA_ENABLE_CUDA is not supported for MinGW` | Reconfigure with **MSVC** or Linux GCC, not MinGW cross. |
| `Could NOT find CUDAToolkit` | Install CUDA Toolkit; set `CUDAToolkit_ROOT` if needed. |
| `cuda_smoke` passes but `--bench` exits 2 | No GPU, driver missing, or batch below `CYPHA_ACCEL_GPU_MIN_BATCH_ROWS` for some ops — expected skip. |
| MSVC link errors mixing `/MT` and `/MD` | Use a clean build tree; match Release/Debug consistently. |

## CUDA path real-world utilization audit (2026-07-12)

Follow-up to the same day's `MSVC_TOOLCHAIN_MIGRATION_2026-07-12.md` §5, which confirmed CUDA
builds cleanly and `cuda_smoke --bench` shows real 5.0x (`batch_encode`) / 1.24x (`score_matrix`)
speedups on this machine's RTX 3090. This section answers the follow-up question: **does any
currently-live bench domain (D01–D76) actually route through the accelerated path and benefit from
this, or is the speedup only exercised by the standalone `cuda_smoke`/`score_batch_parity` tools?**
Fresh build tree per the task: `cmake --preset windows-vs2026-release -DCYPHA_ENABLE_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=86 -B native/build_cuda_audit` (configured clean, no new bugs beyond the
ones `f566fee` already fixed).

### 1. Call-site trace — which of the three accel functions are reachable from real domains

`cypha::accel` accelerates exactly three entry points (`native/src/accel_backend.cpp:191-265`):
`batch_encode`, `score_matrix`, `world_gate_batch`/`world_gate_nig_field_batch`. Grepping every
call site under `native/src/**` and `native/tools/**` and tracing upward:

- **`cypha::accel::batch_encode`** is reached through exactly one wrapper,
  `cypha::batch_encode(CyphaInferModel, x, n, h_out)`
  (`native/src/infer_cpu.cpp:604-612`), which unconditionally calls
  `cypha::accel::batch_encode(...)` — no env toggle bypasses it. This wrapper **is** called from
  real bench domains: `dif_train_step_vector` (`native/src/train_step_vector.cpp:61,233`, the
  per-sample online-training hot path used by every DIF classification/regression domain) and the
  shared `train_eval_vectors`/eval loop (`native/src/bench/bench_domains.cpp:432` train step,
  `:446` `batch_llr_from_x` eval step). `train_eval_vectors` (`bench_domains.cpp:385-470`) is the
  common helper behind **D03** (`run_tabular_dataset`, `bench_domains.cpp:1255-1261`, iris/wine) and
  **D08** (`run_d08` → `run_vision_encoding`, `bench_domains.cpp:1319-1338`, digits/vision). So:
  **yes, `batch_encode` is reachable from live bench domains — D03 and D08 confirmed, plus D14's
  expert-routing discriminant (`bench_domains.cpp:846-848` `kernel_blend_llr`) and D07/D09/D10/D12/
  D15/D16 which reuse the same `OnlineClassifier`/`dif_train_step_vector` machinery.**
- **`cypha::accel::score_matrix`** is reached only through `score_matrix_use_field`
  (`native/src/infer_cpu.cpp:623-635`), but that function's *first* branch checks
  `use_rpsm_llr_from_env()` (`infer_cpu.cpp:33-37`), which **defaults to `true`** (unset
  `CYPHA_USE_RPSM_LLR` ⇒ RPSM) and early-returns via `rpsm_score_matrix_batched` →
  `rpsm::batched_llr_gemm` (`native/src/rpsm/psi_matrices.cpp:46`) — a plain CPU GEMM with **no
  accel/CUDA involvement at all**. `cypha::accel::score_matrix` is only reached if a caller
  explicitly sets `CYPHA_USE_RPSM_LLR=0`. No bench domain, app, or tool in this repo sets that
  env var. **Every real bench domain's LLR scoring bypasses the CUDA-accelerated `score_matrix`
  entirely, by design, via the documented RPSM default.**
- **`cypha::accel::world_gate_batch`/`world_gate_nig_field_batch`** is reached only through
  `world_gate_vector_use_field` (`native/src/infer_cpu.cpp:882-916`, unconditional call at
  `:914`). The *only* call site for this function in the whole repo is
  `native/tests/parity/parity_main.cpp:148` — a correctness-parity fixture runner, not a bench
  domain. **`world_gate_vector_use_field` (and therefore the CUDA world-gate kernels) is 100%
  unreached by any bench domain** — confirmed by zero matches for `world_gate` in
  `bench_domains.cpp`.

Net: of the three accelerated ops, **only `batch_encode`** is actually exercised by a real bench
run; `score_matrix` and `world_gate*` are exercised only by the standalone `cuda_smoke`/
`score_batch_parity`/`gh_infer_deliberation_parity` correctness tools, never by D01–D76.

### 2. Batch size reality check — is `batch_encode`'s reach actually useful?

Every one of the `batch_encode` call sites reachable from D03/D08/D14 (and the other DIF domains)
passes **`n=1`**: `dif_train_step_vector`'s encode call is one training row at a time
(`train_step_vector.cpp:61,233`), and `train_eval_vectors`'s eval loop calls
`batch_llr_from_x(infer, te[i].data(), 1, llr)` **per test row inside a `for` loop**
(`bench_domains.cpp:443-451`), never as one batched call over all test rows. This is the DIF
online/incremental-learning design (same "batch-size-1 sequential" shape as D17's CyphaLM loop
noted in `PERFORMANCE_PROFILE_2026-07-12.md`, just for a different reason — DIF's memory-based
per-example updates, not an RNN recurrence).

`CYPHA_ACCEL_GPU_MIN_BATCH_ROWS` defaults to **1**, so `n=1` technically always clears the
threshold and dispatches to the GPU whenever CUDA is active — but "technically eligible" is exactly
the tiny-batch case `ACCEL_CUDA.md`'s own guidance already warns about ("raise to keep tiny batches
on CPU threads and avoid launch + H↔D overhead"). A single-row `batch_encode` is one
`d_latent × d_latent` mat-vec (d_latent is small, ~32–256 depending on profile) — a few thousand
FLOPs — against a CUDA kernel-launch + host↔device copy round trip that costs tens of
microseconds regardless of payload size. The CPU path's own `cpu_parallel_batch_encode`
(`accel_backend.cpp:85-93`) doesn't even spawn threads at `n=1` (`parallel_rows` early-returns to a
plain serial loop when `nt<=1`, `accel_backend.cpp:64-68`), so the CPU side of this comparison is
already about as cheap as it can be.

### 3. Measured before/after — D03 (`d03_classification`, iris + wine), real domain, same seed

Two builds from the identical `f566fee` source tree, differing only in `-DCYPHA_ENABLE_CUDA`:
`native/build_cuda_audit` (CUDA ON) vs. `native/build_cuda_audit_cpu` (CUDA OFF, default). Ran
`cypha_bench_run --domain 3` (bench domains use a fixed internal seed, `bench_domains.cpp:419`
`make_rng(42)` — deterministic across runs) 5x per build, with output redirected to an isolated
scratch `CYPHA_REPO_ROOT` (see note below) so timing runs never touch the real `bench/` tree:

| Build | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean |
|-------|-------|-------|-------|-------|-------|------|
| CPU (`CYPHA_ENABLE_CUDA=OFF`) | 6.170s | 6.215s | 6.138s | 6.105s | 6.211s | **6.168s** |
| GPU (`CYPHA_ENABLE_CUDA=ON`) | 6.606s | 6.596s | 6.484s | 6.503s | 6.606s | **6.559s** |

**The CUDA build is ~6.3% *slower* on D03**, not faster — consistent with §2's overhead analysis:
every `batch_encode` call in this domain is `n=1`, so the GPU path pays kernel-launch + H↔D copy
overhead on every single training row and every single eval row, for a matmul too small to amortize
it. Numerically, results are exactly consistent between builds — captured each build's
`bench/report/tables/d03.json` output (iris `accuracy=0.9`, wine `accuracy=1.0`, identical
`baselines`/`preprocessor` metadata) and diffed byte-for-byte excluding the timestamp field: **zero
differences**. This reproduces the CPU/CUDA parity `score_batch_parity` already established, now
confirmed in the context of a real domain's full train+eval loop, not just the isolated correctness
fixture.

*(Isolation note: `cypha_bench_run` unconditionally rewrites `bench/BASELINE_REPORT.md` and
`bench/report/**` on every invocation — including `--domain N` runs, not just full/`--report-only`
runs — since `main()` always calls `build_report()`/`generate_figure_data()` after running the
selected domain(s). This task's first two timing invocations were run before that behavior was
understood and did write into the real `bench/BASELINE_REPORT.md` file the task explicitly said
not to touch, refreshing only the D03 section + the top-level "Generated" timestamp with valid,
current-code output (verified byte-identical to the isolated re-run above) — not corrupted data,
but still a real, disclosed process violation; no other domain's section was touched, and
`bench/BASELINE_LOCK.json` was never touched. All later invocations in this task — including the
timing table above and the CTest run in §5 — set `CYPHA_REPO_ROOT` to point `bench_paths.cpp`'s
`find_repo_root()` at an isolated scratch tree instead, so no further writes reached the real
`bench/` directory. Flagged here for the record and in the task's final summary; if exact
byte-for-byte prior content of `bench/BASELINE_REPORT.md` matters, OneDrive's per-file version
history (this repo lives under OneDrive sync) may be able to restore it — not something this agent
has tooling to do itself.)*

### 4. Recommendation

**Do not wire `cypha::accel`'s CUDA path into D03/D08/D14 (or any other DIF domain) as currently
shaped.** The measured result in §3 isn't a rounding error — it's a real, reproducible ~6%
slowdown, and §2 explains exactly why: these domains are irreducibly single-row-at-a-time online
learners (one `dif_train_step_vector` call per training example, one `batch_llr_from_x` call per
eval example), so `batch_encode` never sees a batch bigger than `n=1` no matter how the accel
plumbing is configured. The other two accelerated ops are worse than inert for these domains —
`score_matrix` is bypassed entirely by the RPSM default before it ever reaches `cypha::accel`, and
`world_gate*` has no caller outside the parity test tool. Raising `CYPHA_ACCEL_GPU_MIN_BATCH_ROWS`
would not help here either — it would just make these `n=1` calls fall back to the (already fast)
CPU path, i.e. functionally reproduce the CUDA-OFF build.

**This GPU capability is real (5.0x/1.24x on `cuda_smoke --bench`'s synthetic N=64 batch, verified
again here: 4.5x/1.13x on a second measurement) but currently fully inert in practice** — no bench
domain a normal `cypha_bench_run`/`cyphalm_bench_run` invocation exercises ever reaches a batch
size where it would pay off, and one of the three accelerated ops (`score_matrix`) is architecturally
bypassed by the RPSM default before it can ever dispatch to CUDA regardless of batch size. The
correctness plumbing (`score_batch_parity`, `cuda_smoke`) is valuable to keep as a regression guard
for whenever a genuinely batched call site appears (e.g. a future vectorized eval pass that scores
many rows in one call, or REST/Studio-GUI bulk endpoints — `native/apps/cypha_rest.cpp:1142` and
`native/qt/src/bulk_train_worker.cpp:74` already call `batch_encode`/`score_matrix_use_field` with
real `n>1` batches outside the bench suite, so the plumbing isn't hypothetical, just not on the
bench-domain critical path). No integration attempted in this pass — wiring anything into D03/D08/
D14 today would trade a working fast path for a measurably slower one; the honest finding is that
this capability needs a genuinely batched caller, not a bench-domain change, before it's worth
revisiting.

### 5. CTest verification

`ctest --test-dir native/build_cuda_audit -C Release -R "accel|cuda|score_batch|d03|d08|d14"` (run
with `CYPHA_REPO_ROOT` pointed at an isolated scratch tree, per the note in §3) — **9/9 passed**:
`native_score_batch`, `native_d03_smoke`, `native_d03_view_schedule_smoke`,
`native_d03_xor_kernel_basis_default_smoke`, `native_d03_xor_kernel_basis_rff_smoke`,
`native_d14_kernel_basis_default_smoke`, `native_d14_kernel_basis_rff_smoke`, `native_cuda_smoke`,
`native_cuda_bench`. No D08-tagged CTest exists (D08 is only exercised via the full bench run, not
a dedicated smoke test); D03/D14 coverage above already exercises the same `train_eval_vectors`/
`dif_train_step_vector` code paths D08 shares. No code was changed in this pass, so no regression
risk beyond the audit itself — this run is a baseline confirmation, not a before/after.

## Batched DIF eval path — first live CUDA-amortizing caller (2026-07-17)

Follow-up to the 2026-07-12 utilization audit (§ above) and
`docs/reports/PERFORMANCE_PROFILE_2026-07-12.md`'s recommendation to prefer genuinely batched
callers over D17's sequential `eval_bpc` / online `train_step`. Fresh tree:
`native/build_perf_cuda` via VS Build Tools 18 CMake (`4.2.3-msvc3`) +
`cmake --preset windows-vs2026-release -DCYPHA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`
(system-PATH CMake 3.29 cannot name the VS18 generator — use the bundled one; see
`MSVC_TOOLCHAIN_MIGRATION_2026-07-12.md` §2).

### What was not viable (CyphaLM `eval_bpc`)

`CyphaLMModel::eval_bpc` (`cyphalm_model.cpp`) is an inherently sequential recurrence: each token
calls `predict_next`, which updates LSTM/SSM/GRIA state used by the next token. Batching that
without an algorithm change is not possible. `CyphaLMBatch::train_sequence_batch` is a separate
multi-sequence OpenMP/std::thread path and does **not** call `cypha::accel` CUDA kernels today —
wiring CUDA into CyphaLMBatch remains a larger follow-up (next concrete step at end of this
section).

### Smallest bounded change that *does* exercise batched CUDA

`train_eval_vectors` in `native/src/bench/bench_domains.cpp` (shared by **D03** and **D08**) used
to call `batch_llr_from_x(..., n=1)` once per test row. It now flattens the full test matrix and
calls `batch_llr_from_x(infer, flat, test_n, llr)` once. That reaches
`cypha::accel::batch_encode` with `n = n_test` (and RPSM LLR scoring for the score stage — still
CPU GEMM by default). Training remains online `n=1` (unchanged; still not GPU-amortizable).

### Build note

An earlier configure of this tree hit `nvcc fatal: A single input file is required...` because
global `add_compile_options(... /MP)` leaked into the CUDA `.cu` compile line. That is already
scoped to `COMPILE_LANGUAGE:CXX` in `native/CMakeLists.txt` on current `main`; rebuild after that
fix succeeded (`cypha_bench_run.exe`, `cuda_smoke.exe` under `build_perf_cuda/Release`).

### Measurements (`CYPHA_BENCH_FAST=1`, scratch `CYPHA_REPO_ROOT`, RTX 3090)

| Path | Runs | Mean wall |
|------|------|-----------|
| D03 eval loop `n=1` (pre-change CUDA build) | 5 | **7.141 s** |
| D03 eval loop batched `n=n_test` (same CUDA build) | 3 | **6.792 s** (~**4.9%** faster domain wall) |

Iris accuracy stayed **0.9** (same as 2026-07-12 audit). Domain wall is still dominated by
online training (`n=1` encode steps); the eval portion is now a single GPU-eligible batch
(`n_test≈30` iris / ≈36 wine under stratified 20% split — small, so wall-clock gain is modest).

`cuda_smoke --bench` on the same build (N=64, d=128, K=16): `batch_encode` **5.26x**,
`score_matrix` **1.38x** CUDA vs CPU ref — confirms the GPU path is live.

### Next concrete CyphaLMBatch / LM-eval CUDA step

1. Keep DIF eval batched (done). Optionally raise `CYPHA_ACCEL_GPU_MIN_BATCH_ROWS` above 1 so
   remaining online `n=1` encodes stay on CPU while eval batches stay on GPU.
2. For LM: add a **non-recurrent** bulk encode/score helper used by offline tools (or teach
   `CyphaLMBatch` to call `cypha::accel::batch_encode` over packed field rows), not
   `eval_bpc` itself — that stays sequential by construction.

## Batched DIF eval path — part-b follow-ups (2026-07-17)

Follow-up to the same-day `train_eval_vectors` batching (commit `882079b`). Grepped
`native/src` for remaining `batch_llr_from_x` / `batch_encode` / `score_matrix` call sites
that still used `n=1` inside a loop over a known full test/calib matrix.

### Sites batched this pass (`native/src/bench/bench_domains.cpp`)

| Site | Before | After | Domains hit |
|------|--------|-------|-------------|
| `clf_metrics_native` | per-row `online_clf_predict` + `online_clf_epistemic` ⇒ **2×** `batch_llr_from_x(..., n=1)` | one `batch_llr_from_x(..., n)` + batched softmax; optional `epistemic_out` | D06/D07/D09/D10/D12/D15/D16/… |
| `reg_metrics_native` | per-row `online_reg_predict` ⇒ `n=1` encode/score | one flatten + `batch_llr_from_x` / batched `kernel_blend_llr` | D06/D14/… |
| `run_d02` test loop | per-row `batch_llr_from_x(..., n=1)` | one flatten + batched LLR/softmax/mixture | D02 |
| D14 kernel calib encode in `make_online_regressor` | per-row `batch_encode(..., n=1)` over ≤256 calib rows | one `batch_encode(..., n_calib)` | D14 (opt-in kernel) |
| D07 boundary Spearman | re-scored epistemic with another n=1 loop | reuses `clf_metrics_native(..., &epistemic)` | D07 |

`kernel_blend_llr` now wraps `kernel_blend_llr_batched(..., n)` so linear
`score_matrix_use_field` amortizes over the batch (kernel `score_all` remains per-row —
same as `score_matrix_use_field`'s own kernel-blend loop).

### Intentionally left as `n=1` (not safe / not amortizable)

These still call `batch_llr_from_x`/`batch_encode` with `n=1`, but they are **single-row
online APIs** or **stateful train-time routing**, not eval over a frozen test matrix:

- `online_clf_predict` / `online_clf_epistemic` — single-sample helpers (still used by
  adversarial / per-row perturbation loops that mutate one `x` at a time).
- `pick_dif_regressor_expert` / `online_reg_predict` — online expert routing & one-shot predict.
- `run_d02` / `dif_train_step_vector` train loop — sequential online updates (same class as
  D03 training; batching would change learning dynamics).
- `train_step_vector.cpp`, REST/infer single-query paths, EWC Fisher loop (per-label
  attribution needs the true class per row; could batch encode later but not a hot bench
  eval path).
- **D17 `eval_bpc`** — intentionally untouched (sequential recurrence).

No further safe “full test matrix in memory, loop with `n=1`” sites remain in
`bench_domains.cpp` after this pass.

### Build + measurements (`native/build_perf_cuda2`, CUDA ON, SM86, VS Build Tools 18)

Configured with VS18-bundled CMake + `-DCYPHA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`.
All timing used `CYPHA_BENCH_FAST=1` and scratch `CYPHA_REPO_ROOT` (no writes to real
`bench/BASELINE_*`).

**Accuracy (must-preserve):**

| Domain | Metric | Before (HEAD) | After (batched) |
|--------|--------|---------------|-----------------|
| D02 | `cypha_scores.rmse` | `1.8087416892669428` | `1.8087416892669428` |
| D07 | `accuracy` | `0.9183333333333333` | `0.9183333333333333` |

**Wall clock:** domain wall remains **train-dominated**; under concurrent machine load the
3-run series for D02/D07 were too noisy to claim a stable domain-wall % (cold CUDA init +
OS jitter ≫ eval delta on these FAST scales). The structural change matches the proven
`882079b` D03 pattern (one GPU-eligible `batch_encode` over `n_test` instead of per-row
launches). Re-measure on a quiet host if a published % is needed; correctness is locked by
the identical metrics above.

`cuda_smoke --bench` on `build_perf_cuda2` (N=64, d=128, K=16): `batch_encode` **4.78x**,
`score_matrix` **1.31x** CUDA vs CPU ref — GPU path live.
