# Profile analysis — WSL GPU production run (2026-03-21)

Source artifacts:

- `artifacts/bench/bench_gpu_production_20260321_103515.json`
- `artifacts/tuning/tuning_20260321_103337_{summary.json,results.csv}`
- Console: `gpu_microbench`, `gpu_fullbench`, CuPy burn + coarse tune with GPU stress

## 1. Where time went (wall clock)

| Stage | Wall (s) | Share of total |
|-------|----------|----------------|
| `gpu_microbench` | ~5.0 | 4% |
| `gpu_fullbench` | ~22.6 | 17% |
| `tune_gpu_heavy_coarse` | ~101.9 | 79% |

**Takeaway:** End-to-end “production” cost is dominated by **tuning** (28 trained configs × train + light infer + **heavy GPU stress**), not by the synthetic benches. For CI or quick regression checks, use `bench_gpu_production.py --skip-tune` or a smaller `--gpu-stress-repeats` / `--gpu-batch-n` when scanning hyperparameters.

## 2. Synthetic GPU benches — what they imply

### 2.1 Microbench (8192×256 @ 256×128 GEMM)

Observed: **GPU slower than CPU** (~32 ms vs ~11 ms per iter), zero numeric error.

**Likely causes**

- **Sync + transfer overhead**: CuPy path pays host/device sync every iteration; CPU uses a single well-optimized NumPy/OpenBLAS/MKL region.
- **Problem size / occupancy**: For this shape, the GPU may be **under-fed** relative to launch + memcpy cost.
- **WSL2 GPU**: Extra virtualization layer vs bare-metal Linux (variable impact).

**Improvements**

- Microbench: report **GPU-only kernel time** (sync once before/after the loop, not per matmul) to avoid misleading “speedup”.
- Try **larger N** (e.g. 32k–128k rows) or **batched async** streams before declaring GEMM winner.
- Optional **float32** path for throughput-sensitive workloads (with explicit accuracy gates).

### 2.2 Fullbench (N=4096, d=128, K=32)

- **LLR GPU vs CPU (mocked cuda off):** max |Δ| = **0** — good fp64 parity on the fused path.
- **Encode:** ~**19 ms** CPU vs ~**18 ms** GPU path — essentially **no win**; `batch_encode` is still effectively **CPU + Python per-row** for `VectorEncoder`.
- **Score + softmax + gate:** ~**832 ms** CPU vs ~**16 ms** GPU — **~52×** on that slice; **total** ~**24.8×** vs CPU.

**Takeaway:** The **score_matrix / fused GEMM** path is the right CUDA target; **encoding** is the next large opportunity if batch inference must scale.

**Improvements**

- **Fused encode → score on device**: `batch_infer` keeps `H`/`LLR` on GPU through softmax+gate when **K > 8** (`fused_features_to_device_latent_llr` + `fused_batch_infer_indices_confs_cupy`); only small per-row outputs hit the host.
- **`batch_infer_full`**: when **K > 8**, downloads **LLR** only (for `llrs` / `probs` dicts) plus precomputed **gates** from the device tail — **no full `H` D2H** on that path.
- **True `batch_encode` for VectorEncoder**: one GEMM `(N, d_in) @ W.T` instead of a Python loop (matches comments in `cypha_core` API map).

### 2.3 CuPy burn + tuner probe

- **First GPU GEMM warmup:** ~**420 ms** — typical driver/JIT/first-kernel cost; **production services should warm the GPU once** at startup.
- **64 burn passes** (8192×512×256): ~**0.28 s** total, reported **~498 GFLOP/s** — useful as a **health metric**; track regressions over driver/CuPy versions.

## 3. Tuning rows — quality vs cost

### 3.1 Best classification (coarse grid)

- `val_accuracy` ≈ **0.733**, `wall_total_s` ≈ **2.13 s** per config.
- **Breakdown:** `wall_train_s` ≈ **0.28 s**, `wall_infer_stress_s` ≈ **0.019 s**, **`wall_gpu_stress_s` ≈ 1.83 s** (~**86%** of wall time is **artificial stress**, not training).

### 3.2 Best regression (coarse grid)

- `val_r2` ≈ **0.207** (raw y, California housing), `wall_total_s` ≈ **4.70 s**.
- **`wall_gpu_stress_s` ≈ 4.50 s** — even more dominated by stress (chunked `predict_batch`).

### 3.3 Generation (4 combos)

- Best `gen_match_rate` = **1.0** with only **6×6** generated points — **high variance**; do not treat as stable without more draws or a minimum-sample rule.

**Quality improvements (data / protocol)**

- Run **`--preset medium`** (or `fine` with `--max-combos`) for stronger models; coarse is a sanity grid.
- Align production hyperparameters with **`bench/config/profiled_medium.json`** (already stronger than this coarse best on the earlier full-medium search).
- Generation: require **minimum `gen_n_calls * gen_n_per_call`** when selecting “best”, or report confidence intervals.

**Benchmark hygiene**

- For **fair hyperparameter comparison**, either **disable GPU stress** (`--no-gpu-stress`) or **fix stress config** across runs so `wall_total_s` reflects comparable work.
- Use **`--jobs 1`** on one GPU (already in `tune-gpu-heavy` / `bench_gpu_production`).

## 4. Code-level improvement backlog (prioritized)

| Priority | Item | Status (post-2026-03-21) |
|----------|------|---------------------------|
| P0 | **Reduce host/device copies** — fused **F → H → LLR** on GPU for `batch_infer` / `batch_infer_full` (`VectorEncoder` + CUDA) | **Done** — `fused_features_to_device_latent_llr` / `fused_features_to_latent_and_llr`; `batch_infer_full` avoids **H** D2H when **K > 8** (LLR + gates only). |
| P0 | **Vectorized `batch_encode`** for `VectorEncoder` | **Done** — `np.stack` of rows + locked `W` copy; optional **(N,d) ndarray** passthrough. |
| P1 | **GPU-resident softmax + gate** | **Done** for `batch_infer` when **K > 8** and GH defaults (`fused_batch_infer_indices_confs_cupy`). `batch_infer_full` still runs `_probs_from_llr_matrix` on the host for dict consistency. |
| P1 | **`warmup_cuda()`** | **Done** — `cypha_accel.cuda_util.warmup_cuda`; called at start of `bench_gpu_production.py`. |
| P2 | **Microbench timing** | **Done** — reports loop mean + single matmul+sync line. |
| P2 | **FP32 experimental path** | **Done** — env **`CYPHA_ACCEL_FP32=1`** in `cypha_accel.score_batch` fused CuPy matmuls (outputs promoted to float64 NumPy where applicable). |

## 5. Anomalies to watch

- **CSV row** with `wall_infer_stress_s` ≈ **0.89 s** vs peers ~**0.01–0.02 s** (classification, `field_dim=64`): investigate **cold path**, lock contention, or one-off GC; if reproducible, profile that cell with `cProfile`.
- **Regression stress** slower than classification stress at same batch/repeats: expected from **many `predict_batch` chunks** and repeated encode/score — optimizing `predict_batch` and GPU residency helps most.

---

*Generated from repo artifacts; re-run `bash scripts/wsl_bench_gpu.sh` after changes to validate.*
