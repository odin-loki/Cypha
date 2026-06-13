# GPU production benchmarking

## Requirements

- **Python 3.10â€“3.12** with a CUDA-matched CuPy wheel (official wheels do not support 3.14+ yet).
- NVIDIA driver + CUDA runtime compatible with your `cupy-cuda11x` / `cupy-cuda12x` install.

```bash
native build (`cmake --build`) cupy-cuda12x   # or cupy-cuda11x â€” see https://docs.cupy.dev/en/stable/install.html
```

Verify in the **same** interpreter you use for Cypha:

```bash
native CUDA smoke (`cuda_smoke`)
```

## One-shot bundle

Runs raw GEMM microbench, DIF score-path fullbench, then GPU-heavy coarse tuning (burn + per-cell stress, `--jobs 1`):

```bash
make bench-gpu-prod
# or
```

**WSL** (native CUDA build + `cyphalm_bench_native`; optional `CYPHA_ENABLE_CUDA=0` for CPU-only):

```bash
bash scripts/wsl_bench_gpu.sh
```

Optional:

- `--skip-tune` â€” only micro + fullbench (quick sanity).
- `--medium-extra --medium-max-combos 300` â€” append a **medium** subsample with lighter burn/repeat (still single-GPU safe).

A JSON timing report is written under `artifacts/bench/` (large artifacts; keep out of small bundles if you zip the tree). Tuning CSV/JSON go to `artifacts/tuning/`.

## Piecemeal

| Target | Command |
|--------|---------|
| GEMM only | native parity fixture workflow (see MAINTENANCE.md) |
| Encode + score + gate + **batch_infer** | native parity fixture workflow (see MAINTENANCE.md) |

Optional: set **`CYPHA_ACCEL_FP32=1`** to run fused CuPy matmuls in float32 (still returns float64 `H`/`LLR` to NumPy); useful for throughput experiments on large `N`.
| Tune (heavy GPU) | `make tune-gpu-heavy` |

Always prefer **`--jobs 1`** when GPU stress is enabled so workers do not fight one GPU.

## Reference numbers

Profiled hyperparameters from offline tuning live in `bench/config/profiled_medium.json`; they are independent of GPU benchmarking but useful when repeating quality runs on CUDA hardware.

## After a run: turn numbers into work

See **[`PROFILE_IMPROVEMENTS_20260321_WSL_GPU.md`](PROFILE_IMPROVEMENTS_20260321_WSL_GPU.md)** for a structured read of wall time, micro/fullbench interpretation, tuning CSV/summary, and the improvement backlog (several P0/P1 items are implemented in-tree: fused Fâ†’Hâ†’LLR in `batch_infer`, `warmup_cuda()`, faster `VectorEncoder` `batch_encode`).

### API: hide first-call GPU latency

```python
from cypha_accel import warmup_cuda
warmup_cuda()  # before serving or benchmarking
```

