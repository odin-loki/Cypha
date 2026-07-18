# Full GPU training gap — 2026-07-18

**Plan:** Phase D of [`BACKLOG_EXECUTION_PLAN_2026-07-18.md`](BACKLOG_EXECUTION_PLAN_2026-07-18.md)

## Current truth

| Path | CUDA? | Notes |
|------|-------|-------|
| `cypha::accel::batch_encode` / `score_matrix` | Yes (local) | Infer / bulk encode |
| Online DIF `train_step` (n=1) | No | GPU overhead hurts (~6% slower on D03) |
| `CyphaLMModel::train_step` / `eval_bpc` | No | Sequential CPU |
| Hosted GPU CI | No | Policy: local-only (`docs/native/ACCEL_CUDA.md`) |

## This wave

- Documented gap (this file + BoW §7).
- **Non-goal:** full online train CUDA.
- **Next PR-sized chunk:** wire `batch_encode` into one offline bulk path with `n>1` (tool or REST bulk), then optional local `-DCYPHA_ENABLE_CUDA=ON` smoke.

## Build

```powershell
cmake -S native -B native/build_perf_cuda -DCYPHA_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build native/build_perf_cuda -j8 --target cuda_smoke
```
