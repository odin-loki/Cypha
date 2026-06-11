# Cypha accel — CUDA build (MSVC / Linux)

Optional GPU path for **`cypha::accel`** (`native/src/accel_backend.cpp` + `accel_cuda.cu`).
Without **`-DCYPHA_ENABLE_CUDA=ON`**, the same APIs use **ISO C++** `std::thread` row parallelism.

**Python reference:** `cypha_accel/score_batch.py` — `project_features`, `fused_score_llr` (CuPy when available; NumPy CPU fallback). Native parity: **`score_batch_parity`**, CTest **`native_score_batch`**.

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
.\build-windows-msvc\Release\score_batch_parity.exe ..\parity_fixtures\score_batch\sidecar.json
```

**VS 2026 / Build Tools 18:** use preset `windows-vs2026-release` and `build-windows-vs2026` instead.

**GitHub Actions (blocking CI jobs):**

| Job | Platform | Recipe |
|-----|----------|--------|
| **`windows_cuda_msvc`** | Windows | **`Jimver/cuda-toolkit`** (`sub-packages: ["nvcc","cudart","thrust"]`), **`ilammy/msvc-dev-cmd`**, **Ninja 1.12.1** (pinned), explicit **`CUDA_PATH`** from step outputs, optional copy of `extras/visual_studio_integration` into VS `BuildCustomizations`. |
| **`linux_cuda`** | Ubuntu | Same Jimver toolkit + **`CUDAToolkit_ROOT=$CUDA_PATH`**, GCC + Ninja. |

See [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml). **`native_cuda_bench`** (`--bench`) remains optional (GPU-only, `continue-on-error` when a device is present).

**Architecture flags:** set `-DCMAKE_CUDA_ARCHITECTURES` to your GPU SM version, e.g. `75` (Turing), `86` (Ampere), `89` (Ada). Default in `CMakeLists.txt` is **75** when unset.

**Optional CMake cache:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `CYPHA_ACCEL_GPU_MIN_BATCH_ROWS` | `16` | Minimum batch rows `n` before dispatching encode/score/softmax/gate to CUDA (smaller batches stay on CPU threads to avoid launch + H↔D overhead). |

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
| Fused LLR parity | `score_batch_parity parity_fixtures/score_batch/sidecar.json` — CTest **`native_score_batch`** |
| Pytest | `pytest tests/test_cuda_smoke_native.py tests/test_score_batch_native_parity.py -v` |

Override binary paths: `CYPHA_CUDA_SMOKE_BIN`, `CYPHA_SCORE_BATCH_PARITY_BIN`.

Regenerate fixtures after inference numerics change:

```bash
python scripts/generate_parity_fixtures.py
python scripts/generate_score_batch_fixture.py
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
