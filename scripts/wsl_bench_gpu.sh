#!/usr/bin/env bash
# WSL/Linux native GPU smoke + CyphaLM bench (Python CuPy path removed in P7).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CYPHA_NATIVE_BUILD_DIR:-$ROOT/native/build-wsl-gpu}"
mkdir -p "$BUILD_DIR"
CMAKE_ARGS=(-S "$ROOT/native" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}")
if [[ "${CYPHA_ENABLE_CUDA:-1}" == "1" ]]; then
  CMAKE_ARGS+=(-DCYPHA_ENABLE_CUDA=ON)
fi
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)"
if [[ -x "$BUILD_DIR/cuda_smoke" ]]; then
  "$BUILD_DIR/cuda_smoke" --bench
fi
exec "$BUILD_DIR/cyphalm_bench_native" --mode hybrid --profile d17 \
  --n-train "${N_TRAIN:-5000}" --n-eval "${N_EVAL:-500}" --threads "${THREADS:-1}" "$@"