#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv-wsl/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv-wsl/bin/activate
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if ! python3 -c "import cupy" 2>/dev/null; then
  echo "Installing cupy-cuda12x into venv (first time)..."
  pip install -q "cupy-cuda12x>=13.0"
fi
python3 -c "import cupy; from cypha_accel.cuda_util import cuda_gemm_usable; print('cupy', cupy.__version__, 'cuda_gemm_usable', cuda_gemm_usable())"
exec python3 scripts/bench_gpu_production.py "$@"
