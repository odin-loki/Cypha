#!/usr/bin/env bash
# Large-n CyphaLM observe BPC + optional bit-tree top-k (measured output only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CYPHA_NATIVE_BUILD_DIR:-$ROOT/native/build}"
EXE="$BUILD_DIR/cyphalm_llm_eval"

if [[ ! -x "$EXE" ]]; then
  echo "Build cyphalm_llm_eval first: cmake --build $BUILD_DIR --target cyphalm_llm_eval" >&2
  exit 1
fi

OBSERVE_N="${1:-100000}"
TOPK_N="${2:-8}"
exec "$EXE" --observe-n "$OBSERVE_N" --topk-n "$TOPK_N" --latency-iters 5 "$@"
