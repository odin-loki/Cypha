#!/usr/bin/env bash
# Build (if needed) and run CyphaLM hp profile harness with /usr/bin/time -v.
# Usage: bash scripts/cyphalm_llm_profile.sh [--skip-wiki] [--skip-roundtrip]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CYPHA_NATIVE_BUILD_DIR:-$ROOT/native/build}"
BIN="$BUILD_DIR/cyphalm_llm_profile"
OUT_DIR="${CYPHA_PROFILE_OUT_DIR:-$ROOT/bench/results/cyphalm_llm_profile}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "Building cyphalm_llm_profile in $BUILD_DIR ..."
  cmake -S "$ROOT/native" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-g++}" \
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER:-gcc}" \
    -G "Unix Makefiles"
  cmake --build "$BUILD_DIR" --target cyphalm_llm_profile -j"${CYPHA_BUILD_J:-$(nproc 2>/dev/null || echo 4)}"
fi

RAW="$OUT_DIR/profile_${STAMP}.txt"
TIME_LOG="$OUT_DIR/time_v_${STAMP}.txt"
META="$OUT_DIR/meta_${STAMP}.json"

{
  echo "# cyphalm_llm_profile run $STAMP"
  echo "commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "build_dir=$BUILD_DIR"
  echo "hp_profile=gate24"
  echo "uname=$(uname -a)"
  echo "cpu=$(lscpu | grep 'Model name' | sed 's/^[[:space:]]*//' || true)"
  echo "mem=$(free -h | head -2 | tail -1)"
} | tee "$META"

echo "Running harness (stdout -> $RAW) ..."
/usr/bin/time -v "$BIN" "$@" 2>"$TIME_LOG" | tee "$RAW"

echo ""
echo "time -v log: $TIME_LOG"
echo "harness log: $RAW"
echo "meta: $META"
