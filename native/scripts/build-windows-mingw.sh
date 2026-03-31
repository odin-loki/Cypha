#!/usr/bin/env bash
# Cross-compile native targets for Windows x86_64 using MinGW-w64 (run from WSL or Linux).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
TOOL="$ROOT/toolchains/mingw-w64-x86_64.cmake"
BUILD="${BUILD_DIR:-$ROOT/build-mingw-w64}"
type x86_64-w64-mingw32-g++ >/dev/null 2>&1 || {
  echo "Install: sudo apt-get install -y g++-mingw-w64-x86-64" >&2
  exit 1
}
cmake -S "$ROOT" -B "$BUILD" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOL" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build "$BUILD" -j"$(nproc 2>/dev/null || echo 4)"
echo "Built under: $BUILD"
echo "Run CTest:   cmake --test-dir \"$BUILD\" --output-on-failure"
