#!/usr/bin/env bash
# Default PR CI CTest gate: all native_ tests except cypha_slow label and headless Qt GUI exec.
# Local: CYPHA_NATIVE_BUILD_DIR=... bash scripts/ci_native_fast.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CYPHA_NATIVE_BUILD_DIR:-$ROOT/native/build}"
CTEST_J="${CTEST_J:-2}"

ctest --test-dir "$BUILD_DIR" --output-on-failure -j"$CTEST_J" \
  -R native_ \
  -LE cypha_slow \
  -E 'native_qt_shell_smoke|native_qt_stub_load_reference'

echo "OK: native fast CTest gate ($BUILD_DIR)"
