#!/usr/bin/env bash
# Local mirror of the "Native build + CTest" step from .github/workflows/ci.yml (Linux host or WSL).
# Optional: CYPHA_BUILD_QT=1 and apt install qt6-base-dev → cypha_qt_stub + CTest native_qt_stub_load_reference.
# Optional: CYPHA_QT_CHARTS=1 with qt6-charts-dev (or distro Qt6 Charts) → -DCYPHA_QT_CHARTS=ON for cypha_qt_shell.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CYPHA_NATIVE_BUILD_DIR:-$ROOT/native/build-ci-local}"
J="${CI_NATIVE_J:-$(nproc 2>/dev/null || echo 4)}"
CMAKE_EXTRA=()
if [[ "${CYPHA_BUILD_QT:-0}" == "1" ]]; then
  CMAKE_EXTRA+=(-DCYPHA_BUILD_QT=ON)
fi
if [[ "${CYPHA_QT_CHARTS:-0}" == "1" ]]; then
  CMAKE_EXTRA+=(-DCYPHA_QT_CHARTS=ON)
fi
cmake -S "$ROOT/native" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" "${CMAKE_EXTRA[@]}"
cmake --build "$BUILD_DIR" -j"$J"
CTEST_ARGS=(--test-dir "$BUILD_DIR" --output-on-failure -R native_)
if [[ "${CYPHA_BUILD_QT:-0}" == "1" ]]; then
  # Headless runners cannot exec Qt Widgets smoke; compile-check Qt, skip GUI CTests.
  CTEST_ARGS+=(-E 'native_qt_shell_smoke|native_qt_stub_load_reference')
fi
ctest "${CTEST_ARGS[@]}"
echo "OK: $BUILD_DIR"
