#!/usr/bin/env bash
# Local mirror of the optional "Federated TLS smoke (OpenSSL)" job from .github/workflows/ci.yml.
# Blocking CI gate elsewhere: 98+ CTests (`ctest -R native_`; see scripts/ci_native_linux.sh).
# Builds with -DCYPHA_ENABLE_OPENSSL=ON and runs ctest -R native_federated_tls.
# Exits 0 (skip) when OpenSSL dev libs or the openssl CLI are unavailable; exits non-zero on test failure.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CYPHA_FEDERATED_TLS_BUILD_DIR:-$ROOT/native/build-federated-tls-ci}"
J="${CI_NATIVE_J:-$(nproc 2>/dev/null || echo 4)}"

if ! command -v pkg-config >/dev/null 2>&1; then
  :
elif ! pkg-config --exists openssl 2>/dev/null; then
  echo "SKIP: OpenSSL dev libraries not found (install libssl-dev)"
  exit 0
fi

cmake -S "$ROOT/native" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
  -DCYPHA_ENABLE_OPENSSL=ON

if ! grep -q '^CYPHA_ENABLE_OPENSSL:BOOL=ON' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
  echo "SKIP: CYPHA_ENABLE_OPENSSL not enabled (OpenSSL not found by CMake)"
  exit 0
fi

cmake --build "$BUILD_DIR" -j"$J" --target federated_tls_smoke

if ! command -v openssl >/dev/null 2>&1; then
  echo "SKIP: openssl CLI not on PATH (install openssl package)"
  exit 0
fi

ctest --test-dir "$BUILD_DIR" --output-on-failure -R native_federated_tls
echo "OK: federated TLS smoke $BUILD_DIR"
