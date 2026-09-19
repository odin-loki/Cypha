#!/usr/bin/env bash
# PR CI gate (Linux + macOS): hp / CyphaLM smokes + goldens only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CYPHA_NATIVE_BUILD_DIR:-$ROOT/native/build}"

ctest --test-dir "$BUILD_DIR" --output-on-failure \
  -R 'native_hp_|native_cyphalm_(model_golden|checkpoint_golden|golden_suite)|native_som_golden'

echo "OK: hp/CyphaLM PR smoke gate ($BUILD_DIR)"
