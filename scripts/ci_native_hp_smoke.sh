#!/usr/bin/env bash
# hp / CyphaLM smoke + golden subset (macOS CI and local spot-check).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CYPHA_NATIVE_BUILD_DIR:-$ROOT/native/build}"

ctest --test-dir "$BUILD_DIR" --output-on-failure \
  -R 'native_hp_|native_cyphalm_(model_golden|checkpoint_golden|golden_suite)|native_som_golden' \
  -LE cypha_slow

echo "OK: hp/CyphaLM smoke gate ($BUILD_DIR)"
