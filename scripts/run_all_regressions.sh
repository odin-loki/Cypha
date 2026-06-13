#!/usr/bin/env bash
# Native regression checks: cmake build + ctest -R native_.
# Usage: bash scripts/run_all_regressions.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
exec bash "$ROOT/scripts/ci_native_linux.sh"
