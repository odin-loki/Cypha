#!/usr/bin/env bash
# WSL/native verification: cmake build + ctest -R native_.
# Usage (from repo root, in WSL): bash scripts/wsl_verify.sh
# Optional: CYPHA_BUILD_QT=1 — pass -DCYPHA_BUILD_QT=ON (needs qt6-base-dev).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
exec bash "$ROOT/scripts/ci_native_linux.sh"
