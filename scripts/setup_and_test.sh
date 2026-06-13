#!/usr/bin/env bash
# Unix/WSL: native cmake build + ctest -R native_.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
exec bash "$ROOT/scripts/ci_native_linux.sh"
