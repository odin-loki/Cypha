#!/usr/bin/env bash
# Deprecated wrapper — Windows releases are packaged with native MSVC via PowerShell.
#
# Prefer:
#   pwsh -File scripts/package_release_windows.ps1 -Version <VER> -BuildDir native/build-msvc-release
#
# This bash entrypoint remains for local Git Bash convenience when packaging an MSVC
# build tree (looks under BUILD_DIR and BUILD_DIR/Release).

set -euo pipefail

VERSION="${1:?version required, e.g. 2.3.24}"
BUILD_DIR="${2:?build dir required, e.g. native/build-msvc-release}"
OUT_DIR="${3:-dist}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v pwsh >/dev/null 2>&1; then
  exec pwsh -NoProfile -File "$REPO_ROOT/scripts/package_release_windows.ps1" \
    -Version "$VERSION" -BuildDir "$BUILD_DIR" -OutDir "$OUT_DIR"
fi
if command -v powershell >/dev/null 2>&1; then
  exec powershell -NoProfile -ExecutionPolicy Bypass -File "$REPO_ROOT/scripts/package_release_windows.ps1" \
    -Version "$VERSION" -BuildDir "$BUILD_DIR" -OutDir "$OUT_DIR"
fi

echo "ERROR: PowerShell required to package Windows MSVC release (pwsh or powershell)." >&2
echo "Install PowerShell 7+ or use Windows PowerShell, then re-run." >&2
exit 1
