#!/usr/bin/env bash
# Package Windows x86_64 release ZIP from a MinGW cross-build directory.
#
# Usage:
#   bash scripts/package_release_windows.sh <VERSION> <BUILD_DIR> [OUT_DIR]

set -euo pipefail

VERSION="${1:?version required, e.g. 1.1.0}"
BUILD_DIR="${2:?build dir required, e.g. native/build-mingw-w64}"
OUT_DIR="${3:-dist}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$REPO_ROOT/$OUT_DIR/cypha-${VERSION}-windows-x86_64"
ARCHIVE="$REPO_ROOT/$OUT_DIR/cypha-${VERSION}-windows-x86_64.zip"

BINARIES=(
  cypha_rest.exe
  cyphalm_bench_native.exe
  cyphalm_parity.exe
  cyphalm_checkpoint_parity.exe
  registry_register.exe
  create_model_smoke.exe
)

rm -rf "$STAGING"
mkdir -p "$STAGING/bin" "$STAGING/share/demo_fixtures" "$STAGING/share/examples"

echo "$VERSION" >"$STAGING/VERSION"

for bin in "${BINARIES[@]}"; do
  src="$REPO_ROOT/$BUILD_DIR/$bin"
  if [[ -f "$src" ]]; then
    cp "$src" "$STAGING/bin/$bin"
    echo "  + bin/$bin"
  else
    echo "  skip missing $bin"
  fi
done

cp "$REPO_ROOT/install/install_release_windows.ps1" "$STAGING/install.ps1"

cat >"$STAGING/README.txt" <<EOF
Cypha ${VERSION} — Windows x86_64 native tools (MinGW PE)
=======================================================

Quick install (adds %LOCALAPPDATA%\\Cypha\\${VERSION}\\bin to user PATH):
  powershell -ExecutionPolicy Bypass -File install.ps1

Run native REST:
  cypha_rest.exe --listen 127.0.0.1:8099 --cypha share\\demo_fixtures\\reference.cypha ^
    --f-field-json share\\demo_fixtures\\f_field.json

Run CyphaLM bench:
  cyphalm_bench_native.exe --mode hybrid --profile d17 --n-train 5000 --n-eval 500 --threads 1

Python Studio: clone the repo and run install\\install_windows.ps1 -Studio

These binaries are cross-built with static libgcc/libstdc++ (no separate MinGW DLLs required).
EOF

for f in reference.cypha f_field.json train_hparams.json; do
  if [[ -f "$REPO_ROOT/parity_fixtures/$f" ]]; then
    cp "$REPO_ROOT/parity_fixtures/$f" "$STAGING/share/demo_fixtures/"
  fi
done

if [[ -d "$REPO_ROOT/examples/demo_cyphalm" ]]; then
  cp -r "$REPO_ROOT/examples/demo_cyphalm" "$STAGING/share/examples/"
fi

mkdir -p "$REPO_ROOT/$OUT_DIR"
rm -f "$ARCHIVE"
(
  cd "$(dirname "$STAGING")"
  if command -v zip >/dev/null 2>&1; then
    zip -rq "$ARCHIVE" "$(basename "$STAGING")"
  else
    python3 - <<PY
import pathlib, shutil, sys
root = pathlib.Path(r"$(dirname "$STAGING")")
base = pathlib.Path(r"$(basename "$STAGING")")
shutil.make_archive(r"${ARCHIVE%.zip}", "zip", root, base)
PY
  fi
)
echo "Created $ARCHIVE"
