#!/usr/bin/env bash
# Package Linux x86_64 release tarball from a native CMake build directory.
#
# Usage:
#   bash scripts/package_release_linux.sh <VERSION> <BUILD_DIR> [OUT_DIR]
#
# Example (CI):
#   bash scripts/package_release_linux.sh 1.1.0 native/build dist

set -euo pipefail

VERSION="${1:?version required, e.g. 1.1.0}"
BUILD_DIR="${2:?build dir required, e.g. native/build}"
OUT_DIR="${3:-dist}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$REPO_ROOT/$OUT_DIR/cypha-${VERSION}-linux-x86_64"
ARCHIVE="$REPO_ROOT/$OUT_DIR/cypha-${VERSION}-linux-x86_64.tar.gz"

BINARIES=(
  cypha_rest
  cyphalm_bench_native
  cyphalm_parity
  cyphalm_checkpoint_parity
  registry_register
  create_model_smoke
)

rm -rf "$STAGING"
mkdir -p "$STAGING/bin" "$STAGING/share/demo_fixtures" "$STAGING/share/examples"

echo "$VERSION" >"$STAGING/VERSION"

for bin in "${BINARIES[@]}"; do
  src="$REPO_ROOT/$BUILD_DIR/$bin"
  if [[ -x "$src" ]]; then
    install -m 755 "$src" "$STAGING/bin/$bin"
    echo "  + bin/$bin"
  else
    echo "  skip missing $bin"
  fi
done

cp "$REPO_ROOT/install/install_release_linux.sh" "$STAGING/install.sh"
chmod +x "$STAGING/install.sh"

cat >"$STAGING/README.txt" <<EOF
Cypha ${VERSION} — Linux x86_64 native tools
============================================

Quick install (adds ~/.local/bin/cypha-${VERSION} and symlinks):
  bash install.sh

Run native REST (classifier + CyphaLM routes):
  cypha_rest --listen 127.0.0.1:8099 --cypha share/demo_fixtures/reference.cypha \\
    --f-field-json share/demo_fixtures/f_field.json

Run CyphaLM bench (WikiText profile; needs corpus on PATH or synthetic fallback):
  cyphalm_bench_native --mode hybrid --profile d17 --n-train 5000 --n-eval 500 --threads 1

Python Studio / full stack: clone the repo and run install/install_linux.sh --studio

Runtime deps: libgomp1 (OpenMP), libstdc++6, glibc 2.31+
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
tar -C "$(dirname "$STAGING")" -czf "$ARCHIVE" "$(basename "$STAGING")"
echo "Created $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
