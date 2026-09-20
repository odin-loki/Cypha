#!/usr/bin/env bash
# Package macOS release tarball from a native CMake build directory.
#
# Usage:
#   bash scripts/package_release_macos.sh <VERSION> <BUILD_DIR> [OUT_DIR]
#
# Example (CI macos-latest / arm64):
#   bash scripts/package_release_macos.sh 2.5.0 native/build-macos dist
#
# Archive name: cypha-<VERSION>-macos-<arch>.tar.gz  (arch = uname -m, e.g. arm64)

set -euo pipefail

VERSION="${1:?version required, e.g. 2.5.0}"
BUILD_DIR="${2:?build dir required, e.g. native/build-macos}"
OUT_DIR="${3:-dist}"

ARCH="$(uname -m)"
case "$ARCH" in
  arm64|aarch64) ARCH_LABEL="arm64" ;;
  x86_64|amd64) ARCH_LABEL="x86_64" ;;
  *) ARCH_LABEL="$ARCH" ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$REPO_ROOT/$OUT_DIR/cypha-${VERSION}-macos-${ARCH_LABEL}"
ARCHIVE="$REPO_ROOT/$OUT_DIR/cypha-${VERSION}-macos-${ARCH_LABEL}.tar.gz"

BINARIES=(
  cypha_rest
  cypha_bench_run
  cypha_bench_report
  cypha_diagnostics_run
  cypha_tune_run
  cyphalm_bench_native
  cypha_baseline_lock
  baseline_lock_validate
  registry_register
  create_model_smoke
)

DEV_BINARIES=(
  score_batch_golden
  multilabel_dif_golden
  merge_from_golden
  similarity_index_golden
  embed_table_golden
  retrieval_golden
  som_golden
  kernel_llr_golden
  gh_infer_deliberation_golden
  cyphalm_checkpoint_golden
)

find_bin() {
  local name="$1"
  local cand
  for cand in \
    "$REPO_ROOT/$BUILD_DIR/$name" \
    "$REPO_ROOT/$BUILD_DIR/Release/$name" \
    "$REPO_ROOT/$BUILD_DIR/bin/$name"; do
    if [[ -f "$cand" ]]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

rm -rf "$STAGING"
mkdir -p "$STAGING/bin/dev" "$STAGING/share/demo_fixtures" "$STAGING/share/examples"

echo "$VERSION" >"$STAGING/VERSION"
echo "macos-${ARCH_LABEL}" >"$STAGING/PLATFORM"

for bin in "${BINARIES[@]}"; do
  if ! src="$(find_bin "$bin")"; then
    echo "ERROR: required release binary missing from build dir: $bin (looked in $BUILD_DIR)" >&2
    exit 1
  fi
  cp "$src" "$STAGING/bin/$bin"
  chmod +x "$STAGING/bin/$bin"
  echo "  + bin/$bin"
done

for bin in "${DEV_BINARIES[@]}"; do
  if ! src="$(find_bin "$bin")"; then
    echo "ERROR: required dev release binary missing from build dir: $bin (looked in $BUILD_DIR)" >&2
    exit 1
  fi
  cp "$src" "$STAGING/bin/dev/$bin"
  chmod +x "$STAGING/bin/dev/$bin"
  echo "  + bin/dev/$bin"
done

cp "$REPO_ROOT/packaging/install_release_macos.sh" "$STAGING/install.sh"
chmod +x "$STAGING/install.sh"

cat >"$STAGING/README.txt" <<EOF
Cypha ${VERSION} - macOS ${ARCH_LABEL} native tools
====================================================

Quick install (adds ~/.local/bin symlinks for bin/; dev tools stay in bin/dev/):
  bash install.sh

Run native REST:
  cypha_rest --listen 127.0.0.1:8099 --cypha share/demo_fixtures/reference.cypha \\
    --f-field-json share/demo_fixtures/f_field.json

Run native bench:
  cypha_bench_run --list-domains

Run CyphaLM / hp bench CLI:
  cyphalm_bench_native --help

Dev golden tools (not on PATH): bin/dev/*_golden

Built on macOS ${ARCH_LABEL} (Apple Silicon / Intel matching this archive name).
EOF

for f in reference.cypha f_field.json train_hparams.json; do
  if [[ -f "$REPO_ROOT/fixtures/$f" ]]; then
    cp "$REPO_ROOT/fixtures/$f" "$STAGING/share/demo_fixtures/"
  fi
done

if [[ -d "$REPO_ROOT/examples/demo_cyphalm" ]]; then
  cp -r "$REPO_ROOT/examples/demo_cyphalm" "$STAGING/share/examples/"
fi

mkdir -p "$REPO_ROOT/$OUT_DIR"
tar -C "$(dirname "$STAGING")" -czf "$ARCHIVE" "$(basename "$STAGING")"
echo "Created $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
