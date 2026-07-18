#!/usr/bin/env bash
# Package Linux x86_64 release tarball from a native CMake build directory.
#
# Usage:
#   bash scripts/package_release_linux.sh <VERSION> <BUILD_DIR> [OUT_DIR]
#
# Example (CI):
#   bash scripts/package_release_linux.sh 2.0.0 native/build-release dist

set -euo pipefail

VERSION="${1:?version required, e.g. 2.0.0}"
BUILD_DIR="${2:?build dir required, e.g. native/build}"
OUT_DIR="${3:-dist}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$REPO_ROOT/$OUT_DIR/cypha-${VERSION}-linux-x86_64"
ARCHIVE="$REPO_ROOT/$OUT_DIR/cypha-${VERSION}-linux-x86_64.tar.gz"

# Production binaries (symlinked to PATH by install.sh)
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

# Dev / research golden tools (installed under bin/dev/, not on PATH)
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

cp "$REPO_ROOT/packaging/install_release_linux.sh" "$STAGING/install.sh"
chmod +x "$STAGING/install.sh"

cat >"$STAGING/README.txt" <<EOF
Cypha ${VERSION} - Linux x86_64 native tools (full C++ framework)
=================================================================

Quick install (adds ~/.local/bin symlinks for bin/; dev tools stay in bin/dev/):
  bash install.sh

Run native REST (classifier + CyphaLM + CyphaDIF routes):
  cypha_rest --listen 127.0.0.1:8099 --cypha share/demo_fixtures/reference.cypha \\
    --f-field-json share/demo_fixtures/f_field.json

CyphaDIF REST routes (POST JSON):
  /dif/retrieve   - ranked database hits (input, database, top_k, optional label)
  /dif/generate   - latent samples (mode: langevin | from_observation | retrieval_augmented)

Run native bench domains (d01-d17):
  cypha_bench_run --domain 17

Rebuild bench report from saved tables:
  cypha_bench_report --output ./bench_report

Run native diagnostics (phases 1-4 orchestrator):
  cypha_diagnostics_run --fixtures /path/to/fixtures

Run CyphaLM bench CLI (WikiText profile; needs corpus on PATH or synthetic fallback):
  cyphalm_bench_native --mode hybrid --profile d17 --n-train 5000 --n-eval 500 --threads 1

Run native tuning sweep (dry-run):
  cypha_tune_run --config ../../bench/config/cyphalm_hybrid_lstm_tune_smoke.json --dry-run

Dev golden tools (not on PATH): bin/dev/*_golden

Qt shell: build cypha_qt_shell from native/ (see docs/native/qt/README.md).

Runtime deps: libgomp1 (OpenMP), libstdc++6, glibc 2.31+
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
