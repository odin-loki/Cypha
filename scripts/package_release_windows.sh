#!/usr/bin/env bash
# Package Windows x86_64 release ZIP from a MinGW cross-build directory.
#
# Usage:
#   bash scripts/package_release_windows.sh <VERSION> <BUILD_DIR> [OUT_DIR]

set -euo pipefail

VERSION="${1:?version required, e.g. 2.0.0}"
BUILD_DIR="${2:?build dir required, e.g. native/build-mingw-w64}"
OUT_DIR="${3:-dist}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$REPO_ROOT/$OUT_DIR/cypha-${VERSION}-windows-x86_64"
ARCHIVE="$REPO_ROOT/$OUT_DIR/cypha-${VERSION}-windows-x86_64.zip"

# Production binaries (added to user PATH by install.ps1)
BINARIES=(
  cypha_rest.exe
  cypha_bench_run.exe
  cypha_bench_report.exe
  cypha_diagnostics_run.exe
  cypha_tune_run.exe
  cyphalm_bench_native.exe
  cyphalm_parity.exe
  cyphalm_checkpoint_parity.exe
  gh_infer_deliberation_parity.exe
  kernel_llr_parity.exe
  registry_register.exe
  create_model_smoke.exe
)

# Dev / research parity tools (installed under bin/dev/, not on PATH)
DEV_BINARIES=(
  score_batch_parity.exe
  multilabel_dif_parity.exe
  merge_from_parity.exe
  similarity_index_parity.exe
  embed_table_parity.exe
  retrieval_parity.exe
  som_parity.exe
)

rm -rf "$STAGING"
mkdir -p "$STAGING/bin/dev" "$STAGING/share/demo_fixtures" "$STAGING/share/examples"

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

for bin in "${DEV_BINARIES[@]}"; do
  src="$REPO_ROOT/$BUILD_DIR/$bin"
  if [[ -f "$src" ]]; then
    cp "$src" "$STAGING/bin/dev/$bin"
    echo "  + bin/dev/$bin"
  else
    echo "  skip missing dev/$bin"
  fi
done

cp "$REPO_ROOT/install/install_release_windows.ps1" "$STAGING/install.ps1"

cat >"$STAGING/README.txt" <<EOF
Cypha ${VERSION} — Windows x86_64 native tools (MinGW PE, full C++ framework)
=============================================================================

Quick install (adds %LOCALAPPDATA%\\Cypha\\${VERSION}\\bin to user PATH):
  powershell -ExecutionPolicy Bypass -File install.ps1

Run native REST (classifier + CyphaLM + CyphaDIF routes):
  cypha_rest.exe --listen 127.0.0.1:8099 --cypha share\\demo_fixtures\\reference.cypha ^
    --f-field-json share\\demo_fixtures\\f_field.json

CyphaDIF REST routes (POST JSON):
  /dif/retrieve   — ranked database hits (input, database, top_k, optional label)
  /dif/generate   — latent samples (mode: langevin | from_observation | retrieval_augmented)

Run native bench domains (d01–d17):
  cypha_bench_run.exe --domain 17

Rebuild bench report from saved tables:
  cypha_bench_report.exe --output .\\bench_report

Run native diagnostics (phases 1–4 parity orchestrator):
  cypha_diagnostics_run.exe --fixtures share\\demo_fixtures\\..\\..\\parity_fixtures

Run CyphaLM bench CLI:
  cyphalm_bench_native.exe --mode hybrid --profile d17 --n-train 5000 --n-eval 500 --threads 1

Run native tuning sweep (dry-run):
  cypha_tune_run.exe --config share\\..\\..\\cypha_bench\\config\\cyphalm_hybrid_lstm_tune_smoke.json --dry-run

Dev parity tools (not on PATH): bin\\dev\\score_batch_parity.exe, multilabel_dif_parity.exe, merge_from_parity.exe, similarity_index_parity.exe, embed_table_parity.exe, retrieval_parity.exe, som_parity.exe

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
