#!/usr/bin/env bash
# Install Cypha native release bundle into ~/.local/bin (Linux).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="$(cat "$ROOT/VERSION" 2>/dev/null || echo unknown)"
DEST="$HOME/.local/cypha-${VERSION}"
BIN="$DEST/bin"

mkdir -p "$BIN"
# Production tools only — skip bin/dev (research parity exes stay off PATH)
for exe in "$ROOT/bin"/*; do
  [[ -f "$exe" && -x "$exe" ]] || continue
  install -m 755 "$exe" "$BIN/$(basename "$exe")"
done

required=(
  cypha_rest
  cypha_bench_run
  cypha_bench_report
  cypha_diagnostics_run
  cypha_tune_run
)
for name in "${required[@]}"; do
  if [[ ! -x "$BIN/$name" ]]; then
    echo "error: release bundle missing bin/$name (re-package with scripts/package_release_linux.sh)" >&2
    exit 1
  fi
done

if [[ -d "$ROOT/bin/dev" ]]; then
  mkdir -p "$BIN/dev"
  for exe in "$ROOT/bin/dev"/*; do
    [[ -f "$exe" && -x "$exe" ]] || continue
    install -m 755 "$exe" "$BIN/dev/$(basename "$exe")"
  done
fi

mkdir -p "$DEST/share"
cp -a "$ROOT/share/"* "$DEST/share/" 2>/dev/null || true

mkdir -p "$HOME/.local/bin"
for exe in "$BIN"/*; do
  [[ -f "$exe" && -x "$exe" ]] || continue
  name="$(basename "$exe")"
  ln -sf "$exe" "$HOME/.local/bin/$name"
done

cat <<EOF

Cypha ${VERSION} installed to:
  $DEST

Production binaries linked in ~/.local/bin (ensure ~/.local/bin is on your PATH).
Dev parity tools (if bundled): $BIN/dev/

Try:
  cypha_rest --listen 127.0.0.1:8099 \\
    --cypha $DEST/share/demo_fixtures/reference.cypha \\
    --f-field-json $DEST/share/demo_fixtures/f_field.json

  cypha_bench_run --list-domains
  cypha_bench_run --domain 17
  cypha_bench_run --report-only
  cypha_bench_report --output ./bench_report
  cypha_tune_run --config /path/to/sweep.json --dry-run
  cypha_diagnostics_run --fixtures /path/to/fixtures

CyphaDIF REST (POST JSON on cypha_rest): /dif/retrieve, /dif/generate

EOF
