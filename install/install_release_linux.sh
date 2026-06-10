#!/usr/bin/env bash
# Install Cypha native release bundle into ~/.local/bin (Linux).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="$(cat "$ROOT/VERSION" 2>/dev/null || echo unknown)"
DEST="$HOME/.local/cypha-${VERSION}"
BIN="$DEST/bin"

mkdir -p "$BIN"
cp -a "$ROOT/bin/"* "$BIN/"
mkdir -p "$DEST/share"
cp -a "$ROOT/share/"* "$DEST/share/" 2>/dev/null || true

mkdir -p "$HOME/.local/bin"
for exe in "$BIN"/*; do
  name="$(basename "$exe")"
  ln -sf "$exe" "$HOME/.local/bin/$name"
done

cat <<EOF

Cypha ${VERSION} installed to:
  $DEST

Binaries linked in ~/.local/bin (ensure ~/.local/bin is on your PATH).

Try:
  cypha_rest --listen 127.0.0.1:8099 \\
    --cypha $DEST/share/demo_fixtures/reference.cypha \\
    --f-field-json $DEST/share/demo_fixtures/f_field.json

EOF
