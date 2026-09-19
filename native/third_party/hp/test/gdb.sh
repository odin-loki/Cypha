#!/usr/bin/env bash
# Exercise compress/decompress under gdb (batch mode, no interactive session).
set -euo pipefail
cd "$(dirname "$0")/.."
HP=${HP:-build/hp}
FILE=${1:-data/proxy64k.xml}
MEM=${2:-22}
ARC=/tmp/hp_gdb.arc
OUT=/tmp/hp_gdb.out

if ! command -v gdb >/dev/null 2>&1; then
  echo "SKIP: gdb not installed"
  exit 0
fi

[ -x "$HP" ] || { echo "missing $HP (run make first)"; exit 1; }
[ -f "$FILE" ] || { echo "missing $FILE"; exit 1; }

echo "gdb compress: $FILE"
gdb -batch -q \
  -ex "set pagination off" \
  -ex "run c --mem $MEM $FILE $ARC" \
  -ex "quit" \
  -- "$HP"

echo "gdb decompress: $ARC"
gdb -batch -q \
  -ex "set pagination off" \
  -ex "run d $ARC $OUT" \
  -ex "quit" \
  -- "$HP"

sha_in=$(sha256sum "$FILE" | awk '{print $1}')
sha_out=$(sha256sum "$OUT" | awk '{print $1}')
if [ "$sha_in" != "$sha_out" ]; then
  echo "FAIL gdb roundtrip sha mismatch"
  exit 1
fi
echo "PASS gdb compress/decompress ($sha_in)"
