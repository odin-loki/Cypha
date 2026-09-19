#!/usr/bin/env bash
# Round-trip correctness: compress, decompress, compare SHA-256.
#
# This is the gate. Nothing else about a compressor matters if this fails.
set -u
HP=${HP:-/tmp/hp}
fail=0
for f in "$@"; do
  [ -f "$f" ] || continue
  n=$(stat -c%s "$f")
  "$HP" c "$f" /tmp/_rt.hp 2>/dev/null
  "$HP" d /tmp/_rt.hp /tmp/_rt.out 2>/dev/null
  c=$(stat -c%s /tmp/_rt.hp)
  a=$(sha256sum "$f" | cut -d' ' -f1)
  b=$(sha256sum /tmp/_rt.out | cut -d' ' -f1)
  bpc=$(echo "scale=3; $c*8/$n" | bc)
  if [ "$a" = "$b" ]; then
    printf 'PASS  %-28s %9d -> %8d  %s bpc\n' "$(basename "$f")" "$n" "$c" "$bpc"
  else
    printf 'FAIL  %-28s sha mismatch\n' "$(basename "$f")"; fail=1
  fi
done
exit $fail
