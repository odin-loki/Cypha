#!/usr/bin/env bash
# Cross-build determinism: the real Hutter Prize scenario.
#
# Build the same source under aggressively different optimisation settings,
# then check that (a) every build produces a BYTE-IDENTICAL archive, and
# (b) every build can decode every other build's archive. -ffast-math is
# included deliberately: it is a no-op here, and proving that is the point.
set -u
cd "$(dirname "$0")/.."
IN=${1:?usage: determinism.sh <file>}
TMP=$(mktemp -d)
XSIMD_FLAGS=(-DHP_XSIMD=1 -msse4.1 -Ithird_party/xsimd/include)
FLAGS=("-O0" "-O1" "-O2" "-O3" "-Os" "-O3 -march=native" "-O2 -ffast-math" "-O3 -funroll-loops")
names=()
for i in "${!FLAGS[@]}"; do
  g++ -std=c++17 ${FLAGS[$i]} -Iinclude "${XSIMD_FLAGS[@]}" -o "$TMP/hp$i" src/main.cpp 2>/dev/null || { echo "build $i failed"; exit 1; }
  "$TMP/hp$i" c "$IN" "$TMP/a$i.hp" 2>/dev/null
  names+=("$i")
done

ref=$(sha256sum "$TMP/a0.hp" | cut -d' ' -f1)
ok=1
for i in "${names[@]}"; do
  h=$(sha256sum "$TMP/a$i.hp" | cut -d' ' -f1)
  if [ "$h" = "$ref" ]; then s=same; else s=DIFFER; ok=0; fi
  printf '  archive %-22s %s  %s\n' "${FLAGS[$i]}" "${h:0:16}" "$s"
done
[ $ok -eq 1 ] && echo "PASS: all builds produce identical archives" || { echo "FAIL: archives differ"; exit 1; }

# Cross-decode matrix: archive from build i, decoded by build j.
srchash=$(sha256sum "$IN" | cut -d' ' -f1)
for i in "${names[@]}"; do
  for j in "${names[@]}"; do
    "$TMP/hp$j" d "$TMP/a$i.hp" "$TMP/out" 2>/dev/null
    [ "$(sha256sum "$TMP/out" | cut -d' ' -f1)" = "$srchash" ] || { echo "FAIL: build $j cannot decode build $i"; exit 1; }
  done
done
echo "PASS: full ${#names[@]}x${#names[@]} cross-decode matrix"
rm -rf "$TMP"
