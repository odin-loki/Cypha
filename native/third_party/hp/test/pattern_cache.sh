#!/usr/bin/env bash
# B.1 identity: archives with HP_PATTERN_CACHE=1 and =0 must be byte-identical.
set -eu
cd "$(dirname "$0")/.."
IN=${1:?usage: pattern_cache.sh <file>}
TMP=$(mktemp -d)
XSIMD=(-DHP_XSIMD=1 -msse4.1 -Ithird_party/xsimd/include)
g++ -std=c++17 -O2 -Iinclude "${XSIMD[@]}" -DHP_PATTERN_CACHE=1 -o "$TMP/on"  src/main.cpp
g++ -std=c++17 -O2 -Iinclude "${XSIMD[@]}" -DHP_PATTERN_CACHE=0 -o "$TMP/off" src/main.cpp
"$TMP/on"  c --mem 18 "$IN" "$TMP/on.hp"
"$TMP/off" c --mem 18 "$IN" "$TMP/off.hp"
hon=$(sha256sum "$TMP/on.hp"  | cut -d' ' -f1)
hoff=$(sha256sum "$TMP/off.hp" | cut -d' ' -f1)
if [ "$hon" = "$hoff" ]; then
  echo "PASS: pattern cache on/off archives identical ($hon)"
  rm -rf "$TMP"
  exit 0
else
  echo "FAIL: cache on  $hon"
  echo "FAIL: cache off $hoff"
  rm -rf "$TMP"
  exit 1
fi
