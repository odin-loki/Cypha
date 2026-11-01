#!/bin/bash
# Builds N0's tools from this branch into BIN [~/cypha_build/nn_diag] and runs their selftests.
# Author: Odin Loch
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$HOME/cypha_build/nn_diag}
mkdir -p "$BIN"
cd "$ROOT"
INC=(-I native/include -I native/third_party -I native/third_party/libsais)
gcc -O2 -c native/third_party/libsais/libsais.c -o "$BIN/libsais.o"
g++ -std=c++23 -O3 -march=native -pthread "${INC[@]}" native/tools/cyphalm_tok_gen.cpp native/src/cyphalm/fm_infinigram.cpp "$BIN/libsais.o" -o "$BIN/cyphalm_tok_gen" &
g++ -std=c++23 -O3 -march=native -pthread "${INC[@]}" native/tools/cyphalm_tok.cpp native/src/cyphalm/fm_infinigram.cpp "$BIN/libsais.o" -o "$BIN/cyphalm_tok" &
g++ -std=c++23 -O2 -I native/include native/tools/scribe.cpp -o "$BIN/scribe" &
fail=0
for job in $(jobs -p); do wait "$job" || fail=1; done
[ "$fail" = 0 ] || { echo "build failed"; exit 1; }
"$BIN/scribe" selftest | tail -1
"$BIN/cyphalm_tok" selftest | tail -1
"$BIN/cyphalm_tok_gen" check-t4b
echo "built into $BIN"
