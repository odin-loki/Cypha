#!/bin/bash
# mt8_build.sh BIN -- builds cyphalm_trie and scribe from ~/mt8_src with g++ -O2, as scripts/build_1tb_tools.sh does (libsais.o in scribe, as D2/D4)
set -euo pipefail
BIN=$1; mkdir -p "$BIN"; cd ~/mt8_src
ROOT=$(pwd)
INC=(-I native/include -I native/third_party -I native/third_party/libsais)
gcc -O2 -c native/third_party/libsais/libsais.c -o "$BIN/libsais.o"
g++ -std=c++23 -O2 "${INC[@]}" "-DCYPHA_SIMVEC_DEFAULT=\"$ROOT/data/cyphalm/simvec_ppmi32.bin\"" \
    native/tools/lm/cyphalm_trie.cpp native/src/lm/fm_infinigram.cpp "$BIN/libsais.o" -o "$BIN/cyphalm_trie" &
g++ -std=c++23 -O2 "${INC[@]}" -o "$BIN/scribe" native/tools/eval/scribe.cpp "$BIN/libsais.o" &
fail=0
for job in $(jobs -p); do wait "$job" || fail=1; done
[ "$fail" = 0 ] || { echo "build failed"; exit 1; }
echo "built into $BIN"
