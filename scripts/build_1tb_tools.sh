#!/bin/bash
# Builds the tools docs/reports/TRIE_1TB_PROFILE.md uses into BIN (default ~/bin): the live trainer, its profiler,
# trie_ckpt (verify / bpc / pack), cyphalm_trie (context8k, sample --model), scribe and ppmi_svd.
# Build outside /tmp: WSL clears /tmp when its VM restarts.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${BIN:-$HOME/bin}
mkdir -p "$BIN"
cd "$ROOT"
INC=(-I native/include -I native/third_party -I native/third_party/libsais)
gcc -O2 -c native/third_party/libsais/libsais.c -o "$BIN/libsais.o"
g++ -std=c++23 -O2 -pthread -o "$BIN/trie_stream_train" native/tools/trie_stream_train.cpp &
g++ -std=c++23 -O2 -o "$BIN/trie_stream_profile" native/tools/trie_stream_profile.cpp &
g++ -std=c++23 -O2 "${INC[@]}" -o "$BIN/trie_ckpt" native/tools/trie_ckpt.cpp &
g++ -std=c++23 -O2 "${INC[@]}" "-DCYPHA_SIMVEC_DEFAULT=\"$ROOT/data/cyphalm/simvec_ppmi32.bin\"" \
    native/tools/cyphalm_trie.cpp native/src/cyphalm/fm_infinigram.cpp "$BIN/libsais.o" -o "$BIN/cyphalm_trie" &
g++ -std=c++23 -O2 -I native/include -o "$BIN/scribe" native/tools/scribe.cpp &
g++ -std=c++20 -O2 -o "$BIN/ppmi_svd" native/tools/ppmi_svd.cpp &
fail=0
for job in $(jobs -p); do wait "$job" || fail=1; done
[ "$fail" = 0 ] || { echo "build failed"; exit 1; }
"$BIN/trie_ckpt" selftest
echo "built into $BIN"
