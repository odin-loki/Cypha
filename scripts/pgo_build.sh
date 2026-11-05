#!/bin/bash
# O1 (docs/GENERATION_PLAN.md §13.41, docs/reports/OPT_BUILD.md): a profile-guided fast build of cyphalm_trie and scribe
# whose outputs are the same byte for byte as the plain Release build's.
#
#   scripts/pgo_build.sh [--cc gcc|clang] [--build DIR] [--jobs N] [--no-lto] [--no-arch] [--stage all|generate|train|use]
#
# 1. Configure DIR with CYPHA_FAST_BUILD=ON (-march=native -ffp-contract=off, LTO) and CYPHA_PGO=generate, and build
#    the instrumented binaries.
# 2. Train on short runs that are not the checked ones:
#      cyphalm_trie context8k on a short split (--train 524288 --sample 655360; the small check is 1048576/1179648);
#      cyphalm_trie sample --k5 on 12 holdout prompts (enwik8 [7 MiB, 8 MiB), not the GEN-A dev or test prompts),
#        from the repo root, so the default planner finds bench/fact_store/e9_aug and bench/k5_planner/model;
#      scribe ladder (v2) on that sample, as scripts/gen_eval.sh runs it.
# 3. Clang only: merge the raw profiles (llvm-profdata) into DIR/pgo/cypha.profdata.
# 4. Reconfigure DIR with CYPHA_PGO=use, rebuild, and run both selftests.
# --stage runs one part: generate (1), train (2), use (3 and 4); all (the default) runs them in turn.
#
# Environment (defaults in brackets):
#   ENWIK8      enwik8 [/mnt/d/cypha_llm/enwik8]
#   DOLLY       Dolly 15k [/mnt/d/cypha_llm/corpus/labelled/dolly/databricks-dolly-15k.jsonl]
#   K5_TEXT     K5's training text [/mnt/d/cypha_llm/fuzzy/train.txt]
#   SCRIBE_VEC  SCRIBE's word vectors [/mnt/d/cypha_llm/scribe/v32]
#   K5_THREADS  K5's threads [4]; CYPHA_LSTM_THREADS and CYPHA_NGRAM_THREADS pass through (outputs do not depend on them)
#
# Odin's PC: at most 4 threads per run, so run it under taskset (e.g. taskset -c 16-19 scripts/pgo_build.sh --cc gcc).
#
# Author: Odin Loch
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CC_KIND=gcc
BUILD=""
JOBS=4
LTO=ON
ARCH=ON
STAGE=all
while [ $# -gt 0 ]; do
    case "$1" in
        --cc) CC_KIND=$2; shift 2;;
        --build) BUILD=$2; shift 2;;
        --jobs|-j) JOBS=$2; shift 2;;
        --no-lto) LTO=OFF; shift;;
        --no-arch) ARCH=OFF; shift;;
        --stage) STAGE=$2; shift 2;;
        *) echo "usage: scripts/pgo_build.sh [--cc gcc|clang] [--build DIR] [--jobs N] [--no-lto] [--no-arch] [--stage all|generate|train|use]" >&2; exit 2;;
    esac
done
case "$CC_KIND" in
    gcc) CC=gcc; CXX=g++;;
    clang) CC=clang-18; CXX=clang++-18
           command -v "$CXX" > /dev/null || { CC=clang; CXX=clang++; };;
    *) echo "--cc gcc or clang" >&2; exit 2;;
esac
case "$STAGE" in all|generate|train|use) ;; *) echo "--stage all, generate, train or use" >&2; exit 2;; esac
BUILD=${BUILD:-$HOME/build_pgo_$CC_KIND}
mkdir -p "$BUILD"
BUILD=$(cd "$BUILD" && pwd)
PGO=$BUILD/pgo
TRAIN=$BUILD/pgo-train
ENWIK8=${ENWIK8:-/mnt/d/cypha_llm/enwik8}
DOLLY=${DOLLY:-/mnt/d/cypha_llm/corpus/labelled/dolly/databricks-dolly-15k.jsonl}
K5_TEXT=${K5_TEXT:-/mnt/d/cypha_llm/fuzzy/train.txt}
VEC=${SCRIBE_VEC:-/mnt/d/cypha_llm/scribe/v32}
export K5_THREADS=${K5_THREADS:-4}
for f in "$ENWIK8" "$DOLLY" "$K5_TEXT"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done
log() { echo "[pgo_build $(date +%H:%M:%S)] $*"; }

configure() {  # $1 = generate | use
    cmake -S "$ROOT/native" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        -DCYPHA_FAST_BUILD=ON -DCYPHA_FAST_ARCH="$ARCH" -DCYPHA_FAST_LTO="$LTO" \
        -DCYPHA_PGO="$1" -DCYPHA_PGO_DIR="$PGO" > "$BUILD/pgo_configure_$1.log" 2>&1
}

if [ "$STAGE" = all ] || [ "$STAGE" = generate ]; then
log "1. instrumented build ($CXX, LTO $LTO, -march=native $ARCH) in $BUILD"
rm -rf "$PGO" "$TRAIN"
mkdir -p "$PGO" "$TRAIN"
configure generate
cmake --build "$BUILD" -j "$JOBS" --target cyphalm_trie scribe > "$BUILD/pgo_build_generate.log" 2>&1
fi

if [ "$STAGE" = all ] || [ "$STAGE" = train ]; then
grep -q "^CYPHA_PGO:STRING=generate" "$BUILD/CMakeCache.txt" || { log "$BUILD is not an instrumented build (run --stage generate)"; exit 1; }
log "2a. training: context8k --train 524288 --sample 655360"
"$BUILD/cyphalm_trie" context8k "$ENWIK8" "$DOLLY" --train 524288 --sample 655360 \
    > "$TRAIN/ctx.out" 2> "$TRAIN/ctx.err"
log "2b. training: sample --k5 on 12 holdout prompts (from $ROOT)"
(cd "$ROOT" && "$BUILD/cyphalm_trie" sample "$ENWIK8" "$TRAIN/k5.gen.txt" --prompts 12 --k5 "$K5_TEXT" \
    > "$TRAIN/k5.out" 2> "$TRAIN/k5.err")
log "2c. training: scribe ladder v2 on that sample"
"$BUILD/scribe" ladder --pack "enwik:$ENWIK8:16:48" --ref "enwik:$ENWIK8:48:56" --test "enwik:$ENWIK8:56:64" \
    --vectors "$VEC" --profile v2 --seed 20261027 --gen-trie "$TRAIN/k5.gen.txt" --label pgo \
    > "$TRAIN/scribe.md" 2> "$TRAIN/scribe.err"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = use ]; then
if [ "$CC_KIND" = clang ]; then
    PD=llvm-profdata-18
    command -v "$PD" > /dev/null || PD=llvm-profdata
    log "3. merging $(ls "$PGO"/*.profraw | wc -l) raw profiles"
    "$PD" merge -output="$PGO/cypha.profdata" "$PGO"/*.profraw
else
    log "3. $(find "$PGO" -name '*.gcda' | wc -l) GCC profiles"
fi

log "4. profile-guided build"
configure use
cmake --build "$BUILD" -j "$JOBS" --target cyphalm_trie scribe > "$BUILD/pgo_build_use.log" 2>&1
"$BUILD/cyphalm_trie" selftest > "$BUILD/selftest_cyphalm_trie.log" 2>&1 || { log "cyphalm_trie selftest FAILED"; exit 1; }
"$BUILD/scribe" selftest > "$BUILD/selftest_scribe.log" 2>&1 || { log "scribe selftest FAILED"; exit 1; }
log "done: $BUILD/cyphalm_trie, $BUILD/scribe (selftests pass)"
fi
