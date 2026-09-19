#!/usr/bin/env bash
# Measure compress-equivalent BPC vs archive BPC vs clone-path eval_bpc on the same corpus.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${CYPHA_NATIVE_BUILD_DIR:-$ROOT/native/build}"
OUT="${CYPHA_BPC_GAP_OUT:-$ROOT/bench/results/cyphalm_bpc_gap}"
HP_TOOL="${HP_TOOL:-$BUILD/hp_light}"
mkdir -p "$OUT"

if [[ ! -x "$BUILD/cyphalm_llm_profile" ]]; then
  cmake -S "$ROOT/native" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCYPHA_HP_PROFILE="${CYPHA_HP_PROFILE:-light}"
  cmake --build "$BUILD" --target cyphalm_llm_profile bpc_gap_verify -j"$(nproc 2>/dev/null || echo 4)"
fi

if [[ ! -x "$HP_TOOL" ]]; then
  g++ -std=c++17 -O2 \
    -I "$ROOT/native/third_party/hp/include" \
    -I "$ROOT/native/third_party/hp/third_party/xsimd/include" \
    -DHP_SLOT_MAX=24 -DHP_XSIMD=0 \
    "$ROOT/native/third_party/hp/src/main.cpp" -o "$HP_TOOL"
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CORPUS="${1:-bench/data/enwik8/enwik8.8mb}"
BYTES="${2:-100000}"
CLONE_N="${3:-16}"

echo "# cyphalm_bpc_gap $STAMP corpus=$CORPUS bytes=$BYTES" | tee "$OUT/run_${STAMP}.meta"

python3 "$ROOT/scripts/hp_v78_flag_diff.py" 2>&1 | tee "$OUT/v78_flag_diff_${STAMP}.tsv"

"$BUILD/bpc_gap_verify" --random "${BPC_GAP_VERIFY_RANDOM:-8}" --corpus "$CORPUS" \
  2>&1 | tee "$OUT/verify_${STAMP}.json"

"$BUILD/cyphalm_llm_profile" --bpc-gap-only \
  --corpus-file "$CORPUS" \
  --bpc-gap-bytes "$BYTES" \
  --bpc-gap-clone-n "$CLONE_N" \
  --bpc-gap-clone-train 32 \
  --hp-tool "$HP_TOOL" \
  2>&1 | tee "$OUT/gap_${STAMP}.txt"

echo "Results: $OUT/gap_${STAMP}.txt"
