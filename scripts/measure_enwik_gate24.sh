#!/usr/bin/env bash
# Measure enwik8.8mb BPC for CyphaLM gate24 (archive + observe + profile).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS="${1:-$ROOT/bench/data/enwik8/enwik8.8mb}"
MEASURE_DIR="${CYPHA_MEASURE_DIR:-$ROOT/native/build-gate24-measure}"
OUT="${CYPHA_GATE24_OUT:-$ROOT/bench/results/enwik_gate24_measure}"
mkdir -p "$OUT" "$MEASURE_DIR"

HP_SRC="$ROOT/native/third_party/hp/src/main.cpp"
HP_INC="$ROOT/native/third_party/hp/include"
XSIMD="$ROOT/native/third_party/hp/third_party/xsimd/include"
build_hp_cli() {
  local out="$MEASURE_DIR/hp_gate24"
  [[ -x "$out" ]] && return 0
  echo "Building hp_gate24 CLI..."
  g++ -std=c++17 -O3 -msse4.1 -I "$HP_INC" -I "$XSIMD" -DHP_SLOT_MAX=24 -DHP_XSIMD=1 "$HP_SRC" -o "$out"
}

build_cypha() {
  [[ -x "$MEASURE_DIR/cyphalm_hp_sku_measure" ]] && return 0
  echo "Building Cypha gate24..."
  cmake -S "$ROOT/native" -B "$MEASURE_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ >/dev/null
  cmake --build "$MEASURE_DIR" --target cyphalm_hp_sku_measure -j"$(nproc)" >/dev/null
}

build_hp_cli
build_cypha

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
outfile="$OUT/gate24_${STAMP}.json"
echo "=== Measuring gate24 ==="
set +e
if command -v /usr/bin/time >/dev/null 2>&1; then
  /usr/bin/time -v "$MEASURE_DIR/cyphalm_hp_sku_measure" \
    --corpus "$CORPUS" --hp-tool "$MEASURE_DIR/hp_gate24" \
    --work-dir "$OUT/work_gate24_${STAMP}" --latency-iters 2 \
    2> "$OUT/gate24_${STAMP}_time_v.txt" | tee "$outfile"
else
  "$MEASURE_DIR/cyphalm_hp_sku_measure" \
    --corpus "$CORPUS" --hp-tool "$MEASURE_DIR/hp_gate24" \
    --work-dir "$OUT/work_gate24_${STAMP}" --latency-iters 2 \
    2> "$OUT/gate24_${STAMP}_time_v.txt" | tee "$outfile"
fi
rc=$?
set -e

echo "Result: $outfile (exit $rc)"
exit $rc
