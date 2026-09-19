#!/usr/bin/env bash
# Measure enwik8.8mb BPC for light / gate24 / champ SKUs (archive + observe + profile).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS="${1:-$ROOT/bench/data/enwik8/enwik8.8mb}"
MEASURE_DIR="${CYPHA_MEASURE_DIR:-$ROOT/native/build-sku-measure}"
OUT="${CYPHA_SKU_OUT:-$ROOT/bench/results/enwik_sku_measure}"
mkdir -p "$OUT" "$MEASURE_DIR"

HP_SRC="$ROOT/native/third_party/hp/src/main.cpp"
HP_INC="$ROOT/native/third_party/hp/include"
XSIMD="$ROOT/native/third_party/hp/third_party/xsimd/include"
V78=$(grep -oE '\-DHP_[A-Z0-9_]+=[0-9]+' "$ROOT/native/third_party/hp/tools/v78_flags.ps1" | grep -v HP_SLOT_MAX | tr '\n' ' ')

build_hp_cli() {
  local name=$1 slot=$2
  local out="$MEASURE_DIR/hp_${name}"
  [[ -x "$out" ]] && return 0
  echo "Building hp_${name}..."
  if [[ "$name" == "light" ]]; then
    g++ -std=c++17 -O3 -msse4.1 -I "$HP_INC" -I "$XSIMD" -DHP_SLOT_MAX=24 -DHP_XSIMD=0 \
      "$HP_SRC" -o "$out"
  else
    g++ -std=c++17 -O3 -msse4.1 -I "$HP_INC" -I "$XSIMD" -DHP_SLOT_MAX="$slot" -DHP_XSIMD=1 \
      $V78 "$HP_SRC" -o "$out"
  fi
}

build_cypha() {
  local profile=$1
  local dir="$MEASURE_DIR/cypha_${profile}"
  [[ -x "$dir/cyphalm_hp_sku_measure" ]] && return 0
  echo "Building Cypha ${profile}..."
  cmake -S "$ROOT/native" -B "$dir" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ -DCYPHA_HP_PROFILE="$profile" >/dev/null
  cmake --build "$dir" --target cyphalm_hp_sku_measure -j"$(nproc)" >/dev/null
}

build_hp_cli light 24
build_hp_cli gate24 24
build_hp_cli champ 35

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
python3 "$ROOT/scripts/hp_v78_flag_diff.py" | tee "$OUT/v78_flag_diff_${STAMP}.tsv"

echo "{\"stamp\":\"$STAMP\",\"corpus\":\"$CORPUS\",\"results\":[" > "$OUT/enwik_sku_${STAMP}.json"
first=1
for sku in light gate24 champ; do
  build_cypha "$sku" || { echo "SKIP build $sku"; continue; }
  echo "=== Measuring $sku ==="
  outfile="$OUT/${sku}_${STAMP}.json"
  set +e
  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v "$MEASURE_DIR/cypha_${sku}/cyphalm_hp_sku_measure" \
      --corpus "$CORPUS" --hp-tool "$MEASURE_DIR/hp_${sku}" \
      --work-dir "$OUT/work_${sku}_${STAMP}" --latency-iters 2 \
      2> "$OUT/${sku}_${STAMP}_time_v.txt" | tee "$outfile"
  else
    "$MEASURE_DIR/cypha_${sku}/cyphalm_hp_sku_measure" \
      --corpus "$CORPUS" --hp-tool "$MEASURE_DIR/hp_${sku}" \
      --work-dir "$OUT/work_${sku}_${STAMP}" --latency-iters 2 \
      2> "$OUT/${sku}_${STAMP}_time_v.txt" | tee "$outfile"
  fi
  rc=$?
  set -e
  if [[ $first -eq 0 ]]; then echo "," >> "$OUT/enwik_sku_${STAMP}.json"; fi
  cat "$outfile" >> "$OUT/enwik_sku_${STAMP}.json"
  first=0
  if [[ "$sku" == "champ" && $rc -ne 0 ]]; then
    echo "champ measurement failed (likely OOM)" >> "$OUT/${sku}_${STAMP}_time_v.txt"
  fi
done
echo "]}" >> "$OUT/enwik_sku_${STAMP}.json"

echo "Results in $OUT/enwik_sku_${STAMP}.json"
