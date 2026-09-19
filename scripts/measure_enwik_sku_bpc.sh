#!/usr/bin/env bash
# Measure enwik8.8mb observe + archive BPC for light / gate24 / champ SKUs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS="${1:-$ROOT/bench/data/enwik8/enwik8.8mb}"
BYTES=8388608
HP_SRC="$ROOT/native/third_party/hp/src/main.cpp"
HP_INC="$ROOT/native/third_party/hp/include"
XSIMD="$ROOT/native/third_party/hp/third_party/xsimd/include"
BUILD="$ROOT/native/build-sku-measure"
mkdir -p "$BUILD"

V78_FLAGS=$(grep -oE '\-DHP_[A-Z0-9_]+=[0-9]+' "$ROOT/native/third_party/hp/tools/v78_flags.ps1" \
  | grep -v HP_SLOT_MAX | tr '\n' ' ')

build_hp_cli() {
  local name=$1
  local slot=$2
  local extra=${3:-}
  local out="$BUILD/hp_${name}"
  if [[ ! -x "$out" ]]; then
    echo "Building hp_${name} (SLOT_MAX=$slot)..."
    g++ -std=c++17 -O3 -msse4.1 \
      -I "$HP_INC" -I "$XSIMD" \
      -DHP_SLOT_MAX="$slot" -DHP_XSIMD=1 \
      $V78_FLAGS $extra \
      "$HP_SRC" -o "$out"
  fi
}

build_hp_cli light 24 "-UHP_MIXER_SKIP" 2>/dev/null || \
  g++ -std=c++17 -O3 \
    -I "$HP_INC" -I "$XSIMD" \
    -DHP_SLOT_MAX=24 -DHP_XSIMD=0 \
    "$HP_SRC" -o "$BUILD/hp_light"

# light: bare features.hpp (no v78 -D flags)
if [[ ! -x "$BUILD/hp_light" ]]; then
  echo "Building hp_light (bare)..."
  g++ -std=c++17 -O3 \
    -I "$HP_INC" -I "$XSIMD" \
    -DHP_SLOT_MAX=24 -DHP_XSIMD=0 \
    "$HP_SRC" -o "$BUILD/hp_light"
fi

build_hp_cli gate24 24 ""
build_hp_cli champ 35 ""

run_archive_bpc() {
  local tool=$1
  local label=$2
  local raw="$BUILD/${label}.raw"
  local arc="$BUILD/${label}.cyhp"
  cp "$CORPUS" "$raw"
  local t0=$(date +%s%N)
  local stderr
  stderr=$("$tool" c --mem 22 --lr 2 "$raw" "$arc" 2>&1) || true
  local t1=$(date +%s%N)
  local ms=$(( (t1 - t0) / 1000000 ))
  local in_b=0 arc_b=0 whole=0 frac=0
  local bpc="nan"
  if [[ -f "$arc" ]]; then
    arc_b=$(wc -c <"$arc" | tr -d ' ')
    in_b=$BYTES
    bpc=$(python3 -c "print(f'{$arc_b * 8 / $in_b:.6f}')")
  fi
  # parse hp stderr bpc if present
  local hp_bpc
  hp_bpc=$(echo "$stderr" | grep -oP 'bpc \K[0-9]+\.[0-9]+' | tail -1 || true)
  echo "{\"sku\":\"$label\",\"archive_bytes\":${arc_b:-0},\"archive_bpc\":${bpc},\"hp_stderr_bpc\":\"${hp_bpc:-}\",\"compress_ms\":$ms,\"status\":\"$([ -f "$arc" ] && echo ok || echo fail)\"}"
  rm -f "$raw" "$arc"
}

build_cypha() {
  local profile=$1
  local dir="$BUILD/cypha_${profile}"
  if [[ ! -x "$dir/bpc_gap_measure" ]]; then
    echo "Building Cypha $profile..."
    cmake -S "$ROOT/native" -B "$dir" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=g++ -DCYPHA_HP_PROFILE="$profile" 2>&1 | tail -3
    cmake --build "$dir" --target bpc_gap_measure -j"$(nproc)" 2>&1 | tail -3
  fi
}

run_observe_bpc() {
  local profile=$1
  local dir="$BUILD/cypha_${profile}"
  local hp_tool="$BUILD/hp_${profile}"
  [[ "$profile" == "light" ]] && hp_tool="$BUILD/hp_light"
  "$dir/bpc_gap_measure" --corpus "$CORPUS" --bytes "$BYTES" \
    --hp-tool "$hp_tool" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(json.dumps({
  'sku': '$profile',
  'observe_bpc': d.get('observe_bpc'),
  'archive_bpc': d.get('archive_bpc'),
  'observe_minus_archive': d.get('observe_minus_archive_bpc'),
  'observe_ms': d.get('observe_ms'),
  'compress_ms': d.get('compress_ms'),
  'status': d.get('status'),
}))
"
}

echo "["
for sku in light gate24 champ; do
  build_cypha "$sku" || { echo "SKIP cypha $sku build failed"; continue; }
  obs=$(run_observe_bpc "$sku" 2>/dev/null || echo "{\"sku\":\"$sku\",\"status\":\"observe_fail\"}")
  arc=$(run_archive_bpc "$BUILD/hp_${sku}" "$sku" 2>/dev/null || echo "{\"sku\":\"$sku\",\"status\":\"archive_fail\"}")
  echo "$obs" | python3 -c "
import sys, json
obs = json.load(sys.stdin)
arc = json.loads('''$arc''')
obs['archive_bytes'] = arc.get('archive_bytes', 0)
if arc.get('archive_bpc') and str(arc['archive_bpc']) != 'nan':
    obs['archive_bpc'] = float(arc['archive_bpc'])
obs['compress_ms'] = arc.get('compress_ms', 0)
obs['archive_status'] = arc.get('status', 'fail')
print(json.dumps(obs))
"
  echo ","
done
echo "]"
