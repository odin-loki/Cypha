#!/usr/bin/env bash
# Shard + adapt spike for gate24 hp observe BPC (see docs/reports/CYPHALM_TRAIN_SCALE.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS="${1:-$ROOT/bench/data/canterbury/alice29.txt}"
BUILD_DIR="${CYPHA_SHARD_SPIKE_BUILD:-$ROOT/native/build-shard-spike}"
SPIKE="$BUILD_DIR/cyphalm_hp_shard_spike"
OUT_DIR="${CYPHA_SHARD_SPIKE_OUT:-$ROOT/bench/results/hp_shard_spike}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"

if [[ ! -x "$SPIKE" ]]; then
  echo "Building cyphalm_hp_shard_spike..."
  cmake -S "$ROOT/native" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc >/dev/null
  cmake --build "$BUILD_DIR" --target cyphalm_hp_shard_spike -j"$(nproc)" >/dev/null
fi

EXTRA_ARGS=()
shift || true
while [[ $# -gt 0 ]]; do
  EXTRA_ARGS+=("$1")
  shift
done

OUT_FILE="$OUT_DIR/shard_spike_${STAMP}.json"
echo "=== cyphalm_hp_shard_spike ==="
echo "corpus: $CORPUS"
"$SPIKE" --corpus "$CORPUS" --write-shards "$OUT_DIR/shards_${STAMP}" "${EXTRA_ARGS[@]}" | tee "$OUT_FILE"
echo "wrote $OUT_FILE"
