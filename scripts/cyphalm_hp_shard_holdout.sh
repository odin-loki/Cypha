#!/usr/bin/env bash
# Holdout shard-merge eval: train shards on prefix, eval merged vs single-stream on holdout tail.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${CYPHA_SHARD_SPIKE_BUILD:-$ROOT/native/build-shard-holdout}"
SPIKE="$BUILD_DIR/cyphalm_hp_shard_spike"
OUT_DIR="${CYPHA_SHARD_HOLDOUT_OUT:-$ROOT/bench/results/hp_shard_holdout}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"

CORPUS="${1:-$ROOT/bench/data/canterbury/alice29.txt}"
MAX_BYTES=0
TABLE_BITS=16
HOLDOUT_FRAC=0.2
SHARDS=2
EXTRA=()
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-bytes) MAX_BYTES="$2"; shift 2 ;;
    --table-bits) TABLE_BITS="$2"; shift 2 ;;
    --holdout-frac) HOLDOUT_FRAC="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ ! -x "$SPIKE" ]]; then
  echo "Building cyphalm_hp_shard_spike..."
  cmake -S "$ROOT/native" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc >/dev/null
  cmake --build "$BUILD_DIR" --target cyphalm_hp_shard_spike -j"$(nproc)" >/dev/null
fi

ARGS=(--corpus "$CORPUS" --holdout-frac "$HOLDOUT_FRAC" --shards "$SHARDS" --table-bits "$TABLE_BITS")
if [[ "$MAX_BYTES" -gt 0 ]]; then
  ARGS+=(--max-bytes "$MAX_BYTES")
fi
ARGS+=("${EXTRA[@]}")

OUT_FILE="$OUT_DIR/shard_holdout_${STAMP}.json"
echo "=== cyphalm_hp_shard_holdout ==="
echo "corpus: $CORPUS max_bytes=$MAX_BYTES table_bits=$TABLE_BITS holdout_frac=$HOLDOUT_FRAC"
"$SPIKE" "${ARGS[@]}" | tee "$OUT_FILE"

DOCS_JSON="$ROOT/docs/reports/CYPHALM_SHARD_HOLDOUT.json"
cp "$OUT_FILE" "$DOCS_JSON"
echo "copied to $DOCS_JSON"
