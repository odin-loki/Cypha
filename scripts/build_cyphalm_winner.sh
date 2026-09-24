#!/usr/bin/env bash
# Rebuild the CyphaLM winner models (docs/reports/CYPHALM_LM_QUALITY_REPORT.md,
# "Winner"): 11 slim shard models + a slim 95 MB model + an ∞-gram index over
# the same 95 MB of enwik8, plus the two serving manifests.
#
#   scripts/build_cyphalm_winner.sh ENWIK8 OUT_DIR [BUILD_DIR] [THREADS]
#
# Time on 4 cores: shards ~25 min, 95 MB model ~2.5 h, index ~20 s.
# Disk: ~4.5 GB. Serve: cyphalm_generate --load OUT_DIR/winner.json
set -euo pipefail
ENWIK8=${1:?enwik8 path}
OUT=${2:?output dir}
BUILD=${3:-native/build}
THREADS=${4:-4}
BYTES=95000000
mkdir -p "$OUT"
"$BUILD/cyphalm_infinigram_build" --text "$ENWIK8" --bytes $BYTES --out "$OUT/enwik8_95m.igr" &
"$BUILD/cyphalm_shard_train" --train "$ENWIK8" --bytes $BYTES --shards 11 --tier slim \
    --table-bits 20 --threads "$THREADS" --out "$OUT"
"$BUILD/cyphalm_lm_quality" --tier slim --train "$ENWIK8" --train-bytes $BYTES \
    --save "$OUT/slim95" --gen-bytes 0 > "$OUT/slim95_train.json"
wait
cp "$(dirname "$0")/../models/cyphalm_winner/winner.json" "$(dirname "$0")/../models/cyphalm_winner/winner_light.json" "$OUT/"
echo "built $OUT: serve with --load $OUT/winner.json (or winner_light.json)"
