#!/usr/bin/env bash
# Rebuild the CyphaLM winner v2 (docs/reports/CYPHALM_LM_QUALITY_REPORT.md,
# "Winner v2"): 11 lean shard models trained with the upstream mixer settings
# and folded by occupancy, an ∞-gram index over the same 95 MB of enwik8, a
# byte LSTM neural expert, and a slim 95 MB model for the light manifest.
#
#   scripts/build_cyphalm_winner.sh ENWIK8 OUT_DIR [BUILD_DIR] [THREADS]
#
# Time on 4 cores: shards ~24 min, fold ~5 min, LSTM 60 min (PyTorch, CPU),
# 95 MB slim model ~2.5 h (light manifest only), index ~20 s.
# Disk: ~5 GB peak. Serve: cyphalm_generate --load OUT_DIR/winner.json
set -euo pipefail
ENWIK8=${1:?enwik8 path}
OUT=${2:?output dir}
BUILD=${3:-native/build}
THREADS=${4:-4}
HERE=$(cd "$(dirname "$0")/.." && pwd)
BYTES=95000000
mkdir -p "$OUT/raw"
"$BUILD/cyphalm_infinigram_build" --text "$ENWIK8" --bytes $BYTES --out "$OUT/enwik8_95m.igr" &
# Upstream mixer gains (CompressionAlgorithm H33 / v91 / v93).
CYPHA_HP_LR1_SCALE=40 CYPHA_HP_MIXER_SCALE=49152 CYPHA_HP_MIXER_SKIP=56 CYPHA_HP_MIXER_SKIP_L1=80 \
    "$BUILD/cyphalm_shard_train" --train "$ENWIK8" --bytes $BYTES --shards 11 --tier lean \
    --table-bits 20 --threads "$THREADS" --out "$OUT/raw"
# Fold each table to <= 80% projected occupancy and drop sentst_ (context model 16).
for i in $(seq 0 10); do
    "$BUILD/cyphalm_lm_quality" --load "$OUT/raw/shard_$i.json" --fold-auto 0.8 --drop 65536 \
        --save "$OUT/shard_$i" --gen-bytes 0 > /dev/null
done
rm -rf "$OUT/raw"
# Neural expert: 2 x LSTM 512 byte model, 60 min on the same data.
python3 "$HERE/bench/lm_compare/byte_lm.py" train --arch lstm --data "$ENWIK8" --d 512 --layers 2 \
    --ctx 128 --batch 32 --lr 2e-3 --budget-min 60 --out "$OUT/lstm_ckpt"
python3 "$HERE/bench/lm_compare/byte_lm.py" export --ckpt "$OUT/lstm_ckpt/final.pt" --out "$OUT/lstm.blm"
"$BUILD/cyphalm_lm_quality" --tier slim --train "$ENWIK8" --train-bytes $BYTES \
    --save "$OUT/slim95" --gen-bytes 0 > "$OUT/slim95_train.json"
wait
cp "$HERE"/models/cyphalm_winner/winner*.json "$OUT/"
echo "built $OUT: serve with --load $OUT/winner.json (4 shards; winner_full.json, winner_light.json)"
