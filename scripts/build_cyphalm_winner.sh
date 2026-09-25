#!/usr/bin/env bash
# Rebuild the CyphaLM winner (docs/reports/CYPHALM_LM_QUALITY_REPORT.md,
# "Winner v3"): 11 lean shard models trained with the upstream mixer settings
# and folded by occupancy, a byte LSTM and a byte Transformer as neural
# experts, and a slim 95 MB model for the light manifest. The ∞-gram index
# is not stored: the manifests point at the corpus, which is indexed at load.
#
#   scripts/build_cyphalm_winner.sh ENWIK8 OUT_DIR [BUILD_DIR] [THREADS]
#
# Time on 4 cores: shards ~24 min, fold ~5 min, LSTM ~60 min and Transformer
# ~60 min (PyTorch, CPU; fixed byte budgets, so a busy machine only takes
# longer), 95 MB slim model ~2.5 h (light manifest only).
# Disk: ~5 GB peak. Serve: cyphalm_generate --load OUT_DIR/winner.json
set -euo pipefail
ENWIK8=${1:?enwik8 path}
OUT=${2:?output dir}
BUILD=${3:-native/build}
THREADS=${4:-4}
HERE=$(cd "$(dirname "$0")/.." && pwd)
BYTES=95000000
mkdir -p "$OUT/raw"
# The manifests name OUT/enwik8. Link it unless it already is the corpus
# (ENWIK8 given as OUT/enwik8, or linked before): ln -sf onto the corpus or
# its link would leave a link to itself.
if [ ! "$OUT/enwik8" -ef "$ENWIK8" ]; then
    ln -sf "$(cd "$(dirname "$ENWIK8")" && pwd)/$(basename "$ENWIK8")" "$OUT/enwik8"
fi
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
# Neural experts on the same 95 MB, stopped at the byte counts the published
# 60-minute runs reached (bench/lm_compare, docs/reports/CYPHALM_VS_NEURAL_LM.md).
LM="python3 $HERE/bench/lm_compare/byte_lm.py"
$LM train --arch lstm --data "$ENWIK8" --d 512 --layers 2 --ctx 128 --batch 32 --lr 2e-3 \
    --budget-bytes 81137664 --out "$OUT/lstm_ckpt"
$LM export --ckpt "$OUT/lstm_ckpt/final.pt" --out "$OUT/lstm.blm"
$LM train --arch gpt --data "$ENWIK8" --d 256 --layers 4 --heads 8 --ctx 512 --batch 16 --lr 1e-3 \
    --budget-bytes 68960256 --out "$OUT/gpt_ckpt"
$LM export --ckpt "$OUT/gpt_ckpt/final.pt" --out "$OUT/gpt.bgt"
"$BUILD/cyphalm_lm_quality" --tier slim --train "$ENWIK8" --train-bytes $BYTES \
    --save "$OUT/slim95" --gen-bytes 0 > "$OUT/slim95_train.json"
cp "$HERE"/models/cyphalm_winner/winner*.json "$OUT/"
echo "built $OUT: serve with --load $OUT/winner.json (4 shards; winner_full.json, winner_light.json)"
