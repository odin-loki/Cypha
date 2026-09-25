#!/usr/bin/env bash
# Rebuild the CyphaLM winner (docs/reports/CYPHALM_LM_QUALITY_REPORT.md,
# "Winner v4"): lean shard models on enwik8 and on PG-19 books, trained with
# the upstream mixer settings and folded by occupancy; a byte LSTM and a byte
# Transformer on enwik8 plus a copy of each fine-tuned on a wiki + books mix
# (four neural experts); a slim 95 MB model for the light manifest. The ∞-gram
# index is not stored: the manifests point at the corpus (95 MB of enwik8 +
# books), which is indexed at load.
#
#   scripts/build_cyphalm_winner.sh ENWIK8 OUT_DIR [BUILD_DIR] [THREADS]
#
# Needs network access once (PG-19 books, ~410 MB, from the public bucket;
# scripts/cyphalm_books.py checks their digests). Time on 4 cores: shards
# ~15 min, folds ~5 min, LSTM ~60 min and Transformer ~60 min (PyTorch, CPU;
# fixed byte budgets, so a busy machine only takes longer), fine-tuning ~35 min,
# 95 MB slim model ~2.5 h (light manifest only).
# Disk: ~6 GB peak. Serve: cyphalm_generate --load OUT_DIR/winner.json
set -euo pipefail
ENWIK8=${1:?enwik8 path}
OUT=${2:?output dir}
BUILD=${3:-native/build}
THREADS=${4:-4}
HERE=$(cd "$(dirname "$0")/.." && pwd)
BYTES=95000000
SHARD=8636363  # 95 MB / 11, the v3 shard size
mkdir -p "$OUT/raw"
# The manifests name OUT/enwik8. Link it unless it already is the corpus
# (ENWIK8 given as OUT/enwik8, or linked before): ln -sf onto the corpus or
# its link would leave a link to itself.
if [ ! "$OUT/enwik8" -ef "$ENWIK8" ]; then
    ln -sf "$(cd "$(dirname "$ENWIK8")" && pwd)/$(basename "$ENWIK8")" "$OUT/enwik8"
fi
# Books: books_a (289 PG-19 train books), books_b (715 more), the ∞-gram
# datastores wiki95_books.txt / wiki95_books500.txt and the experts' fine-tuning
# mix; decontaminated against the Canterbury texts used for evaluation.
python3 "$HERE/scripts/cyphalm_books.py" fetch "$OUT/books" "$ENWIK8"
# Upstream mixer gains (CompressionAlgorithm H33 / v91 / v93).
export CYPHA_HP_LR1_SCALE=40 CYPHA_HP_MIXER_SCALE=49152 CYPHA_HP_MIXER_SKIP=56 CYPHA_HP_MIXER_SKIP_L1=80
# The first three of v3's eleven 8.6 MB wiki shards (same slices).
"$BUILD/cyphalm_shard_train" --train "$ENWIK8" --bytes $((3 * SHARD)) --shards 3 --tier lean \
    --table-bits 20 --threads "$THREADS" --out "$OUT/raw"
# Book shards: the first 8.6 MB of books_a (book_0) and of books_b (book_1).
"$BUILD/cyphalm_lm_quality" --tier lean --table-bits 20 --train "$OUT/books/books_a.txt" \
    --train-bytes $SHARD --save "$OUT/raw/book_0" --gen-bytes 0 > /dev/null &
B0=$!
"$BUILD/cyphalm_lm_quality" --tier lean --table-bits 20 --train "$OUT/books/books_b.txt" \
    --train-bytes $SHARD --save "$OUT/raw/book_1" --gen-bytes 0 > /dev/null &
wait $B0 $!  # exits (set -e) if either failed
unset CYPHA_HP_LR1_SCALE CYPHA_HP_MIXER_SCALE CYPHA_HP_MIXER_SKIP CYPHA_HP_MIXER_SKIP_L1
# Fold each table to <= 80% projected occupancy and drop sentst_ (context model 16).
for m in shard_0 shard_1 shard_2 book_0 book_1; do
    "$BUILD/cyphalm_lm_quality" --load "$OUT/raw/$m.json" --fold-auto 0.8 --drop 65536 \
        --save "$OUT/$m" --gen-bytes 0 > /dev/null
done
rm -rf "$OUT/raw"
# Neural experts on the same 95 MB, stopped at the byte counts the published
# 60-minute runs reached (bench/lm_compare, docs/reports/CYPHALM_VS_NEURAL_LM.md),
# then a copy of each continued on 100 MiB of interleaved wiki and book blocks.
LM="python3 $HERE/bench/lm_compare/byte_lm.py"
MIX=(--data "$OUT/books/mix_wiki_books.txt" --data-bytes 104857600 --val-offset 104857600 --val-bytes 65536 --warmup 50)
$LM train --arch lstm --data "$ENWIK8" --d 512 --layers 2 --ctx 128 --batch 32 --lr 2e-3 \
    --budget-bytes 81137664 --out "$OUT/lstm_ckpt"
$LM export --ckpt "$OUT/lstm_ckpt/final.pt" --out "$OUT/lstm.blm"
$LM train --arch gpt --data "$ENWIK8" --d 256 --layers 4 --heads 8 --ctx 512 --batch 16 --lr 1e-3 \
    --budget-bytes 68960256 --out "$OUT/gpt_ckpt"
$LM export --ckpt "$OUT/gpt_ckpt/final.pt" --out "$OUT/gpt.bgt"
$LM train --arch lstm "${MIX[@]}" --d 512 --layers 2 --ctx 128 --batch 32 --lr 6e-4 \
    --budget-bytes 25000000 --init "$OUT/lstm_ckpt/final.pt" --out "$OUT/lstm_mix_ckpt"
$LM export --ckpt "$OUT/lstm_mix_ckpt/final.pt" --out "$OUT/lstm_mix.blm"
$LM train --arch gpt "${MIX[@]}" --d 256 --layers 4 --heads 8 --ctx 512 --batch 16 --lr 3e-4 \
    --budget-bytes 20000000 --init "$OUT/gpt_ckpt/final.pt" --out "$OUT/gpt_mix_ckpt"
$LM export --ckpt "$OUT/gpt_mix_ckpt/final.pt" --out "$OUT/gpt_mix.bgt"
"$BUILD/cyphalm_lm_quality" --tier slim --train "$ENWIK8" --train-bytes $BYTES \
    --save "$OUT/slim95" --gen-bytes 0 > "$OUT/slim95_train.json"
cp "$HERE"/models/cyphalm_winner/winner*.json "$OUT/"
echo "built $OUT: serve with --load $OUT/winner.json (winner_books.json, winner_light.json)"
