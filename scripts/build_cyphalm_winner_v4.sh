#!/usr/bin/env bash
# Rebuild the CyphaLM winner (docs/reports/lm_quality/v4/NOTES.md): lean shard
# models on enwik8 and on PG-19 books, trained with the upstream mixer settings
# and folded by occupancy; a byte LSTM and a byte Transformer on enwik8 plus a
# copy of each fine-tuned on a wiki + books mix (four neural experts); a slim
# 95 MB model for the light manifest. The ∞-gram index is not stored: the
# manifests point at the corpus, which is indexed at load.
#
#   scripts/build_cyphalm_winner_v4.sh ENWIK8 OUT_DIR [BUILD_DIR] [THREADS]
#
# Needs network access once (PG-19 books, ~410 MB). Time on 4 cores: shards
# ~15 min, folds ~5 min, LSTM ~60 min and Transformer ~60 min (PyTorch, CPU),
# fine-tuning ~35 min, 95 MB slim model ~2.5 h (light manifest only).
# Disk: ~6 GB peak. Serve: cyphalm_generate --load OUT_DIR/winner.json
set -euo pipefail
ENWIK8=${1:?enwik8 path}
OUT=${2:?output dir}
BUILD=${3:-native/build}
THREADS=${4:-4}
PYTHON=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")/.." && pwd)
BYTES=95000000
SHARD=8636363  # 95 MB / 11, the v3 shard size
mkdir -p "$OUT/raw"
if [ ! "$OUT/enwik8" -ef "$ENWIK8" ]; then
    ln -sf "$(cd "$(dirname "$ENWIK8")" && pwd)/$(basename "$ENWIK8")" "$OUT/enwik8"
fi
echo "fetching PG-19 books"
"$PYTHON" "$HERE/scripts/cyphalm_books.py" fetch "$OUT/books" "$ENWIK8"
echo "training wiki shards"
export CYPHA_HP_LR1_SCALE=40 CYPHA_HP_MIXER_SCALE=49152 CYPHA_HP_MIXER_SKIP=56 CYPHA_HP_MIXER_SKIP_L1=80
"$BUILD/cyphalm_shard_train" --train "$ENWIK8" --bytes $((3 * SHARD)) --shards 3 --tier lean \
    --table-bits 20 --threads "$THREADS" --out "$OUT/raw"
echo "training book shards"
"$BUILD/cyphalm_lm_quality" --tier lean --table-bits 20 --train "$OUT/books/books_a.txt" \
    --train-bytes $SHARD --save "$OUT/raw/book_0" --gen-bytes 0 > /dev/null &
B0=$!
"$BUILD/cyphalm_lm_quality" --tier lean --table-bits 20 --train "$OUT/books/books_b.txt" \
    --train-bytes $SHARD --save "$OUT/raw/book_1" --gen-bytes 0 > /dev/null &
wait $B0 $!
unset CYPHA_HP_LR1_SCALE CYPHA_HP_MIXER_SCALE CYPHA_HP_MIXER_SKIP CYPHA_HP_MIXER_SKIP_L1
echo "folding shards"
for m in shard_0 shard_1 shard_2 book_0 book_1; do
    "$BUILD/cyphalm_lm_quality" --load "$OUT/raw/$m.json" --fold-auto 0.8 --drop 65536 \
        --save "$OUT/$m" --gen-bytes 0 > /dev/null
done
rm -rf "$OUT/raw"
LM="$PYTHON $HERE/bench/lm_compare/byte_lm.py"
MIX=(--data "$OUT/books/mix_wiki_books.txt" --data-bytes 104857600 --val-offset 104857600 --val-bytes 65536 --warmup 50)
echo "training LSTM"
$LM train --arch lstm --data "$ENWIK8" --d 512 --layers 2 --ctx 128 --batch 32 --lr 2e-3 \
    --budget-bytes 81137664 --out "$OUT/lstm_ckpt"
$LM export --ckpt "$OUT/lstm_ckpt/final.pt" --out "$OUT/lstm.blm"
echo "training Transformer"
$LM train --arch gpt --data "$ENWIK8" --d 256 --layers 4 --heads 8 --ctx 512 --batch 16 --lr 1e-3 \
    --budget-bytes 68960256 --out "$OUT/gpt_ckpt"
$LM export --ckpt "$OUT/gpt_ckpt/final.pt" --out "$OUT/gpt.bgt"
echo "fine-tuning experts on wiki+books"
$LM train --arch lstm "${MIX[@]}" --d 512 --layers 2 --ctx 128 --batch 32 --lr 6e-4 \
    --budget-bytes 25000000 --init "$OUT/lstm_ckpt/final.pt" --out "$OUT/lstm_mix_ckpt"
$LM export --ckpt "$OUT/lstm_mix_ckpt/final.pt" --out "$OUT/lstm_mix.blm"
$LM train --arch gpt "${MIX[@]}" --d 256 --layers 4 --heads 8 --ctx 512 --batch 16 --lr 3e-4 \
    --budget-bytes 20000000 --init "$OUT/gpt_ckpt/final.pt" --out "$OUT/gpt_mix_ckpt"
$LM export --ckpt "$OUT/gpt_mix_ckpt/final.pt" --out "$OUT/gpt_mix.bgt"
echo "training slim95"
"$BUILD/cyphalm_lm_quality" --tier slim --train "$ENWIK8" --train-bytes $BYTES \
    --save "$OUT/slim95" --gen-bytes 0 > "$OUT/slim95_train.json"
# c1: 3 wiki shards + book_0, wiki95+books_a, four experts.
# light: slim95 + book_0, same index and experts.
# books: 2 wiki + both book shards, wiki95+books_a+books_b.
"$PYTHON" - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
neural = ["lstm.blm", "gpt.bgt", "lstm_mix.blm", "gpt_mix.bgt"]
common = {
    "cyphalm_ensemble": 1,
    "learning_rate": 0.01,
    "neural": neural,
    "neural_learning_rate": 0.05,
    "neural_adapt": 0.002,
}
def write(name, members, index):
    meta = dict(common)
    meta["members"] = [{"checkpoint": m + ".json"} for m in members]
    meta["infinigram"] = index
    meta["note"] = "Winner v4 (" + name + "). Built by scripts/build_cyphalm_winner_v4.sh."
    with open(out + "/" + name + ".json", "w", newline="\n") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
write("winner", ["shard_0", "shard_1", "shard_2", "book_0"], "books/wiki95_books.txt")
write("winner_light", ["slim95", "book_0"], "books/wiki95_books.txt")
write("winner_books", ["shard_0", "shard_1", "book_0", "book_1"], "books/wiki95_books500.txt")
PY
echo "built $OUT: serve with --load $OUT/winner.json (winner_books.json, winner_light.json)"
