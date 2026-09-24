#!/usr/bin/env bash
# Full CyphaLM vs LSTM vs Transformer comparison on one machine. Jobs run one
# at a time so every timing has the whole CPU.
#
#   bench/lm_compare/run_all.sh ENWIK8 CANT_DIR WORK [BUILD]
#
# Budget: each system trains for <= 60 min on 4 cores (CyphaLM: 11 shards +
# index, ~25 min; the NNs stop at 60 min). ~5 h end to end.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export ENWIK8=${1:?enwik8} CANT=${2:?canterbury dir} WORK=${3:?work dir}
export CYPHA_BUILD=${4:-native/build}
export WINNER=$WORK/cyphalm NN=$WORK/nn
mkdir -p "$WINNER" "$NN"
T="python3 $HERE/byte_lm.py train --data $ENWIK8 --budget-min 60"
[ -f "$NN/gpt/final.pt" ] || $T --arch gpt --d 256 --layers 4 --heads 8 --ctx 512 --batch 16 --lr 1e-3 --out "$NN/gpt" > "$NN/gpt_train.log"
[ -f "$NN/lstm/final.pt" ] || $T --arch lstm --d 512 --layers 2 --ctx 128 --batch 32 --lr 2e-3 --out "$NN/lstm" > "$NN/lstm_train.log"
if [ ! -f "$WINNER/winner.json" ]; then
    python3 "$HERE/timed.py" "$WINNER/train_time.json" \
        "$CYPHA_BUILD/cyphalm_shard_train" --train "$ENWIK8" --bytes 95000000 --shards 11 --tier slim \
        --table-bits 20 --threads 4 --out "$WINNER"
    python3 "$HERE/timed.py" "$WINNER/index_time.json" \
        "$CYPHA_BUILD/cyphalm_infinigram_build" --text "$ENWIK8" --bytes 95000000 --out "$WINNER/enwik8_95m.igr"
    cp "$HERE/../../models/cyphalm_winner/winner.json" "$WINNER/"
fi
cd "$HERE"
for stage in evals bench gens judge curve; do python3 compare.py $stage; done
python3 report.py > "$WORK/report.md"
