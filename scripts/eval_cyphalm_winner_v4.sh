#!/usr/bin/env bash
# Score the v4 manifests written by scripts/build_cyphalm_winner_v4.sh.
# Tuning slices choose the mix; test slices are reported separately.
# 16 KiB each, matching docs/reports/lm_quality/v4/NOTES.md.
#
#   scripts/eval_cyphalm_winner_v4.sh OUT_DIR [BUILD_DIR]
set -euo pipefail
OUT=${1:?output dir}
BUILD=${2:-native/build}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python3}
EVAL="$BUILD/cyphalm_lm_quality"
N=16384
mkdir -p "$OUT/eval"
ALICE="$HERE/bench/data/canterbury/alice29.txt"
LCET="$HERE/bench/data/canterbury/lcet10.txt"
WIKI="$OUT/enwik8"
PG_TUNE="$OUT/books/pg19/33756.txt"
PG_TEST="$OUT/books/pg19/9931.txt"

one() {
    local tag=$1 manifest=$2 path=$3 offset=$4
    shift 4
    local dest="$OUT/eval/${tag}.json"
    if [ -f "$dest" ]; then
        echo "skip $tag"
        return 0
    fi
    echo "eval $tag"
    "$EVAL" --load "$OUT/$manifest" --eval "$path" --eval-offset "$offset" --eval-bytes "$N" \
        --gen-bytes 0 "$@" > "$dest"
    "$PYTHON" - "$dest" "$tag" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
e = d["eval"]
print(f"{sys.argv[2]}\t{e['nll_bits_per_byte']:.4f}\t{e.get('ms_per_byte')}", flush=True)
PY
}

# Tuning sweep on winner.json (c1, four experts).
for spec in \
    "tune_none wiki $WIKI 97000000" \
    "tune_none alice $ALICE 100000" \
    "tune_none lcet $LCET 200000" \
    "tune_none pg $PG_TUNE 100000"
do
    set -- $spec
    one "$1_$2" winner.json "$3" "$4"
done
for spec in \
    "tune_t09 wiki $WIKI 97000000" \
    "tune_t09 alice $ALICE 100000" \
    "tune_t09 lcet $LCET 200000" \
    "tune_t09 pg $PG_TUNE 100000"
do
    set -- $spec
    one "$1_$2" winner.json "$3" "$4" --final-temp 0.9
done
for spec in \
    "tune_t09_lr01 wiki $WIKI 97000000" \
    "tune_t09_lr01 alice $ALICE 100000" \
    "tune_t09_lr01 lcet $LCET 200000" \
    "tune_t09_lr01 pg $PG_TUNE 100000"
do
    set -- $spec
    one "$1_$2" winner.json "$3" "$4" --final-temp 0.9 --final-temp-lr 0.01
done
for spec in \
    "tune_log wiki $WIKI 97000000" \
    "tune_log alice $ALICE 100000" \
    "tune_log lcet $LCET 200000" \
    "tune_log pg $PG_TUNE 100000"
do
    set -- $spec
    one "$1_$2" winner.json "$3" "$4" --neural-mix log
done
for spec in \
    "tune_switch wiki $WIKI 97000000" \
    "tune_switch alice $ALICE 100000" \
    "tune_switch lcet $LCET 200000" \
    "tune_switch pg $PG_TUNE 100000"
do
    set -- $spec
    one "$1_$2" winner.json "$3" "$4" --neural-mix switch
done
for spec in \
    "tune_gate wiki $WIKI 97000000" \
    "tune_gate alice $ALICE 100000" \
    "tune_gate lcet $LCET 200000" \
    "tune_gate pg $PG_TUNE 100000"
do
    set -- $spec
    one "$1_$2" winner.json "$3" "$4" --ensemble-gate
done
for spec in \
    "tune_longest16 wiki $WIKI 97000000" \
    "tune_longest16 alice $ALICE 100000" \
    "tune_longest16 lcet $LCET 200000" \
    "tune_longest16 pg $PG_TUNE 100000"
do
    set -- $spec
    one "$1_$2" winner.json "$3" "$4" --infinigram-mode longest16
done

# Test slices: plain manifests, then the same with final temperature 0.9.
for manifest in winner winner_light winner_books; do
    for spec in \
        "wiki $WIKI 96000000" \
        "alice $ALICE 20000" \
        "lcet $LCET 50000" \
        "pg $PG_TEST 100000"
    do
        set -- $spec
        one "test_${manifest}_$1" "$manifest.json" "$2" "$3"
        one "test_${manifest}_t09_$1" "$manifest.json" "$2" "$3" --final-temp 0.9
    done
done

echo "eval done: $OUT/eval"
