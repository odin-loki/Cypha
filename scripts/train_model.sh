#!/bin/bash
# Train model N on the 1 TB corpus, verify it, and compare it with the model before it (docs/TRAIN_1TB_RUNBOOK.md).
#
#   bash scripts/train_model.sh          # model 2, compared with model 1
#   bash scripts/train_model.sh 3        # model 3, compared with model 2
#
# Model 1 is the live run: <corpus>/trie/checkpoint.bin (scripts/run_train.sh and trie_stream_train.cpp, unchanged).
# Model N (2 or more): <corpus>/trie/model<N>/ (checkpoint.bin, trained.txt, snapshots, done.txt, report/).
# Training resumes if stopped and ends when every finished corpus file is trained. Then the script verifies model N,
# verifies the model before it, and compares the two on held-out text. Reports: <corpus>/trie/model<N>/report/.
#
# Settings (environment): CYPHA_REPO, CYPHA_CORPUS, CYPHA_CORE (the CPU core), CYPHA_SNAPSHOTS (sizes to keep a
# copy of the checkpoint at), CYPHA_HOLDOUTS (FILE or FILE:FROM:LEN, space-separated; default every *.txt in
# <corpus>/../holdout), CYPHA_ENWIK8 (adds enwik8's 8 MiB-protocol holdout), CYPHA_DEEP=1 (re-read the corpus to
# check the bytes seen; slow).
set -uo pipefail
N=${1:-2}
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 2 ]; then
  echo "usage: bash scripts/train_model.sh [N]   (N >= 2; model 1 is the live run)"
  exit 2
fi
REPO=${CYPHA_REPO:-/mnt/c/Users/odinl/OneDrive/Desktop/Cypha}
CORPUS=${CYPHA_CORPUS:-/mnt/d/cypha_llm/corpus}
CORE=${CYPHA_CORE:-0}
SNAPSHOTS=${CYPHA_SNAPSHOTS:-8M,64M,1G,10G,100G}
model_ckpt() { if [ "$1" = 1 ]; then echo "$CORPUS/trie/checkpoint.bin"; else echo "$CORPUS/trie/model$1/checkpoint.bin"; fi; }
DIR=$CORPUS/trie/model$N
REP=$DIR/report
PREV=$((N - 1))
PREV_CKPT=$(model_ckpt "$PREV")
mkdir -p "$REP"
cd "$REPO" || exit 1

echo "== build"
g++ -std=c++23 -O2 -I native/include -I native/third_party -o /tmp/cypha_train_model native/tools/trie_stream_train_v2.cpp || exit 1
g++ -std=c++23 -O2 -I native/include -I native/third_party -o /tmp/cypha_trie_ckpt native/tools/trie_ckpt.cpp || exit 1
/tmp/cypha_trie_ckpt selftest > "$REP/selftest.txt" 2>&1 || { cat "$REP/selftest.txt"; echo "selftest failed"; exit 1; }
tail -1 "$REP/selftest.txt"

DEEP=""; [ "${CYPHA_DEEP:-0}" = 1 ] && DEEP="--deep"

echo "== verify model $PREV ($PREV_CKPT)"
if [ -f "$PREV_CKPT" ]; then
  /tmp/cypha_trie_ckpt verify "$PREV_CKPT" --corpus "$CORPUS" $DEEP | tee "$REP/verify_model$PREV.txt"
else
  echo "no model $PREV yet" | tee "$REP/verify_model$PREV.txt"
fi

echo "== train model $N ($DIR); full log: $REP/train.log"
PIN=""; command -v taskset > /dev/null && PIN="taskset -c $CORE"
$PIN /tmp/cypha_train_model "$CORPUS" --out "$DIR" --once --snapshots "$SNAPSHOTS" 2>&1 | tee -a "$REP/train.log" |
  grep --line-buffered -E '^(new checkpoint|resumed|trained|snapshot|done|cannot|checkpoint failed)'
if [ "${PIPESTATUS[0]}" != 0 ]; then echo "training stopped with an error; see $REP/train.log"; exit 1; fi

echo "== verify model $N"
/tmp/cypha_trie_ckpt verify "$DIR/checkpoint.bin" --corpus "$CORPUS" $DEEP | tee "$REP/verify_model$N.txt"

HOLD=${CYPHA_HOLDOUTS:-}
if [ -z "$HOLD" ] && [ -d "$CORPUS/../holdout" ]; then
  for f in "$CORPUS"/../holdout/*.txt; do [ -f "$f" ] && HOLD="$HOLD $f"; done
fi
[ -n "${CYPHA_ENWIK8:-}" ] && HOLD="$HOLD ${CYPHA_ENWIK8}:7340032:1048576"
if [ -z "${HOLD// /}" ]; then
  echo "== no held-out text: set CYPHA_HOLDOUTS or CYPHA_ENWIK8 to compare (docs/TRAIN_1TB_RUNBOOK.md)"
  exit 0
fi
echo "== compare model $PREV (A) with model $N (B)"
if [ -f "$PREV_CKPT" ]; then
  /tmp/cypha_trie_ckpt compare "$PREV_CKPT" "$DIR/checkpoint.bin" $HOLD 2>/dev/null | tee "$REP/compare_model${PREV}_model${N}.txt"
else
  /tmp/cypha_trie_ckpt bpc "$DIR/checkpoint.bin" $HOLD 2>/dev/null | tee "$REP/bpc_model$N.txt"
fi
echo "== reports in $REP"
