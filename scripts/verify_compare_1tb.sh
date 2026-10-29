#!/bin/bash
# Verify the 1 TB runs and compare them (docs/TRAIN_1TB_RUNBOOK.md).
#   A = the live run's checkpoint ($CORPUS/trie/checkpoint.bin, TRI1)
#   B = the second run's checkpoint ($CORPUS/trie/v2/checkpoint.bin, TRI2), when it exists
# Holdouts: CYPHA_HOLDOUTS (space-separated FILE[:FROM:LEN] specs), else every *.txt in $CORPUS/../holdout,
# plus enwik8's 8 MiB-protocol holdout when CYPHA_ENWIK8 names the enwik8 file. Scores only text neither run trained
# on if the holdout files sit outside the corpus. CYPHA_DEEP=1 also re-reads the corpus to check the bytes seen (slow).
set -uo pipefail
REPO=${CYPHA_REPO:-/mnt/c/Users/odinl/OneDrive/Desktop/Cypha}
CORPUS=${CYPHA_CORPUS:-/mnt/d/cypha_llm/corpus}
OUT=${CYPHA_OUT:-$CORPUS/trie/compare_$(date +%Y%m%d_%H%M)}
cd "$REPO" || exit 1
g++ -std=c++23 -O2 -I native/include -I native/third_party -o /tmp/trie_ckpt native/tools/trie_ckpt.cpp || exit 1
/tmp/trie_ckpt selftest || { echo "trie_ckpt selftest failed: not using it"; exit 1; }
mkdir -p "$OUT"
A=$CORPUS/trie/checkpoint.bin
B=$CORPUS/trie/v2/checkpoint.bin
DEEP=""; [ "${CYPHA_DEEP:-0}" = 1 ] && DEEP="--deep"
echo "== A: $A" | tee "$OUT/verify_A.txt"
/tmp/trie_ckpt verify "$A" --corpus "$CORPUS" $DEEP | tee -a "$OUT/verify_A.txt"
HAVE_B=0
if [ -f "$B" ]; then
  HAVE_B=1
  echo "== B: $B" | tee "$OUT/verify_B.txt"
  /tmp/trie_ckpt verify "$B" --corpus "$CORPUS" $DEEP | tee -a "$OUT/verify_B.txt"
fi
HOLD=${CYPHA_HOLDOUTS:-}
if [ -z "$HOLD" ] && [ -d "$CORPUS/../holdout" ]; then
  for f in "$CORPUS"/../holdout/*.txt; do [ -f "$f" ] && HOLD="$HOLD $f"; done
fi
[ -n "${CYPHA_ENWIK8:-}" ] && HOLD="$HOLD ${CYPHA_ENWIK8}:7340032:1048576"
if [ -z "${HOLD// /}" ]; then
  echo "no holdout files: set CYPHA_HOLDOUTS or CYPHA_ENWIK8 (see docs/TRAIN_1TB_RUNBOOK.md)" | tee "$OUT/compare.txt"
  exit 0
fi
if [ "$HAVE_B" = 1 ]; then
  /tmp/trie_ckpt compare "$A" "$B" $HOLD | tee "$OUT/compare.txt"
else
  /tmp/trie_ckpt bpc "$A" $HOLD | tee "$OUT/bpc_A.txt"
fi
echo "reports in $OUT"
