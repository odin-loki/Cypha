#!/bin/bash
# Compare model A with model B on held-out text (docs/TRAIN_1TB_RUNBOOK.md). Model 1 is the live run.
#   CYPHA_ENWIK8=/path/to/enwik8 bash scripts/compare_models.sh 1 2
# Held-out text: CYPHA_HOLDOUTS (FILE or FILE:FROM:LEN, space-separated), else every *.txt in <corpus>/../holdout,
# plus enwik8's 8 MiB-protocol holdout when CYPHA_ENWIK8 is set. Each row: A's and B's bits per byte (full counts and
# the packed form) and B - A with a 95% interval over 4 KiB blocks; negative means B predicts better.
# Report: <corpus>/trie/reports/compare_model<A>_model<B>.txt.
set -uo pipefail
source "$(dirname "$0")/model_common.sh"
[ $# -eq 2 ] || { echo "usage: bash scripts/compare_models.sh A B"; exit 2; }
need_number "$1" 1
need_number "$2" 1
CA=$(model_ckpt "$1")
CB=$(model_ckpt "$2")
for c in "$CA" "$CB"; do [ -f "$c" ] || { echo "no checkpoint at $c"; exit 1; }; done
HOLD=$(holdouts)
if [ -z "${HOLD// /}" ]; then
  echo "no held-out text: set CYPHA_HOLDOUTS or CYPHA_ENWIK8, or put *.txt files in $CORPUS/../holdout"
  exit 2
fi
build_checker
OUT=$REPORTS/compare_model$1_model$2.txt
echo "comparing model $1 (A) with model $2 (B)"
/tmp/cypha_trie_ckpt compare "$CA" "$CB" $HOLD 2>/dev/null | tee "$OUT"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "compare failed"; exit 1; }
echo "report: $OUT"
