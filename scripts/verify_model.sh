#!/bin/bash
# Verify models by number (docs/TRAIN_1TB_RUNBOOK.md): the file, its structure, whether 32-bit counts wrapped,
# and that every finished corpus file was trained. Model 1 is the live run.
#   bash scripts/verify_model.sh 1
#   bash scripts/verify_model.sh 1 2
# Reports: <corpus>/trie/reports/verify_model<N>.txt. CYPHA_DEEP=1 also re-reads the corpus to check the bytes seen
# (hours). Exit code 0 only if every model given is VERIFIED.
set -uo pipefail
source "$(dirname "$0")/model_common.sh"
[ $# -ge 1 ] || { echo "usage: bash scripts/verify_model.sh N [N ...]"; exit 2; }
for n in "$@"; do need_number "$n" 1; done
build_checker
DEEP=""; [ "${CYPHA_DEEP:-0}" = 1 ] && DEEP="--deep"
rc=0
for n in "$@"; do
  C=$(model_ckpt "$n")
  echo "== model $n ($C)"
  if [ ! -f "$C" ]; then
    echo "no model $n at $C" | tee "$REPORTS/verify_model$n.txt"
    rc=1
    continue
  fi
  /tmp/cypha_trie_ckpt verify "$C" --corpus "$CORPUS" $DEEP | tee "$REPORTS/verify_model$n.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
done
exit $rc
