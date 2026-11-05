#!/bin/bash
# One writing arm of docs/reports/K5_STRUCTURE.md on GEN-A dev (SPLIT=test for the one test read): scripts/gen_eval.sh
# (sample --k5, SCRIBE v2 and v3.1), timed. Usage: arm.sh LABEL [cyphalm_trie sample flags ...]
# e.g. arm.sh W1_s0 --k5-struct /mnt/d/cypha_llm/k5struct/S200.R.chosen.model
set -euo pipefail
LABEL=$1; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$HOME/cypha_build/k5s}
OUT=${OUT:-/mnt/d/cypha_llm/k5struct/gen}
export CYPHALM_TRIE=$BIN/cyphalm_trie SCRIBE=$BIN/scribe WIKI=/mnt/d/cypha_llm/enwik8 ENWIK8=/mnt/d/cypha_llm/enwik8 \
       SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=${K5_THREADS:-8}
mkdir -p "$OUT"
/usr/bin/time -v -o "$OUT/$LABEL.time" "$ROOT/scripts/gen_eval.sh" "${SPLIT:-dev}" "$LABEL" "$OUT" --prompts 250 \
    --k5 /mnt/d/cypha_llm/fuzzy/train.txt "$@" > "$OUT/$LABEL.log" 2>&1
grep -m1 "^| trie samples (pooled)" "$OUT/$LABEL.scribe.md" | awk -F'|' -v l="$LABEL" '{print l, "v2", $4}'
grep -m1 "^| trie samples (pooled)" "$OUT/$LABEL.scribe31.md" | awk -F'|' -v l="$LABEL" '{print l, "v3.1", $4}' || true
grep -h "GEN_EVAL copy32" "$OUT/$LABEL.sample.out" || true
grep -h "generated" "$OUT/$LABEL.sample.err" || true
