#!/bin/bash
# One planner arm of docs/reports/K5_PLANNER.md on GEN-A dev: scripts/gen_eval.sh (sample --k5, SCRIBE v2 and v3.1),
# timed, with the planner's per-prompt log. Usage: arm.sh LABEL STOREDIR MODEL full|k5   (LABEL K5 with no more args:
# today's K5). Env: BIN (the build), OUT, K5_THREADS.
set -euo pipefail
LABEL=$1; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$HOME/build_pl2}
OUT=${OUT:-/mnt/d/cypha_llm/k5plan/gen}
export CYPHALM_TRIE=$BIN/cyphalm_trie SCRIBE=$BIN/scribe WIKI=/mnt/d/cypha_llm/enwik8 ENWIK8=/mnt/d/cypha_llm/enwik8 \
       SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=${K5_THREADS:-8} K5_PLAN_LOG=$OUT/$LABEL.plan.log
mkdir -p "$OUT"
EXTRA=()
if [ $# -ge 3 ]; then EXTRA=(--k5-plan "$1" --k5-plan-model "$2" --k5-plan-gate "$3" --plan-drop 75497472:100000000); fi
/usr/bin/time -v -o "$OUT/$LABEL.time" "$ROOT/scripts/gen_eval.sh" dev "$LABEL" "$OUT" --prompts 250 \
    --k5 /mnt/d/cypha_llm/fuzzy/train.txt "${EXTRA[@]}" > "$OUT/$LABEL.log" 2>&1
echo "$LABEL done" > "$OUT/$LABEL.done"
