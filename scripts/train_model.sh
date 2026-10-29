#!/bin/bash
# Train model N (2 or more) on the 1 TB corpus into <corpus>/trie/model<N>/ (docs/TRAIN_1TB_RUNBOOK.md).
#   bash scripts/train_model.sh 2
# It resumes if stopped and ends when every finished corpus file is trained (done.txt). Full log:
# <corpus>/trie/reports/train_model<N>.log. Settings: CYPHA_REPO, CYPHA_CORPUS, CYPHA_CORE (CPU core, default 0),
# CYPHA_SNAPSHOTS (sizes to keep a checkpoint copy at, default 8M,64M,1G,10G,100G).
set -uo pipefail
source "$(dirname "$0")/model_common.sh"
N=${1:-}
need_number "$N" 2
CORE=${CYPHA_CORE:-0}
SNAPSHOTS=${CYPHA_SNAPSHOTS:-8M,64M,1G,10G,100G}
DIR=$(model_dir "$N")
mkdir -p "$DIR" "$REPORTS"
build_trainer
LOG=$REPORTS/train_model$N.log
echo "training model $N into $DIR (full log: $LOG)"
PIN=""; command -v taskset > /dev/null && PIN="taskset -c $CORE"
$PIN /tmp/cypha_train_model "$CORPUS" --out "$DIR" --once --snapshots "$SNAPSHOTS" 2>&1 | tee -a "$LOG" |
  grep --line-buffered -E '^(new checkpoint|resumed|trained|snapshot|done|cannot|checkpoint failed)'
if [ "${PIPESTATUS[0]}" != 0 ]; then echo "training stopped with an error; see $LOG"; exit 1; fi
echo "model $N trained. Next: bash scripts/verify_model.sh $N; bash scripts/compare_models.sh $((N - 1)) $N"
