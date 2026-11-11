#!/bin/bash
# arm.sh LABEL TRAIN [env VAR=..] -- [sample flags]
set -euo pipefail
LABEL=$1; TRAIN=$2; shift 2
cd ~/gq_src
export CYPHALM_TRIE=~/gq_bin2/cyphalm_trie SCRIBE=~/gq_bin2/scribe WIKI=/mnt/d/cypha_llm/enwik8 ENWIK8=/mnt/d/cypha_llm/enwik8 SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=${K5_THREADS:-6} K5_PLAN_LOG=~/gq_runs/$LABEL.plan.log
/usr/bin/time -v -o ~/gq_runs/$LABEL.time scripts/gen_eval.sh dev $LABEL ~/gq_runs --prompts 250 --k5 $TRAIN "$@" > ~/gq_runs/$LABEL.log 2>&1
echo done > ~/gq_runs/$LABEL.done
