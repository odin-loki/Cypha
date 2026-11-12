#!/bin/bash
# skf_arm.sh LABEL [sample flags]  -- one K5_SKELFREE dev arm; waits for >= 16 GB available first
set -uo pipefail
LABEL=$1; shift
while :; do
  av=$(free -g | awk '/^Mem:/{print $7}')
  [ "$av" -ge 16 ] && break
  sleep 30
done
echo "start $LABEL avail=${av}G $(date +%T)"
cd ~/skf_src
export CYPHALM_TRIE=$HOME/skf_bin/cyphalm_trie SCRIBE=$HOME/skf_bin/scribe WIKI=/mnt/d/cypha_llm/enwik8 ENWIK8=/mnt/d/cypha_llm/enwik8 SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=6 K5_PLAN_LOG=$HOME/skf_runs/$LABEL.plan.log
/usr/bin/time -v -o ~/skf_runs/$LABEL.time scripts/gen_eval.sh dev $LABEL ~/skf_runs --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt "$@" > ~/skf_runs/$LABEL.log 2>&1
echo "rc=$? $LABEL $(date +%T)"
grep -E "trie samples \(pooled\)|copy32" ~/skf_runs/$LABEL.summary.md
grep -E "Elapsed|Maximum resident" ~/skf_runs/$LABEL.time
grep -E "k5-skelfree" ~/skf_runs/$LABEL.log | tail -3
