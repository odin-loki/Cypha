#!/bin/bash
# mt_arm.sh BIN SRC LABEL [sample flags] -- one MIXDEC_TUNE dev arm (250 prompts); refuses to start under 16 GB available
set -uo pipefail
BIN=$1; SRC=$2; LABEL=$3; shift 3
while [ "$(free -g | awk "/^Mem:/{print \$7}")" -lt 16 ]; do sleep 60; done
echo "start $LABEL avail=$(free -g | awk "/^Mem:/{print \$7}")G $(date +%T)"
cd $SRC
export CYPHALM_TRIE=$BIN/cyphalm_trie SCRIBE=$BIN/scribe WIKI=/mnt/d/cypha_llm/enwik8 ENWIK8=/mnt/d/cypha_llm/enwik8 SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=6
/usr/bin/time -v -o ~/mt_runs/$LABEL.time scripts/gen_eval.sh dev $LABEL ~/mt_runs --prompts ${NPROMPTS:-250} --k5 /mnt/d/cypha_llm/fuzzy/train.txt "$@" > ~/mt_runs/$LABEL.log 2>&1
echo "rc=$? $LABEL $(date +%T)"
grep -E "trie samples \(pooled\)|copy32" ~/mt_runs/$LABEL.summary.md
grep -E "Elapsed|Maximum resident" ~/mt_runs/$LABEL.time
