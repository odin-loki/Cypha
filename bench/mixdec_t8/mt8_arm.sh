#!/bin/bash
# mt8_arm.sh BIN LABEL [sample flags] -- one MIXDEC_T8 dev run (250 prompts); refuses to start under 16 GB available
set -uo pipefail
BIN=$1; LABEL=$2; shift 2
av=$(free -g | awk "/^Mem:/{print \$7}")
if [ "$av" -lt 16 ]; then echo "only ${av} GB available; not started"; exit 3; fi
echo "start $LABEL avail=${av}G $(date +%T)"
cd ~/mt8_src
export CYPHALM_TRIE=$BIN/cyphalm_trie SCRIBE=$BIN/scribe WIKI=/mnt/d/cypha_llm/enwik8 ENWIK8=/mnt/d/cypha_llm/enwik8 SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=6
/usr/bin/time -v -o ~/mt8_runs/$LABEL.time scripts/gen_eval.sh dev $LABEL ~/mt8_runs --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt "$@" > ~/mt8_runs/$LABEL.log 2>&1
echo "rc=$? $LABEL $(date +%T)"
grep -E "trie samples \(pooled\)|copy32" ~/mt8_runs/$LABEL.summary.md
grep -E "Elapsed|Maximum resident" ~/mt8_runs/$LABEL.time
