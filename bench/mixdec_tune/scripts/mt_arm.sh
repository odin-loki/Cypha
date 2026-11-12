#!/bin/bash
# mt_arm.sh BIN SRC LABEL [sample flags] -- one MIXDEC_TUNE dev arm (250 prompts). Takes a shared heavy slot
# (/tmp/cypha_heavy_{0,1}.lock), then waits inside it for 16 GB available. POST=1: the GEN file exists; SCRIBE and summary only.
set -uo pipefail
BIN=$1; SRC=$2; LABEL=$3; shift 3
export CYPHALM_TRIE=$BIN/cyphalm_trie SCRIBE=$BIN/scribe WIKI=/mnt/d/cypha_llm/enwik8 ENWIK8=/mnt/d/cypha_llm/enwik8 SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=6
export BIN SRC LABEL
LK=/tmp/cypha_heavy_$((RANDOM%2)).lock
~/heavy.sh bash -c '
  av() { free -g | awk "/^Mem:/{print \$7}"; }
  while [ "$(av)" -lt 16 ]; do sleep 60; done
  echo "start $LABEL avail=$(av)G lock=$0 $(date +%T)"
  cd $SRC
  if [ "${POST:-0}" = 1 ]; then
    E=/mnt/d/cypha_llm/enwik8; G=~/mt_runs/$LABEL.gen.txt; O=~/mt_runs
    /usr/bin/time -v -o $O/$LABEL.post.time bash -c "$SCRIBE ladder --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:56:64 --vectors $SCRIBE_VEC --profile v2 --seed 20261027 --gen-trie $G --label $LABEL > $O/$LABEL.scribe.md 2> $O/$LABEL.scribe.err; $SCRIBE ladder --pack enwik:$E:16:48 --ref enwik:$E:48:56 --test enwik:$E:56:64 --vectors $SCRIBE_VEC --profile v31 --seed 20261027 --gen-trie $G --label $LABEL > $O/$LABEL.scribe31.md 2> $O/$LABEL.scribe31.err; python3 -I scripts/gen_eval_summary.py $O $LABEL dev > $O/$LABEL.summary.md"
  else
    /usr/bin/time -v -o ~/mt_runs/$LABEL.time scripts/gen_eval.sh dev $LABEL ~/mt_runs --prompts ${NPROMPTS:-250} --k5 /mnt/d/cypha_llm/fuzzy/train.txt "$@" > ~/mt_runs/$LABEL.log 2>&1
  fi
  echo "rc=$? $LABEL $(date +%T)"
' heavy "$@"
grep -E "trie samples \(pooled\)|copy32" ~/mt_runs/$LABEL.summary.md
grep -E "Elapsed|Maximum resident" ~/mt_runs/$LABEL.time 2>/dev/null
