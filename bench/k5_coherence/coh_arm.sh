#!/bin/bash
# coh_arm.sh BIN LABEL [sample flags]  -- one K5_COHERENCE dev arm (gen_eval.sh); waits for >= 14 GB available first
# coh_arm.sh BIN LABEL --long [sample flags] -- the long reading: sample alone, --bytes 1800 --k5-max-words 600
set -uo pipefail
BIN=$1; LABEL=$2; shift 2
LONG=0
if [ "${1:-}" = "--long" ]; then LONG=1; shift; fi
while :; do
  av=$(free -g | awk '/^Mem:/{print $7}')
  [ "$av" -ge 14 ] && break
  sleep 30
done
echo "start $LABEL avail=${av}G $(date +%T)"
cd ~/coh_src
export CYPHALM_TRIE=$BIN/cyphalm_trie SCRIBE=$BIN/scribe WIKI=/mnt/d/cypha_llm/enwik8 ENWIK8=/mnt/d/cypha_llm/enwik8 SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=6 K5_PLAN_LOG=$HOME/coh_runs/$LABEL.plan.log
if [ $LONG = 0 ]; then
  /usr/bin/time -v -o ~/coh_runs/$LABEL.time scripts/gen_eval.sh dev $LABEL ~/coh_runs --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt "$@" > ~/coh_runs/$LABEL.log 2>&1
  echo "rc=$? $LABEL $(date +%T)"
  grep -E "trie samples \(pooled\)|copy32" ~/coh_runs/$LABEL.summary.md
else
  /usr/bin/time -v -o ~/coh_runs/$LABEL.time nice -n 5 $BIN/cyphalm_trie sample $WIKI ~/coh_runs/$LABEL.gen.txt --prompts-file bench/eval/gen_prompts_dev.tsv \
      --plan-drop 75497472:100000000 --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt --bytes 1800 --k5-max-words 600 "$@" \
      > ~/coh_runs/$LABEL.sample.out 2> ~/coh_runs/$LABEL.sample.err
  echo "rc=$? $LABEL $(date +%T)"
fi
grep -E "Elapsed|Maximum resident" ~/coh_runs/$LABEL.time
