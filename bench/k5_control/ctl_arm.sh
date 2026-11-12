#!/bin/bash
# ctl_arm.sh BIN LABEL [--long] [sample flags] -- one K5_CONTROL dev arm, 250 prompts, 6 threads.
#   short: scripts/gen_eval.sh (sample, SCRIBE v2 and v3.1, summary)
#   --long: cyphalm_trie sample alone with --bytes 1800 --k5-max-words 600 (K5_COHERENCE's long setup)
# Refuses to start under 16 GB available. Writes ~/ctl_runs/LABEL.done with the exit code when it ends.
set -uo pipefail
BIN=$1; LABEL=$2; shift 2
LONG=0
if [ "${1:-}" = "--long" ]; then LONG=1; shift; fi
R=~/ctl_runs; mkdir -p $R; rm -f $R/$LABEL.done
av=$(free -g | awk '/^Mem:/{print $7}')
if [ "$av" -lt 16 ]; then echo "only ${av} GB available; not started" > $R/$LABEL.done; exit 3; fi
echo "start $LABEL avail=${av}G $(date +%T)" > $R/$LABEL.start
cd ~/ctl_src
export CYPHALM_TRIE=$BIN/cyphalm_trie SCRIBE=$BIN/scribe WIKI=/mnt/d/cypha_llm/enwik8 ENWIK8=/mnt/d/cypha_llm/enwik8 SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=6
if [ $LONG = 0 ]; then
  /usr/bin/time -v -o $R/$LABEL.time scripts/gen_eval.sh dev $LABEL $R --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt "$@" > $R/$LABEL.log 2>&1
  rc=$?
else
  /usr/bin/time -v -o $R/$LABEL.time nice -n 5 $BIN/cyphalm_trie sample $WIKI $R/$LABEL.gen.txt --prompts-file bench/eval/gen_prompts_dev.tsv \
      --plan-drop 75497472:100000000 --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt --bytes 1800 --k5-max-words 600 "$@" \
      > $R/$LABEL.sample.out 2> $R/$LABEL.sample.err
  rc=$?
fi
echo "rc=$rc $LABEL avail_at_start=${av}G $(date +%T)" > $R/$LABEL.done
