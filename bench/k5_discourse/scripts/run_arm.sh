#!/bin/bash
# usage: run_arm.sh LABEL BIN [sample flags...]   (dev split, K5 on fuzzy/train.txt, 8 threads)
LABEL=$1; BIN=$2; shift 2
export CYPHALM_TRIE=$BIN SCRIBE=$HOME/cypha_build/disc/scribe WIKI=/mnt/d/cypha_llm/enwik8 ENWIK8=/mnt/d/cypha_llm/enwik8 SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=${K5_THREADS:-6}
cd /home/odin/disc_wt
/usr/bin/time -v -o /mnt/d/cypha_llm/disc/out/$LABEL.time scripts/gen_eval.sh dev $LABEL /mnt/d/cypha_llm/disc/out --k5 /mnt/d/cypha_llm/fuzzy/train.txt "$@" > /mnt/d/cypha_llm/disc/out/$LABEL.log 2>&1
echo DONE > /mnt/d/cypha_llm/disc/out/$LABEL.done
