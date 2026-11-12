#!/bin/bash
# mt_eval.sh LABEL SPEC [extra sample flags] -- MIXDEC_TUNE part 1: the learner bits a byte on the 250 dev continuations
set -uo pipefail
L=$1; SPEC=$2; shift 2
BIN=${BIN:-$HOME/mt_bin1}
cd ~/mt_src
CYPHA_MIXDEC_EVAL=250 CYPHA_MIXDEC_ARTS=$HOME/mt_meas/arts CYPHA_MIXDEC_EVAL_ONLY=1 K5_THREADS=6 /usr/bin/time -v -o ~/mt_meas/$L.time \
  $BIN/cyphalm_trie sample /mnt/d/cypha_llm/enwik8 ~/mt_meas/$L.gen.txt --prompts-file bench/eval/gen_prompts_dev.tsv --plan-drop 75497472:100000000 \
  --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt --mixdec "$SPEC" "$@" > ~/mt_meas/$L.out 2> ~/mt_meas/$L.err
echo "$L rc=$? $(grep -h "mixdec_eval_art\|base primed" ~/mt_meas/$L.err | tr "\n" " ") $(grep -h Elapsed ~/mt_meas/$L.time)" >> ~/mt_meas/results.txt
