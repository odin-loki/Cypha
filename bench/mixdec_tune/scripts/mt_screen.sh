#!/bin/bash
# MIXDEC_TUNE screen measures: fact precision and hook_screen (T8MR first, the baseline; T8 beside it) for the finished arms
set -uo pipefail
cd ~/mt_src
R=~/mt_runs
OUT=bench/mixdec_tune/screen
mkdir -p $OUT
arms=(T8MR=$R/T8MR.gen.txt T8=bench/k5_decode/runs/T8.gen.txt)
for l in TL TL5 TL2 TM5; do [ -s $R/$l.summary.md ] && arms+=($l=$R/$l.gen.txt); done
nice python3 -I scripts/gen_fact_check.py run --v2 --out $OUT/facts "${arms[@]}" > $OUT/facts.log 2>&1
nice python3 -I bench/para_plan/hook_screen.py --surprise scripts/surprise_profile.py --out $OUT/hook "${arms[@]}" > $OUT/hook.log 2>&1
head -30 $OUT/facts/summary.md
cat $OUT/hook.md
