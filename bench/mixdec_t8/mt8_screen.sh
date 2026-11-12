#!/bin/bash
# MIXDEC_T8.md screen measures: fact precision and hook_screen (T8 first, the baseline) for the finished arms
set -uo pipefail
cd ~/mt8_src
R=~/mt8_runs
OUT=bench/mixdec_t8/screen
mkdir -p $OUT bench/mixdec_t8/runs
arms=(T8=bench/k5_decode/runs/T8.gen.txt)
for l in T8MR T8MRJ; do [ -s $R/$l.summary.md ] && arms+=($l=$R/$l.gen.txt); done
nice python3 -I scripts/gen_fact_check.py run --v2 --out $OUT/facts "${arms[@]}" > $OUT/facts.log 2>&1
nice python3 -I bench/para_plan/hook_screen.py --surprise scripts/surprise_profile.py --out $OUT/hook "${arms[@]}" > $OUT/hook.log 2>&1
head -30 $OUT/facts/summary.md
cat $OUT/hook.md
