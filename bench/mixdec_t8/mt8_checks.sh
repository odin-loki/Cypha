#!/bin/bash
# MIXDEC_T8 step 1: the merged build against T8, K1, MR and MRJ (cmp of GEN lines, MODEL aside)
B=~/mt8_bin; R=~/mt8_runs; S=~/mt8_src
cmpg() { cmp <(grep -v "^MODEL" $1) <(grep -v "^MODEL" $2) && echo "$3 CMP_EQUAL" || echo "$3 DIFFERS"; }
while ! ~/mt8_arm.sh $B C_T8; do sleep 60; done
cmpg $R/C_T8.gen.txt $S/bench/k5_decode/runs/T8.gen.txt C_T8
while ! ~/mt8_arm.sh $B C_K1 --temps 1.0; do sleep 60; done
cmpg $R/C_K1.gen.txt $S/bench/k5_retrieve/runs/K1.gen.txt C_K1
git -C $S show gh/wip/k5-mixdec:bench/k5_mixdec/runs/MR.gen.txt > $R/D4_MR.gen.txt
git -C $S show gh/wip/k5-mixdec:bench/k5_mixdec/runs/MRJ.gen.txt > $R/D4_MRJ.gen.txt
while ! ~/mt8_arm.sh $B C_MR --temps 1.0 --mixdec ratio=0.3,copy=0; do sleep 60; done
cmpg $R/C_MR.gen.txt $R/D4_MR.gen.txt C_MR
while ! ~/mt8_arm.sh $B C_MRJ --temps 1.0 --mixdec ratio=0.3,judge=1,copy=0; do sleep 60; done
cmpg $R/C_MRJ.gen.txt $R/D4_MRJ.gen.txt C_MRJ
touch $R/checks.done
echo checks done
