#!/bin/bash
B=$HOME/mt_bin4; S=$HOME/mt_src
L="cls=1,bkn=10:4:2:1,ngo=1:2:3:4:5:6:7:8:12:16,ngcap=8,nglr=0.000625,mix2=2,ngpre=8388608,ngtb=20,artk=4,art=32768"
[ -s ~/mt_runs/OFF.summary.md ] || ~/mt_arm.sh $B $S OFF
[ -s ~/mt_runs/T8MR.summary.md ] || ~/mt_arm.sh $B $S T8MR --mixdec ratio=0.3,copy=0
[ -s ~/mt_runs/TL.summary.md ] || ~/mt_arm.sh $B $S TL --mixdec ratio=0.3,copy=0,$L
[ -s ~/mt_runs/TM5.summary.md ] || ~/mt_arm.sh $B $S TM5 --mixdec ratio=0.5,copy=0
[ -s ~/mt_runs/TL5.summary.md ] || ~/mt_arm.sh $B $S TL5 --mixdec ratio=0.5,copy=0,$L
[ -s ~/mt_runs/TL2.summary.md ] || ~/mt_arm.sh $B $S TL2 --mixdec ratio=0.2,copy=0,$L
echo done > ~/mt_runs/arms.done
