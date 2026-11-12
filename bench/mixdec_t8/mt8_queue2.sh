#!/bin/bash
# MIXDEC_T8: T8MRJ again (WSL restarted during its first run), then the screen
B=~/mt8_bin
while ! ~/mt8_arm.sh $B T8MRJ --mixdec ratio=0.3,judge=1,copy=0; do sleep 60; done
~/mt8_screen.sh
touch ~/mt8_runs/arms.done
echo queue done
