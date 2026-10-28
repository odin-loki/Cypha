#!/bin/bash
# Hard stop of the baseline queue at the deadline (epoch seconds in bl/deadline).
SP=/tmp/claude-0/-home-user-Cypha/36c53d01-6fbd-5b4b-903b-8b30032ddc8e/scratchpad
DL=$(cat $SP/bl/deadline)
while [ $(date +%s) -lt $DL ]; do sleep 20; done
echo "$(date '+%Y-%m-%d %H:%M:%S') deadline reached: stopping the queue" >> $SP/run/baselines/logs/queue.log
pkill -f "u queue[.]py"
sleep 1
pkill -f "scripts/scribe_eval/(baselines|judge)[.]py"
echo "$(date '+%Y-%m-%d %H:%M:%S') stopped" >> $SP/run/baselines/logs/queue.log
