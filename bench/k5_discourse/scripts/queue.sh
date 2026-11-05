#!/bin/bash
# Runs queue.txt lines "LABEL flags..." in order, each once; waits for new lines until STOP exists.
cd /mnt/d/cypha_llm/disc
while [ ! -e STOP ]; do
  line=$(while read -r l; do lab=${l%% *}; [ -n "$lab" ] && [ ! -e out/$lab.done ] && [ ! -e out/$lab.started ] && { echo "$l"; break; }; done < queue.txt)
  if [ -z "$line" ]; then sleep 20; continue; fi
  lab=${line%% *}; flags=${line#* }; [ "$flags" = "$lab" ] && flags=""
  touch out/$lab.started
  ./run_arm.sh $lab /home/odin/cypha_build/disc/cyphalm_trie $flags
done
