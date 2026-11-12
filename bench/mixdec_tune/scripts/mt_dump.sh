#!/bin/bash
cd ~/mt_src
CYPHA_MIXDEC_DUMP=$HOME/mt_meas/arts K5_THREADS=6 /usr/bin/time -v -o ~/mt_meas/dump.time ~/mt_bin1/cyphalm_trie sample /mnt/d/cypha_llm/enwik8 ~/mt_meas/dump.gen.txt --prompts-file bench/eval/gen_prompts_dev.tsv --plan-drop 75497472:100000000 --prompts 250 --bytes 100 --k5 /mnt/d/cypha_llm/fuzzy/train.txt --mixdec ad=1,artk=4 > ~/mt_meas/dump.out 2> ~/mt_meas/dump.err
echo done > ~/mt_meas/dump.done
