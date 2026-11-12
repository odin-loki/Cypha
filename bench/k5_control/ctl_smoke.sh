#!/bin/bash
# ctl_smoke.sh BIN LABEL N [flags] -- cyphalm_trie sample on the first N dev prompts (no SCRIBE), for timing and the knob log
BIN=$1; LABEL=$2; N=$3; shift 3
D=~/ctl_runs/smoke; mkdir -p $D
av=$(free -g | awk '/^Mem:/{print $7}')
if [ "$av" -lt 16 ]; then echo "only ${av} GB available; not started" > $D/$LABEL.done; exit 3; fi
cd ~/ctl_src
export K5_THREADS=6
/usr/bin/time -v -o $D/$LABEL.time $BIN/cyphalm_trie sample /mnt/d/cypha_llm/enwik8 $D/$LABEL.gen.txt --prompts-file bench/eval/gen_prompts_dev.tsv \
    --plan-drop 75497472:100000000 --prompts $N --k5 /mnt/d/cypha_llm/fuzzy/train.txt "$@" > $D/$LABEL.out 2> $D/$LABEL.err
echo "rc=$? avail_at_start=${av}" > $D/$LABEL.done
