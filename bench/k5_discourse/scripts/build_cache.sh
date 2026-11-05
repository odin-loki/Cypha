#!/bin/bash
# usage: build_cache.sh TEXT K ITERS : trains (and caches) the entity grid and LDA K on TEXT
TEXT=$1 K=$2 IT=$3
tag=$(basename $TEXT)_K${K}_it${IT}
K5_THREADS=1 /usr/bin/time -v -o /mnt/d/cypha_llm/disc/cache/$tag.time ~/cypha_build/disc/cyphalm_trie sample /mnt/d/cypha_llm/enwik8 /mnt/d/cypha_llm/disc/cache/$tag.gen.txt --prompts-file /mnt/d/cypha_llm/disc/chk/p12.tsv --prompts 1 --k5 /mnt/d/cypha_llm/fuzzy/train.txt --k5-lda $K --k5-lda-iters $IT --k5-lda-w 1 --k5-eg 1 --k5-disc-text $TEXT --k5-disc-cache /mnt/d/cypha_llm/disc/cache > /mnt/d/cypha_llm/disc/cache/$tag.log 2>&1
echo DONE >> /mnt/d/cypha_llm/disc/cache/$tag.log
