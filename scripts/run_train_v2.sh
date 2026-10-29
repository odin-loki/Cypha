#!/bin/bash
# Second 1 TB run (docs/TRAIN_1TB_RUNBOOK.md). Trains the same model as the live run (scripts/run_train.sh and
# native/tools/trie_stream_train.cpp, both unchanged) on the same corpus files, with counts that halve instead of
# wrapping. Output: $CORPUS/trie/v2/ (checkpoint.bin, trained.txt, snapshots, done.txt). It stops when every
# finished corpus file is trained; run it again later to add newly finished files.
#
#   bash scripts/run_train_v2.sh            # paths below; override with CYPHA_REPO / CYPHA_CORPUS / CYPHA_CORE
set -euo pipefail
REPO=${CYPHA_REPO:-/mnt/c/Users/odinl/OneDrive/Desktop/Cypha}
CORPUS=${CYPHA_CORPUS:-/mnt/d/cypha_llm/corpus}
CORE=${CYPHA_CORE:-0}
SNAPSHOTS=${CYPHA_SNAPSHOTS:-8M,64M,1G,10G,100G}
cd "$REPO"
g++ -std=c++23 -O2 -I native/include -I native/third_party -o /tmp/trie_stream_train_v2 native/tools/trie_stream_train_v2.cpp
exec taskset -c "$CORE" /tmp/trie_stream_train_v2 "$CORPUS" --once --snapshots "$SNAPSHOTS" "$@"
