#!/bin/bash
# Shared by scripts/train_model.sh, verify_model.sh and compare_models.sh (sourced, not run). docs/TRAIN_1TB_RUNBOOK.md
# Models are numbered: model 1 is the live run (<corpus>/trie/checkpoint.bin, scripts/run_train.sh, unchanged);
# model N (2 or more) is <corpus>/trie/model<N>/checkpoint.bin. Reports go to <corpus>/trie/reports/.
REPO=${CYPHA_REPO:-/mnt/c/Users/odinl/OneDrive/Desktop/Cypha}
CORPUS=${CYPHA_CORPUS:-/mnt/d/cypha_llm/corpus}
REPORTS=${CYPHA_REPORTS:-$CORPUS/trie/reports}

model_dir() { if [ "$1" = 1 ]; then echo "$CORPUS/trie"; else echo "$CORPUS/trie/model$1"; fi; }
model_ckpt() { echo "$(model_dir "$1")/checkpoint.bin"; }

need_number() {  # need_number N MIN
  if ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -lt "$2" ]; then
    echo "bad model number '$1' (model 1 is the live run; trained models are 2 or more)"
    exit 2
  fi
}

build_trainer() {
  (cd "$REPO" && g++ -std=c++23 -O2 -pthread -I native/include -I native/third_party \
     -o /tmp/cypha_train_model native/tools/trie_stream_train_v2.cpp) || { echo "trainer build failed"; exit 1; }
}

build_checker() {
  (cd "$REPO" && g++ -std=c++23 -O2 -pthread -I native/include -I native/third_party \
     -o /tmp/cypha_trie_ckpt native/tools/trie_ckpt.cpp) || { echo "checker build failed"; exit 1; }
  mkdir -p "$REPORTS"
  /tmp/cypha_trie_ckpt selftest > "$REPORTS/selftest.txt" 2>&1 || { cat "$REPORTS/selftest.txt"; echo "checker selftest failed"; exit 1; }
}

# Held-out text: CYPHA_HOLDOUTS (FILE or FILE:FROM:LEN, space-separated), else every *.txt in <corpus>/../holdout;
# plus enwik8's 8 MiB-protocol holdout when CYPHA_ENWIK8 names the enwik8 file.
holdouts() {
  local h=${CYPHA_HOLDOUTS:-}
  if [ -z "$h" ] && [ -d "$CORPUS/../holdout" ]; then
    for f in "$CORPUS"/../holdout/*.txt; do [ -f "$f" ] && h="$h $f"; done
  fi
  [ -n "${CYPHA_ENWIK8:-}" ] && h="$h ${CYPHA_ENWIK8}:7340032:1048576"
  echo "$h"
}
