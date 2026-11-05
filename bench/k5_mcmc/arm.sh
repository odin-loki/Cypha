#!/bin/bash
# One arm of docs/reports/K5_MCMC.md: scripts/gen_eval.sh on GEN-A dev (SCRIBE v2 and v3.1), K5 on its clean text,
# wall time and peak memory, the arm's GEN file and summary copied here, one row in $OUT/results.tsv.
# Usage: bench/k5_mcmc/arm.sh NAME [cyphalm_trie sample flags ...]   e.g. arm.sh R1 --k5-mcmc 1000
# Author: Odin Loch
set -euo pipefail
NAME=$1; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$HOME/cypha_build/mcmc}
OUT=${OUT:-/mnt/d/cypha_llm/k5mcmc}
export ENWIK8=${ENWIK8:-/mnt/d/cypha_llm/enwik8} WIKI=${WIKI:-/mnt/d/cypha_llm/enwik8} SCRIBE_VEC=${SCRIBE_VEC:-/mnt/d/cypha_llm/scribe/v32}
export K5_THREADS=${K5_THREADS:-8} CYPHALM_TRIE=$BIN/cyphalm_trie SCRIBE=$BIN/scribe
TRAIN=${TRAIN:-/mnt/d/cypha_llm/fuzzy/train.txt}
mkdir -p "$OUT"
/usr/bin/time -f "%e %M" -o "$OUT/$NAME.time" "$ROOT/scripts/gen_eval.sh" dev "$NAME" "$OUT" --prompts 250 --k5 "$TRAIN" "$@" > /dev/null
cp "$OUT/$NAME.gen.txt" "$OUT/$NAME.summary.md" "$ROOT/bench/k5_mcmc/"
python3 -I "$ROOT/bench/k5_mcmc/collect.py" "$OUT" "$NAME" "$*" | tee -a "$OUT/results.tsv"
