#!/bin/bash
# One K5 arm of docs/reports/K5_UPGRADES.md: sample on GEN-A dev (or test with SPLIT=test), SCRIBE v2, one row in
# $OUT/results.tsv. Usage: arm.sh NAME [cyphalm_trie sample flags ...]   (e.g. arm.sh T09 --temps 0.9)
set -euo pipefail
NAME=$1; shift
BIN=${BIN:-$HOME/cypha_build/k5}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
E=${ENWIK8:-/mnt/d/cypha_llm/enwik8}
TRAIN=${TRAIN:-/mnt/d/cypha_llm/fuzzy/train.txt}
VEC=${VEC:-/mnt/d/cypha_llm/scribe/v32}
OUT=${OUT:-/mnt/d/cypha_llm/k5up}
SPLIT=${SPLIT:-dev}
mkdir -p "$OUT"; cd "$OUT"
if [ -z "${ROWONLY:-}" ]; then
"$BIN/cyphalm_trie" sample "$E" "$NAME.gen.txt" --prompts-file "$ROOT/bench/eval/gen_prompts_$SPLIT.tsv" --prompts 250 \
    --k5 "$TRAIN" "$@" > "$NAME.eval" 2> "$NAME.log"
"$BIN/scribe" ladder --pack "enwik:$E:16:48" --ref "enwik:$E:48:56" --test "enwik:$E:56:64" --vectors "$VEC" --profile v2 \
    --seed 20261027 --gen-trie "$NAME.gen.txt" --label "$NAME" > "$NAME.scribe.md" 2> "$NAME.scribe.err"
fi
# One temperature: the pooled row is that temperature's (scribe prints a T= row only for 0.6, 0.8 and 1.0).
row=$(grep -m1 "^| trie samples (pooled)" "$NAME.scribe.md")
sc=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$4); print $4}')
ss=$(echo "$row" | awk -F'|' '{gsub(/^ +| +$/,"",$6); print $6}')
bpb=$(grep -o "k5_cont_bpb [0-9.]*" "$NAME.eval" | awk '{print $2}')
cp=$(grep -o "copy32 T[0-9.]* [0-9.]*" "$NAME.eval" | awk '{print $3}')
v13=$(grep -o "verbatim13 [0-9.]* ([0-9]* of" "$NAME.eval" | awk '{print $3}' | tr -d '(')
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$SPLIT" "$sc" "${ss%% *}" "$cp" "$v13" "$bpb" "$*" >> results.tsv
printf "%s\t%s\tscribe %s\tS %s\tcopy32 %s\tv13 %s\tbpb %s\n" "$NAME" "$SPLIT" "$sc" "${ss%% *}" "$cp" "$v13" "$bpb"
