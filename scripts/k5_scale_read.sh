#!/bin/bash
# One K5 read for docs/reports/K5_SCALE.md: K5 trained on TEXT writes GEN-A (250 prompts of SPLIT) at seed SEED, and
# SCRIBE v2 scores it. Writes LABEL.gen.txt, LABEL.sample.log (K5's build line, copy32, verbatim13, /usr/bin/time -v)
# and LABEL.scribe.md into the current directory, and prints one summary line.
#
# usage: scripts/k5_scale_read.sh LABEL TEXT SEED SPLIT [extra cyphalm_trie flags, e.g. --k5-vmax 65536]
# env: BIN (cyphalm_trie and scribe), ENWIK8, VEC32. SPLIT is dev; test only for the one registered read.
#
# Author: Odin Loch
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LABEL=$1 TEXT=$2 SEED=$3 SPLIT=$4
shift 4
BIN=${BIN:-$HOME/cypha_build/k5scale} E=${ENWIK8:-/mnt/d/cypha_llm/enwik8} V=${VEC32:-/mnt/d/cypha_llm/scribe/v32}
/usr/bin/time -v "$BIN/cyphalm_trie" sample "$E" "$LABEL.gen.txt" --prompts-file "$ROOT/bench/eval/gen_prompts_$SPLIT.tsv" \
    --prompts 250 --k5 "$TEXT" --k5-seed "$SEED" "$@" > "$LABEL.sample.log" 2>&1
"$BIN/scribe" ladder --pack "enwik:$E:16:48" --ref "enwik:$E:48:56" --test "enwik:$E:56:64" --vectors "$V" --profile v2 \
    --seed 20261027 --gen-trie "$LABEL.gen.txt" --label "$LABEL" > "$LABEL.scribe.md" 2> "$LABEL.scribe.log"
sc=$(grep '^| trie samples T=1.0 | ' "$LABEL.scribe.md" | head -1 | cut -d'|' -f4 | xargs)
bl=$(grep -o 'vocab [0-9]*; entries.*built in [0-9.]* s' "$LABEL.sample.log" | head -1)
cp=$(grep -o 'copy32 T1.0 [0-9.]* verbatim13 [0-9.]* ([0-9]* of [0-9]*)' "$LABEL.sample.log")
rss=$(grep 'Maximum resident' "$LABEL.sample.log" | awk '{print $NF}')
wall=$(grep 'Elapsed (wall' "$LABEL.sample.log" | awk '{print $NF}')
unk=$(grep -o 'unk [0-9.]*' "$LABEL.sample.log" | head -1)
echo "READ $LABEL split=$SPLIT seed=$SEED scribe=$sc | $cp | $unk | $bl | peak_rss_kb=$rss wall=$wall"
