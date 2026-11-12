#!/bin/bash
# GEN-W's harness (docs/reports/GEN_WEB.md): scripts/gen_eval.sh's three steps on the web benchmark.
#
#   scripts/gen_eval_web.sh dev|test LABEL OUTDIR [cyphalm_trie sample flags...]
#
# 1. cyphalm_trie sample on bench/eval/gen_web_{dev,test}.tsv (250 prompts), flags passed through, as gen_eval.sh.
# 2. scribe ladder with SCRIBE-W: C4 pack, reference and rungs (shards 00737, 00736, 00735), vectors from the pack,
#    --dedup, seed 20261112; v2 (the score of record) and v3.1 beside it.
# 3. scripts/gen_eval_summary.py, OUTDIR/LABEL.summary.md.
#
# The test split is read once per change; log the read in docs/reports/GEN_WEB.md.
#
# Environment (defaults in brackets): CYPHALM_TRIE [native/build/cyphalm_trie], SCRIBE [native/build/scribe],
#   WIKI (the corpus `sample` trains on) [/root/cdata/enwik8], GENW (pack.jsonl, ref.jsonl, rungs.jsonl, vecW)
#   [/mnt/d/cypha_llm/genw].
#
# Author: Odin Loch
set -euo pipefail
if [ $# -lt 3 ]; then
    echo "usage: scripts/gen_eval_web.sh dev|test LABEL OUTDIR [cyphalm_trie sample flags...]" >&2
    exit 2
fi
SPLIT=$1; LABEL=$2; OUT=$3; shift 3
case "$SPLIT" in dev|test) ;; *) echo "split must be dev or test" >&2; exit 2;; esac
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TRIE=${CYPHALM_TRIE:-$ROOT/native/build/cyphalm_trie}
SC=${SCRIBE:-$ROOT/native/build/scribe}
WIKI=${WIKI:-/root/cdata/enwik8}
GW=${GENW:-/mnt/d/cypha_llm/genw}
PROMPTS=$ROOT/bench/eval/gen_web_$SPLIT.tsv
SW=(--pack "raw:$GW/pack.jsonl" --ref "raw:$GW/ref.jsonl" --test "raw:$GW/rungs.jsonl" --vectors "$GW/vecW" --dedup --seed 20261112)
mkdir -p "$OUT"
G=$OUT/$LABEL.gen.txt
if [ "$SPLIT" = test ]; then echo "gen_eval_web: TEST read ($LABEL); log it in docs/reports/GEN_WEB.md" >&2; fi
# --plan-drop as gen_eval.sh, so the planner reads the same facts on both benchmarks.
nice -n 5 "$TRIE" sample "$WIKI" "$G" --prompts-file "$PROMPTS" --plan-drop 75497472:100000000 "$@" > "$OUT/$LABEL.sample.out" 2> "$OUT/$LABEL.sample.err"
nice -n 5 "$SC" ladder "${SW[@]}" --profile v2 --gen-trie "$G" --label "$LABEL" > "$OUT/$LABEL.scribe.md" 2> "$OUT/$LABEL.scribe.err"
nice -n 5 "$SC" ladder "${SW[@]}" --profile v31 --gen-trie "$G" --label "$LABEL" > "$OUT/$LABEL.scribe31.md" 2> "$OUT/$LABEL.scribe31.err" || true
python3 -I "$ROOT/scripts/gen_eval_summary.py" "$OUT" "$LABEL" "$SPLIT" | tee "$OUT/$LABEL.summary.md"
