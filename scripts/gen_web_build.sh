#!/bin/bash
# GEN-W (the web benchmark) and C1 (clean C4 prose), docs/reports/GEN_WEB.md.
#
# usage: scripts/gen_web_build.sh STEP...    steps: extract clean vectors pool bench ladder c1prep c1 manifest (or all)
# Environment (defaults in brackets):
#   BIN     scribe, ppmi_svd and k5_overlap [~/genweb_bin]
#   C4      the C4 shards [/mnt/d/cypha_llm/corpus/articles/c4]
#   GENW    GEN-W's large files [/mnt/d/cypha_llm/genw]
#   C1DIR   C1's output [/mnt/d/cypha_llm/c4prose]
#   ENWIK8  [/mnt/d/cypha_llm/enwik8]; K5_TEXT [/mnt/d/cypha_llm/fuzzy/train.txt];
#   K5_MORE [/mnt/d/cypha_llm/k5scale/clean/train_all.txt]; WIKI_CLEAN (SCRIBE's cleaned Wikipedia pack and
#   reference/test text, scribe_pack.txt and scribe_reftest.txt) [/mnt/d/cypha_llm/k5scale/clean]
# At most four heavy processes run at once (the PC's disk is shared).
# Author: Odin Loch
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${BIN:-$HOME/genweb_bin}
C4=${C4:-/mnt/d/cypha_llm/corpus/articles/c4}
G=${GENW:-/mnt/d/cypha_llm/genw}
C=${C1DIR:-/mnt/d/cypha_llm/c4prose}
E8=${ENWIK8:-/mnt/d/cypha_llm/enwik8}
K5_TEXT=${K5_TEXT:-/mnt/d/cypha_llm/fuzzy/train.txt}
K5_MORE=${K5_MORE:-/mnt/d/cypha_llm/k5scale/clean/train_all.txt}
WC=${WIKI_CLEAN:-/mnt/d/cypha_llm/k5scale/clean}
P="python3 -I $ROOT/scripts/gen_web_prep.py"
SEED=20261112
shard() { echo "$C4/c4-train.$1-of-01024.jsonl"; }
mkdir -p "$G" "$C"

step_extract() {   # the four GEN-W regions, each from its own shard
  $P extract "$(shard 00738)" "$G/src_raw.jsonl" genw --docs 20000 &
  $P extract "$(shard 00737)" "$G/pack_raw.jsonl" genw_pack --bytes 41943040 &
  $P extract "$(shard 00736)" "$G/ref_raw.jsonl" genw_ref --docs 5000 &
  $P extract "$(shard 00735)" "$G/rungs_raw.jsonl" genw_rungs --docs 5000 &
  wait
}

step_clean() {     # SCRIBE's plain-text cleaner, then exact repeats of the cleaned text; line i is document i
  for f in src pack ref rungs; do
    "$BIN/scribe" docs "$G/${f}_raw.jsonl" --set "genw_$f" --out "$G/${f}_cl.jsonl"
    $P dedup "$G/${f}_cl.jsonl" "$G/$f.jsonl"
    $P lines "$G/$f.jsonl" "$G/$f.txt"
  done
}

step_vectors() {   # SCRIBE-W's word vectors, from the pack (as for WebText)
  python3 -I "$ROOT/scripts/scribe_eval/runs/v3_paper_tools/wrap.py" "$G/pack.jsonl" "$G/pack.xml"
  "$BIN/ppmi_svd" "$G/pack.xml" none none - "$G/vecW" --no-ri --no-ablation > "$G/vecW.log" 2>&1
}

step_pool() {
  "$BIN/scribe" prompts "raw:$G/src.jsonl" --n 100000 --seed $SEED --set genw --out "$G/pool.jsonl"
}

step_bench() {
  $P texts "$G/gena_texts.txt" "$ROOT/bench/eval/gen_prompts_dev.tsv" "$ROOT/bench/eval/gen_prompts_test.tsv"
  head -c 8388608 "$E8" > "$G/enwik8_0_8.raw"
  $P bench "$G/pool.jsonl" "$ROOT/bench/eval" 250 --train "$K5_TEXT" "$K5_MORE" "$G/enwik8_0_8.raw" \
     --scribe "$G/pack.txt" "$G/ref.txt" "$G/rungs.txt" "$G/gena_texts.txt" | tee "$G/bench.log"
  $P texts "$G/genw_texts.txt" "$ROOT/bench/eval/gen_web_dev.tsv" "$ROOT/bench/eval/gen_web_test.tsv"
}

step_ladder() {    # SCRIBE's control ladder on GEN-W's source region (S, without the test rows' documents) and the rungs (R)
  mkdir -p "$G/ladder"
  $P without "$G/src.jsonl" "$ROOT/bench/eval/gen_web_test.tsv" "$G/ladder_S.jsonl"
  for prof in v2 v31; do
    "$BIN/scribe" ladder --pack "raw:$G/pack.jsonl" --ref "raw:$G/ref.jsonl" --test "raw:$G/ladder_S.jsonl" --vectors "$G/vecW" \
      --profile $prof --seed $SEED --dedup --label "S_$prof" > "$G/ladder/S_$prof.md" 2> "$G/ladder/S_$prof.err" &
    "$BIN/scribe" ladder --pack "raw:$G/pack.jsonl" --ref "raw:$G/ref.jsonl" --test "raw:$G/rungs.jsonl" --vectors "$G/vecW" \
      --profile $prof --seed $SEED --dedup --label "R_$prof" > "$G/ladder/R_$prof.md" 2> "$G/ladder/R_$prof.err" &
  done
  wait
}

C1_SHARDS="00000 00001 00002"

step_c1prep() {    # C1's shards cleaned, one document per line (needs no GEN-W file)
  mkdir -p "$C/work"
  one() {
    local s=$1 W=$C/work
    $P extract "$(shard "$s")" "$W/c4_${s}_raw.jsonl" "c1_$s" --docs 100000000
    "$BIN/scribe" docs "$W/c4_${s}_raw.jsonl" --set "c1_$s" --out "$W/c4_${s}_cl.jsonl"
    $P dedup "$W/c4_${s}_cl.jsonl" "$W/c4_${s}.jsonl"
    $P lines "$W/c4_${s}.jsonl" "$W/c4_${s}.txt"
  }
  for s in $C1_SHARDS; do one $s > "$C/work/prep_$s.log" 2>&1 & done
  wait
}

step_c1() {        # C1: the overlap rule, joined and cut at 1.5 GB, checked again
  cat "$G/genw_texts.txt" "$G/gena_texts.txt" > "$C/work/forbid_bench.txt"
  cat "$G/pack.txt" "$G/ref.txt" "$G/rungs.txt" > "$C/work/forbid_scribe_web.txt"
  cat "$WC/scribe_pack.txt" "$WC/scribe_reftest.txt" > "$C/work/forbid_scribe_wiki.txt"
  for s in $C1_SHARDS; do
    "$BIN/k5_overlap" clean "$C/work/c4_${s}.txt" "$C/work/kept_${s}.txt" "$C/work/overlap_${s}.txt" \
      "$C/work/forbid_bench.txt" "$C/work/forbid_scribe_web.txt" "$C/work/forbid_scribe_wiki.txt" > "$C/work/overlap_$s.log" 2>&1 &
  done
  wait
  $P join "$C/c1_train.txt" 1500000000 "$C/work/kept_00000.txt" "$C/work/kept_00001.txt" "$C/work/kept_00002.txt" | tee "$C/join.log"
  # The second, independent check of the written file, in three line-split parts at once (one part takes ~20 min).
  ( cd "$C/work" && split -n l/3 -d ../c1_train.txt chk_ )
  for p in chk_00 chk_01 chk_02; do
    "$BIN/k5_overlap" check "$C/work/$p" "$C/work/forbid_bench.txt" "$C/work/forbid_scribe_web.txt" \
      "$C/work/forbid_scribe_wiki.txt" > "$C/work/$p.check" 2>&1 &
  done
  wait
  cat "$C"/work/chk_0?.check | tee "$C/recheck.txt"
  cat "$C"/work/chk_00 "$C"/work/chk_01 "$C"/work/chk_02 | cmp - "$C/c1_train.txt"
}

step_manifest() {
  ( cd "$G" && sha256sum src.jsonl pack.jsonl ref.jsonl rungs.jsonl pool.jsonl vecW.vectors.f32 vecW.vocab.tsv ladder_S.jsonl ) > "$G/sha256.txt"
  ( cd "$C" && sha256sum c1_train.txt ) > "$C/sha256.txt"
}

for s in "$@"; do
  if [ "$s" = all ]; then for t in extract clean vectors pool bench ladder c1prep c1 manifest; do echo "== $t"; "step_$t"; done
  else echo "== $s"; "step_$s"; fi
done
