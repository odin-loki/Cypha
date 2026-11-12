#!/bin/bash
# DECODE_LOOP.md: one candidate of the decode loop. Odin Loch, 2026-11-13.
#
#   bench/decode_loop/loop_arm.sh LABEL [cyphalm_trie sample flags...]
#
# Writes on the 150 loop prompts (subset A, then the 50 rater prompts; bench/decode_loop/loop_prompts.tsv), with the
# default writer's command (`sample --k5 /mnt/d/cypha_llm/fuzzy/train.txt`) plus the flags given. Then:
#   LABEL.gen.txt    every GEN line, PROMPT ids put back to GEN-A dev numbers (so blind_rate.py reads it);
#   LABEL.A.gen.txt  subset A only (the metrics); LABEL.R.gen.txt the 50 rater prompts only (the packets);
#   SCRIBE v2 and v3.1 on LABEL.A.gen.txt; fact precision (gen_fact_check.py run --v2) on it;
#   copy32 and verbatim13 from sample's GEN_EVAL line (all 150 prompts).
# Waits for >= 16 GB available before it starts; the sample runs under ~/heavy.sh (one of the PC's 2 heavy slots). Leaves LABEL.done holding the exit code.
#
# Environment: DLOOP_BIN (cyphalm_trie, scribe) [~/dloop_bin], DLOOP_RUNS [~/dloop_runs], K5_THREADS [6].
set -uo pipefail
LABEL=$1; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${DLOOP_BIN:-$HOME/dloop_bin}
R=${DLOOP_RUNS:-$HOME/dloop_runs}
E=/mnt/d/cypha_llm/enwik8
VEC=/mnt/d/cypha_llm/scribe/v32
mkdir -p "$R"
rm -f "$R/$LABEL.done"
while :; do
  av=$(free -g | awk '/^Mem:/{print $7}')
  [ "$av" -ge 16 ] && break
  sleep 30
done
echo "start $LABEL avail=${av}G $(date +%T) flags: $*" > "$R/$LABEL.arm.log"
export K5_THREADS=${K5_THREADS:-6}
cd "$ROOT"
"$HOME/heavy.sh" /usr/bin/time -v -o "$R/$LABEL.time" nice -n 5 "$BIN/cyphalm_trie" sample "$E" "$R/$LABEL.raw.gen.txt" \
    --prompts-file bench/decode_loop/loop_prompts.tsv --prompts 150 --plan-drop 75497472:100000000 \
    --k5 /mnt/d/cypha_llm/fuzzy/train.txt "$@" > "$R/$LABEL.sample.out" 2> "$R/$LABEL.sample.err"
rc=$?
if [ $rc = 0 ]; then
  python3 -I bench/decode_loop/loop_split.py "$R/$LABEL.raw.gen.txt" "$R/$LABEL" || rc=10
fi
if [ $rc = 0 ]; then
  nice -n 5 "$BIN/scribe" ladder --pack "enwik:$E:16:48" --ref "enwik:$E:48:56" --test "enwik:$E:56:64" --vectors "$VEC" \
      --profile v2 --seed 20261027 --gen-trie "$R/$LABEL.A.gen.txt" --label "$LABEL" > "$R/$LABEL.scribe.md" 2> "$R/$LABEL.scribe.err" || rc=11
  nice -n 5 "$BIN/scribe" ladder --pack "enwik:$E:16:48" --ref "enwik:$E:48:56" --test "enwik:$E:56:64" --vectors "$VEC" \
      --profile v31 --seed 20261027 --gen-trie "$R/$LABEL.A.gen.txt" --label "$LABEL" > "$R/$LABEL.scribe31.md" 2> "$R/$LABEL.scribe31.err" || true
  python3 -I scripts/gen_fact_check.py run --v2 --out "$R/facts_$LABEL" "$LABEL=$R/$LABEL.A.gen.txt" > "$R/$LABEL.facts.out" 2>&1 || rc=12
fi
echo "rc=$rc $LABEL $(date +%T)" >> "$R/$LABEL.arm.log"
grep -E "GEN_EVAL copy32" "$R/$LABEL.sample.out" >> "$R/$LABEL.arm.log"
grep -E "Elapsed|Maximum resident" "$R/$LABEL.time" >> "$R/$LABEL.arm.log"
echo $rc > "$R/$LABEL.done"
