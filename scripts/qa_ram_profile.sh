#!/bin/bash
# M18 (PLAN Track P): where qa's RAM peaks sit. Odin Loch, 2026-11-01.
# Runs three qa configurations, each traced every 0.25 s (RSS, high-water mark, last stderr line):
#   R1 built (C1L + defaults, --save), R2 served (--model, R1's store), R3 the frozen-test configuration.
# It changes no code and trains nothing. R3 repeats BKN2's frozen-test configuration only to measure its peak; its
# answers are BKN2's, already read, and the summary prints only the peak.
# Usage: bash scripts/qa_ram_profile.sh            (about 1-2 h)
# Settings: CYPHA_ENWIK8, CYPHA_DOLLY (found under /mnt/d/cypha_llm, ~ and /root if unset),
#           CYPHA_M18_OUT (default /mnt/d/cypha_llm/profile/m18), CYPHA_BUILD (default ~/cypha_build).
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
OUT=${CYPHA_M18_OUT:-/mnt/d/cypha_llm/profile/m18}
BLD=${CYPHA_BUILD:-$HOME/cypha_build}
find1() { find /mnt/d/cypha_llm "$HOME" /root -maxdepth 4 -type f -name "$1" 2>/dev/null | head -1; }
E=${CYPHA_ENWIK8:-$(find1 enwik8)}
D=${CYPHA_DOLLY:-$(find1 dolly.jsonl)}
[ -f "$E" ] || { echo "enwik8 not found: set CYPHA_ENWIK8"; exit 1; }
[ -f "$D" ] || { echo "dolly.jsonl not found: set CYPHA_DOLLY"; exit 1; }
mkdir -p "$OUT/in" "$BLD"
tar -xzf "$R/bench/eval/qa_inputs.tar.gz" -C "$OUT/in"
cmake -S "$R/native" -B "$BLD" -DCMAKE_BUILD_TYPE=Release > "$OUT/cmake.log" 2>&1
cmake --build "$BLD" --target cyphalm_trie -j"$(nproc)" >> "$OUT/cmake.log" 2>&1
B="$BLD/cyphalm_trie"
I=$OUT/in
C1LD="--dev $R/bench/eval/t_qa_wiki_dev.tsv --leak $R/bench/eval/t_qa_wiki.tsv --parse $I/parse_dev_span.tsv --dk $I/dk_mention.tsv --dk-ratio 0.20 --fam after --fam-stat ent_title --fam-train low --shard $I/shard_facts.tsv --base $I/a0_rows.tsv"
TEST="--dev $R/bench/eval/t_qa_wiki_dev.tsv --test $R/bench/eval/t_qa_wiki.tsv --parse $I/parse_all_span.tsv --dk $I/dk_mention.tsv --dk-ratio 0.20 --fam after --fam-stat ent_title --fam-train low --shard $I/shard_facts.tsv --base $I/a0_rows.tsv"
trace() {  # trace NAME cmd... : NAME.out, NAME.err, NAME.prof (s rss_kb hwm_kb | stderr), NAME.peak
  local p=$OUT/$1; shift
  local t0; t0=$(date +%s.%N)
  "$@" > "$p.out" 2> "$p.err" &
  local pid=$! hwm=0 s h
  : > "$p.prof"
  while kill -0 $pid 2>/dev/null; do
    s=$(awk '/VmRSS/{r=$2} /VmHWM/{h=$2} END{print r" "h}' /proc/$pid/status 2>/dev/null || true)
    h=${s#* }
    [ -n "$h" ] && [ "$h" -gt "$hwm" ] 2>/dev/null && hwm=$h
    echo "$(awk -v a="$(date +%s.%N)" -v b="$t0" 'BEGIN{printf "%.2f", a-b}') $s | $(tail -n1 "$p.err" 2>/dev/null | cut -c1-90)" >> "$p.prof"
    sleep 0.25
  done
  local rc=0; wait $pid || rc=$?
  echo "peak_rss_kb $hwm seconds $(awk -v a="$(date +%s.%N)" -v b="$t0" 'BEGIN{printf "%.2f", a-b}') rc $rc" > "$p.peak"
}
rm -rf "$OUT/store"
trace R1 "$B" qa "$E" "$D" $C1LD --save "$OUT/store" --out "$OUT/R1.rows.tsv"
trace R2 "$B" qa "$E" "$D" $C1LD --model "$OUT/store" --out "$OUT/R2.rows.tsv"
trace R3 "$B" qa "$E" "$D" $TEST --out "$OUT/R3.rows.tsv"
{
  echo "M18 qa RAM profile, $(date -u +%FT%TZ), $(git -C "$R" rev-parse --short HEAD), $(uname -n)"
  for n in R1 R2 R3; do
    pk=$(awk '{print $2}' "$OUT/$n.peak")
    echo "== $n: $(cat "$OUT/$n.peak") ($(awk -v k="$pk" 'BEGIN{printf "%.1f", k/1024}') MB)"
    echo "   first sample within 1 MB of the peak, and the phase (last stderr line) there:"
    awk -v pk="$pk" '$2+1024>=pk {print "   "$0; exit}' "$OUT/$n.prof"
    echo "   RSS at each new phase (first sample of each stderr line), largest 12:"
    awk -F'|' '{split($1,a," "); ph=$2; if(!(ph in seen)){seen[ph]=1; print a[2]"\t"a[1]"s\t"ph}}' "$OUT/$n.prof" | sort -rn | head -12 | sed 's/^/   /'
  done
  echo "== rows: R1 vs R2 identical? $(cmp -s "$OUT/R1.rows.tsv" "$OUT/R2.rows.tsv" && echo yes || echo NO)"
} > "$OUT/summary.txt"
cat "$OUT/summary.txt"
