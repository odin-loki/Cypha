#!/bin/bash
# hp_harness.sh - measurement harness used for RECORD H33-H36.
#
#   source hp/tools/hp_harness.sh
#   hp_build myexe -DHP_SLOT_MAX=24 -DHP_LR1_SCALE=60
#   hp_run   myexe data/enwik8.2mb 22        # -> "bytes rss_kb ms"
#   hp_suite myexe 22                        # -> 12x256KB spread windows
#
# Calibrated screens (RECORD H34):
#   data/enwik8.2mb  @ SLOT_MAX=24   92 s  1.5 GB  Spearman +0.84, 13/13 signs
#   data/enwik8.8mb  @ SLOT_MAX=24  500 s  1.6 GB  Pearson  +0.96 vs SLOT_MAX=35
# Champ SLOT_MAX=35 peaks 13.3 GB RSS: needs >=16 GB.
# Windows: python3 - <<'E'  (see RECORD H34) or reuse data/win/w*.xml.
# hp measurement harness. Box: 4 cores / 15 GB. Champ cfg (SLOT_MAX=35) peaks 13.3 GB -> OOM here.
REPO=${HP_REPO:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}
H=${HP_HARNESS_TMP:-/tmp/hp_harness}; mkdir -p "$H"
BASE=$(grep -oE '\-DHP_[A-Z0-9_]+=[0-9]+' $REPO/hp/tools/v78_flags.ps1 | grep -v 'HP_SLOT_MAX' | tr '\n' ' ')
export BASE
hp_build() { local n=$1; shift
  g++ -O3 -std=c++17 -msse4.1 -I $REPO/hp/include -I $REPO/hp/third_party/xsimd/include \
      $BASE "$@" $REPO/hp/src/main.cpp -o $REPO/hp/build/hp_$n.exe 2>&1 | head -5
  [ -x $REPO/hp/build/hp_$n.exe ] && echo "BUILD_OK $n" || echo "BUILD_FAIL $n"; }
# hp_run <exe> <input> <mem> [extra args] -> "bytes rss_kb ms"   (unique temp out; safe in parallel)
hp_run() { local e=$1 i=$2 m=$3; shift 3
  local o; o=$(mktemp "$H/out.XXXXXXXX.hp")
  local t0=$(date +%s%N)
  "$REPO/hp/build/hp_$e.exe" c --mem "$m" "$@" "$i" "$o" >/dev/null 2>&1 & local p=$!
  local mx=0 r
  while kill -0 $p 2>/dev/null; do r=$(awk '/VmHWM/{print $2}' /proc/$p/status 2>/dev/null); [ -n "$r" ] && [ "$r" -gt "$mx" ] && mx=$r; sleep 0.2; done
  wait $p; local rc=$?
  local t1=$(date +%s%N) b=0
  [ $rc -eq 0 ] && b=$(stat -c%s "$o" 2>/dev/null || echo 0)
  rm -f "$o"
  echo "$b $mx $(( (t1-t0)/1000000 ))"; }
# hp_suite <exe> <mem> -> "total rss_kb ms" over the 12 spread 256KB windows
hp_suite() { local e=$1 m=$2 tot=0 mx=0 ms=0 b r s
  for w in $REPO/data/win/w*.xml; do
    read b r s <<< "$(hp_run $e "$w" $m)"
    [ "$b" -eq 0 ] && { echo "0 0 0"; return 1; }
    tot=$((tot+b)); [ "$r" -gt "$mx" ] && mx=$r; ms=$((ms+s))
  done
  echo "$tot $mx $ms"; }
