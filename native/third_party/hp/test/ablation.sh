#!/usr/bin/env bash
# GRIA ablation. The ONLY difference between the two runs is whether the
# alpha gate carries regime information or is pinned to a constant. Model
# set, parameter count, memory and every other gate are identical, so the
# difference is attributable.
set -u
HP=${HP:-/tmp/hp}
printf '%-28s %10s %10s %9s %8s\n' file gria_on gria_off delta pct
for f in "$@"; do
  [ -f "$f" ] || continue
  on=$( "$HP" c          "$f" /tmp/_ab.hp 2>&1 | grep -o 'out [0-9]*' | cut -d' ' -f2)
  off=$("$HP" c --no-gria "$f" /tmp/_ab.hp 2>&1 | grep -o 'out [0-9]*' | cut -d' ' -f2)
  d=$((off-on)); pct=$(echo "scale=3; 100*$d/$off" | bc)
  printf '%-28s %10d %10d %+9d %7s%%\n' "$(basename "$f")" "$on" "$off" "$d" "$pct"
done
