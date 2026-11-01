#!/bin/bash
# The self-organising structure bench (docs/GENERATION_PLAN.md §13.9) on one machine: waits for the fuzzy bake-off to
# finish (its prep and K5 are shared; SO_NOWAIT=1 skips the wait), then one job per core, then the report.
# Odin Loch, 2026-11-01.
# Settings: FZ_WORK (default /mnt/d/cypha_llm/fuzzy), ENWIK8, SCRIBE, SCRIBE_VEC, PY, SO_SHIFT (plain text for SO5,
#           default /mnt/d/cypha_llm/profile/c4_heldout.txt, the unseen C4 shard), SO_CORE0 (default 53).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export FZ_WORK=${FZ_WORK:-/mnt/d/cypha_llm/fuzzy}
export SO_SHIFT=${SO_SHIFT:-/mnt/d/cypha_llm/profile/c4_heldout.txt}
PY=${PY:-python3}
C0=${SO_CORE0:-53}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "${SO_NOWAIT:-0}" != 1 ]; then
    while [ ! -f "$FZ_WORK/summary.md" ]; do sleep 120; done
fi
mkdir -p "$FZ_WORK/so"
head -c 8388608 "$SO_SHIFT" > "$FZ_WORK/so/shift.txt"
export SO_SHIFT=$FZ_WORK/so/shift.txt
i=0
for j in so1x so1k so2p so2m so3a so3k so4 curve1 curve2 curve4 so5; do
    taskset -c $((C0 + i)) nohup "$PY" -u "$here/so_bench.py" "$j" > "$FZ_WORK/so/job_$j.log" 2>&1 &
    i=$((i + 1))
done
wait
"$PY" "$here/so_bench.py" report
