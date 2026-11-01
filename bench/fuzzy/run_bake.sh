#!/bin/bash
# The fuzzy bake-off (docs/GENERATION_PLAN.md §13.7-13.8) on one machine. Prep runs first. Then one job per arm,
# each pinned to its own core with taskset and one BLAS thread. Then the report. Odin Loch, 2026-11-01.
# Settings: FZ_WORK (default /mnt/d/cypha_llm/fuzzy), ENWIK8, SCRIBE, SCRIBE_VEC, PY (default python3),
#           FZ_CORE0 (the first core, default 40; arms use FZ_CORE0 .. FZ_CORE0+12).
# Outputs: FZ_WORK/summary.md, res_ARM.json, arm_ARM.log, gen/ARM.gen.txt and ARM.scribe.md.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export FZ_WORK=${FZ_WORK:-/mnt/d/cypha_llm/fuzzy}
PY=${PY:-python3}
C0=${FZ_CORE0:-40}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$FZ_WORK"
taskset -c "$C0" "$PY" -u "$here/fuzzy_bake.py" prep > "$FZ_WORK/prep.log" 2>&1
i=0
for a in K5 F1 F2 F3 F4 F5 F6 F7 F8a F8b F9 F10 ALL; do
    taskset -c $((C0 + i)) nohup "$PY" -u "$here/fuzzy_bake.py" arm "$a" > "$FZ_WORK/arm_$a.log" 2>&1 &
    i=$((i + 1))
done
wait
"$PY" "$here/fuzzy_bake.py" report
