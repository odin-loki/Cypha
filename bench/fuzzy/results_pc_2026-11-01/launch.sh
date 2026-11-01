#!/bin/bash
# run_bake.sh's steps, unchanged except ALL starts after the other 12 arms finish (free RAM was 23 GB < the 25-30 GB peak).
set -euo pipefail
here=$HOME/fuzzy_wt/bench/fuzzy
export FZ_WORK=/mnt/d/cypha_llm/fuzzy ENWIK8=/mnt/d/cypha_llm/enwik8 SCRIBE=$HOME/cypha_build/nn_diag/scribe SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32
PY=$HOME/nn_venv/bin/python; C0=40
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c "$C0" "$PY" -u "$here/fuzzy_bake.py" prep > "$FZ_WORK/prep.log" 2>&1
i=0
for a in K5 F1 F2 F3 F4 F5 F6 F7 F8a F8b F9 F10; do
    taskset -c $((C0 + i)) nohup "$PY" -u "$here/fuzzy_bake.py" arm "$a" > "$FZ_WORK/arm_$a.log" 2>&1 &
    i=$((i + 1))
done
wait
taskset -c $((C0 + 12)) "$PY" -u "$here/fuzzy_bake.py" arm ALL > "$FZ_WORK/arm_ALL.log" 2>&1
"$PY" "$here/fuzzy_bake.py" report
echo "bake-off done $(date '+%F %T')"
