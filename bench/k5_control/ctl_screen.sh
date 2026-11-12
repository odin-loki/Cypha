#!/bin/bash
# ctl_screen.sh -- K5_CONTROL's screen measures on the dev runs in ~/ctl_runs (written to bench/k5_control/screen/)
set -uo pipefail
cd ~/ctl_src
R=~/ctl_runs; O=bench/k5_control/screen; mkdir -p $O
T8=bench/k5_decode/runs/T8.gen.txt
A=(); L=()
for x in CL CW CLW; do
  [ -f $R/$x.gen.txt ] && A+=("$x=$R/$x.gen.txt")
  [ -f $R/${x}_long.gen.txt ] && L+=("${x}_long=$R/${x}_long.gen.txt")
done
python3 -I bench/k5_control/ctl_measures.py --cut 460 --human T8=$T8 "${A[@]}" --json $O/props460.json > $O/props460.txt
python3 -I bench/k5_control/ctl_measures.py --cut 1500 --human T8_long=$R/T8_long.gen.txt "${L[@]}" --json $O/props1500.json > $O/props1500.txt
python3 -I bench/k5_control/ctl_measures.py --cut 1500 --ids bench/blind/coh_long/prompts.json --human T8_long=$R/T8_long.gen.txt "${L[@]}" > $O/props1500_coh25.txt
python3 -I bench/para_plan/hook_screen.py --surprise scripts/surprise_profile.py --train /mnt/d/cypha_llm/fuzzy/train.txt --out $O/hook T8=$T8 "${A[@]}" 2> $O/hook.err
python3 -I scripts/gen_fact_check.py run --v2 --out $O/facts T8=$T8 "${A[@]}" > $O/facts.log 2>&1
{
  echo "| arm | controlled sentences | temperature mean | at 0.70 | at 0.90 | rel mean | wl mean | wl at +1 | wl at 0 | wl below 0 |"
  echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
  for x in CL CW CLW CL_long CW_long CLW_long; do
    f=$R/$x.sample.err; [ -f $f ] || continue
    grep '^K5CTL' $f | awk -F'\t' -v n=$x '{c++; t+=$3; r+=$4; w+=$5; if($3<=0.7001)tl++; if($3>=0.8999)th++; if($5>=0.9999)wh++; if($5==0)w0++; if($5<0)wn++}
      END{printf "| %s | %d | %.3f | %.2f | %.2f | %.3f | %.3f | %.2f | %.2f | %.2f |\n", n, c, t/c, tl/c, th/c, r/c, w/c, wh/c, w0/c, wn/c}'
  done
} > $O/knobs.md
for x in OFF T8_long CL CL_long CW CW_long CLW CLW_long; do
  echo "== $x"; grep -h 'GEN_EVAL copy32' $R/$x.sample.out 2>/dev/null; grep -hE 'Elapsed|Maximum resident' $R/$x.time 2>/dev/null
done > $O/copy_time.txt
echo screen done
